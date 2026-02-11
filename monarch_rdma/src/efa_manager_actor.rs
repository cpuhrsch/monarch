/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! # EFA Manager Actor
//!
//! Manages EFA (Elastic Fabric Adapter) connections and operations using `hyperactor`
//! for asynchronous messaging.
//!
//! ## Architecture
//!
//! `EfaManagerActor` is a per-host entity that:
//! - Manages an EFA endpoint for RDMA-like operations
//! - Handles connection setup with peer EfaManagerActors
//! - Manages memory registration and data transfer
//!
//! ## Core Operations
//!
//! - Memory region registration
//! - Peer connection establishment
//! - RDMA-like read/write operations
//! - Completion notification via tagged send/recv
//!
//! ## Differences from RdmaManagerActor
//!
//! - Uses libfabric EFA instead of ibverbs
//! - Point-to-point connections instead of queue pairs
//! - Tagged messaging for completion notification instead of CQ polling

use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::Context;
use hyperactor::HandleClient;
use hyperactor::Handler;
use hyperactor::Instance;
use hyperactor::OncePortRef;
use hyperactor::RefClient;
use hyperactor::RemoteSpawn;
use hyperactor::supervision::ActorSupervisionEvent;
use serde::Deserialize;
use serde::Serialize;
use typeuri::Named;

use crate::efa_primitives::EfaEndpoint;
use crate::efa_components::EfaBuffer;
use crate::efa_supported;

/// Yield control back to the async runtime, allowing other tasks
/// (including hyperactor session heartbeats) to make progress.
///
/// This is equivalent to `tokio::task::yield_now()` but uses only
/// `std::future` primitives to avoid a direct tokio dependency.
async fn async_yield_now() {
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    struct YieldNow(bool);

    impl Future for YieldNow {
        type Output = ();
        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
            if self.0 {
                Poll::Ready(())
            } else {
                self.0 = true;
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }

    YieldNow(false).await
}

/// Poll the EFA completion queue with spin-polling and async yielding.
///
/// Spins for up to `SPINS_BEFORE_YIELD` iterations before yielding to the
/// async runtime, keeping latency low while allowing other tasks (like
/// hyperactor session heartbeats) to make progress.
async fn poll_for_completion(
    endpoint: &EfaEndpoint,
    timeout: u64,
    operation: &str,
) -> Result<i32, anyhow::Error> {
    let timeout_duration = Duration::from_secs(timeout);
    let start_time = std::time::Instant::now();
    const SPINS_BEFORE_YIELD: u32 = 10_000;
    let mut spin_count: u32 = 0;
    loop {
        let completions = endpoint.poll_cq(0).map_err(|e| {
            anyhow::anyhow!("Failed to poll for {} completion: {}", operation, e)
        })?;
        if completions > 0 {
            return Ok(completions);
        }
        spin_count += 1;
        if start_time.elapsed() >= timeout_duration {
            return Err(anyhow::anyhow!(
                "Timeout waiting for {} completion",
                operation
            ));
        }
        if spin_count >= SPINS_BEFORE_YIELD {
            spin_count = 0;
            async_yield_now().await;
        }
    }
}

/// Messages handled by EfaManagerActor
#[derive(Handler, HandleClient, RefClient, Debug, Serialize, Deserialize, Named)]
pub enum EfaManagerMessage {
    /// Request a buffer to be registered with the EFA endpoint
    RequestBuffer {
        addr: usize,
        size: usize,
        #[reply]
        reply: OncePortRef<EfaBuffer>,
    },
    /// Release a previously registered buffer
    ReleaseBuffer {
        buffer: EfaBuffer,
    },
    /// Read data from a remote peer into a local buffer
    ReadFromPeer {
        local_mr_id: usize,
        size: usize,
        remote_buffer: EfaBuffer,
        timeout: u64,
        #[reply]
        reply: OncePortRef<bool>,
    },
    /// Write data from a local buffer to a remote peer (fire-and-forget)
    ///
    /// This is a one-way message with no reply. The caller should poll for
    /// the tagged receive completion instead of waiting for a response.
    /// This is necessary because EFA requires FI_PROGRESS_MANUAL, so both
    /// sides need to be actively polling their CQs for progress.
    WriteToPeer {
        local_mr_id: usize,
        size: usize,
        remote_buffer: EfaBuffer,
        timeout: u64,
        /// Tag for completion notification (must match the tag used by receiver's trecv)
        notification_tag: u64,
    },
}
wirevalue::register_type!(EfaManagerMessage);

/// Internal representation of a registered memory region
#[derive(Debug)]
struct MemoryRegion {
    addr: usize,
    size: usize,
    key: u64,
}

/// EFA Manager Actor - manages EFA endpoint and operations for a single host
#[derive(Debug)]
#[hyperactor::export(
    spawn = true,
    handlers = [
        EfaManagerMessage,
    ],
)]
pub struct EfaManagerActor {
    /// The EFA endpoint for this host
    endpoint: Option<EfaEndpoint>,

    /// Registered memory regions: mr_id -> MemoryRegion
    memory_regions: HashMap<usize, MemoryRegion>,

    /// Next memory region ID
    next_mr_id: usize,

    /// Tag counter for tagged messaging
    next_tag: u64,

    /// Cached peer addresses: raw endpoint addr -> fi_addr_t handle
    known_peers: HashMap<Vec<u8>, u64>,

    /// Cached local endpoint address to avoid per-transfer allocation
    cached_local_addr: Vec<u8>,
}

impl Drop for EfaManagerActor {
    fn drop(&mut self) {
        // Deregister all memory regions
        if let Some(ref endpoint) = self.endpoint {
            for (mr_id, mr) in self.memory_regions.drain() {
                if let Err(e) = endpoint.deregister_mr(mr.key) {
                    tracing::error!(
                        "Failed to deregister MR {} with key {}: {}",
                        mr_id,
                        mr.key,
                        e
                    );
                }
            }
        }
    }
}

#[async_trait]
impl RemoteSpawn for EfaManagerActor {
    type Params = ();

    async fn new(_params: Self::Params) -> Result<Self, anyhow::Error> {
        if !efa_supported() {
            return Err(anyhow::anyhow!(
                "Cannot create EfaManagerActor because EFA is not supported on this machine"
            ));
        }

        let endpoint = EfaEndpoint::new("efa", 0).map_err(|e| {
            anyhow::anyhow!("Failed to create EFA endpoint: {}", e)
        })?;

        // Cache the local endpoint address once at creation time
        let cached_local_addr = endpoint.get_local_addr().map_err(|e| {
            anyhow::anyhow!("Failed to get local endpoint address: {}", e)
        })?;

        tracing::info!("EfaManagerActor created with EFA endpoint");

        Ok(Self {
            endpoint: Some(endpoint),
            memory_regions: HashMap::new(),
            next_mr_id: 1, // Start at 1 so mr_id=0 is never used (0 means "not specified")
            next_tag: 0,
            known_peers: HashMap::new(),
            cached_local_addr,
        })
    }
}

#[async_trait]
impl Actor for EfaManagerActor {
    async fn init(&mut self, _this: &Instance<Self>) -> Result<(), anyhow::Error> {
        Ok(())
    }

    async fn handle_supervision_event(
        &mut self,
        _cx: &Instance<Self>,
        event: &ActorSupervisionEvent,
    ) -> Result<bool, anyhow::Error> {
        tracing::error!("EfaManagerActor supervision event: {:?}", event);
        tracing::error!("EfaManagerActor error occurred, stopping worker process");
        std::process::exit(1);
    }
}

#[async_trait]
#[hyperactor::forward(EfaManagerMessage)]
impl EfaManagerMessageHandler for EfaManagerActor {
    /// Registers a memory region and returns an EfaBuffer handle.
    /// If the same (addr, size) is already registered, returns the cached handle.
    async fn request_buffer(
        &mut self,
        cx: &Context<Self>,
        addr: usize,
        size: usize,
    ) -> Result<EfaBuffer, anyhow::Error> {
        let endpoint = self.endpoint.as_ref().ok_or_else(|| {
            anyhow::anyhow!("EFA endpoint not initialized")
        })?;

        // Check if we already have an MR for this exact (addr, size)
        for (&mr_id, mr) in &self.memory_regions {
            if mr.addr == addr && mr.size == size {
                tracing::info!(
                    "[EFA] request_buffer: CACHED mr_id={}, addr=0x{:x}, key={}",
                    mr_id, addr, mr.key
                );
                return Ok(EfaBuffer {
                    owner: cx.bind().clone(),
                    mr_id,
                    endpoint_addr: self.cached_local_addr.clone(),
                    mr_addr: addr,
                    mr_size: size,
                    mr_key: mr.key,
                });
            }
        }

        // Register new memory region
        let key = endpoint.register_mr(addr, size).map_err(|e| {
            anyhow::anyhow!("Failed to register memory region: {}", e)
        })?;

        let mr_id = self.next_mr_id;
        self.next_mr_id += 1;

        self.memory_regions.insert(
            mr_id,
            MemoryRegion { addr, size, key },
        );

        let buffer = EfaBuffer {
            owner: cx.bind().clone(),
            mr_id,
            endpoint_addr: self.cached_local_addr.clone(),
            mr_addr: addr,
            mr_size: size,
            mr_key: key,
        };
        tracing::info!(
            "[EFA] request_buffer: mr_id={}, addr=0x{:x}, size={}, key={}",
            mr_id, addr, size, key
        );
        Ok(buffer)
    }

    /// Deregisters a memory region.
    async fn release_buffer(
        &mut self,
        _cx: &Context<Self>,
        buffer: EfaBuffer,
    ) -> Result<(), anyhow::Error> {
        tracing::info!(
            "[EFA] release_buffer called: mr_id={}, addr=0x{:x}, key={}",
            buffer.mr_id, buffer.mr_addr, buffer.mr_key
        );
        if let Some(mr) = self.memory_regions.remove(&buffer.mr_id) {
            if let Some(ref endpoint) = self.endpoint {
                endpoint.deregister_mr(mr.key).map_err(|e| {
                    anyhow::anyhow!("Failed to deregister memory region: {}", e)
                })?;
            }
            tracing::debug!("Released EFA buffer: mr_id={}", buffer.mr_id);
        }
        Ok(())
    }

    /// Coordinates an RDMA transfer. This handler runs on the SOURCE actor.
    ///
    /// Called via source_buffer.owner.read_from_peer(...) from efa_components.rs.
    /// local_mr_id = source's data MR (LOCAL to us)
    /// remote_buffer = dest's recv buffer (REMOTE)
    ///
    /// Flow:
    /// 1. Tell dest to post trecv + poll (write_to_peer, fire-and-forget)
    /// 2. Insert dest's peer address (cached)
    /// 3. fi_write FROM our data TO dest's recv buffer
    /// 4. Poll write CQ (EFA needs both sides polling for progress)
    /// 5. fi_tsend completion notification to dest
    /// 6. Fast-poll tsend CQ (1-byte message, completes in microseconds)
    async fn read_from_peer(
        &mut self,
        cx: &Context<Self>,
        local_mr_id: usize,
        size: usize,
        remote_buffer: EfaBuffer,
        timeout: u64,
    ) -> Result<bool, anyhow::Error> {
        let local_mr = self.memory_regions.get(&local_mr_id).ok_or_else(|| {
            anyhow::anyhow!("Local memory region {} not found", local_mr_id)
        })?;

        let endpoint = self.endpoint.as_ref().ok_or_else(|| {
            anyhow::anyhow!("EFA endpoint not initialized")
        })?;

        let tag = self.next_tag;
        self.next_tag += 1;

        let our_buffer = EfaBuffer {
            owner: cx.bind().clone(),
            mr_id: local_mr_id,
            endpoint_addr: self.cached_local_addr.clone(),
            mr_addr: local_mr.addr,
            mr_size: local_mr.size,
            mr_key: local_mr.key,
        };

        // Tell dest to start CQ polling FIRST. The dest must be polling
        // for fi_write to make progress (EFA FI_PROGRESS_MANUAL).
        remote_buffer.owner.write_to_peer(
            cx,
            0,
            size,
            our_buffer,
            timeout,
            tag,
        ).await?;

        // Insert dest's peer address (get or reuse fi_addr_t handle)
        let peer = if let Some(&fi_addr) = self.known_peers.get(&remote_buffer.endpoint_addr) {
            fi_addr
        } else {
            let fi_addr = endpoint.insert_peer_addr(&remote_buffer.endpoint_addr).map_err(|e| {
                anyhow::anyhow!("Failed to insert peer address: {}", e)
            })?;
            self.known_peers.insert(remote_buffer.endpoint_addr.clone(), fi_addr);
            fi_addr
        };

        // fi_write: push FROM our local data TO dest's recv buffer
        endpoint.write(
            local_mr.addr,
            size,
            remote_buffer.mr_addr as u64,
            remote_buffer.mr_key,
            peer,
        ).map_err(|e| {
            anyhow::anyhow!("Failed to perform EFA write: {}", e)
        })?;

        // Poll for write completion
        poll_for_completion(endpoint, timeout, "write").await?;

        // Send completion notification to dest
        endpoint.tsend(tag, peer).map_err(|e| {
            anyhow::anyhow!("Failed to send completion notification: {}", e)
        })?;

        // Poll for tsend completion
        poll_for_completion(endpoint, timeout, "tsend").await?;

        Ok(true)
    }

    /// Destination-side handler: post trecv and poll for source's notification.
    /// EFA requires both sides to be actively polling for fi_write to complete.
    async fn write_to_peer(
        &mut self,
        _cx: &Context<Self>,
        _local_mr_id: usize,
        _size: usize,
        remote_buffer: EfaBuffer,
        timeout: u64,
        notification_tag: u64,
    ) -> Result<(), anyhow::Error> {
        let endpoint = self.endpoint.as_ref().ok_or_else(|| {
            anyhow::anyhow!("EFA endpoint not initialized")
        })?;

        // Insert source's peer address (get or reuse fi_addr_t handle)
        if !self.known_peers.contains_key(&remote_buffer.endpoint_addr) {
            let fi_addr = endpoint.insert_peer_addr(&remote_buffer.endpoint_addr).map_err(|e| {
                anyhow::anyhow!("Failed to insert peer address: {}", e)
            })?;
            self.known_peers.insert(remote_buffer.endpoint_addr.clone(), fi_addr);
        }

        // Post trecv and poll for source's completion notification
        endpoint.trecv(notification_tag).map_err(|e| {
            anyhow::anyhow!("Failed to post trecv: {}", e)
        })?;

        poll_for_completion(endpoint, timeout, "trecv").await?;

        Ok(())
    }
}
