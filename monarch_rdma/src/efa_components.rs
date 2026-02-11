/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! # EFA Components
//!
//! This module provides the core EFA (Elastic Fabric Adapter) building blocks
//! for establishing and managing RDMA-like connections on AWS EFA instances.
//!
//! ## Core Components
//!
//! * `EfaBuffer` - Serializable buffer handle for EFA RDMA operations
//!
//! ## EFA vs ibverbs
//!
//! Unlike ibverbs which uses queue pairs for many-to-many communication,
//! EFA uses libfabric endpoints with explicit peer connections (point-to-point).
//! Notifications are done via tagged send/recv (tsend/trecv) rather than
//! completion queue polling.

use hyperactor::ActorRef;
use hyperactor::context;
use serde::Deserialize;
use serde::Serialize;
use typeuri::Named;

use crate::EfaManagerActor;
use crate::efa_manager_actor::EfaManagerMessageClient;

/// Represents a reference to a remote EFA buffer that can be accessed via RDMA-like operations.
///
/// This struct encapsulates all the information needed to identify and access a memory region
/// on a remote host using EFA. Unlike the raw `EfaEndpoint`, this struct is serializable
/// and can be sent to remote actors.
///
/// # Fields
///
/// * `owner` - Reference to the EfaManagerActor that owns this buffer
/// * `mr_id` - Memory region ID for deregistration
/// * `endpoint_addr` - Serialized endpoint address for connection exchange
/// * `mr_addr` - Memory region address
/// * `mr_size` - Memory region size
/// * `mr_key` - Memory registration key for RDMA access
#[derive(Debug, Serialize, Deserialize, Named, Clone)]
pub struct EfaBuffer {
    pub owner: ActorRef<EfaManagerActor>,
    pub mr_id: usize,
    pub endpoint_addr: Vec<u8>,
    pub mr_addr: usize,
    pub mr_size: usize,
    pub mr_key: u64,
}
wirevalue::register_type!(EfaBuffer);

impl EfaBuffer {
    /// Returns the size of the buffer in bytes.
    pub fn size(&self) -> usize {
        self.mr_size
    }

    /// Read from the EfaBuffer into the provided local memory.
    ///
    /// This method transfers data from the remote buffer into the local memory region.
    /// The operation is coordinated through the EfaManagerActor system.
    ///
    /// # Arguments
    /// * `client` - The actor performing the read
    /// * `local_buffer` - Local EfaBuffer to read into
    /// * `timeout` - Timeout in seconds for the operation
    ///
    /// # Returns
    /// `Ok(())` if the operation completed successfully.
    pub async fn read_into(
        &self,
        client: &impl context::Actor,
        local_buffer: EfaBuffer,
        timeout: u64,
    ) -> Result<(), anyhow::Error> {
        tracing::debug!(
            "[efa_buffer] reading from {:?} into local ({:?})",
            self.owner.actor_id(),
            local_buffer.owner.actor_id(),
        );

        // The local buffer's owner performs the read operation
        // It needs to:
        // 1. Connect to the remote peer (self.owner) if not already connected
        // 2. Have the remote peer write data to our buffer
        // 3. Wait for completion notification

        local_buffer
            .owner
            .read_from_peer(
                client,
                local_buffer.mr_id,
                local_buffer.mr_size,
                self.clone(),
                timeout,
            )
            .await?;

        Ok(())
    }

    /// Write from local memory into the EfaBuffer.
    ///
    /// This method transfers data from the local memory region to the remote buffer.
    /// The operation is coordinated through the EfaManagerActor system.
    ///
    /// # Arguments
    /// * `client` - The actor performing the write
    /// * `source_buffer` - Source EfaBuffer to write from
    /// * `timeout` - Timeout in seconds for the operation
    ///
    /// # Returns
    /// `Ok(())` if the operation completed successfully.
    pub async fn write_from(
        &self,
        client: &impl context::Actor,
        source_buffer: EfaBuffer,
        timeout: u64,
    ) -> Result<(), anyhow::Error> {
        tracing::info!(
            "[efa_components] write_from: dest_owner={:?}, dest_mr_id={}, source_owner={:?}, source_mr_id={}",
            self.owner.actor_id(),
            self.mr_id,
            source_buffer.owner.actor_id(),
            source_buffer.mr_id,
        );

        tracing::info!("[efa_components] write_from: calling source_buffer.owner.read_from_peer");
        source_buffer.owner
            .read_from_peer(
                client,
                source_buffer.mr_id,
                self.mr_size.min(source_buffer.mr_size),
                self.clone(),
                timeout,
            )
            .await?;

        Ok(())
    }

    /// Drop the buffer and release remote handles.
    ///
    /// This method calls the owning EfaManagerActor to release the buffer and clean up
    /// associated memory regions.
    ///
    /// # Arguments
    /// * `client` - The actor requesting the release
    ///
    /// # Returns
    /// `Ok(())` if the operation completed successfully.
    pub async fn drop_buffer(&self, client: &impl context::Actor) -> Result<(), anyhow::Error> {
        tracing::debug!("[efa_buffer] dropping buffer {:?}", self);
        self.owner.release_buffer(client, self.clone()).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_efa_buffer_serialization() {
        // EfaBuffer should be serializable/deserializable
        // This is a compile-time check more than a runtime test
        let _: fn() = || {
            fn assert_serde<T: Serialize + for<'de> Deserialize<'de>>() {}
            assert_serde::<EfaBuffer>();
        };
    }
}
