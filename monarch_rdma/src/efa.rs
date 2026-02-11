/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Safe Rust wrappers for EFA (Elastic Fabric Adapter) operations.
//!
//! This module provides safe wrappers around the C EFA functions from rdmaxcel_sys,
//! enabling RDMA-like operations on AWS EFA instances where traditional ibverbs
//! is not available.

use std::ffi::CString;

use rdmaxcel_sys::{
    rdmaxcel_efa_available, rdmaxcel_efa_deregister_mr,
    rdmaxcel_efa_ep_create, rdmaxcel_efa_ep_destroy, rdmaxcel_efa_ep_t,
    rdmaxcel_efa_error_string, rdmaxcel_efa_get_local_addr,
    rdmaxcel_efa_insert_peer_addr, rdmaxcel_efa_poll_cq, rdmaxcel_efa_read,
    rdmaxcel_efa_register_mr, rdmaxcel_efa_write,
    rdmaxcel_efa_tsend, rdmaxcel_efa_trecv,
};

// EFA error code constants (matching rdmaxcel_efa.h)
const EFA_SUCCESS: i32 = 0;
const EFA_ERROR_NOT_AVAILABLE: i32 = -1;
const EFA_ERROR_INVALID_PARAMS: i32 = -14;
const EFA_ERROR_EP_FAILED: i32 = -5;

/// Maximum size for EFA endpoint addresses.
const EFA_MAX_ADDR_SIZE: usize = 64;

/// Error type for EFA operations.
#[derive(Debug, Clone)]
pub struct EfaError {
    pub code: i32,
    pub message: String,
}

impl std::fmt::Display for EfaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EFA error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for EfaError {}

impl From<i32> for EfaError {
    fn from(code: i32) -> Self {
        let message = unsafe {
            let ptr = rdmaxcel_efa_error_string(code);
            if ptr.is_null() {
                format!("Unknown error code {}", code)
            } else {
                std::ffi::CStr::from_ptr(ptr)
                    .to_string_lossy()
                    .into_owned()
            }
        };
        EfaError { code, message }
    }
}

/// Result type for EFA operations.
pub type EfaResult<T> = Result<T, EfaError>;

/// Check if EFA is available on this system.
pub fn efa_available() -> bool {
    unsafe { rdmaxcel_efa_available() != 0 }
}

/// A safe wrapper around an EFA endpoint.
///
/// This struct manages the lifecycle of an EFA endpoint, ensuring proper
/// cleanup when dropped.
pub struct EfaEndpoint {
    ep: *mut rdmaxcel_efa_ep_t,
    cached_local_addr: Vec<u8>,
}

// EFA endpoints are thread-safe for concurrent operations
unsafe impl Send for EfaEndpoint {}
unsafe impl Sync for EfaEndpoint {}

impl std::fmt::Debug for EfaEndpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EfaEndpoint")
            .field("ep", &format!("{:p}", self.ep))
            .finish()
    }
}

impl EfaEndpoint {
    /// Create a new EFA endpoint.
    ///
    /// # Arguments
    /// * `provider` - Provider name, typically "efa" for AWS EFA
    /// * `buffer_size` - Size of internal buffer to allocate (0 for no buffer)
    ///
    /// # Returns
    /// A new EfaEndpoint on success, or an error if creation fails.
    pub fn new(provider: &str, buffer_size: usize) -> EfaResult<Self> {
        if !efa_available() {
            return Err(EfaError {
                code: EFA_ERROR_NOT_AVAILABLE,
                message: "EFA is not available on this system".to_string(),
            });
        }

        let provider_cstr = CString::new(provider).map_err(|_| EfaError {
            code: EFA_ERROR_INVALID_PARAMS,
            message: "Invalid provider name".to_string(),
        })?;

        let ep = unsafe { rdmaxcel_efa_ep_create(provider_cstr.as_ptr(), buffer_size) };

        if ep.is_null() {
            return Err(EfaError {
                code: EFA_ERROR_EP_FAILED,
                message: "Failed to create EFA endpoint".to_string(),
            });
        }

        // Fetch and cache the local address at creation time
        let mut addr_buf = vec![0u8; EFA_MAX_ADDR_SIZE];
        let mut addr_len = EFA_MAX_ADDR_SIZE;
        let ret = unsafe {
            rdmaxcel_efa_get_local_addr(
                ep,
                addr_buf.as_mut_ptr() as *mut _,
                &mut addr_len,
            )
        };
        if ret != EFA_SUCCESS {
            unsafe { rdmaxcel_efa_ep_destroy(ep); }
            return Err(EfaError::from(ret));
        }
        addr_buf.truncate(addr_len);

        Ok(EfaEndpoint { ep, cached_local_addr: addr_buf })
    }

    /// Get the local endpoint address for connection exchange.
    ///
    /// This address is cached at creation time, so this method is allocation-free
    /// (returns a clone of the cached Vec).
    ///
    /// # Returns
    /// A byte vector containing the local address.
    pub fn get_local_addr(&self) -> EfaResult<Vec<u8>> {
        Ok(self.cached_local_addr.clone())
    }

    /// Insert a peer's address into the address vector.
    ///
    /// This must be called before performing RDMA operations to that peer.
    ///
    /// # Arguments
    /// * `peer_addr` - The peer's address obtained from their `get_local_addr()`
    ///
    /// # Returns
    /// The fi_addr_t handle for the peer, to be passed to `write`/`read`/`tsend`.
    pub fn insert_peer_addr(&self, peer_addr: &[u8]) -> EfaResult<u64> {
        let mut fi_addr: u64 = 0;
        let ret = unsafe {
            rdmaxcel_efa_insert_peer_addr(
                self.ep,
                peer_addr.as_ptr() as *const _,
                peer_addr.len(),
                &mut fi_addr,
            )
        };

        if ret != EFA_SUCCESS {
            return Err(EfaError::from(ret));
        }

        Ok(fi_addr)
    }

    /// Register a memory region for RDMA operations.
    ///
    /// # Arguments
    /// * `addr` - Start address of the memory region
    /// * `size` - Size of the memory region in bytes
    ///
    /// # Returns
    /// A memory registration key that can be used for remote access.
    ///
    /// # Safety
    /// The memory region must remain valid and pinned while registered.
    pub fn register_mr(&self, addr: usize, size: usize) -> EfaResult<u64> {
        let mut key: u64 = 0;

        let ret =
            unsafe { rdmaxcel_efa_register_mr(self.ep, addr as *mut _, size, &mut key) };

        if ret != EFA_SUCCESS {
            return Err(EfaError::from(ret));
        }

        Ok(key)
    }

    /// Deregister a previously registered memory region.
    ///
    /// # Arguments
    /// * `key` - The memory registration key from `register_mr()`
    pub fn deregister_mr(&self, key: u64) -> EfaResult<()> {
        let ret = unsafe { rdmaxcel_efa_deregister_mr(self.ep, key) };

        if ret != EFA_SUCCESS {
            return Err(EfaError::from(ret));
        }

        Ok(())
    }

    /// Perform an RDMA write operation.
    ///
    /// Writes data from a local buffer to a remote memory region.
    ///
    /// # Arguments
    /// * `local_addr` - Local buffer address
    /// * `size` - Size of data to write
    /// * `remote_addr` - Remote memory address
    /// * `remote_key` - Remote memory registration key
    /// * `peer` - Peer fi_addr_t handle from `insert_peer_addr()`
    pub fn write(
        &self,
        local_addr: usize,
        size: usize,
        remote_addr: u64,
        remote_key: u64,
        peer: u64,
    ) -> EfaResult<()> {
        let ret = unsafe {
            rdmaxcel_efa_write(self.ep, local_addr as *mut _, size, remote_addr, remote_key, peer)
        };

        if ret != EFA_SUCCESS {
            return Err(EfaError::from(ret));
        }

        Ok(())
    }

    /// Perform an RDMA read operation.
    ///
    /// Reads data from a remote memory region into a local buffer.
    ///
    /// # Arguments
    /// * `local_addr` - Local buffer address to read into
    /// * `size` - Size of data to read
    /// * `remote_addr` - Remote memory address
    /// * `remote_key` - Remote memory registration key
    /// * `peer` - Peer fi_addr_t handle from `insert_peer_addr()`
    pub fn read(
        &self,
        local_addr: usize,
        size: usize,
        remote_addr: u64,
        remote_key: u64,
        peer: u64,
    ) -> EfaResult<()> {
        let ret = unsafe {
            rdmaxcel_efa_read(self.ep, local_addr as *mut _, size, remote_addr, remote_key, peer)
        };

        if ret != EFA_SUCCESS {
            return Err(EfaError::from(ret));
        }

        Ok(())
    }

    /// Poll for completion of RDMA operations.
    ///
    /// # Arguments
    /// * `timeout_ms` - Timeout in milliseconds (-1 for infinite)
    ///
    /// # Returns
    /// The number of completions on success, or an error on failure.
    pub fn poll_cq(&self, timeout_ms: i32) -> EfaResult<i32> {
        let ret = unsafe { rdmaxcel_efa_poll_cq(self.ep, timeout_ms) };

        if ret < 0 {
            return Err(EfaError::from(ret));
        }

        Ok(ret)
    }

    /// Send a tagged message for completion notification.
    ///
    /// This sends a minimal (1-byte) tagged message to the peer,
    /// typically used to signal that an RDMA write has completed.
    ///
    /// # Arguments
    /// * `tag` - Message tag (used to match with trecv on the peer)
    /// * `peer` - Peer fi_addr_t handle from `insert_peer_addr()`
    pub fn tsend(&self, tag: u64, peer: u64) -> EfaResult<()> {
        let ret = unsafe { rdmaxcel_efa_tsend(self.ep, tag, peer) };

        if ret != EFA_SUCCESS {
            return Err(EfaError::from(ret));
        }

        Ok(())
    }

    /// Post a tagged receive for completion notification.
    ///
    /// This posts a receive buffer to accept a tagged message from the peer,
    /// typically used to wait for notification that an RDMA write has completed.
    ///
    /// # Arguments
    /// * `tag` - Expected message tag (must match the tag used in tsend)
    pub fn trecv(&self, tag: u64) -> EfaResult<()> {
        let ret = unsafe { rdmaxcel_efa_trecv(self.ep, tag) };

        if ret != EFA_SUCCESS {
            return Err(EfaError::from(ret));
        }

        Ok(())
    }

    /// Get the raw endpoint pointer for advanced usage.
    ///
    /// # Safety
    /// The returned pointer must not outlive this EfaEndpoint.
    pub unsafe fn as_raw(&self) -> *mut rdmaxcel_efa_ep_t {
        self.ep
    }
}

impl Drop for EfaEndpoint {
    fn drop(&mut self) {
        if !self.ep.is_null() {
            unsafe {
                rdmaxcel_efa_ep_destroy(self.ep);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_efa_available_check() {
        // This just tests that the function doesn't crash
        let _ = efa_available();
    }

    #[test]
    fn test_efa_error_display() {
        let err = EfaError {
            code: -1,
            message: "Test error".to_string(),
        };
        assert!(err.to_string().contains("Test error"));
    }
}
