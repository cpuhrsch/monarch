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
    rdmaxcel_efa_insert_peer_addr,
    rdmaxcel_efa_register_mr,
    rdmaxcel_efa_push_data, rdmaxcel_efa_wait_for_data,
};

const EFA_SUCCESS: i32 = 0;
const EFA_ERROR_NOT_AVAILABLE: i32 = -1;
const EFA_ERROR_INVALID_PARAMS: i32 = -14;
const EFA_ERROR_EP_FAILED: i32 = -5;
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

pub type EfaResult<T> = Result<T, EfaError>;

/// Call an EFA FFI function that returns EFA_SUCCESS on success.
macro_rules! efa_call {
    ($call:expr) => {{
        let ret = unsafe { $call };
        if ret != EFA_SUCCESS {
            return Err(EfaError::from(ret));
        }
        Ok(())
    }};
}

/// Check if EFA is available on this system.
pub fn efa_available() -> bool {
    unsafe { rdmaxcel_efa_available() != 0 }
}

/// A safe wrapper around an EFA endpoint.
///
/// Manages the lifecycle of an EFA endpoint, ensuring proper cleanup when dropped.
pub struct EfaEndpoint {
    ep: *mut rdmaxcel_efa_ep_t,
    cached_local_addr: Vec<u8>,
}

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
    pub fn new(provider: &str) -> EfaResult<Self> {
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

        let ep = unsafe { rdmaxcel_efa_ep_create(provider_cstr.as_ptr()) };

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
            rdmaxcel_efa_get_local_addr(ep, addr_buf.as_mut_ptr() as *mut _, &mut addr_len)
        };
        if ret != EFA_SUCCESS {
            unsafe { rdmaxcel_efa_ep_destroy(ep); }
            return Err(EfaError::from(ret));
        }
        addr_buf.truncate(addr_len);

        Ok(EfaEndpoint { ep, cached_local_addr: addr_buf })
    }

    pub fn get_local_addr(&self) -> EfaResult<Vec<u8>> {
        Ok(self.cached_local_addr.clone())
    }

    /// Insert a peer address and return its fi_addr_t handle.
    pub fn insert_peer_addr(&self, peer_addr: &[u8]) -> EfaResult<u64> {
        let mut fi_addr: u64 = 0;
        let ret = unsafe {
            rdmaxcel_efa_insert_peer_addr(
                self.ep, peer_addr.as_ptr() as *const _, peer_addr.len(), &mut fi_addr,
            )
        };
        if ret != EFA_SUCCESS {
            return Err(EfaError::from(ret));
        }
        Ok(fi_addr)
    }

    /// Register a memory region, returning its key.
    pub fn register_mr(&self, addr: usize, size: usize) -> EfaResult<u64> {
        let mut key: u64 = 0;
        let ret = unsafe { rdmaxcel_efa_register_mr(self.ep, addr as *mut _, size, &mut key) };
        if ret != EFA_SUCCESS {
            return Err(EfaError::from(ret));
        }
        Ok(key)
    }

    pub fn deregister_mr(&self, key: u64) -> EfaResult<()> {
        efa_call!(rdmaxcel_efa_deregister_mr(self.ep, key))
    }

    /// Push data to a remote peer: fi_write + poll + fi_tsend + poll.
    /// Blocks until complete. The destination must call wait_for_data with the same tag.
    pub fn push_data(&self, local_addr: usize, size: usize, remote_addr: u64, remote_key: u64, peer: u64, tag: u64) -> EfaResult<()> {
        efa_call!(rdmaxcel_efa_push_data(self.ep, local_addr as *mut _, size, remote_addr, remote_key, peer, tag))
    }

    /// Wait for data from a remote peer: fi_trecv + poll.
    /// Blocks until complete. The source must call push_data with the same tag.
    pub fn wait_for_data(&self, tag: u64) -> EfaResult<()> {
        efa_call!(rdmaxcel_efa_wait_for_data(self.ep, tag))
    }
}

impl Drop for EfaEndpoint {
    fn drop(&mut self) {
        if !self.ep.is_null() {
            unsafe { rdmaxcel_efa_ep_destroy(self.ep); }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_efa_available_check() {
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
