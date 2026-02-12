/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef RDMAXCEL_EFA_H
#define RDMAXCEL_EFA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// EFA Error Codes
typedef enum {
  EFA_SUCCESS = 0,
  EFA_ERROR_NOT_AVAILABLE = -1,
  EFA_ERROR_INIT_FAILED = -2,
  EFA_ERROR_FABRIC_FAILED = -3,
  EFA_ERROR_DOMAIN_FAILED = -4,
  EFA_ERROR_EP_FAILED = -5,
  EFA_ERROR_CQ_FAILED = -6,
  EFA_ERROR_AV_FAILED = -7,
  EFA_ERROR_MR_FAILED = -8,
  EFA_ERROR_CONNECT_FAILED = -9,
  EFA_ERROR_WRITE_FAILED = -10,
  EFA_ERROR_READ_FAILED = -11,
  EFA_ERROR_POLL_FAILED = -12,
  EFA_ERROR_TIMEOUT = -13,
  EFA_ERROR_INVALID_PARAMS = -14
} efa_error_code_t;

// Opaque EFA endpoint handle
typedef struct rdmaxcel_efa_ep rdmaxcel_efa_ep_t;

// Check if EFA is available on this system
// Returns: 1 if available, 0 if not
int rdmaxcel_efa_available(void);

// Create EFA endpoint
// Parameters:
//   provider: Provider name, typically "efa" for AWS EFA
// Returns: Pointer to endpoint on success, NULL on failure
rdmaxcel_efa_ep_t* rdmaxcel_efa_ep_create(const char* provider);

// Destroy EFA endpoint and free resources
void rdmaxcel_efa_ep_destroy(rdmaxcel_efa_ep_t* ep);

// Get the local endpoint address for connection
// Parameters:
//   ep: EFA endpoint
//   addr_out: Buffer to write address to
//   addr_len: In: size of buffer, Out: actual address length
// Returns: EFA_SUCCESS on success, error code on failure
int rdmaxcel_efa_get_local_addr(
    rdmaxcel_efa_ep_t* ep,
    void* addr_out,
    size_t* addr_len);

// Insert a peer address into the address vector
// Parameters:
//   ep: EFA endpoint
//   peer_addr: Peer's address (from rdmaxcel_efa_get_local_addr)
//   addr_len: Length of peer_addr
//   fi_addr_out: Output parameter for the fi_addr_t handle (used in write/read/tsend)
// Returns: EFA_SUCCESS on success, error code on failure
int rdmaxcel_efa_insert_peer_addr(
    rdmaxcel_efa_ep_t* ep,
    const void* peer_addr,
    size_t addr_len,
    uint64_t* fi_addr_out);

// Register a memory region for RDMA operations
// Parameters:
//   ep: EFA endpoint
//   addr: Start address of memory region
//   size: Size of memory region in bytes
//   key_out: Output parameter for the memory registration key
// Returns: EFA_SUCCESS on success, error code on failure
int rdmaxcel_efa_register_mr(
    rdmaxcel_efa_ep_t* ep,
    void* addr,
    size_t size,
    uint64_t* key_out);

// Deregister a memory region
// Parameters:
//   ep: EFA endpoint
//   key: Memory registration key from rdmaxcel_efa_register_mr
// Returns: EFA_SUCCESS on success, error code on failure
int rdmaxcel_efa_deregister_mr(
    rdmaxcel_efa_ep_t* ep,
    uint64_t key);

// Perform RDMA write operation
// Parameters:
//   ep: EFA endpoint
//   local_addr: Local buffer address
//   size: Size of data to write
//   remote_addr: Remote memory address
//   remote_key: Remote memory registration key
//   peer: Peer fi_addr_t handle from rdmaxcel_efa_insert_peer_addr
// Returns: EFA_SUCCESS on success, error code on failure
int rdmaxcel_efa_write(
    rdmaxcel_efa_ep_t* ep,
    void* local_addr,
    size_t size,
    uint64_t remote_addr,
    uint64_t remote_key,
    uint64_t peer);

// Perform RDMA read operation
// Parameters:
//   ep: EFA endpoint
//   local_addr: Local buffer address to read into
//   size: Size of data to read
//   remote_addr: Remote memory address
//   remote_key: Remote memory registration key
//   peer: Peer fi_addr_t handle from rdmaxcel_efa_insert_peer_addr
// Returns: EFA_SUCCESS on success, error code on failure
int rdmaxcel_efa_read(
    rdmaxcel_efa_ep_t* ep,
    void* local_addr,
    size_t size,
    uint64_t remote_addr,
    uint64_t remote_key,
    uint64_t peer);

// Poll for completion of RDMA operations (non-blocking)
// Parameters:
//   ep: EFA endpoint
// Returns: Number of completions (0 = none available), negative error code on failure
int rdmaxcel_efa_poll_cq(rdmaxcel_efa_ep_t* ep);

// Send a tagged message (for completion notification)
// Parameters:
//   ep: EFA endpoint
//   tag: Message tag (used to match with trecv)
//   peer: Peer fi_addr_t handle from rdmaxcel_efa_insert_peer_addr
// Returns: EFA_SUCCESS on success, error code on failure
int rdmaxcel_efa_tsend(
    rdmaxcel_efa_ep_t* ep,
    uint64_t tag,
    uint64_t peer);

// Post a tagged receive (for completion notification)
// Parameters:
//   ep: EFA endpoint
//   tag: Expected message tag
// Returns: EFA_SUCCESS on success, error code on failure
int rdmaxcel_efa_trecv(
    rdmaxcel_efa_ep_t* ep,
    uint64_t tag);

// Get error string for EFA error code
const char* rdmaxcel_efa_error_string(int error_code);

#ifdef __cplusplus
}
#endif

#endif // RDMAXCEL_EFA_H
