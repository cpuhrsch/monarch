# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Monarch RDMA API - RFC for next version public interface.

This file documents the user-facing API surface only.
Implementation details are elided with `...`.
Sugar methods show their implementation in terms of other public API.

Memory Management Model
========================

RDMABuffer wraps an address and size representing a region of memory,
which may reside in CPU or GPU memory. The buffer can be constructed from:

- torch.Tensor: Wraps the tensor's underlying storage
- memoryview: Python buffer protocol object that exposes a pointer to memory
  with custom lifetime management

Buffer Creation Performance
---------------------------

While RDMABuffer creation may be slow at process startup (due to memory
registration and connection setup), it quickly converges to O(1) creation
time in steady state—roughly equivalent to a hash lookup. This means you
do not need to manage long-lived buffer pools or cache RDMABuffers if your
application does not already require them. Creating a fresh RDMABuffer for
each transfer is efficient once the system has warmed up.


Ownership Semantics
-------------------

Once created, an RDMABuffer takes a reference to the memory region. The buffer
must be explicitly released by calling drop() to drop this reference.
This is a manual operation:

drop() is NOT reference-counted and will invalidate all instances of that RDMABuffer
object that have been sent around the system. The RDMABuffer is not the same thing
as the underlying memoryview/tensor. That memory will remain if there are still other
references to it.

We deliberately avoid distributed reference counting because maintaining
consistent counts across remote failures is prohibitively complex. Instead,
we require explicit manual management and expect higher-level libraries to
build resource management abstractions on top of this primitive.

Local vs Remote Memory
----------------------

RDMABuffer represents the REMOTE side of RDMA transactions. When performing
operations like read_into() or write_from(), the RDMABuffer is the remote
memory region, while local memory is passed directly as torch.Tensor or
memoryview arguments. The local memory is held by the operation until the
RDMA action completes.

Data Hazards
------------

While an RDMA operation is in flight, the local memory is being accessed
concurrently by the RDMA engine. Accessing this memory from your code
before the operation completes can cause undefined behavior:

- If the RDMA engine is READING from local memory (write_from), do not
  modify that memory until the Future completes. The remote side may
  receive partially updated or corrupted data. Concurrent reads are safe.

- If the RDMA engine is WRITING to local memory (read_into), do not read
  or write that memory until the Future completes. You may see incomplete
  data, or your writes may be overwritten by the RDMA engine.

Await or block on the returned Future before performing any writes to memory
involved in a write_from operation, or any reads/writes to memory involved
in a read_into operation.
"""

from __future__ import annotations

from typing import List, Literal, Optional, TYPE_CHECKING

import torch
from monarch._src.actor.future import Future
from typing_extensions import Self


# ============================================================================
# Core API
# ============================================================================

# Local memory in RDMA operations. Passed directly to read_into()/write_from()
# and held until the operation completes.
LocalMemory = torch.Tensor | memoryview


class RDMABuffer:
    """
    RDMA buffer wrapping a 1d contiguous region of memory.

    Represents the remote side of RDMA transactions between actors. The buffer
    wraps an address/size pair that may point to CPU or GPU memory.

    Construction accepts:
    - torch.Tensor: Must be 1d and contiguous (including views/slices)
    - memoryview: Must be 1d and c-contiguous; as a PyBuffer, memoryview
      allows wrapping arbitrary memory with custom lifecycle management

    Once created, the RDMABuffer takes ownership of the memory region.
    You must call drop() to release it. See module docstring for ownership
    semantics.
    """

    def __init__(self, data: torch.Tensor | memoryview) -> None:
        """
        Create an RDMA buffer from a tensor or memoryview.

        The buffer takes ownership of the memory region. You must call
        drop() to release it when done.

        Args:
            data: 1d contiguous torch.Tensor or c-contiguous memoryview.

        Raises:
            ValueError: If data is not 1d contiguous or has size 0.
            RuntimeError: If RDMA is not available.
        """
        ...

    def size(self) -> int:
        """Return the size of the buffer in bytes."""
        ...

    def read_into(
        self,
        dst: torch.Tensor | memoryview,
        *,
        transport: Transport = "best",
        timeout: int = 3,
    ) -> Future[Optional[None]]:
        """
        Read data from this RDMA buffer into a local destination.

        Args:
            dst: Destination tensor or memoryview.

        Keyword Args:
            transport: Transport type to use. See Transport type. Defaults to 'best'.
            timeout: Timeout in seconds. Defaults to 3.

        Returns:
            Future that completes when the read is done.

        Raises:
            ValueError: If dst size < buffer size.
        """
        # Sugar: creates an RDMAAction and submits it
        return RDMAAction().read_into(self, dst, transport).submit()

    def write_from(
        self,
        src: torch.Tensor | memoryview,
        *,
        transport: Transport = "best",
        timeout: int = 3,
    ) -> Future[None]:
        """
        Write data from a local source into this RDMA buffer.

        Args:
            src: Source tensor or memoryview.

        Keyword Args:
            transport: Transport type to use. See Transport type. Defaults to 'best'.
            timeout: Timeout in seconds. Defaults to 3.

        Returns:
            Future that completes when the write is done.

        Raises:
            ValueError: If src size > buffer size.
        """
        # Sugar: creates an RDMAAction and submits it
        return RDMAAction().write_from(self, src, transport).submit()

    def drop(self) -> Future[None]:
        """
        Release ownership of the memory region.

        This is NOT reference-counted. Calling drop() will invalidate
        all copies of this buffer. Manual management is required.

        Returns:
            Future that completes when the buffer is released.
        """
        ...

    @property
    def owner(self) -> str:
        """The owner actor reference."""
        ...

    def local_ptr(self, device: str) -> Optional[int]:
        """
        NEW!!
        Get a pointer to the buffer's memory if accessible from the specified device.

        Returns the raw pointer to the underlying memory if it can be directly
        read from the given device without an RDMA transfer. This is useful for
        zero-copy access when memory is co-located or connected via NVLink.

        Args:
            device: Device specifier string. Examples:
                - "cpu" - CPU memory on the local host
                - "cuda" - Default CUDA device
                - "cuda:0", "cuda:1" - Specific CUDA device

        Returns:
            The pointer as an integer if the memory is directly accessible from
            the specified device, or None if an RDMA transfer would be required.

        Examples:
            - CPU buffer on local host, device="cpu" -> returns pointer
            - GPU buffer accessible via NVLink, device="cuda:0" -> returns pointer
            - Remote buffer not accessible locally -> returns None
        """
        ...

    def nic_transport(self, device: str) -> Optional[NicTransportInfo]:
        """
        NEW!!
        Get low-level NIC transport info for direct RDMA operations to this buffer.

        Returns a cffi struct containing the ibverbs or libfabric transport
        information needed to perform RDMA operations directly, such as from
        a GPU kernel. This enables advanced use cases like GPU-initiated RDMA
        (GPUDirect RDMA).

        The returned cffi struct contains the local QP handle, remote key,
        remote address, and routing information needed to post RDMA work requests
        directly via ibv_post_send() or libfabric equivalents.

        Args:
            device: Device specifier for the local side of the transfer.
                - "cpu" - Transfer from/to CPU memory
                - "cuda", "cuda:0" - Transfer from/to GPU memory (GPUDirect)

        Returns:
            cffi CData pointer to NicTransportInfo struct if this buffer can be
            accessed via NIC transport from the specified device, or None if NIC
            transport is not available (e.g., buffer is local, or no RDMA NIC).

        Note:
            The `id` field in the returned info uniquely identifies the underlying
            queue pair. Buffers that return the same `id` share a queue pair and
            can be batched together for optimal performance.

        Example:
            info = buffer.nic_transport("cuda:0")
            if info:
                # Access fields directly
                print(info.rkey, info.remote_addr)
                # Pass to C/CUDA kernel
                launch_rdma_kernel(ffi.addressof(info), local_buf)
        """
        ...


class RDMAAction:
    """
    Batch multiple RDMA operations for optimized bulk transfers.

    Provides an opportunity to optimize bulk RDMA transactions
    without exposing complexity to users.
    """

    def __init__(self) -> None:
        """Create an empty action batch."""
        ...

    def read_into(
        self,
        src: RDMABuffer,
        dst: LocalMemory | List[LocalMemory],
        transport: Transport = "best",
    ) -> Self:
        """
        Queue a read from src RDMA buffer into dst memory.

        Args:
            src: Source RDMA buffer to read from.
            dst: Destination local memory. If a list, data is concatenated.
            transport: Transport type to use. See Transport type. Defaults to 'best'.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If dst size < src buffer size.
        """
        ...

    def write_from(
        self,
        src: RDMABuffer,
        dst: LocalMemory | List[LocalMemory],
        transport: Transport = "best",
    ) -> Self:
        """
        Queue a write from dst memory to src RDMA buffer.

        Args:
            src: Destination RDMA buffer to write to.
            dst: Source local memory. If a list, data is concatenated.
            transport: Transport type to use. See Transport type. Defaults to 'best'.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If dst size > src buffer size.
        """
        ...

    def fetch_add(
        self,
        src: RDMABuffer,
        dst: LocalMemory,
        add: int,
        transport: Transport = "best",
    ) -> Self:
        """
        Queue atomic fetch-and-add on src RDMA buffer.

        Atomically: *dst = *src; *src = *src + add

        Args:
            src: RDMA buffer to operate on (8 bytes).
            dst: Local memory to store original value (8 bytes).
            add: Value to add.
            transport: Transport type to use. See Transport type. Defaults to 'best'.

        Returns:
            Self for method chaining.
        """
        ...

    def compare_and_swap(
        self,
        src: RDMABuffer,
        dst: LocalMemory,
        compare: int,
        swap: int,
        transport: Transport = "best",
    ) -> Self:
        """
        Queue atomic compare-and-swap on src RDMA buffer.

        Atomically: *dst = *src; if (*src == compare) *src = swap

        Args:
            src: RDMA buffer to operate on (8 bytes).
            dst: Local memory to store original value (8 bytes).
            compare: Value to compare against.
            swap: Value to swap in if comparison succeeds.
            transport: Transport type to use. See Transport type. Defaults to 'best'.

        Returns:
            Self for method chaining.
        """
        ...

    def submit(self) -> Future[None]:
        """
        Submit all queued operations.

        Can be called multiple times to schedule the same work repeatedly.
        Operations to different owners execute concurrently.

        Returns:
            Future that completes when all operations are done.
        """
        # NEW!! We truly submit RDMA transactions as a batch;
        # the RDMABuffer methods just desugar to a single action.
        ...


# ============================================================================
# Appendix: Types
# ============================================================================

# NEW!!
# Transport specifies which transport type to use for RDMA operations.
#
# Values (ordered slowest to fastest: tcp -> nic -> nvlink):
#   'best'   - Select the best available transport
#   'tcp'    - TCP/IP sockets
#   'nic'    - RDMA NIC (RoCE, InfiniBand, EFA)
#   'nvlink' - NVLink (GPU-to-GPU)
#
# Modifiers:
#   '+' suffix - "at least this fast or better" (e.g., 'nic+' means nic or nvlink)
#   '-' suffix - "this speed or slower" (e.g., 'nic-' means tcp or nic)
Transport = Literal[
    "best",
    "tcp",
    "tcp+",
    "tcp-",
    "nic",
    "nic+",
    "nic-",
    "nvlink",
    "nvlink+",
    "nvlink-",
]

# NicTransportInfo is a cffi struct exposing low-level NIC transport details.
#
# C definition:
#
#     typedef struct {
#         uint64_t id;              // unique endpoint identifier (for batching)
#         struct ibv_qp *qp;        // local QP handle (already connected)
#         uint32_t rkey;            // remote memory key
#         uint64_t remote_addr;     // remote buffer address
#         uint64_t size;            // buffer size in bytes
#     } NicTransportInfo;
#
# For libfabric/EFA, the struct contains equivalent fields:
#   - id: unique endpoint identifier
#   - ep: endpoint handle (instead of qp)
#   - rkey: memory region key
#   - remote_addr, size: as above
#
# Access fields directly: info.qp, info.rkey, info.remote_addr, etc.
# Pass to C/CUDA via ffi.buffer(info) or ffi.addressof(info).
#
# Buffers sharing the same `id` use the same underlying queue pair and can
# be batched together for optimal performance.
NicTransportInfo = "ffi.CData"  # cffi struct pointer


# ============================================================================
# Appendix: Utilities and Warnings
# ============================================================================


def is_rdma_available() -> bool:
    """Check if RDMA is available on this platform."""
    ...


class RDMAReadTransferWarning(Warning):
    """Warning for RDMA read transfer performance issues."""

    pass


class RDMAWriteTransferWarning(Warning):
    """Warning for RDMA write transfer performance issues."""

    pass
