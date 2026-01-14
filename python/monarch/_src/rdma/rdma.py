# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import ctypes
import functools
import logging
import warnings
from collections import defaultdict
from typing import Any, cast, List, Literal, Optional, Tuple, Union

import torch
from monarch._rust_bindings.monarch_hyperactor.pytokio import PythonTask, Shared
from monarch._src.actor.proc_mesh import ProcMesh
from typing_extensions import Self

try:
    from monarch._rust_bindings.rdma import _RdmaBuffer, _RdmaManager
except ImportError as e:
    logging.error("RDMA is not available: {}".format(e))
    raise e
from enum import Enum
from typing import Dict

from monarch._src.actor.actor_mesh import Actor, context
from monarch._src.actor.endpoint import endpoint
from monarch._src.actor.future import Future
from monarch._src.actor.proc_mesh import get_or_spawn_controller
from pyre_extensions import none_throws


# RDMARead/WriteTransferWarnings are warnings that are only printed once per process.
# Remove these once GPU support is added.
class RDMAReadTransferWarning(Warning):
    pass


class RDMAWriteTransferWarning(Warning):
    pass


warnings.simplefilter("once", RDMAReadTransferWarning)
warnings.simplefilter("once", RDMAWriteTransferWarning)


# Transport type for selecting data transfer mechanism
# - "best": Auto-select best available (RDMA NIC if available, else TCP)
# - "tcp": Force TCP transport (uses actor messaging)
# - "nic": Force RDMA NIC (error if unavailable)
Transport = Literal["best", "tcp", "nic"]

# Per-process registry mapping buffer IDs to tensor data for TCP fallback
# Key is the buffer's unique identifier, value is the original tensor/memoryview
_buffer_registry: Dict[str, Union[torch.Tensor, memoryview]] = {}


def is_rdma_available():
    return _RdmaBuffer.rdma_supported()


# Cached so that we don't have to call out to the root client every time,
# which may be on a different host.
@functools.cache
def _ensure_init_rdma_manager() -> Shared[None]:
    async def task() -> None:
        # Ensure the proc mesh is initialized before we can send it over the wire,
        # since pickling the proc mesh before it is initiliazed would block the
        # tokio runtime and cause a panic.
        await context().actor_instance.proc_mesh.initialized
        await (
            await get_or_spawn_controller("rdma_controller", RdmaController)
        ).init_rdma_on_mesh.call_one(none_throws(context().actor_instance.proc_mesh))

    return PythonTask.from_coroutine(task()).spawn()


def _get_error(buf) -> ValueError:
    return ValueError(
        "RDMABuffer only supports 1d contiguous torch.Tensor or 1d c-contiguous memoryview. Got: {}".format(
            buf
        )
    )


def _assert_1d_contiguous(buf: torch.Tensor | memoryview) -> None:
    if isinstance(buf, torch.Tensor):
        if buf.dim() != 1 or not buf.is_contiguous():
            raise _get_error(buf)
    elif isinstance(buf, memoryview):
        if buf.ndim != 1 or not buf.c_contiguous:
            raise _get_error(buf)
    else:
        raise _get_error(buf)


def _get_memoryview_addr_and_size(buf: memoryview) -> tuple[int, int]:
    addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    size = buf.nbytes
    return addr, size


def _get_tensor_addr_and_size(tensor: torch.Tensor) -> tuple[int, int]:
    data_ptr: int = tensor.untyped_storage().data_ptr()
    # Calculate the actual starting address of the tensor data
    # storage_offset() can return either int or torch.SymInt in newer PyTorch versions
    try:
        storage_offset = int(tensor.storage_offset())
    except Exception as e:
        raise RuntimeError("Failed to convert tensor.storage_offset() to int.") from e
    offset: int = storage_offset * tensor.element_size()
    addr: int = data_ptr + offset
    size: int = tensor.element_size() * tensor.numel()
    return addr, size


def _get_addr_and_size(buf: torch.Tensor | memoryview) -> tuple[int, int]:
    _assert_1d_contiguous(buf)
    if isinstance(buf, memoryview):
        return _get_memoryview_addr_and_size(buf)
    elif isinstance(buf, torch.Tensor):
        return _get_tensor_addr_and_size(buf)
    # This shouldn't happen unless there is a bug, handle the type in caller.
    raise RuntimeError(
        "Trying to get address and size of unsupported type. Expected memoryview or torch.Tensor. Got: {}".format(
            type(buf)
        )
    )


class RdmaController(Actor):
    def __init__(self) -> None:
        self._manager_futures: Dict[ProcMesh, Future[_RdmaManager]] = {}

    @endpoint
    async def init_rdma_on_mesh(self, proc_mesh: ProcMesh) -> None:
        # Note: RdmaController acts as coordinator and can run on any node
        # The RDMA support check should happen on the target proc_mesh nodes, not on RdmaController's node

        if proc_mesh not in self._manager_futures:

            async def create_manager() -> _RdmaManager:
                proc_mesh_result = await Future(
                    coro=cast("PythonTask[Any]", proc_mesh._proc_mesh.task())
                )
                return none_throws(
                    await _RdmaManager.create_rdma_manager_nonblocking(
                        proc_mesh_result, context().actor_instance
                    )
                )

            self._manager_futures[proc_mesh] = Future(coro=create_manager())

        await self._manager_futures[proc_mesh]


class TcpDataActor(Actor):
    """
    Per-process actor for handling TCP-based data transfer as fallback for RDMA.

    This actor provides endpoints for fetching and writing buffer data when
    RDMA NIC is not available or when TCP transport is explicitly requested.
    """

    def __init__(self) -> None:
        # Local buffers registered for TCP access on this process
        self._local_buffers: Dict[str, Union[torch.Tensor, memoryview]] = {}

    @endpoint
    async def register_buffer(
        self, buffer_id: str, data: torch.Tensor
    ) -> None:
        """Register a buffer for TCP access."""
        self._local_buffers[buffer_id] = data

    @endpoint
    async def unregister_buffer(self, buffer_id: str) -> None:
        """Unregister a buffer from TCP access."""
        self._local_buffers.pop(buffer_id, None)

    @endpoint
    async def fetch_buffer_data(self, buffer_id: str) -> bytes:
        """
        Fetch buffer data for TCP transfer.

        Returns the buffer contents as bytes.
        """
        data = self._local_buffers.get(buffer_id)
        if data is None:
            raise ValueError(f"Buffer {buffer_id} not found in TCP registry")

        if isinstance(data, torch.Tensor):
            # Convert tensor to bytes
            return data.numpy().tobytes()
        else:
            # memoryview - convert to bytes
            return bytes(data)

    @endpoint
    async def write_buffer_data(self, buffer_id: str, data: bytes) -> None:
        """
        Write data to buffer via TCP.

        Copies the provided bytes into the registered buffer.
        """
        buf = self._local_buffers.get(buffer_id)
        if buf is None:
            raise ValueError(f"Buffer {buffer_id} not found in TCP registry")

        if isinstance(buf, torch.Tensor):
            # Create tensor from bytes and copy
            src = torch.frombuffer(bytearray(data), dtype=buf.dtype).reshape(buf.shape)
            buf.copy_(src)
        else:
            # memoryview - direct copy
            buf[:] = data


# Cached helper to get or spawn the per-process TcpDataActor
@functools.cache
def _get_tcp_data_actor_future() -> "Future[TcpDataActor]":
    """Get or spawn the TcpDataActor for this process (returns Future)."""
    return get_or_spawn_controller("tcp_data_actor", TcpDataActor)


def _get_tcp_data_actor_blocking() -> "TcpDataActor":
    """Get or spawn the TcpDataActor for this process (blocking)."""
    return _get_tcp_data_actor_future().get()


def pt_cuda_allocator_compatibility() -> bool:
    """
    Check if PyTorch CUDA caching allocator is compatible with RDMA.

    This checks if both the CUDA caching allocator is enabled AND expandable
    segments are enabled, which is required for RDMA operations with CUDA tensors.

    Returns:
        bool: True if both conditions are met, False otherwise
    """
    if not torch.cuda.is_available():
        return False

    # Get allocator snapshot which contains settings
    snapshot = torch.cuda.memory._snapshot()
    allocator_settings = snapshot.get("allocator_settings", {})

    # Check if expandable_segments is enabled
    return allocator_settings.get("expandable_segments", False)


@functools.cache
def _check_cuda_expandable_segments_enabled() -> bool:
    """
    Check if PyTorch CUDA caching allocator is using expandable segments.

    Returns:
        bool: True if expandable segments are enabled, False otherwise
    """
    try:
        # Call the Python implementation of pt_cuda_allocator_compatibility
        pt_cuda_compat = pt_cuda_allocator_compatibility()

        if not pt_cuda_compat:
            warnings.warn(
                "CUDA caching allocator is not using expandable segments.\n"
                "This is required to maximize RDMA performance with CUDA tensors.\n\n"
                "To fix this, set the environment variable BEFORE importing PyTorch:\n"
                "1. In shell:\n"
                '   export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"\n'
                "2. Or in Python script (BEFORE any PyTorch imports):\n"
                "   import os\n"
                '   os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"\n'
                "   import torch  # Must come after setting the env var\n\n",
                UserWarning,
                stacklevel=2,
            )
            return False
        return True

    except Exception as e:
        warnings.warn(
            "Unable to verify CUDA allocator configuration.\n"
            "Please ensure expandable segments are enabled for best RDMA performance with CUDA tensors:\n"
            '   export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"\n'
            "Set this environment variable before importing PyTorch.",
            UserWarning,
            stacklevel=2,
        )
        return False


class RDMABuffer:
    def __init__(
        self,
        data: torch.Tensor | memoryview,
    ) -> None:
        """
        RDMABuffer supports 1d contiguous tensors (including tensor views/slices) or 1d c-contiguous memoryviews.

        Args:
            data: torch.Tensor or memoryview to create the buffer from. Must be 1d and contiguous.
                  If provided, addr and size must not be specified.

        Raises:
            ValueError: If data is not 1d contiguous, if size is 0, or if data is a GPU tensor.
            RuntimeError: If RDMA is not available on this platform.

        Note:
            Currently only CPU tensors are supported. GPU tensor support will be added in the future.

        TODO: Create TensorBuffer, which will be main user API supporting non-contiguous tensors
        """
        if isinstance(data, torch.Tensor) and data.device.type == "cuda":
            # Check if CUDA caching allocator is using expandable segments
            _check_cuda_expandable_segments_enabled()

        assert is_rdma_available(), (
            "Tried to create an RDMABuffer, but RDMA is not available on this platform."
        )

        # We need to ensure that _RdmaManager is initialized at this point, because under the hood
        # _RdmaBuffer.create_rdma_buffer_blocking relies on this being the case.
        _ensure_init_rdma_manager().block_on()

        addr, size = _get_addr_and_size(data)

        try:
            if size == 0:
                raise ValueError("Cannot create RDMABuffer with size 0.")
            ctx = context()
            self._buffer: _RdmaBuffer = _RdmaBuffer.create_rdma_buffer_blocking(
                addr=addr,
                size=size,
                proc_id=ctx.actor_instance.proc_id,
                client=ctx.actor_instance,
            )

            # Store metadata for TCP fallback transport
            # Use the Rust buffer's name as unique identifier
            self._buffer_id: str = self._buffer.name
            self._dtype: Optional[torch.dtype] = (
                data.dtype if isinstance(data, torch.Tensor) else None
            )

            # Register in global buffer registry for local TCP access
            _buffer_registry[self._buffer_id] = data

            # Get the TcpDataActor for this process and register the buffer
            # The actor reference is serializable and can be sent to remote processes
            self._tcp_data_actor: TcpDataActor = _get_tcp_data_actor_blocking()
            # Register asynchronously (fire and forget for now)
            # The data is also in _buffer_registry for immediate local access
            self._tcp_data_actor.register_buffer.call_one(self._buffer_id, data)

        # TODO - specific exception
        except Exception as e:
            logging.error("Failed to create buffer %s", e)
            raise e

    def size(self) -> int:
        return self._buffer.size()

    def read_into(
        self,
        dst: torch.Tensor | memoryview,
        *,
        timeout: int = 3,
        transport: Transport = "best",
    ) -> Future[Optional[int]]:
        """
        Read data from the RDMABuffer into a destination tensor.

        The destination tensor must be contiguous (including tensor views/slices).
        Args:
            dst: Destination tensor or memoryview to read into.
        Keyword Args:
            timeout (int, optional): Timeout in seconds for the operation. Defaults to 3s.
            transport (Transport, optional): Transport to use for the operation.
                - "best": Auto-select best available (RDMA NIC if available, else TCP)
                - "tcp": Force TCP transport (uses actor messaging)
                - "nic": Force RDMA NIC (error if unavailable)
                Defaults to "best".
        Returns:
            Future[Optional[int]]: A Monarch Future that can be awaited or called with .get() for blocking operation.

        Raises:
            ValueError: If the destination tensor size is smaller than the RDMA buffer size.
            RuntimeError: If transport="nic" but RDMA is not available.

        Note:
            Currently only CPU tensors are fully supported. GPU tensors will be temporarily
            copied to CPU, which may impact performance.
        """
        dst_addr, dst_size = _get_addr_and_size(dst)

        if self.size() > dst_size:
            raise ValueError(
                f"Destination tensor size ({dst_size}) must be >= RDMA buffer size ({self.size()})"
            )

        # Determine which transport to use
        use_tcp = self._should_use_tcp(transport)

        if use_tcp:
            return self._read_into_tcp(dst, timeout)
        else:
            return self._read_into_rdma(dst, dst_addr, dst_size, timeout)

    def _should_use_tcp(self, transport: Transport) -> bool:
        """Determine if TCP transport should be used based on transport setting."""
        if transport == "tcp":
            return True
        elif transport == "nic":
            if not is_rdma_available():
                raise RuntimeError(
                    "Transport 'nic' requested but RDMA is not available on this platform"
                )
            return False
        else:  # "best"
            return not is_rdma_available()

    def _read_into_rdma(
        self,
        dst: torch.Tensor | memoryview,
        dst_addr: int,
        dst_size: int,
        timeout: int,
    ) -> Future[Optional[int]]:
        """Read using RDMA NIC transport (existing implementation)."""
        local_proc_id = context().actor_instance.proc_id
        client = context().actor_instance

        async def read_into_nonblocking() -> Optional[int]:
            await _ensure_init_rdma_manager()

            res = await self._buffer.read_into(
                addr=dst_addr,
                size=dst_size,
                local_proc_id=local_proc_id,
                client=client,
                timeout=timeout,
            )
            return res

        return Future(coro=read_into_nonblocking())

    def _read_into_tcp(
        self,
        dst: torch.Tensor | memoryview,
        timeout: int,
    ) -> Future[Optional[int]]:
        """Read using TCP transport (actor messaging fallback)."""
        buffer_id = self._buffer_id
        dtype = self._dtype
        tcp_actor = self._tcp_data_actor

        async def read_into_tcp_impl() -> Optional[int]:
            # Try local registry first (fast path for same-process)
            local_data = _buffer_registry.get(buffer_id)

            if local_data is not None:
                # Local access - direct copy
                if isinstance(dst, torch.Tensor) and isinstance(local_data, torch.Tensor):
                    dst.copy_(local_data)
                elif isinstance(dst, memoryview) and isinstance(local_data, memoryview):
                    dst[:] = local_data
                elif isinstance(dst, torch.Tensor):
                    src_tensor = torch.frombuffer(bytearray(local_data), dtype=dst.dtype)
                    dst.copy_(src_tensor)
                else:
                    dst[:] = local_data.numpy().tobytes()
            else:
                # Remote access - call TcpDataActor on owner's process
                data_bytes: bytes = await tcp_actor.fetch_buffer_data.call_one(buffer_id)

                # Copy received bytes to destination
                if isinstance(dst, torch.Tensor):
                    src_tensor = torch.frombuffer(
                        bytearray(data_bytes), dtype=dtype or dst.dtype
                    )
                    dst.copy_(src_tensor)
                else:
                    dst[:] = data_bytes

            return None

        return Future(coro=read_into_tcp_impl())

    def write_from(
        self,
        src: torch.Tensor | memoryview,
        *,
        timeout: int = 3,
        transport: Transport = "best",
    ) -> Future[None]:
        """
        Write data from a source tensor into the RDMABuffer.

        Args:
            src: Source tensor containing data to be written to the RDMA buffer.
                                Must be a contiguous tensor (including tensor views/slices).
                                Either src or addr/size must be provided.
        Keyword Args:
            timeout (int, optional): Timeout in seconds for the operation. Defaults to 3s.
            transport (Transport, optional): Transport to use for the operation.
                - "best": Auto-select best available (RDMA NIC if available, else TCP)
                - "tcp": Force TCP transport (uses actor messaging)
                - "nic": Force RDMA NIC (error if unavailable)
                Defaults to "best".

        Returns:
            Future[None]: A Monarch Future object that can be awaited or called with .get()
                         for blocking operation. Returns None when completed successfully.

        Raises:
            ValueError: If the source tensor size exceeds the RDMA buffer size.
            RuntimeError: If transport="nic" but RDMA is not available.

        Note:
            Currently only CPU tensors are fully supported. GPU tensors will be temporarily
            copied to CPU, which may impact performance.
        """

        src_addr, src_size = _get_addr_and_size(src)

        if src_size > self.size():
            raise ValueError(
                f"Source tensor size ({src_size}) must be <= RDMA buffer size ({self.size()})"
            )

        # Determine which transport to use
        use_tcp = self._should_use_tcp(transport)

        if use_tcp:
            return self._write_from_tcp(src, timeout)
        else:
            return self._write_from_rdma(src, src_addr, src_size, timeout)

    def _write_from_rdma(
        self,
        src: torch.Tensor | memoryview,
        src_addr: int,
        src_size: int,
        timeout: int,
    ) -> Future[None]:
        """Write using RDMA NIC transport (existing implementation)."""
        local_proc_id = context().actor_instance.proc_id
        client = context().actor_instance

        async def write_from_nonblocking() -> None:
            await _ensure_init_rdma_manager()

            res = await self._buffer.write_from(
                addr=src_addr,
                size=src_size,
                local_proc_id=local_proc_id,
                client=client,
                timeout=timeout,
            )
            return res

        return Future(coro=write_from_nonblocking())

    def _write_from_tcp(
        self,
        src: torch.Tensor | memoryview,
        timeout: int,
    ) -> Future[None]:
        """Write using TCP transport (actor messaging fallback)."""
        buffer_id = self._buffer_id
        tcp_actor = self._tcp_data_actor

        async def write_from_tcp_impl() -> None:
            # Try local registry first (fast path for same-process)
            dst_data = _buffer_registry.get(buffer_id)

            if dst_data is not None:
                # Local access - direct copy
                if isinstance(dst_data, torch.Tensor) and isinstance(src, torch.Tensor):
                    dst_data.copy_(src)
                elif isinstance(dst_data, memoryview) and isinstance(src, memoryview):
                    dst_data[:] = src
                elif isinstance(dst_data, torch.Tensor):
                    src_tensor = torch.frombuffer(bytearray(src), dtype=dst_data.dtype)
                    dst_data.copy_(src_tensor)
                else:
                    dst_data[:] = src.numpy().tobytes()
            else:
                # Remote access - call TcpDataActor on owner's process
                # Convert source to bytes
                if isinstance(src, torch.Tensor):
                    src_bytes = src.numpy().tobytes()
                else:
                    src_bytes = bytes(src)

                await tcp_actor.write_buffer_data.call_one(buffer_id, src_bytes)

        return Future(coro=write_from_tcp_impl())

    def drop(self) -> Future[None]:
        """
        Release the handle on the memory that the src holds to this memory.

        This also removes the buffer from the TCP registry if it was registered.
        """
        local_proc_id = context().actor_instance.proc_id
        client = context().actor_instance
        buffer_id = self._buffer_id
        tcp_actor = self._tcp_data_actor

        async def drop_nonblocking() -> None:
            await _ensure_init_rdma_manager()

            await self._buffer.drop(
                local_proc_id=local_proc_id,
                client=client,
            )

            # Clean up local TCP registry entry
            _buffer_registry.pop(buffer_id, None)

            # Unregister from TcpDataActor
            await tcp_actor.unregister_buffer.call_one(buffer_id)

        return Future(coro=drop_nonblocking())

    @property
    def owner(self) -> str:
        """
        The owner reference (str)
        """
        return self._buffer.owner_actor_id()


LocalMemory = torch.Tensor | memoryview


class RDMAAction:
    """
    Schedule a bunch of actions at once. This provides an opportunity to
    optimize bulk RDMA transactions without exposing complexity to users.

    Args:
        transport: Default transport to use for all operations. Can be overridden
            per-operation. Defaults to "best".
    """

    class RDMAOp(Enum):
        """Enumeration of RDMA operation types."""

        READ_INTO = "read_into"
        WRITE_FROM = "write_from"
        FETCH_ADD = "fetch_add"
        COMPARE_AND_SWAP = "compare_and_swap"

    def __init__(self, transport: Transport = "best") -> None:
        self._instructs: List[
            Tuple[RDMAAction.RDMAOp, RDMABuffer, LocalMemory, Transport]
        ] = []
        self._memory_dependencies: Dict[Tuple[int, int], RDMAAction.RDMAOp] = {}
        self._default_transport: Transport = transport

    def _check_and_merge_overlapping_range(
        self, addr: int, size: int, op: "RDMAAction.RDMAOp"
    ) -> None:
        """
        Check for overlapping ranges and merge if found.

        Returns the final range to use (either new_range or expanded merged range).
        Updates self._memory_dependencies in place if merging occurs.
        """
        new_start, new_end = addr, addr + size

        # Find overlapping range
        overlapping_range = None
        for existing_start, existing_end in self._memory_dependencies:
            # Check if ranges overlap
            if not (new_end <= existing_start or existing_end <= new_start):
                overlapping_range = (existing_start, existing_end)
                break

        # No overlap found - good to go
        if overlapping_range is None:
            self._memory_dependencies[(new_start, new_end)] = op
            return

        # Overlap found - merge ranges
        existing_op = self._memory_dependencies[overlapping_range]

        # Merge ops, only safe if neither is write_from at the moment
        if existing_op == self.RDMAOp.WRITE_FROM or op == self.RDMAOp.WRITE_FROM:
            raise ValueError(
                f"Same data range already has a write_from within RDMAAction: {existing_op} vs {op}"
            )

        # Create expanded range that covers both
        expanded_range = (
            min(overlapping_range[0], new_start),
            max(overlapping_range[1], new_end),
        )

        # range is unchanged - no need to update
        if expanded_range == (new_start, new_end):
            return

        # Update dictionary: remove old range, add expanded range
        del self._memory_dependencies[overlapping_range]
        self._memory_dependencies[expanded_range] = op

        # now since merged, possible need to merge again
        return self._check_and_merge_overlapping_range(
            expanded_range[0], expanded_range[1] - expanded_range[0], op
        )

    def read_into(
        self,
        src: RDMABuffer,
        dst: LocalMemory | List[LocalMemory],
        transport: Optional[Transport] = None,
    ) -> Self:
        """
        Read from src RDMA buffer into dst memory.

        Args:
            src: Source RDMA buffer to read from
            dst: Destination local memory to read into
                   If dst is a list, it is the concatenation of the data in the list
            transport: Transport to use for this operation. If None, uses the
                default transport set in __init__. Defaults to None.
        """
        # Throw NotImplementedError for lists to simplify logic
        if isinstance(dst, list):
            raise NotImplementedError("List destinations not yet supported")

        addr, size = _get_addr_and_size(dst)

        if size < src.size():
            raise ValueError(
                f"dst memory size ({size}) must be >= src buffer size ({src.size()})"
            )

        self._check_and_merge_overlapping_range(addr, size, self.RDMAOp.READ_INTO)

        effective_transport = transport if transport is not None else self._default_transport
        self._instructs.append((self.RDMAOp.READ_INTO, src, dst, effective_transport))

        return self

    def write_from(
        self,
        src: RDMABuffer,
        dst: LocalMemory | List[LocalMemory],
        transport: Optional[Transport] = None,
    ) -> Self:
        """
        Write from dst memory to src RDMA buffer.

        Args:
            src: Destination RDMA buffer to write to
            dst: Source local memory to write from
                   If local is a list, it is the concatenation of the data in the list
            transport: Transport to use for this operation. If None, uses the
                default transport set in __init__. Defaults to None.
        """
        # Throw NotImplementedError for lists to simplify logic
        if isinstance(dst, list):
            raise NotImplementedError("List sources not yet supported")

        addr, size = _get_addr_and_size(dst)

        if size > src.size():
            raise ValueError(
                f"Local memory size ({size}) must be <= src buffer size ({src.size()})"
            )

        self._check_and_merge_overlapping_range(addr, size, self.RDMAOp.WRITE_FROM)

        effective_transport = transport if transport is not None else self._default_transport
        self._instructs.append((self.RDMAOp.WRITE_FROM, src, dst, effective_transport))

        return self

    def fetch_add(self, src: RDMABuffer, dst: LocalMemory, add: int) -> Self:
        """
        Perform atomic fetch-and-add operation on src RDMA buffer.

        Args:
            src: src RDMA buffer to perform operation on
            dst: Local memory to store the original value
            add: Value to add to the src buffer

        Atomically:
            *dst = *src
            *src = *src + add

        Note: src/dst are 8 bytes
        """
        raise NotImplementedError("Not yet supported")

    def compare_and_swap(
        self, src: RDMABuffer, dst: LocalMemory, compare: int, swap: int
    ) -> Self:
        """
        Perform atomic compare-and-swap operation on src RDMA buffer.

        Args:
            src: src RDMA buffer to perform operation on
            dst: Local memory to store the original value
            compare: Value to compare against
            swap: Value to swap in if comparison succeeds

        Atomically:
            *dst = *src;
            if (*src == compare) {
                *src = swap
            }

        Note: src/dst are 8 bytes
        """
        raise NotImplementedError("Not yet supported")

    def submit(self) -> Future[None]:
        """
        Schedules the work (can be called multiple times to schedule the same work more than once).
        Future completes when all the work is done.

        Executes futures for each src actor independently and concurrently for optimal performance.
        """

        async def submit_all_work() -> None:
            if not self._instructs:
                return

            work = defaultdict(list)

            # Group operations by owner for concurrent execution per owner
            for op, src, dst, transport in self._instructs:
                if op == self.RDMAOp.READ_INTO:
                    fut = src.read_into(dst, transport=transport)
                elif op == self.RDMAOp.WRITE_FROM:
                    fut = src.write_from(dst, transport=transport)
                else:
                    raise NotImplementedError(f"Unknown RDMA operation: {op}")
                work[src.owner].append(fut)

            # Create a list of tasks, one per owner, that wait for all that owner's futures sequentially
            owner_tasks = []

            for _, futures in work.items():
                # Create a coroutine that processes all futures for a qp sequentially
                async def process_owner_futures(owner_futures_list=futures):
                    """Process all futures for a single qp sequentially"""
                    for future in owner_futures_list:
                        await future

                # Convert to PythonTask for Monarch's native concurrency
                owner_task = PythonTask.from_coroutine(process_owner_futures())
                owner_tasks.append(owner_task)

            # Spawn all owner tasks concurrently and collect their shared handles
            shared_tasks = [task.spawn() for task in owner_tasks]

            # Wait for all owner tasks to complete concurrently
            for shared_task in shared_tasks:
                await shared_task

        return Future(coro=submit_all_work())
