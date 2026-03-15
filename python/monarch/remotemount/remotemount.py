# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import subprocess
import time
from typing import Optional

from monarch._rust_bindings.monarch_extension.fast_pack import (
    load_file_into_buffer as _c_load_file_into_buffer,
    pack_files_to_shm as _c_pack_files_to_shm,
    pack_files_with_offsets as _c_pack_files,
)
from monarch.actor import Actor, endpoint

logger = logging.getLogger(__name__)

CHUNK_SIZE = (1024 * 1024 * 1024) * 8
HASH_BLOCK_SIZE = 64 * 1024 * 1024  # 64MB blocks for incremental diffing
RDMA_PARALLEL_TLS_THRESHOLD = 8  # blocks <= this: TLS to all workers; above: RDMA fan-out
CACHE_DIR = "/tmp/monarch_remotemount_cache"
FRAG_THRESHOLD = 0.2  # max dead-space ratio before sequential repack


def block_hashes(data_mv, block_size=HASH_BLOCK_SIZE):
    """Compute xxhash per block of a packed memoryview."""
    import xxhash

    hashes = []
    for i in range(0, len(data_mv), block_size):
        hashes.append(xxhash.xxh64(bytes(data_mv[i : i + block_size])).hexdigest())
    return hashes


def classify_workers(client_hashes, client_total_size, worker_states):
    """Classify workers as fresh, partial, or stale.

    Args:
        client_hashes: list of block hash strings (or None for assumed-clean
            blocks in incremental mode) from client
        client_total_size: total packed data size on client
        worker_states: list of (remote_hashes, remote_size) tuples

    Returns:
        (fresh_ranks, worker_dirty) where:
        - fresh_ranks: list of rank indices that are up-to-date
        - worker_dirty: dict {rank: list[int] | None} — dirty block
          indices for partial workers, or None for stale workers
    """
    fresh_ranks = []
    worker_dirty = {}
    for rank, (remote_hashes, remote_size) in enumerate(worker_states):
        if (
            remote_hashes
            and len(remote_hashes) == len(client_hashes)
            and remote_size == client_total_size
            and all(
                ch is None or ch == rh
                for ch, rh in zip(client_hashes, remote_hashes)
            )
        ):
            fresh_ranks.append(rank)
        elif remote_hashes:
            # Partial: compare overlapping blocks, mark new/changed as dirty.
            # Skip blocks where client hash is None (assumed unchanged in
            # incremental mode).
            min_blocks = min(len(remote_hashes), len(client_hashes))
            dirty = [
                i
                for i in range(min_blocks)
                if client_hashes[i] is not None
                and remote_hashes[i] != client_hashes[i]
            ]
            # Any new blocks beyond the old count that have real hashes.
            dirty.extend(
                i
                for i in range(min_blocks, len(client_hashes))
                if client_hashes[i] is not None
            )
            # If size changed, the last overlapping block likely changed
            # (partial block at the boundary may have different content).
            if remote_size != client_total_size and min_blocks > 0:
                last = min_blocks - 1
                if last not in dirty and client_hashes[last] is not None:
                    dirty.append(last)
                    dirty.sort()
            worker_dirty[rank] = dirty
        else:
            worker_dirty[rank] = None
    return fresh_ranks, worker_dirty


def _load_pack_index(path):
    """Load JSON pack index from disk. Returns dict or None."""
    import json

    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _save_pack_index(path, index_data):
    """Write pack index as JSON."""
    import json

    try:
        with open(path, "w") as f:
            json.dump(index_data, f)
    except OSError:
        logger.warning(f"Failed to save pack index to {path}", exc_info=True)


def _compute_file_hashes(staging_mv, file_entries, offset_map):
    """Compute xxh64 per file from packed buffer. Returns {vpath: hash_hex}."""
    import xxhash

    hashes = {}
    for vpath, _full_path, file_len, _mtime_ns in file_entries:
        offset = offset_map[vpath]
        # Hash directly from memoryview slice — avoids a full bytes() copy.
        hashes[vpath] = xxhash.xxh64(
            staging_mv[offset : offset + file_len]
        ).hexdigest()
    return hashes


def _assign_offsets(file_entries, previous_index):
    """Append-only offset assignment.

    Unchanged files keep their original offsets. Changed/new files are
    appended after the previous total size.

    Args:
        file_entries: [(vpath, full_path, file_len, mtime_ns), ...]
        previous_index: dict with 'total_size' and 'files' keys

    Returns:
        (offset_map, new_total_size, dead_space)
    """
    prev_files = previous_index.get("files", {})
    prev_total = previous_index.get("total_size", 0)

    offset_map = {}
    dead_space = 0
    append_offset = prev_total

    current_vpaths = set()
    for vpath, _full_path, file_len, mtime_ns in file_entries:
        current_vpaths.add(vpath)
        prev = prev_files.get(vpath)
        if prev and prev["size"] == file_len and prev["mtime_ns"] == mtime_ns:
            # File unchanged — keep old offset
            offset_map[vpath] = prev["offset"]
        else:
            # File changed or new — append at the end
            if prev:
                dead_space += prev["size"]
            offset_map[vpath] = append_offset
            append_offset += file_len

    # Deleted files contribute dead space
    for vpath, info in prev_files.items():
        if vpath not in current_vpaths:
            dead_space += info["size"]

    return offset_map, append_offset, dead_space


def pack_directory_chunked(source_path, chunk_size=None, use_shm=False, previous_index=None):
    """Walk a directory, pack all files into contiguous mmap chunks.

    When *previous_index* is provided (from a prior run's pack index), files
    whose ``(mtime_ns, size)`` match the index keep their original offsets and
    changed/new files are appended at the end of the buffer.  If the resulting
    dead-space ratio exceeds ``FRAG_THRESHOLD``, the layout falls back to
    sequential packing.

    Returns (fs_metadata, staging_mv, chunks, shm_path, block_hashes_list, pack_index)
    where:
    - fs_metadata: dict mapping virtual paths to stat/offset metadata
    - staging_mv: memoryview over the packed data
    - chunks: list of chunk-sized memoryview slices
    - shm_path: path to named pack file if use_shm=True, else None
    - block_hashes_list: list of xxh64 hex digest strings per 64 MB block
    - pack_index: dict with per-file offset/size/mtime/hash for incremental packing
    """
    if chunk_size is None:
        chunk_size = CHUNK_SIZE

    fs_metadata = {}
    file_entries = []  # [(vpath, full_path, file_len, mtime_ns)]

    source_path = os.path.abspath(source_path)

    for root, dirs, files in os.walk(source_path):
        rel_path = root[len(source_path) :]
        if rel_path == "":
            rel_path = "/"

        # Directory Metadata
        st = os.stat(root)
        fs_metadata[rel_path] = {
            "attr": {
                key: getattr(st, key)
                for key in (
                    "st_atime",
                    "st_ctime",
                    "st_gid",
                    "st_mode",
                    "st_mtime",
                    "st_nlink",
                    "st_size",
                    "st_uid",
                )
            },
            "children": dirs + files,
        }

        for f in files:
            full_path = os.path.join(root, f)
            virtual_path = (rel_path + "/" + f) if rel_path != "/" else ("/" + f)

            lst = os.lstat(full_path)
            is_symlink = (lst.st_mode & 0o170000) == 0o120000

            if is_symlink:
                fs_metadata[virtual_path] = {
                    "attr": {
                        key: getattr(lst, key)
                        for key in (
                            "st_atime",
                            "st_ctime",
                            "st_gid",
                            "st_mode",
                            "st_mtime",
                            "st_nlink",
                            "st_size",
                            "st_uid",
                        )
                    },
                    "link_target": os.readlink(full_path),
                }
            else:
                file_len = lst.st_size
                mtime_ns = lst.st_mtime_ns
                attr = {
                    key: getattr(lst, key)
                    for key in (
                        "st_atime",
                        "st_ctime",
                        "st_gid",
                        "st_mode",
                        "st_mtime",
                        "st_nlink",
                        "st_size",
                        "st_uid",
                    )
                }
                attr["st_size"] = file_len

                # Defer global_offset — assigned after offset-assignment phase.
                fs_metadata[virtual_path] = {
                    "attr": attr,
                    "file_len": file_len,
                }

                file_entries.append((virtual_path, full_path, file_len, mtime_ns))

    # --- Offset assignment phase ---
    use_append = False
    if previous_index and previous_index.get("files"):
        offset_map, total_size, dead_space = _assign_offsets(
            file_entries, previous_index
        )
        if total_size > 0 and dead_space / total_size > FRAG_THRESHOLD:
            logger.info(
                f"Fragmentation {dead_space / total_size:.1%} exceeds threshold "
                f"{FRAG_THRESHOLD:.0%}, repacking sequentially"
            )
        else:
            n_reused = sum(
                1
                for vpath, _, flen, mns in file_entries
                if previous_index["files"].get(vpath, {}).get("size") == flen
                and previous_index["files"].get(vpath, {}).get("mtime_ns") == mns
            )
            logger.info(
                f"Append-only layout: {n_reused}/{len(file_entries)} files reused, "
                f"dead_space={dead_space // 1024}KiB "
                f"({dead_space / total_size:.1%} of {total_size // (1024**2)}MiB)"
            )
            use_append = True

    if not use_append:
        # Sequential offsets (current behavior)
        offset_map = {}
        current_offset = 0
        for vpath, _full_path, file_len, _mtime_ns in file_entries:
            offset_map[vpath] = current_offset
            current_offset += file_len
        total_size = current_offset

    # Set global_offset in fs_metadata and build file_list for Rust packer.
    file_list = []
    for vpath, full_path, file_len, _mtime_ns in file_entries:
        fs_metadata[vpath]["global_offset"] = offset_map[vpath]
        file_list.append((full_path, offset_map[vpath], file_len))

    logger.info(f"Packing {total_size // (1024**2)}MiB, {len(file_list)} files")

    if total_size == 0:
        return fs_metadata, None, [], None, [], None

    # Always use anonymous mmap (no /tmp file write needed for TLS transfer).
    staging_mv, hashes = _c_pack_files(file_list, total_size)
    shm_path = None

    chunks = [
        staging_mv[i : i + chunk_size] for i in range(0, len(staging_mv), chunk_size)
    ]

    # Compute per-file content hashes and build pack index.
    file_hashes = _compute_file_hashes(staging_mv, file_entries, offset_map)
    new_pack_index = {
        "total_size": total_size,
        "files": {
            vpath: {
                "offset": offset_map[vpath],
                "size": file_len,
                "mtime_ns": mtime_ns,
                "content_hash": file_hashes[vpath],
            }
            for vpath, _full_path, file_len, mtime_ns in file_entries
        },
    }

    return fs_metadata, staging_mv, chunks, shm_path, list(hashes), new_pack_index


class FUSEActor(Actor):
    def __init__(self, chunk_size, backend="slurm"):
        self.chunk_size = chunk_size
        self.backend = backend
        self.meta = None
        self.chunks = []
        self._chunk_storage = None
        self._chunk_offsets = None
        self._next_chunk_idx = 0
        self._block_hashes = []
        self._total_size = 0
        self._fuse_handle = None
        self._cache_path = None
        self._tls_receiver = None
        self._pack_index = {}
        self._rdma_staging = None
        self._rdma_staging_mv = None

    @endpoint
    def try_load_cache(self, cache_key):
        """Load cached chunk data from a previous run, if available.

        Sets ``_cache_path`` so that subsequent ``init_chunk_storage`` and
        ``collect_shards`` calls use file-backed mmap. If a cache file
        already exists, mmaps it and computes block hashes so the client
        can classify this worker as fresh or partial.
        """
        import mmap

        os.makedirs(CACHE_DIR, exist_ok=True)
        self._cache_path = os.path.join(CACHE_DIR, cache_key)

        try:
            if os.path.exists(self._cache_path):
                size = os.path.getsize(self._cache_path)
                if size > 0:
                    fd = os.open(self._cache_path, os.O_RDWR)
                    try:
                        self._chunk_storage = mmap.mmap(fd, size)
                    finally:
                        os.close(fd)
                    self._chunk_storage_mv = memoryview(self._chunk_storage)
                    self._total_size = size
                    self._block_hashes = list(
                        _c_load_file_into_buffer(self._cache_path, self._chunk_storage_mv)
                    )
                    self._pack_index = (
                        _load_pack_index(self._cache_path + ".index") or {}
                    )

                    # Build chunks list so mount() works with cached data.
                    self.chunks = []
                    self._chunk_offsets = []
                    remaining = size
                    off = 0
                    idx = 0
                    while remaining > 0:
                        sz = min(remaining, self.chunk_size)
                        self._chunk_offsets.append((off, sz))
                        self.chunks.append(self._chunk_storage_mv[off : off + sz])
                        off += sz
                        remaining -= sz
                        idx += 1
                    self._next_chunk_idx = idx

                    logger.info(
                        f"[CACHE] Loaded {self._cache_path}: "
                        f"{size // (1024**2)}MiB, "
                        f"{len(self._block_hashes)} block hashes"
                    )
        except Exception:
            logger.warning(
                f"[CACHE] Failed to load cache {self._cache_path}, "
                "will do full transfer",
                exc_info=True,
            )
            self._block_hashes = []
            self._total_size = 0

    @endpoint
    def set_meta(self, meta):
        self.meta = meta

    @endpoint
    def init_chunk_storage(self, chunk_sizes):
        import mmap

        # Reset state from any previous transfer.
        self.chunks = []
        self._next_chunk_idx = 0

        total_size = sum(chunk_sizes)
        if self._cache_path:
            fd = os.open(self._cache_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                os.ftruncate(fd, total_size)
                self._chunk_storage = mmap.mmap(fd, total_size)
            finally:
                os.close(fd)
        else:
            self._chunk_storage = mmap.mmap(
                -1, total_size, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
            )
        self._chunk_storage_mv = memoryview(self._chunk_storage)
        self._chunk_offsets = []
        offset = 0
        for size in chunk_sizes:
            self._chunk_offsets.append((offset, size))
            offset += size
        self._next_chunk_idx = 0

        # EFA requires explicit initialization of its manager actor on each worker.
        # Unlike ibverbs which lazily initializes its RDMA context when creating
        # buffers, EFA uses an actor-based approach (EfaManagerActor) that must be
        # spawned before workers can register destination buffers for RDMA transfers.
        # Only needed on SLURM (EFA), not MAST (ibverbs).
        if self.backend == "slurm":
            from monarch.rdma import is_rdma_available

            if not is_rdma_available():
                try:
                    logger.debug("[WORKER] init_chunk_storage: starting EFA init")
                    from monarch._src.rdma.rdma import _ensure_init_efa_manager

                    _ensure_init_efa_manager().block_on()
                    logger.debug("[WORKER] init_chunk_storage: EFA init complete")
                except ImportError:
                    logger.debug(
                        "[WORKER] init_chunk_storage: EFA APIs not available, skipping"
                    )

    @endpoint
    def fetch_chunk_rdma(self, rdma_buffer, chunk_size: int, timeout: int = 300):
        """Receive RDMABuffer (works with both ibverbs and EFA) and read from it."""
        import mmap as _mmap

        idx = self._next_chunk_idx

        # Copy into pre-allocated mmap via RDMA (ibverbs or TCP fallback).
        offset, _ = self._chunk_offsets[idx]
        dst_mv = self._chunk_storage_mv[offset : offset + chunk_size]
        t1 = time.time()

        if self._cache_path:
            # RDMA memory registration fails on file-backed MAP_SHARED pages.
            # Receive into an anonymous buffer, then copy to the cache file.
            # Don't explicitly close the anonymous mmap — it can raise
            # BufferError if the Rust RDMABuffer still holds a reference.
            anon = _mmap.mmap(-1, chunk_size, _mmap.MAP_PRIVATE | _mmap.MAP_ANONYMOUS)
            anon_mv = memoryview(anon)
            rdma_buffer.read_into(anon_mv, timeout=timeout).get()
            dst_mv[:] = anon_mv
        else:
            rdma_buffer.read_into(dst_mv, timeout=timeout).get()

        t2 = time.time()
        self.chunks.append(dst_mv)
        self._next_chunk_idx += 1
        gbps = (chunk_size * 8.0 / 1e9) / max(t2 - t1, 1e-9)
        logger.info(
            f"[WORKER] fetch_chunk {idx}: {chunk_size / (1024**2):.0f}MiB "
            f"in {t2 - t1:.3f}s ({gbps:.1f} Gbps)"
        )

    @endpoint
    def fanout_chunk_rdma(
        self, peer_actors, chunk_size: int, chunk_idx: int = -1, timeout: int = 300
    ):
        """Fan out a chunk to peer workers via RDMA.

        Args:
            peer_actors: Mesh of peer FUSEActors to receive the chunk.
            chunk_size: Size of the chunk in bytes.
            chunk_idx: Which chunk to fan out. Defaults to -1 (last received).
            timeout: RDMA timeout in seconds.
        """
        import mmap

        from monarch.rdma import RDMABuffer

        t0 = time.time()
        idx = chunk_idx if chunk_idx >= 0 else self._next_chunk_idx - 1
        offset, _ = self._chunk_offsets[idx]
        src_mv = self._chunk_storage_mv[offset : offset + chunk_size]

        # RDMA memory registration (ibv_reg_mr) can fail on file-backed
        # MAP_SHARED mmap pages.  Copy into an anonymous buffer so the
        # RDMABuffer always uses anonymous memory.
        anon = None
        if self._cache_path:
            anon = mmap.mmap(-1, chunk_size, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            anon_mv = memoryview(anon)
            anon_mv[:] = src_mv
            rdma_buffer = RDMABuffer(anon_mv)
        else:
            rdma_buffer = RDMABuffer(src_mv)
        flat_peers = peer_actors.flatten("rank")
        t1 = time.time()
        futures = []
        for rank in range(len(flat_peers)):
            peer = flat_peers.slice(rank=rank)
            futures.append(peer.fetch_chunk_rdma.call(rdma_buffer, chunk_size, timeout))
        t2 = time.time()
        for f in futures:
            f.get()
        t3 = time.time()

        # Anonymous mmap is reclaimed on GC; explicit close() can raise
        # BufferError if the Rust RDMABuffer still holds a reference.

        n = len(flat_peers)
        gbps = (chunk_size * n * 8.0 / 1e9) / max(t3 - t1, 1e-9)
        logger.info(
            f"[WORKER] fanout_chunk {idx}: setup={t1 - t0:.3f}s, "
            f"dispatch={t2 - t1:.3f}s, wait={t3 - t2:.3f}s, "
            f"total={t3 - t0:.3f}s ({gbps:.1f} Gbps aggregate, {n} peers)"
        )

    @endpoint
    def get_block_rdma_buffer(self, block_idx, total_size):
        """Copy a block into a reusable anonymous staging buffer and return an RDMABuffer.

        Uses a single 64MB anonymous mmap that is reused across calls.
        The caller MUST wait for all peers to finish reading before
        calling this again (the staging buffer is overwritten each time).
        """
        import mmap as _mmap

        from monarch.rdma import RDMABuffer

        block_size = min(HASH_BLOCK_SIZE, total_size - block_idx * HASH_BLOCK_SIZE)
        offset = block_idx * HASH_BLOCK_SIZE

        # Allocate staging once, reuse for all blocks.
        if self._rdma_staging is None:
            self._rdma_staging = _mmap.mmap(
                -1, HASH_BLOCK_SIZE, _mmap.MAP_PRIVATE | _mmap.MAP_ANONYMOUS
            )
            self._rdma_staging_mv = memoryview(self._rdma_staging)

        # Copy block data into anonymous staging buffer.
        staging_slice = self._rdma_staging_mv[:block_size]
        staging_slice[:] = self._chunk_storage_mv[offset : offset + block_size]

        return RDMABuffer(staging_slice)

    @endpoint
    def get_blocks_rdma_buffer(self, block_indices, total_size):
        """Copy multiple blocks into a contiguous staging buffer, return RDMABuffer.

        Blocks are packed contiguously (no gaps). The caller MUST wait for
        all peers to finish reading before calling this again.
        """
        import mmap as _mmap

        from monarch.rdma import RDMABuffer

        # Compute total staging size needed.
        staging_needed = 0
        for bi in block_indices:
            staging_needed += min(HASH_BLOCK_SIZE, total_size - bi * HASH_BLOCK_SIZE)

        # Grow staging buffer if needed.
        if self._rdma_staging is None or len(self._rdma_staging) < staging_needed:
            self._rdma_staging = _mmap.mmap(
                -1, staging_needed, _mmap.MAP_PRIVATE | _mmap.MAP_ANONYMOUS
            )
            self._rdma_staging_mv = memoryview(self._rdma_staging)

        # Copy blocks contiguously into staging.
        pos = 0
        for bi in block_indices:
            block_size = min(HASH_BLOCK_SIZE, total_size - bi * HASH_BLOCK_SIZE)
            offset = bi * HASH_BLOCK_SIZE
            self._rdma_staging_mv[pos : pos + block_size] = (
                self._chunk_storage_mv[offset : offset + block_size]
            )
            pos += block_size

        return RDMABuffer(self._rdma_staging_mv[:staging_needed])

    @endpoint
    def ensure_storage(self, total_size):
        """Ensure storage is allocated at the given size for RDMA reception."""
        import mmap as _mmap

        if self._chunk_storage is not None and self._total_size == total_size:
            return

        self._chunk_storage_mv = None
        self.chunks = []
        self._chunk_offsets = None

        if self._cache_path:
            fd = os.open(self._cache_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                os.ftruncate(fd, total_size)
                if self._chunk_storage is not None:
                    self._chunk_storage.close()
                self._chunk_storage = _mmap.mmap(fd, total_size)
            finally:
                os.close(fd)
        else:
            if self._chunk_storage is not None:
                self._chunk_storage.close()
            self._chunk_storage = _mmap.mmap(
                -1, total_size, _mmap.MAP_PRIVATE | _mmap.MAP_ANONYMOUS
            )

        self._chunk_storage_mv = memoryview(self._chunk_storage)
        self._total_size = total_size

        self.chunks = []
        self._chunk_offsets = []
        remaining = total_size
        off = 0
        while remaining > 0:
            sz = min(remaining, self.chunk_size)
            self._chunk_offsets.append((off, sz))
            self.chunks.append(self._chunk_storage_mv[off : off + sz])
            off += sz
            remaining -= sz
        self._next_chunk_idx = len(self._chunk_offsets)

    @endpoint
    def replace_block(
        self, block_idx: int, rdma_buffer, block_size: int, timeout: int = 300
    ):
        """Overwrite a single hash block in existing storage via RDMA.

        Uses a reusable anonymous staging buffer to avoid registering
        file-backed MAP_SHARED pages with ibv_reg_mr (which crashes).
        """
        import mmap as _mmap

        offset = block_idx * HASH_BLOCK_SIZE

        if self._cache_path:
            # Allocate staging once, reuse for all blocks.
            if self._rdma_staging is None:
                self._rdma_staging = _mmap.mmap(
                    -1, HASH_BLOCK_SIZE, _mmap.MAP_PRIVATE | _mmap.MAP_ANONYMOUS
                )
                self._rdma_staging_mv = memoryview(self._rdma_staging)

            staging_slice = self._rdma_staging_mv[:block_size]
            rdma_buffer.read_into(staging_slice, timeout=timeout).get()
            self._chunk_storage_mv[offset : offset + block_size] = staging_slice
        else:
            dst_mv = self._chunk_storage_mv[offset : offset + block_size]
            rdma_buffer.read_into(dst_mv, timeout=timeout).get()

    @endpoint
    def replace_blocks(
        self, block_indices, total_size, rdma_buffer, staging_size: int, timeout: int = 300
    ):
        """Overwrite multiple hash blocks from a single contiguous RDMA buffer.

        The rdma_buffer contains blocks packed contiguously (matching the
        order in block_indices). This reduces RDMA round-trips vs per-block.
        """
        import mmap as _mmap

        # Always need staging: RDMA reads into contiguous buffer, then we
        # scatter into the correct offsets in chunk storage.
        if self._rdma_staging is None or len(self._rdma_staging) < staging_size:
            self._rdma_staging = _mmap.mmap(
                -1, staging_size, _mmap.MAP_PRIVATE | _mmap.MAP_ANONYMOUS
            )
            self._rdma_staging_mv = memoryview(self._rdma_staging)

        staging_slice = self._rdma_staging_mv[:staging_size]
        rdma_buffer.read_into(staging_slice, timeout=timeout).get()

        # Scatter from contiguous staging into correct offsets.
        pos = 0
        for bi in block_indices:
            block_size = min(HASH_BLOCK_SIZE, total_size - bi * HASH_BLOCK_SIZE)
            offset = bi * HASH_BLOCK_SIZE
            self._chunk_storage_mv[offset : offset + block_size] = (
                staging_slice[pos : pos + block_size]
            )
            pos += block_size

    @endpoint
    def mount(self, mount_point, new_block_hashes=None, total_size=0, pack_index=None):
        import json

        from monarch._rust_bindings.monarch_extension.chunked_fuse import (
            mount_chunked_fuse,
        )

        # Flush mmap to disk so the cache file persists across actor restarts.
        if self._cache_path and self._chunk_storage is not None:
            try:
                self._chunk_storage.flush()
            except Exception:
                pass

        # Persist pack index alongside cached data.
        if pack_index is not None and self._cache_path:
            self._pack_index = pack_index
            _save_pack_index(self._cache_path + ".index", pack_index)

        self._fuse_handle = mount_chunked_fuse(
            json.dumps(self.meta),
            self.chunks,
            self.chunk_size,
            mount_point,
        )
        self._block_hashes = new_block_hashes or []
        self._total_size = total_size
        return 0

    @endpoint
    def get_block_hashes(self):
        """Return per-block hashes and total size of the mounted data."""
        return (self._block_hashes, self._total_size)

    @endpoint
    def get_pack_index(self):
        """Return the pack index for append-only packing."""
        return self._pack_index

    @endpoint
    def get_cache_path(self):
        """Return the cache file path for this worker."""
        return self._cache_path

    @endpoint
    def prepare_receiver(self, num_streams, total_size):
        """Create a Rust TLS receiver and return its address.

        Allocates chunk storage (file-backed or anonymous mmap) and creates
        a TlsReceiver that will write received blocks directly into it.
        """
        import mmap

        from monarch._rust_bindings.monarch_extension.tls_receiver import TlsReceiver

        # Allocate storage if not already present, or resize if needed.
        if self._chunk_storage is None or self._total_size != total_size:
            # Release existing memoryview/chunks before closing old mmap.
            self._chunk_storage_mv = None
            self.chunks = []
            self._chunk_offsets = None
            if self._cache_path:
                # Resize without O_TRUNC to preserve existing cached blocks.
                fd = os.open(
                    self._cache_path, os.O_RDWR | os.O_CREAT, 0o600
                )
                try:
                    os.ftruncate(fd, total_size)
                    if self._chunk_storage is not None:
                        self._chunk_storage.close()
                    self._chunk_storage = mmap.mmap(fd, total_size)
                finally:
                    os.close(fd)
            else:
                self._chunk_storage = mmap.mmap(
                    -1, total_size, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
                )
            self._chunk_storage_mv = memoryview(self._chunk_storage)
            self._total_size = total_size

            # Build chunks list for mount().
            self.chunks = []
            self._chunk_offsets = []
            remaining = total_size
            off = 0
            while remaining > 0:
                sz = min(remaining, self.chunk_size)
                self._chunk_offsets.append((off, sz))
                self.chunks.append(self._chunk_storage_mv[off : off + sz])
                off += sz
                remaining -= sz
            self._next_chunk_idx = len(self._chunk_offsets)

        self._tls_receiver = TlsReceiver(num_streams)
        return self._tls_receiver.addr

    @endpoint
    def receive_blocks(self):
        """Block until the TLS receiver has finished receiving all blocks."""
        if self._tls_receiver is None:
            raise RuntimeError("prepare_receiver() not called")
        self._tls_receiver.wait(self._chunk_storage_mv)
        self._tls_receiver = None
        return True

    @endpoint
    def run_commands(self, commands):
        result = subprocess.run(commands, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr


class MountHandler:
    def __init__(
        self,
        host_mesh,
        sourcepath: str,
        mntpoint: Optional[str] = None,
        chunk_size=None,
        backend: str = "slurm",
        num_parallel_streams: int = 8,
        tls_addresses=None,
        on_tls_tunnel_error=None,
    ):
        self.sourcepath = sourcepath
        if mntpoint is None:
            mntpoint = sourcepath
        self.mntpoint = mntpoint
        self.fuse_actors = None
        self.host_mesh = host_mesh
        self.procs = None
        self.chunk_size = chunk_size
        self.backend = backend
        if num_parallel_streams < 1:
            raise ValueError(
                f"num_parallel_streams must be >= 1, got {num_parallel_streams}"
            )
        self.num_parallel_streams = num_parallel_streams
        self.tls_addresses = tls_addresses
        self.on_tls_tunnel_error = on_tls_tunnel_error
        self._staging_mv = None

    def open(self):
        t_open_start = time.time()

        # Reuse existing actors if available (preserves block hashes
        # and pack index for incremental update checks).
        if self.fuse_actors is None:
            self.procs = self.host_mesh.spawn_procs(per_host={"gpus": 1})
            self.fuse_actors = self.procs.spawn(
                "FUSEActor", FUSEActor, self.chunk_size, self.backend
            )
            self.fuse_actors.run_commands.call(["mkdir", "-p", self.mntpoint]).get()

            import xxhash

            cache_key = xxhash.xxh64(
                (self.sourcepath + ":" + self.mntpoint).encode()
            ).hexdigest()
            self.fuse_actors.try_load_cache.call(cache_key).get()

        t_actors_ready = time.time()

        # Fire RPCs before packing so the network round-trips overlap
        # with the CPU-bound walk+pack+hash step.
        flat_actors = self.fuse_actors.flatten("rank")
        num_workers = len(flat_actors)
        hashes_future = self.fuse_actors.get_block_hashes.call()
        index_future = self.fuse_actors.get_pack_index.call()

        # Get pack index from workers (first non-empty).
        # This is small JSON so the wait is fast.
        try:
            index_result = index_future.get()
            previous_index = next(
                (idx for _, idx in index_result if idx and idx.get("files")),
                None,
            )
        except Exception:
            previous_index = None

        t_index_ready = time.time()

        (
            meta,
            self._staging_mv,
            chunks,
            self._pack_shm_path,
            client_hashes,
            new_pack_index,
        ) = pack_directory_chunked(
            self.sourcepath,
            self.chunk_size,
            use_shm=False,
            previous_index=previous_index,
        )
        staging_mv = self._staging_mv
        client_total_size = len(staging_mv) if staging_mv is not None else 0

        t_pack_done = time.time()

        # Collect worker hashes (should already be available after packing).
        try:
            result = hashes_future.get()
            worker_states = [
                (remote_hashes, remote_size)
                for _point, (remote_hashes, remote_size) in result
            ]
            fresh_ranks, worker_dirty = classify_workers(
                client_hashes, client_total_size, worker_states
            )
            # Debug: log hash comparison details
            for rank, (rh, rs) in enumerate(worker_states):
                if rh:
                    mismatches = [
                        i for i in range(min(len(rh), len(client_hashes)))
                        if client_hashes[i] != rh[i]
                    ]
                    logger.info(
                        f"[DEBUG] worker {rank}: "
                        f"client_size={client_total_size} worker_size={rs} "
                        f"client_blocks={len(client_hashes)} worker_blocks={len(rh)} "
                        f"mismatched={len(mismatches)}/{min(len(rh), len(client_hashes))} "
                        f"first_mismatch={mismatches[:3] if mismatches else 'none'}"
                    )
                    if mismatches:
                        i = mismatches[0]
                        logger.info(
                            f"[DEBUG] block {i}: client={client_hashes[i][:16]} "
                            f"worker={rh[i][:16]}"
                        )
        except Exception as e:
            logger.info(f"Block hash query failed: {e}")
            fresh_ranks = []
            worker_dirty = {rank: None for rank in range(num_workers)}

        t_classify_done = time.time()

        # Always send metadata so newly spawned actors (which loaded
        # block data from the persistent cache) have filesystem layout.
        self.fuse_actors.set_meta.call(meta).get()

        t_meta_done = time.time()

        if not worker_dirty:
            # Lazy-unmount any stale FUSE left by a killed process.
            try:
                self.fuse_actors.run_commands.call(
                    ["fusermount3", "-uz", self.mntpoint]
                ).get()
            except Exception:
                pass
            self.fuse_actors.mount.call(
                self.mntpoint, client_hashes, client_total_size, new_pack_index
            ).get()
            t_mount_done = time.time()
            logger.info(
                f"All {num_workers} workers up-to-date — skipping transfer, re-mounting. "
                f"Timings: actors={t_actors_ready - t_open_start:.2f}s, "
                f"get_index={t_index_ready - t_actors_ready:.2f}s, "
                f"pack+hash={t_pack_done - t_index_ready:.2f}s "
                f"({client_total_size / (1024**2):.0f}MiB), "
                f"classify={t_classify_done - t_pack_done:.2f}s, "
                f"set_meta={t_meta_done - t_classify_done:.2f}s, "
                f"mount={t_mount_done - t_meta_done:.2f}s, "
                f"total={t_mount_done - t_open_start:.2f}s"
            )
            return self

        n_partial = sum(1 for v in worker_dirty.values() if v is not None)
        n_stale = sum(1 for v in worker_dirty.values() if v is None)
        logger.info(
            f"{len(fresh_ranks)} fresh, {n_partial} partial, "
            f"{n_stale} stale out of {num_workers} workers"
        )

        # Unmount workers that need updating.  Use lazy unmount (-uz) so
        # stale mounts from killed processes are cleaned up.
        for rank in worker_dirty:
            try:
                flat_actors.slice(rank=rank).run_commands.call(
                    ["fusermount3", "-uz", self.mntpoint]
                ).get()
            except Exception:
                pass

        t_unmount_done = time.time()

        # Compute dirty blocks: union of all non-fresh workers.
        # Stale workers (None) need all blocks; partial workers need their list.
        all_blocks = list(range(len(client_hashes)))
        dirty_blocks = set()
        for rank, d in worker_dirty.items():
            if d is None:
                dirty_blocks = set(all_blocks)
                break
            dirty_blocks.update(d)
        dirty_blocks = sorted(dirty_blocks)

        target_ranks = sorted(worker_dirty.keys())

        if dirty_blocks and target_ranks:
            logger.info(
                f"{len(dirty_blocks)}/{len(client_hashes)} blocks dirty "
                f"across {len(target_ranks)} workers"
            )
            # TLS to leader, RDMA fan-out to all peers.
            self._transfer_fanout(
                flat_actors, target_ranks, dirty_blocks, client_total_size
            )

        # Clean up the client-side pack file after all transfers.
        if self._pack_shm_path is not None:
            try:
                os.unlink(self._pack_shm_path)
            except OSError:
                pass
            self._pack_shm_path = None

        t_transfer_done = time.time()

        # Remount all workers (fresh ones for metadata update).
        self.fuse_actors.mount.call(
            self.mntpoint, client_hashes, client_total_size, new_pack_index
        ).get()

        t_mount_done = time.time()

        logger.info(
            f"open() timings: actors={t_actors_ready - t_open_start:.2f}s, "
            f"get_index={t_index_ready - t_actors_ready:.2f}s, "
            f"pack+hash={t_pack_done - t_index_ready:.2f}s "
            f"({client_total_size / (1024**2):.0f}MiB), "
            f"classify={t_classify_done - t_pack_done:.2f}s, "
            f"set_meta={t_meta_done - t_classify_done:.2f}s, "
            f"unmount={t_unmount_done - t_meta_done:.2f}s, "
            f"transfer={t_transfer_done - t_unmount_done:.2f}s, "
            f"mount={t_mount_done - t_transfer_done:.2f}s, "
            f"total={t_mount_done - t_open_start:.2f}s"
        )
        return self

    def _transfer_blocks_rust_tls(self, fuse_actor, dirty_blocks, total_size, rank=0):
        """Transfer dirty blocks to a single worker using Rust TLS.

        Sends blocks directly from ``self._staging_mv`` (the buffer produced
        by ``pack_directory_chunked``) so no second pack step is needed.

        Retries once on failure — the relay tunnel can lose data on transient
        kubectl port-forward hiccups (sender writes succeed because data fits
        in the local TCP buffer, but the relay doesn't deliver it all).

        Flow:
          1. Worker: prepare_receiver() → creates TlsReceiver, returns address
          2. Client: send_blocks_from_buffer() → parallel TLS connections
          3. Worker: receive_blocks() → waits for all data
        """
        if not dirty_blocks:
            return

        from monarch._rust_bindings.monarch_extension.tls_sender import (
            send_blocks_from_buffer,
        )

        num_streams = self.num_parallel_streams

        # When direct TLS tunnels are available, the number of streams
        # is driven by the number of tunnel addresses (not the default).
        if self.tls_addresses and rank < len(self.tls_addresses):
            num_streams = len(self.tls_addresses[rank])

        # Get cache path from the FUSEActor.
        cache_result = fuse_actor.get_cache_path.call().get()
        cache_path = [v for _, v in cache_result][0]
        if cache_path is None:
            cache_path = ""

        total_bytes = sum(
            min(HASH_BLOCK_SIZE, total_size - bi * HASH_BLOCK_SIZE)
            for bi in dirty_blocks
        )

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # 1. Start receiver on worker.
                t_start = time.time()
                addr_result = fuse_actor.prepare_receiver.call(num_streams, total_size).get()

                # Use direct TLS tunnel addresses if available (parallel kubectl
                # port-forwards bypass the relay for higher throughput), otherwise
                # fall back to replicating the relay-tunneled address.
                if self.tls_addresses and rank < len(self.tls_addresses):
                    addresses = self.tls_addresses[rank]
                    logger.info(
                        f"Using {len(addresses)} direct TLS tunnels for rank {rank}"
                    )
                else:
                    addr = [v for _, v in addr_result][0]
                    addresses = [addr] * num_streams

                # 2. Fire receive_blocks (non-blocking) so worker starts waiting.
                recv_future = fuse_actor.receive_blocks.call()

                # 3. Send blocks directly from the staging buffer.
                t_setup = time.time()
                send_blocks_from_buffer(
                    self._staging_mv, total_size, dirty_blocks, addresses, cache_path
                )
                t_send = time.time()

                # 4. Wait for receiver to finish.
                recv_future.get()
            except Exception:
                if attempt < max_attempts - 1:
                    logger.warning(
                        f"TLS transfer failed (attempt {attempt + 1}/{max_attempts}), "
                        f"retrying...",
                        exc_info=True,
                    )
                    # Let the caller restart dead tunnels before retry.
                    if self.on_tls_tunnel_error:
                        self.on_tls_tunnel_error()
                    time.sleep(2)
                    continue
                raise
            break

        t_done = time.time()

        gbps = (total_bytes * 8.0 / 1e9) / max(t_send - t_setup, 1e-9)
        logger.info(
            f"Rust TLS block transfer ({len(dirty_blocks)} blocks, "
            f"{num_streams} streams): {total_bytes // (1024**2)}MiB "
            f"in {t_send - t_setup:.1f}s ({gbps:.1f} Gbps), "
            f"setup={t_setup - t_start:.2f}s, "
            f"total={t_done - t_start:.1f}s"
        )

    def _transfer_fanout(
        self, flat_actors, target_ranks, dirty_blocks, total_size
    ):
        """Transfer dirty blocks: TLS to leader, RDMA fan-out to peers.

        Small incremental transfers (≤ RDMA_PARALLEL_TLS_THRESHOLD blocks)
        go directly via TLS to each worker in parallel — faster than any
        RDMA round-trip for small payloads.

        Large transfers use TLS to the leader, then a single RDMA fan-out
        to all peers (one staging + one RDMA read per peer).
        """
        t0 = time.time()

        total_bytes = sum(
            min(HASH_BLOCK_SIZE, total_size - bi * HASH_BLOCK_SIZE)
            for bi in dirty_blocks
        )

        leader_rank = target_ranks[0]
        peer_ranks = target_ranks[1:]
        leader = flat_actors.slice(rank=leader_rank)

        # Small incremental: TLS to every worker in parallel (no RDMA).
        if len(dirty_blocks) <= RDMA_PARALLEL_TLS_THRESHOLD or not peer_ranks:
            tls_futures = []
            for rank in target_ranks:
                worker = flat_actors.slice(rank=rank)
                tls_futures.append((rank, worker))

            # Ensure all workers have storage allocated.
            ensure_futures = []
            for rank, worker in tls_futures:
                ensure_futures.append(worker.ensure_storage.call(total_size))
            for f in ensure_futures:
                f.get()

            # TLS to each worker in parallel (non-blocking sends).
            import threading
            errors = []
            def _tls_to_worker(rank, worker):
                try:
                    self._transfer_blocks_rust_tls(
                        worker, dirty_blocks, total_size, rank=rank
                    )
                except Exception as e:
                    errors.append((rank, e))

            threads = []
            for rank, worker in tls_futures:
                t = threading.Thread(target=_tls_to_worker, args=(rank, worker))
                t.start()
                threads.append(t)
            for t in threads:
                t.join()

            t_done = time.time()
            if errors:
                raise errors[0][1]
            logger.info(
                f"Parallel TLS to {len(target_ranks)} workers: "
                f"{total_bytes // (1024**2)}MiB "
                f"({len(dirty_blocks)} blocks) in {t_done - t0:.1f}s"
            )
            return

        # Large transfer: TLS to leader, single RDMA fan-out to peers.

        # Ensure peers have storage allocated (parallel with TLS to leader).
        ensure_futures = []
        for rank in peer_ranks:
            peer = flat_actors.slice(rank=rank)
            ensure_futures.append(peer.ensure_storage.call(total_size))

        # Step 1: TLS transfer to leader.
        self._transfer_blocks_rust_tls(leader, dirty_blocks, total_size, rank=leader_rank)
        t_tls = time.time()

        # Wait for peer storage allocation.
        for f in ensure_futures:
            f.get()

        # Step 2: Single RDMA fan-out — all dirty blocks in one shot.
        # One RPC to leader (stage all blocks), one RPC per peer (read all).
        staging_size = total_bytes

        result = leader.get_blocks_rdma_buffer.call(dirty_blocks, total_size).get()
        rdma_buf = [v for _, v in result][0]

        read_futures = []
        for rank in peer_ranks:
            peer = flat_actors.slice(rank=rank)
            read_futures.append(
                peer.replace_blocks.call(dirty_blocks, total_size, rdma_buf, staging_size)
            )
        for f in read_futures:
            f.get()

        t_rdma = time.time()
        tls_gbps = (total_bytes * 8.0 / 1e9) / max(t_tls - t0, 1e-9)
        rdma_gbps = (
            (total_bytes * len(peer_ranks) * 8.0 / 1e9)
            / max(t_rdma - t_tls, 1e-9)
        )
        logger.info(
            f"TLS→leader: {total_bytes // (1024**2)}MiB in {t_tls - t0:.1f}s "
            f"({tls_gbps:.1f} Gbps), "
            f"RDMA fan-out: {len(dirty_blocks)} blocks to {len(peer_ranks)} "
            f"peers in {t_rdma - t_tls:.1f}s ({rdma_gbps:.1f} Gbps)"
        )

    def close(self):
        """Unmount FUSE but keep actors alive for incremental updates."""
        if self.fuse_actors is not None:
            try:
                self.fuse_actors.run_commands.call(
                    ["fusermount3", "-uz", self.mntpoint]
                ).get()
            except Exception:
                pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False  # Don't suppress exceptions


def remotemount(
    host_mesh,
    sourcepath: str,
    mntpoint: Optional[str] = None,
    chunk_size=None,
    backend: str = "slurm",
    num_parallel_streams: int = 8,
    tls_addresses=None,
    on_tls_tunnel_error=None,
):
    """Mount a local directory on remote hosts via RDMA transfer and FUSE."""
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    return MountHandler(
        host_mesh, sourcepath, mntpoint, chunk_size, backend, num_parallel_streams,
        tls_addresses=tls_addresses,
        on_tls_tunnel_error=on_tls_tunnel_error,
    )
