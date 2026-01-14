# TCP Transport Implementation Plan

## Summary

Add TCP as a fallback transport for RDMA operations when RDMA NIC is unavailable.

**Key decisions:**
- Reuse hyperactor TCP infrastructure (`FrameReader`/`FrameWrite`)
- Per-operation transport selection via `transport` parameter
- Automatic fallback: `"best"` uses NIC if available, TCP otherwise

## Transport Types

```python
Transport = Literal["best", "tcp", "tcp+", "nic", "nic+", "nvlink"]
```

| Value | Meaning |
|-------|---------|
| `best` | Auto-select best available (default) |
| `tcp` | Force TCP |
| `tcp+` | TCP or faster |
| `nic` | Force RDMA NIC (error if unavailable) |
| `nic+` | RDMA NIC or faster |
| `nvlink` | NVLink (future) |

## API Changes

```python
# Before
buffer.read_into(dst, timeout=3)

# After
buffer.read_into(dst, timeout=3, transport="best")
```

## Architecture

```
Python API (transport param)
    ↓
Rust Bindings (parse transport string)
    ↓
RdmaBuffer.read_into_with_transport()
    ↓
select_transport() → Transport::Nic or Transport::Tcp
    ↓
┌─────────────────────┬─────────────────────┐
│  RDMA NIC Path      │  TCP Fallback Path  │
│  (existing code)    │  (new)              │
│  - QueuePair.put()  │  - TcpDataTransport │
│  - ibverbs ops      │  - FrameWrite       │
└─────────────────────┴─────────────────────┘
```

## Files to Create

| File | Purpose |
|------|---------|
| `monarch_rdma/src/transport.rs` | Transport enum and selection logic |
| `monarch_rdma/src/tcp_transport.rs` | TCP bulk data transfer |
| `python/tests/test_rdma_tcp_transport.py` | Integration tests |

## Files to Modify

| File | Changes |
|------|---------|
| `monarch_rdma/src/lib.rs` | Add new modules |
| `monarch_rdma/src/rdma_components.rs` | Add `*_with_transport` methods |
| `monarch_rdma/src/rdma_manager_actor.rs` | TCP transport + message handlers |
| `monarch_rdma/extension/lib.rs` | Add `transport` param to bindings |
| `python/monarch/_src/rdma/rdma.py` | Add `transport` param to API |
| `python/monarch/_rust_bindings/rdma.pyi` | Update type stubs |

## Implementation Phases

### Phase 1: Transport Abstraction
Create `transport.rs` with:
- `Transport` enum
- `select_transport(requested, local_has_nic, remote_has_nic) -> Transport`

### Phase 2: TCP Transport
Create `tcp_transport.rs`:
- Reuse `hyperactor::channel::net::framed::{FrameReader, FrameWrite}`
- Connection pooling per remote actor
- `TcpDataTransport::send_data()` / `recv_data()`

### Phase 3: RdmaBuffer Extension
Add to `rdma_components.rs`:
- `RdmaBuffer::read_into_with_transport()`
- `RdmaBuffer::write_from_with_transport()`
- Private `read_into_tcp()` / `write_from_tcp()` methods

### Phase 4: Manager Actor
Add to `rdma_manager_actor.rs`:
- `tcp_transport: Option<TcpDataTransport>` field
- `RequestTcpTransfer` message handler
- `ExchangeTcpEndpoint` for capability discovery

### Phase 5: Python Bindings
Update `extension/lib.rs`:
- Add `transport: &str` parameter (default `"best"`)
- Add `parse_transport()` helper

### Phase 6: Python API
Update `rdma.py`:
- Add `Transport` type alias
- Add `transport` parameter to `read_into()`, `write_from()`
- Update `RDMAAction` methods

## Testing

```bash
# Existing tests (regression check)
pytest python/tests/test_rdma.py -v

# New TCP transport tests
pytest python/tests/test_rdma_tcp_transport.py -v
```

Test cases:
1. Explicit TCP transport works
2. `"best"` falls back to TCP when NIC unavailable
3. `"nic"` fails when RDMA unavailable
4. Large data transfers over TCP
5. Concurrent TCP operations
