# TCP Transport Implementation

## Summary

TCP transport has been added as a fallback for RDMA operations, implemented entirely in Python using existing Monarch actor APIs. The implementation supports both local (same-process) and remote (cross-process) data transfer.

## Transport Types

```python
Transport = Literal["best", "tcp", "nic"]
```

| Value | Behavior |
|-------|----------|
| `"best"` | Auto-select: RDMA NIC if available, else TCP (default) |
| `"tcp"` | Force TCP transport |
| `"nic"` | Force RDMA NIC (raises `RuntimeError` if unavailable) |

## API Changes

### RDMABuffer

```python
# New transport parameter (default: "best")
buffer.read_into(dst, timeout=3, transport="best")
buffer.write_from(src, timeout=3, transport="best")
```

### RDMAAction

```python
# Default transport in constructor
action = RDMAAction(transport="best")

# Per-operation override
action.read_into(buffer, dst, transport="tcp")
action.write_from(buffer, src, transport="nic")
```

## Architecture

### Cross-Process TCP Transport

```
Host A (Buffer Owner)                    Host B (Caller)
─────────────────────                    ───────────────
RDMABuffer created
  → registers with TcpDataActor
  → stores actor ref in buffer

buffer sent to Host B ──────────────────→ buffer received
                                           (with TcpDataActor ref)

                                         buffer.read_into(dst, transport="tcp")
                                           │
                                           ├─ Check local registry (miss)
                                           │
                                           └─ Call tcp_actor.fetch_buffer_data()
                                                      │
tcp_actor receives call ←──────────────────────────────┘
  → reads from _local_buffers
  → returns bytes

                                         ←─ receives bytes
                                         ←─ copies to dst
```

### Components

1. **TcpDataActor**: Per-process actor that holds buffer data and handles remote fetch/write requests
2. **Buffer Registry**: Local dict for fast same-process access
3. **Actor Reference**: Stored in RDMABuffer, serializes across processes

## Implementation Details

### Buffer Creation

```python
class RDMABuffer:
    def __init__(self, data):
        # ... RDMA setup ...

        # Store buffer ID and dtype for TCP
        self._buffer_id = self._buffer.name
        self._dtype = data.dtype if isinstance(data, torch.Tensor) else None

        # Register in local registry (fast path)
        _buffer_registry[self._buffer_id] = data

        # Get TcpDataActor and register (for remote access)
        self._tcp_data_actor = _get_tcp_data_actor_blocking()
        self._tcp_data_actor.register_buffer.call_one(self._buffer_id, data)
```

### TCP Read (read_into)

```python
async def read_into_tcp_impl():
    # Fast path: check local registry
    local_data = _buffer_registry.get(buffer_id)

    if local_data is not None:
        # Same process - direct copy
        dst.copy_(local_data)
    else:
        # Cross process - call remote TcpDataActor
        data_bytes = await tcp_actor.fetch_buffer_data.call_one(buffer_id)
        dst.copy_(torch.frombuffer(bytearray(data_bytes), dtype=dtype))
```

### TCP Write (write_from)

```python
async def write_from_tcp_impl():
    # Fast path: check local registry
    dst_data = _buffer_registry.get(buffer_id)

    if dst_data is not None:
        # Same process - direct copy
        dst_data.copy_(src)
    else:
        # Cross process - call remote TcpDataActor
        src_bytes = src.numpy().tobytes()
        await tcp_actor.write_buffer_data.call_one(buffer_id, src_bytes)
```

## Files Modified

- `python/monarch/_src/rdma/rdma.py`:
  - Added `Transport` type alias
  - Added `_buffer_registry` for local fast path
  - Added `TcpDataActor` class with `register_buffer`, `unregister_buffer`, `fetch_buffer_data`, `write_buffer_data` endpoints
  - Added `_get_tcp_data_actor_future()` and `_get_tcp_data_actor_blocking()` helpers
  - Modified `RDMABuffer.__init__` to register with TcpDataActor
  - Added `transport` parameter to `read_into()` and `write_from()`
  - Added `_should_use_tcp()`, `_read_into_tcp()`, `_write_from_tcp()` methods
  - Updated `drop()` to unregister from TcpDataActor
  - Modified `RDMAAction` to support transport parameter

## Usage Examples

### Basic Cross-Process Usage

```python
# Host A: Parameter Server
class ParameterServer(Actor):
    def __init__(self):
        self.weights = torch.randn(1000, 1000)
        self.buffer = RDMABuffer(self.weights.view(torch.uint8).flatten())

    @endpoint
    async def get_weights_buffer(self) -> RDMABuffer:
        return self.buffer

# Host B: Worker
class Worker(Actor):
    @endpoint
    async def fetch_weights(self, server: ParameterServer):
        buffer = await server.get_weights_buffer.call_one()

        local_weights = torch.zeros(1000, 1000)
        # Works with TCP transport - calls remote TcpDataActor
        await buffer.read_into(
            local_weights.view(torch.uint8).flatten(),
            transport="tcp"
        )
```

### Batch Operations

```python
from monarch._src.rdma.rdma import RDMAAction

# Create action with default TCP transport
action = RDMAAction(transport="tcp")
action.read_into(buffer_a, dst_a)
action.write_from(buffer_b, src_b)
action.submit().get()
```

## Performance Characteristics

| Aspect | RDMA (NIC) | TCP (Same Process) | TCP (Cross Process) |
|--------|------------|--------------------|--------------------|
| Latency | Very low | Very low (direct copy) | Higher (serialization + network) |
| Throughput | High | High | Lower (Python serialization) |
| CPU Usage | Minimal | Low | Higher (data conversion) |
| Dependencies | RDMA NIC | None | None |

## Notes

- **torch is a required dependency** (package is named "torchmonarch")
- TCP transport adds overhead for cross-process transfers but works universally
- Same-process TCP is optimized via local registry (no serialization)
- The TcpDataActor is spawned lazily on first buffer creation
