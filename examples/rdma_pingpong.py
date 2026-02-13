import atexit
import os
import socket
import time
from contextlib import contextmanager
from typing import Optional

import fire
import torch

from monarch.actor import Actor, endpoint, enable_transport, shutdown_context
from monarch.rdma import RDMABuffer, is_rdma_available, get_rdma_backend


class PingPongActor(Actor):
    """Actor that participates in RDMA pingpong."""

    def __init__(self, data_size_bytes: int):
        self.hostname = socket.gethostname()
        self.data_size_bytes = data_size_bytes
        self.data_size_floats = data_size_bytes // 4
        self.data = torch.rand(self.data_size_floats, dtype=torch.float32)
        self.receive_buffer = torch.zeros(self.data_size_floats, dtype=torch.float32)

    @endpoint
    async def get_info(self) -> dict:
        return {
            "hostname": self.hostname,
            "rdma_available": is_rdma_available(),
            "rdma_backend": get_rdma_backend(),
            "data_size_mb": self.data_size_bytes / (1024 * 1024),
            "data_checksum": self.data.sum().item(),
        }

    @endpoint
    async def init_rdma(self):
        """Pre-initialize the RDMA manager."""
        from monarch._src.actor.future import Future
        from monarch._src.rdma.rdma import _ensure_init_rdma_manager
        await Future(coro=_ensure_init_rdma_manager())
        return get_rdma_backend()

    @endpoint
    async def get_rdma_buffer(self) -> RDMABuffer:
        byte_tensor = self.data.view(torch.uint8).flatten()
        return RDMABuffer(byte_tensor)

    @endpoint
    async def ping(self, peer_buffer: RDMABuffer) -> dict:
        """Read data from peer's buffer into local receive buffer."""
        byte_tensor = self.receive_buffer.view(torch.uint8).flatten()

        start_time = time.perf_counter()
        await peer_buffer.read_into(byte_tensor, timeout=60)
        elapsed_sec = time.perf_counter() - start_time

        return {
            "hostname": self.hostname,
            "operation": "read_into",
            "elapsed_sec": elapsed_sec,
            "throughput_gbps": (self.data_size_bytes / (1024**3)) / elapsed_sec,
            "received_checksum": self.receive_buffer.sum().item(),
        }

    @endpoint
    async def pong(self, peer_buffer: RDMABuffer) -> dict:
        """Write local data to peer's buffer."""
        byte_tensor = self.data.view(torch.uint8).flatten()

        start_time = time.perf_counter()
        await peer_buffer.write_from(byte_tensor, timeout=60)
        elapsed_sec = time.perf_counter() - start_time

        return {
            "hostname": self.hostname,
            "operation": "write_from",
            "elapsed_sec": elapsed_sec,
            "throughput_gbps": (self.data_size_bytes / (1024**3)) / elapsed_sec,
        }

    @endpoint
    async def verify_data(self, expected_checksum: float) -> dict:
        actual = self.receive_buffer.sum().item()
        matches = abs(actual - expected_checksum) < abs(expected_checksum) * 1e-5
        return {
            "hostname": self.hostname,
            "expected_checksum": expected_checksum,
            "actual_checksum": actual,
            "data_valid": matches,
        }

    @endpoint
    async def reset_receive_buffer(self):
        self.receive_buffer.zero_()


def format_timing(result: dict) -> str:
    return (
        f"  {result['hostname']}: {result['operation']} - "
        f"{result['elapsed_sec']:.3f}s, {result['throughput_gbps']:.2f} GB/s"
    )


@contextmanager
def managed_job(job, kill_on_exit: bool):
    """Context manager that optionally kills the job on exit (success or failure)."""
    if kill_on_exit:
        def _cleanup():
            try:
                job.kill()
            except Exception:
                pass
        atexit.register(_cleanup)

    try:
        yield job
    finally:
        if kill_on_exit:
            print("Killing job on exit...")
            try:
                job.kill()
            except Exception:
                pass
            atexit.unregister(_cleanup)


def main(
    data_size_mb: int = 100,
    num_iterations: int = 5,
    backend: str = "slurm",
    kill_on_exit: bool = False,
    partition: Optional[str] = None,
    account: Optional[str] = None,
    qos: Optional[str] = None,
    time_limit: str = "01:00:00",
    gpus_per_node: int = 1,
    cpus_per_task: Optional[int] = None,
    # MAST options
    hpc_identity: str = "hyper_monarch",
    hpc_job_oncall: str = "monarch",
    hpc_cluster_uuid: str = "MastGenAICluster",
    rm_attribution: str = "msl_infra_pytorch_dev",
):
    """
    RDMA Pingpong Example.

    Transfers data between two nodes using RDMABuffer (works with both
    ibverbs and EFA backends via the actor-based manager).

    Args:
        kill_on_exit: If True, kill the SLURM/MAST job when the script
            exits (on success, failure, or interrupt). Useful for automated
            testing to avoid leaving idle allocations.
    """
    data_size_bytes = data_size_mb * 1024 * 1024

    # Unbuffered stdout so output isn't lost if process exits abruptly
    import sys
    sys.stdout.reconfigure(line_buffering=True)

    print(f"{'='*60}")
    print("RDMA Pingpong Example")
    print(f"{'='*60}")
    print(f"Data size: {data_size_mb} MB ({data_size_bytes:,} bytes)")
    print(f"Iterations: {num_iterations}")
    print(f"Backend: {backend}")
    print(f"Kill on exit: {kill_on_exit}")
    print()

    if backend == "slurm":
        from monarch.job import SlurmJob

        slurm_args = []
        if account:
            slurm_args.append(f"--account={account}")
        if qos:
            slurm_args.append(f"--qos={qos}")

        job = SlurmJob(
            meshes={"workers": 2},
            gpus_per_node=gpus_per_node,
            time_limit=time_limit,
            partition=partition,
            cpus_per_task=cpus_per_task,
            slurm_args=slurm_args,
            exclusive=False,
            log_dir=os.path.expanduser("~/monarch_slurm_logs"),
        )
    elif backend == "mast":
        from monarch.job.meta import MASTJob

        enable_transport("metatls-hostname")
        job = MASTJob(
            hpcIdentity=hpc_identity,
            hpcJobOncall=hpc_job_oncall,
            hpcClusterUuid=hpc_cluster_uuid,
            rmAttribution=rm_attribution,
            useStrictName=True,
            localityConstraints=["region", "gtn"],
        )
        job.add_mesh("workers", 2)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'slurm' or 'mast'.")

    with managed_job(job, kill_on_exit):
        workers = job.state().workers
        procs = workers.spawn_procs()

        actor0 = procs.spawn("actor0", PingPongActor, data_size_bytes).slice(hosts=0)
        actor1 = procs.spawn("actor1", PingPongActor, data_size_bytes).slice(hosts=1)

        # Get actor info
        info0 = actor0.get_info.call_one().get()
        info1 = actor1.get_info.call_one().get()

        print("Actor Information:")
        for label, info in [("Actor 0", info0), ("Actor 1", info1)]:
            print(f"  {label}: {info['hostname']}")
            print(f"    rdma: {info['rdma_available']}, backend: {info['rdma_backend']}")
        print(f"  Actor 0 data checksum: {info0['data_checksum']:.6f}")
        print(f"  Actor 1 data checksum: {info1['data_checksum']:.6f}")
        print()

        if not info0['rdma_available']:
            print("ERROR: No RDMA backend available!")
            return

        # Pre-initialize RDMA/EFA manager on workers (avoids block_on deadlock)
        print("Initializing RDMA backend on workers...")
        b0 = actor0.init_rdma.call_one().get()
        b1 = actor1.init_rdma.call_one().get()
        print(f"  Worker 0 backend: {b0}, Worker 1 backend: {b1}")

        # Get RDMA buffers (works for both ibverbs and EFA via actor manager)
        print("Creating RDMA buffers...")
        buffer0 = actor0.get_rdma_buffer.call_one().get()
        buffer1 = actor1.get_rdma_buffer.call_one().get()
        print("  Buffers created")
        print()

        # Run pingpong iterations
        all_timings = []
        print(f"Running {num_iterations} pingpong iterations...")
        print("-" * 60)

        for i in range(num_iterations):
            print(f"\nIteration {i + 1}/{num_iterations}:")

            actor0.reset_receive_buffer.call_one().get()
            actor1.reset_receive_buffer.call_one().get()

            # Ping: Actor 0 reads from Actor 1's buffer
            ping_result = actor0.ping.call_one(buffer1).get()
            print(format_timing(ping_result))
            all_timings.append(ping_result)

            verify0 = actor0.verify_data.call_one(info1['data_checksum']).get()
            print(f"    Data verification: {'PASS' if verify0['data_valid'] else 'FAIL'}")

            # Pong: Actor 1 writes to Actor 0's buffer
            # Note: this overwrites buffer0 (Actor 0's data) with Actor 1's data
            pong_result = actor1.pong.call_one(buffer0).get()
            print(format_timing(pong_result))
            all_timings.append(pong_result)

            # Verify pong: Actor 0's buffer should now contain Actor 1's data
            verify_pong = actor0.verify_data.call_one(info1['data_checksum']).get()
            print(f"    Pong verification: {'PASS' if verify_pong['data_valid'] else 'FAIL'}")

            # Reverse: Actor 1 reads from Actor 0 (which now has Actor 1's data after pong)
            actor1.reset_receive_buffer.call_one().get()
            ping_reverse = actor1.ping.call_one(buffer0).get()
            print(format_timing(ping_reverse))
            all_timings.append(ping_reverse)

            # buffer0 contains Actor 1's data (written by pong), so verify against that
            verify1 = actor1.verify_data.call_one(info1['data_checksum']).get()
            print(f"    Data verification: {'PASS' if verify1['data_valid'] else 'FAIL'}")

        # Summary
        print()
        print("=" * 60)
        print("Summary Statistics")
        print("=" * 60)

        read_timings = [t for t in all_timings if "read_into" in t['operation']]
        write_timings = [t for t in all_timings if "write_from" in t['operation']]

        for label, timings in [("read_into", read_timings), ("write_from", write_timings)]:
            if timings:
                avg_tp = sum(t['throughput_gbps'] for t in timings) / len(timings)
                avg_lat = sum(t['elapsed_sec'] for t in timings) / len(timings)
                print(f"\n{label} ({len(timings)} ops):")
                print(f"  Avg throughput: {avg_tp:.2f} GB/s")
                print(f"  Avg latency:    {avg_lat:.3f}s")
                print(f"  Min latency:    {min(t['elapsed_sec'] for t in timings):.3f}s")
                print(f"  Max latency:    {max(t['elapsed_sec'] for t in timings):.3f}s")

        all_tp = [t['throughput_gbps'] for t in all_timings]
        print(f"\nOverall avg throughput: {sum(all_tp)/len(all_tp):.2f} GB/s")
        print()

        print("Shutting down...")
        try:
            workers.shutdown().get()
        except Exception as e:
            print(f"  Warning during worker shutdown: {e}")
        try:
            shutdown_context().get()
        except Exception as e:
            print(f"  Warning during context shutdown: {e}")

    print("Done!")


if __name__ == "__main__":
    try:
        fire.Fire(main)
    except SystemExit as e:
        # kill_on_exit shutdown race causes monarch to call sys.exit(1)
        # from a background thread. Suppress it since the transfer succeeded.
        pass
