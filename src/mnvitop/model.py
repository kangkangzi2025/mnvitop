"""Data model shared by the probe parser, the pollers, the digests and the renderer."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field


@dataclass
class Gpu:
    """One GPU as reported by nvidia-smi; memory in MiB, power in W, temperature in C."""

    index: int
    uuid: str
    name: str
    util: int | None = None
    mem_used: int | None = None
    mem_total: int | None = None
    temp: int | None = None
    power: float | None = None
    power_limit: float | None = None

    @property
    def mem_free(self) -> int | None:
        """Free memory in MiB, when both used and total are known."""
        if self.mem_used is None or self.mem_total is None:
            return None
        return max(0, self.mem_total - self.mem_used)


@dataclass
class Proc:
    """A compute process on one GPU, merged with ps output when the host could resolve the pid."""

    pid: int
    gpu_index: int | None = None
    gpu_uuid: str | None = None
    gpu_mem: int | None = None
    user: str | None = None
    etimes: int | None = None
    cpu: float | None = None
    rss: int | None = None
    command: str | None = None


@dataclass
class Snapshot:
    """Everything one probe run learned about one host."""

    host: str
    hostname: str | None = None
    load: tuple[float, float, float] | None = None
    ncpu: int | None = None
    mem_total: int | None = None
    mem_avail: int | None = None
    gpus: list[Gpu] = field(default_factory=list)
    procs: list[Proc] = field(default_factory=list)
    error: str | None = None
    taken_at: float = field(default_factory=time.time)
    latency: float = 0.0

    @property
    def ok(self) -> bool:
        """True when the probe ran without a fatal error."""
        return self.error is None

    def procs_on(self, index: int) -> list[Proc]:
        """Processes attached to one GPU, largest GPU memory first."""
        procs = [p for p in self.procs if p.gpu_index == index]
        return sorted(procs, key=lambda p: -(p.gpu_mem or 0))

    def in_use(self, gpu: Gpu) -> bool:
        """A GPU counts as in use when a process holds it or more than five percent of its memory is taken."""
        if any(proc.gpu_index == gpu.index for proc in self.procs):
            return True
        if gpu.mem_used is None or not gpu.mem_total:
            return False
        return gpu.mem_used / gpu.mem_total > 0.05

    def to_dict(self) -> dict:
        """Plain representation for JSON output."""
        return asdict(self)


@dataclass
class HostState:
    """Latest probe result, the last good snapshot and a utilisation history per GPU."""

    latest: Snapshot | None = None
    good: Snapshot | None = None
    meta_error: str | None = None
    history: dict[str, deque[int | None]] = field(default_factory=dict)
    history_len: int = 120

    @property
    def error(self) -> str | None:
        """Why the display is stale: the latest sample failed, or the process refresh did."""
        if self.latest is not None and not self.latest.ok:
            return self.latest.error
        return self.meta_error

    def update(self, snap: Snapshot) -> None:
        """Record a full or streamed sample; failed polls keep the previous good data."""
        self.latest = snap
        if not snap.ok:
            return
        self.good = snap
        for gpu in snap.gpus:
            self.history.setdefault(gpu.uuid, deque(maxlen=self.history_len)).append(gpu.util)

    def refresh_meta(self, snap: Snapshot) -> None:
        """Take processes and host metadata from a full probe while streamed samples own the GPU columns."""
        if not snap.ok:
            self.meta_error = snap.error
            if self.good is None:
                self.latest = snap
            return
        self.meta_error = None
        if self.good is None:
            self.update(snap)
            return
        by_uuid = {gpu.uuid: gpu.index for gpu in self.good.gpus}
        for proc in snap.procs:
            if proc.gpu_index is None and proc.gpu_uuid in by_uuid:
                proc.gpu_index = by_uuid[proc.gpu_uuid]
        good = self.good
        good.procs = snap.procs
        good.hostname = snap.hostname
        good.load = snap.load
        good.ncpu = snap.ncpu
        good.mem_total = snap.mem_total
        good.mem_avail = snap.mem_avail
        good.latency = snap.latency

    def mark_error(self, host: str, message: str) -> None:
        """Record a failed sample without touching the last good data."""
        self.latest = Snapshot(host=host, error=message)
