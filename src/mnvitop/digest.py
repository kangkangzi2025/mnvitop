"""Compact fleet digests for agents: the few numbers needed to decide where a job can run.

Every dict carries a one-line ``summary`` so a reader can stop there, and
structured fields underneath for the cases that need them.  Commands are
truncated and only the largest processes per GPU are kept, because the point
is a decision, not a dump.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from .model import Gpu, Proc, Snapshot

BUSY_UTIL = 5


def host_digest(snap: Snapshot, processes: bool = True, max_procs: int = 2, cmd_len: int = 60) -> dict:
    """One host: free and busy GPU counts, free VRAM, load, and per GPU state."""
    if not snap.ok:
        return {"host": snap.host, "ok": False, "error": snap.error, "summary": f"{snap.host}: unreachable ({snap.error})"}
    names = {gpu.name.removeprefix("NVIDIA ") for gpu in snap.gpus}
    model = names.pop() if len(names) == 1 else "mixed"
    gpus = []
    free = 0
    vram_free = vram_total = 0.0
    for gpu in snap.gpus:
        busy = snap.in_use(gpu) or (gpu.util or 0) >= BUSY_UTIL
        free += not busy
        vram_free += _gb(gpu.mem_free) or 0.0
        vram_total += _gb(gpu.mem_total) or 0.0
        entry: dict = {
            "gpu": gpu.index,
            "busy": busy,
            "util": gpu.util,
            "vram_used_gb": _gb(gpu.mem_used),
            "vram_free_gb": _gb(gpu.mem_free),
            "temp_c": gpu.temp,
        }
        if processes:
            entry["procs"] = [_proc_digest(proc, cmd_len) for proc in snap.procs_on(gpu.index)[:max_procs]]
        gpus.append(entry)
    load = snap.load[0] if snap.load else None
    ram_free = _gb_from_kb(snap.mem_avail)
    parts = [f"{free}/{len(snap.gpus)} {model} free", f"{vram_free:.0f}/{vram_total:.0f} GB VRAM free"]
    if load is not None:
        parts.append(f"load {load:.1f}/{snap.ncpu or '?'}")
    return {
        "host": snap.host,
        "ok": True,
        "gpu_model": model,
        "gpus": len(snap.gpus),
        "free": free,
        "busy": len(snap.gpus) - free,
        "vram_free_gb": round(vram_free, 1),
        "vram_total_gb": round(vram_total, 1),
        "load": load,
        "cpus": snap.ncpu,
        "ram_free_gb": ram_free,
        "age_s": max(0, int(time.time() - snap.taken_at)),
        "summary": f"{snap.host}: " + ", ".join(parts),
        "gpu": gpus,
    }


def fleet_digest(snaps: list[Snapshot], processes: bool = True, max_procs: int = 2) -> dict:
    """All hosts plus fleet totals."""
    hosts = [host_digest(snap, processes, max_procs) for snap in snaps]
    up = [host for host in hosts if host["ok"]]
    gpus = sum(host["gpus"] for host in up)
    free = sum(host["free"] for host in up)
    vram_free = sum(host["vram_free_gb"] for host in up)
    down = [host["host"] for host in hosts if not host["ok"]]
    summary = f"{len(up)}/{len(hosts)} hosts up; {free}/{gpus} GPUs free; {vram_free:.0f} GB VRAM free"
    if down:
        summary += f"; unreachable: {', '.join(down)}"
    return {"as_of": _now(), "summary": summary, "hosts": hosts}


def find_gpus(snaps: list[Snapshot], count: int = 1, min_free_gb: float = 0.0) -> dict:
    """Where ``count`` GPUs with at least ``min_free_gb`` free VRAM each are idle right now, best host first."""
    count = max(1, count)
    placements = []
    alternatives = []
    for snap in snaps:
        if not snap.ok:
            continue
        idle = [gpu for gpu in snap.gpus if not snap.in_use(gpu) and (gpu.util or 0) < BUSY_UTIL and (_gb(gpu.mem_free) or 0) >= min_free_gb]
        shared = [gpu for gpu in snap.gpus if gpu not in idle and (_gb(gpu.mem_free) or 0) >= min_free_gb]
        load = snap.load[0] if snap.load else 0.0
        load_frac = load / snap.ncpu if snap.ncpu else 0.0
        model = _model(snap)
        if len(idle) >= count:
            chosen = sorted(idle, key=lambda gpu: -(gpu.mem_free or 0))[:count]
            placements.append({
                "host": snap.host,
                "gpus": sorted(gpu.index for gpu in chosen),
                "gpu_model": model,
                "vram_free_gb_min": round(min(_gb(gpu.mem_free) or 0 for gpu in chosen), 1),
                "idle_gpus_on_host": len(idle),
                "load": round(load, 1),
                "_rank": (-len(idle) >= -count, load_frac),
            })
        elif idle or shared:
            alternatives.append({
                "host": snap.host,
                "gpu_model": model,
                "idle_gpus": sorted(gpu.index for gpu in idle),
                "shared_gpus": [{"gpu": gpu.index, "vram_free_gb": _gb(gpu.mem_free), "util": gpu.util} for gpu in shared],
                "load": round(load, 1),
            })
    placements.sort(key=lambda item: item["_rank"])
    for item in placements:
        item.pop("_rank")
    if placements:
        best = placements[0]
        summary = f"{best['host']}: GPUs {best['gpus']} idle ({best['gpu_model']}, >= {best['vram_free_gb_min']} GB free each)"
        if len(placements) > 1:
            summary += f"; also possible on {', '.join(item['host'] for item in placements[1:])}"
    else:
        summary = f"no host has {count} idle GPU(s) with {min_free_gb:g} GB free"
        if alternatives:
            summary += "; see alternatives (shared GPUs still have free VRAM but someone is using them)"
    return {"as_of": _now(), "found": bool(placements), "summary": summary, "placements": placements, "alternatives": alternatives}


def _proc_digest(proc: Proc, cmd_len: int) -> dict:
    """The few fields that identify a job."""
    command = proc.command or "?"
    if len(command) > cmd_len:
        command = command[: cmd_len - 1] + "~"
    return {
        "user": proc.user,
        "pid": proc.pid,
        "vram_gb": _gb(proc.gpu_mem),
        "age": _age(proc.etimes),
        "cmd": command,
    }


def _model(snap: Snapshot) -> str:
    """GPU model name shared by the host, or 'mixed'."""
    names = {gpu.name.removeprefix("NVIDIA ") for gpu in snap.gpus}
    return names.pop() if len(names) == 1 else "mixed"


def _gb(mib: int | None) -> float | None:
    """GiB with one decimal from MiB."""
    return None if mib is None else round(mib / 1024, 1)


def _gb_from_kb(kb: int | None) -> float | None:
    """GiB with no decimals from kB."""
    return None if kb is None else round(kb / 2**20)


def _age(seconds: int | None) -> str | None:
    """Compact duration such as 45s, 12m, 3h or 2d."""
    if seconds is None:
        return None
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    return f"{seconds // 86400}d{(seconds % 86400) // 3600:02d}h"


def _now() -> str:
    """ISO timestamp in UTC without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
