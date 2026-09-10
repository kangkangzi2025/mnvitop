"""Remote probe script and the parser for its output.

The probe is a small shell script that every host runs through ``bash -s``.
It needs only ``nvidia-smi`` and ``ps`` on the remote side and prints one
record per line, tagged by its first field: HOST, GPU, APP, PS or ERR.
"""

from __future__ import annotations

from .model import Gpu, Proc, Snapshot

GPU_FIELDS = "index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit"

_HEAD = r"""
export LC_ALL=C
host=$(hostname 2>/dev/null || cat /proc/sys/kernel/hostname 2>/dev/null)
load=$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null)
ncpu=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null)
mem=$(awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} END{print t "|" a}' /proc/meminfo 2>/dev/null)
echo "HOST|$host|$load|$ncpu|$mem"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERR|nvidia-smi not found in PATH"
  exit 0
fi
"""

_GPUS = "nvidia-smi --query-gpu=__FIELDS__ --format=csv,noheader,nounits 2>&1 | sed 's/^/GPU|/'\n"

_APPS = r"""apps=$(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits 2>/dev/null)
if [ -n "$apps" ]; then
  printf '%s\n' "$apps" | sed 's/^/APP|/'
  pids=$(printf '%s\n' "$apps" | cut -d, -f1 | tr -d ' ' | sort -u | paste -sd, -)
  ps -o pid=,user:32=,etimes=,pcpu=,rss=,args= -p "$pids" 2>/dev/null | sed 's/^/PS|/'
fi
"""


def probe_script(include_gpus: bool = True) -> str:
    """The remote shell; without GPU rows it only refreshes processes and host metadata."""
    parts = [_HEAD]
    if include_gpus:
        parts.append(_GPUS.replace("__FIELDS__", GPU_FIELDS))
    parts.append(_APPS)
    return "".join(parts)


PROBE_SCRIPT = probe_script(True)


def parse_probe_output(host: str, text: str) -> Snapshot:
    """Build a Snapshot from the probe's stdout; never raises on malformed lines."""
    snap = Snapshot(host=host)
    apps: list[list[str]] = []
    ps: dict[int, dict] = {}
    errors: list[str] = []
    seen_host = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        tag, _, body = line.partition("|")
        if tag == "HOST":
            seen_host = True
            _fill_host(snap, body)
        elif tag == "GPU":
            gpu = parse_gpu_row(body)
            if gpu is None:
                errors.append(body.strip())
            else:
                snap.gpus.append(gpu)
        elif tag == "APP":
            apps.append(_csv(body))
        elif tag == "PS":
            entry = _parse_ps(body)
            if entry is not None:
                ps[entry["pid"]] = entry
        elif tag == "ERR":
            errors.append(body.strip())
    by_uuid = {gpu.uuid: gpu.index for gpu in snap.gpus}
    for fields in apps:
        if len(fields) < 3:
            continue
        pid = _to_int(fields[0])
        if pid is None:
            continue
        extra = {key: value for key, value in ps.get(pid, {}).items() if key != "pid"}
        snap.procs.append(
            Proc(pid=pid, gpu_index=by_uuid.get(fields[1]), gpu_uuid=fields[1], gpu_mem=_to_int(fields[2]), **extra)
        )
    if errors and not snap.gpus:
        snap.error = "; ".join(errors)
    elif not seen_host:
        snap.error = "no output from probe"
    return snap


def parse_gpu_row(body: str) -> Gpu | None:
    """Parse one nvidia-smi --query-gpu csv row; None when the row is not a GPU."""
    fields = _csv(body)
    if len(fields) < 9:
        return None
    index = _to_int(fields[0])
    if index is None:
        return None
    return Gpu(
        index=index,
        uuid=fields[1],
        name=fields[2],
        util=_to_int(fields[3]),
        mem_used=_to_int(fields[4]),
        mem_total=_to_int(fields[5]),
        temp=_to_int(fields[6]),
        power=_to_float(fields[7]),
        power_limit=_to_float(fields[8]),
    )


def _fill_host(snap: Snapshot, body: str) -> None:
    """Fill hostname, load, cpu count and memory from a HOST record."""
    fields = body.split("|")
    fields += [""] * (5 - len(fields))
    snap.hostname = fields[0].strip() or None
    parts = fields[1].split()
    if len(parts) >= 3:
        try:
            snap.load = (float(parts[0]), float(parts[1]), float(parts[2]))
        except ValueError:
            snap.load = None
    snap.ncpu = _to_int(fields[2])
    snap.mem_total = _to_int(fields[3])
    snap.mem_avail = _to_int(fields[4])


def _parse_ps(body: str) -> dict | None:
    """Parse one ps row: pid user etimes pcpu rss args."""
    parts = body.split(None, 5)
    if len(parts) < 5:
        return None
    pid = _to_int(parts[0])
    if pid is None:
        return None
    return {
        "pid": pid,
        "user": parts[1],
        "etimes": _to_int(parts[2]),
        "cpu": _to_float(parts[3]),
        "rss": _to_int(parts[4]),
        "command": parts[5].strip() if len(parts) > 5 else None,
    }


def _csv(body: str) -> list[str]:
    """Split a comma separated nvidia-smi row into stripped fields."""
    return [field.strip() for field in body.split(",")]


def _to_int(value: str | None) -> int | None:
    """Integer from a field, None for [N/A], [Not Supported] and friends."""
    try:
        return int(float(value.strip()))
    except (AttributeError, TypeError, ValueError):
        return None


def _to_float(value: str | None) -> float | None:
    """Float from a field, None when it is not numeric."""
    try:
        return float(value.strip())
    except (AttributeError, TypeError, ValueError):
        return None
