"""Run the probe on hosts through the system ssh client, reusing connections with ControlMaster."""

from __future__ import annotations

import asyncio
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path

from .config import HostSpec
from .model import Snapshot
from .probe import PROBE_SCRIPT, parse_probe_output


def default_control_dir() -> Path:
    """Directory for ControlMaster sockets: ~/.ssh when it exists, else the temp dir."""
    ssh_dir = Path.home() / ".ssh"
    if ssh_dir.is_dir():
        return ssh_dir
    return Path(tempfile.gettempdir())


def control_options(control_dir: Path) -> list[str]:
    """ssh options that share one connection per host for a few minutes."""
    return [
        "-o", "ControlMaster=auto",
        "-o", f"ControlPath={control_dir / 'mnvitop-%C'}",
        "-o", "ControlPersist=300",
    ]


def ssh_command(host: HostSpec, timeout: float, control_dir: Path | None) -> list[str]:
    """Argument vector that runs ``bash -s`` on the host."""
    if host.local:
        return ["bash", "-s"]
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={max(1, int(timeout))}", "-o", "LogLevel=ERROR"]
    if control_dir is not None:
        cmd += control_options(control_dir)
    cmd += [*host.ssh_args, host.ssh, "bash", "-s"]
    return cmd


async def run_probe(host: HostSpec, timeout: float, control_dir: Path | None, script: str = PROBE_SCRIPT) -> Snapshot:
    """Execute a probe script once and return a Snapshot; host failures become ``snap.error``."""
    started = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *ssh_command(host, timeout, control_dir),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        return Snapshot(host=host.name, error=f"cannot start ssh: {exc}")
    try:
        out, err = await asyncio.wait_for(proc.communicate(script.encode()), timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return Snapshot(host=host.name, error=f"timeout after {timeout:.0f}s", latency=time.monotonic() - started)
    snap = parse_probe_output(host.name, out.decode(errors="replace"))
    snap.latency = time.monotonic() - started
    if proc.returncode != 0 and not snap.gpus:
        snap.error = describe_failure(err.decode(errors="replace"), proc.returncode or 0)
    return snap


def describe_failure(stderr: str, code: int) -> str:
    """The last non-empty stderr line, or the exit status when ssh said nothing."""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    return lines[-1] if lines else f"ssh exited with status {code}"


async def close_masters(hosts: Iterable[HostSpec], control_dir: Path) -> None:
    """Ask the ControlMaster connections opened for these hosts to exit."""
    for host in hosts:
        if host.local:
            continue
        try:
            proc = await asyncio.create_subprocess_exec(
                "ssh", "-O", "exit", "-o", f"ControlPath={control_dir / 'mnvitop-%C'}", *host.ssh_args, host.ssh,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), 5)
        except (OSError, TimeoutError):
            pass
