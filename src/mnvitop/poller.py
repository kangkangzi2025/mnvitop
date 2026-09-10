"""Keeps host states fresh.

With streaming on, every host gets a persistent ssh session in which nvidia-smi
loops at the display interval, plus a slower full probe that refreshes the
process list and host metadata.  With streaming off, the full probe runs at
the display interval and is the only source of data.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from pathlib import Path

from .config import HostSpec, Settings
from .model import HostState, Snapshot
from .probe import probe_script
from .ssh import close_masters, describe_failure, run_probe, ssh_command
from .stream import BlockParser, stream_script

RECONNECT_MIN = 1.0
RECONNECT_MAX = 30.0
HEALTHY_AFTER = 60.0


class Poller:
    """Owns the polling tasks for all hosts and the states they write into."""

    def __init__(self, settings: Settings, control_dir: Path | None, stream: bool = True) -> None:
        self.settings = settings
        self.control_dir = control_dir
        self.stream = stream
        self.states = {spec.name: HostState() for spec in settings.hosts}
        self._tasks: list[asyncio.Task] = []
        self._procs: dict[str, asyncio.subprocess.Process] = {}

    def start(self) -> None:
        """Launch the loops; call from inside a running event loop."""
        for spec in self.settings.hosts:
            self._tasks.append(asyncio.create_task(self._meta_loop(spec)))
            if self.stream:
                self._tasks.append(asyncio.create_task(self._stream_loop(spec)))

    async def close(self) -> None:
        """Cancel the loops, end the stream sessions and close the shared ssh connections."""
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        for proc in list(self._procs.values()):
            await _end(proc)
        self._procs.clear()
        if self.control_dir is not None:
            await close_masters(self.settings.hosts, self.control_dir)

    async def _meta_loop(self, spec: HostSpec) -> None:
        """Full probe on the slow interval (or the display interval without streaming)."""
        state = self.states[spec.name]
        interval = self.settings.proc_interval if self.stream else self.settings.interval
        script = probe_script(include_gpus=not self.stream)
        while True:
            snap = await run_probe(spec, self.settings.timeout, self.control_dir, script)
            if self.stream:
                state.refresh_meta(snap)
            else:
                state.update(snap)
            await asyncio.sleep(interval)

    async def _stream_loop(self, spec: HostSpec) -> None:
        """Keep one streaming session alive, reconnecting with backoff when it ends."""
        state = self.states[spec.name]
        backoff = RECONNECT_MIN
        while True:
            started = time.monotonic()
            error = await self._stream_once(spec, state)
            state.mark_error(spec.name, error)
            if time.monotonic() - started > HEALTHY_AFTER:
                backoff = RECONNECT_MIN
            await asyncio.sleep(backoff)
            backoff = min(RECONNECT_MAX, backoff * 2)

    async def _stream_once(self, spec: HostSpec, state: HostState) -> str:
        """Run one streaming session until it ends and say why."""
        interval = self.settings.interval
        stall = max(self.settings.timeout, interval * 5)
        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_command(spec, self.settings.timeout, self.control_dir),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            return f"cannot start ssh: {exc}"
        self._procs[spec.name] = proc
        assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
        try:
            proc.stdin.write(stream_script(int(interval * 1000)).encode())
            await proc.stdin.drain()
            proc.stdin.close()
            parser = BlockParser()
            while True:
                try:
                    line = await asyncio.wait_for(proc.stdout.readline(), stall)
                except TimeoutError:
                    return f"stream stalled for {stall:.0f}s"
                if not line:
                    break
                sample = parser.feed(line.decode(errors="replace"))
                if sample is not None:
                    self._apply_sample(spec, state, sample)
            stderr = (await proc.stderr.read()).decode(errors="replace")
            await proc.wait()
            return describe_failure(stderr, proc.returncode or 0)
        finally:
            self._procs.pop(spec.name, None)
            await _end(proc)

    def _apply_sample(self, spec: HostSpec, state: HostState, gpus: list) -> None:
        """Merge a streamed GPU sample over the last full snapshot's processes and metadata."""
        template = state.good
        if template is None:
            snap = Snapshot(host=spec.name, gpus=gpus)
        else:
            snap = replace(template, gpus=gpus, taken_at=time.time(), error=None)
        state.update(snap)


async def _end(proc: asyncio.subprocess.Process) -> None:
    """Terminate a subprocess, escalating to kill after three seconds."""
    if proc.returncode is not None:
        return
    try:
        proc.terminate()
        await asyncio.wait_for(proc.wait(), 3)
    except TimeoutError:
        proc.kill()
        await proc.wait()
    except ProcessLookupError:
        pass
