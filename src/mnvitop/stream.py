"""Streaming GPU samples: a remote nvidia-smi loop over one ssh session and the parser for its rows."""

from __future__ import annotations

from .model import Gpu
from .probe import GPU_FIELDS, parse_gpu_row

MIN_INTERVAL_MS = 100


def stream_script(interval_ms: int) -> str:
    """Shell that streams one CSV block per sample forever, line buffered when stdbuf is available."""
    cmd = f"nvidia-smi --query-gpu={GPU_FIELDS} --format=csv,noheader,nounits -lms {max(MIN_INTERVAL_MS, int(interval_ms))}"
    return f"export LC_ALL=C\nif command -v stdbuf >/dev/null 2>&1; then exec stdbuf -oL {cmd}; fi\nexec {cmd}\n"


class BlockParser:
    """Groups streamed CSV rows into samples; a sample ends when the index column wraps back to zero."""

    def __init__(self) -> None:
        self._rows: list[Gpu] = []

    def feed(self, line: str) -> list[Gpu] | None:
        """Add one line; returns the completed sample when this line starts the next one."""
        gpu = parse_gpu_row(line.strip())
        if gpu is None:
            return None
        if gpu.index == 0 and self._rows:
            sample, self._rows = self._rows, [gpu]
            return sample
        self._rows.append(gpu)
        return None
