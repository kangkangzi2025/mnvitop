"""Rich renderables for the dashboard: a header line and one block per host.

A block is one rule line carrying the host summary, a grey column header and
one row per GPU.  The lean layout shows utilisation, memory, history, power
and the largest process on each GPU; ``detail`` adds temperature, process cpu
and memory, the hostname and more processes per GPU.  Bars and the history
are green, yellow or red by level; secondary text is mid grey so it reads on
light and dark terminals alike; a host rule turns yellow while it shows stale
data and red when it has none.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from functools import lru_cache

from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.measure import Measurement
from rich.table import Table
from rich.text import Text

from .config import HostSpec, Settings
from .model import HostState, Proc, Snapshot

FILLED = "\u2501"
TRACK = "\u2504"
RULE = "\u2500"
SPARKS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
SEP = " \u00b7 "
MUTED = "grey50"
ACCENT = "bold cyan"
HISTORY_MIN_WIDTH = 100
HISTORY_MIN_WIDTH_DETAIL = 140
MAX_BAR = 32


class Bar:
    """A level bar that grows with its table cell, followed by a right-aligned label."""

    def __init__(self, fraction: float | None, label: str, label_width: int, min_bar: int = 6) -> None:
        self.fraction = fraction
        self.label = label.rjust(label_width)
        self.min_bar = min_bar

    def __rich_measure__(self, console: Console, options: ConsoleOptions) -> Measurement:
        return Measurement(self.min_bar + 1 + len(self.label), options.max_width)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = min(MAX_BAR, max(self.min_bar, options.max_width - len(self.label) - 1))
        if self.fraction is None:
            yield Text.assemble((TRACK * width, MUTED), " ", (self.label, MUTED))
            return
        fraction = min(1.0, max(0.0, self.fraction))
        filled = round(fraction * width)
        style = _level_style(fraction)
        yield Text.assemble((FILLED * filled, style), (TRACK * (width - filled), MUTED), " ", (self.label, style))


class Sparkline:
    """Recent utilisation samples as block characters, newest at the right, as wide as the cell allows."""

    def __init__(self, values: Sequence[int | None], min_width: int = 4) -> None:
        self.values = list(values)
        self.min_width = min_width

    def __rich_measure__(self, console: Console, options: ConsoleOptions) -> Measurement:
        return Measurement(min(self.min_width, options.max_width), options.max_width)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = max(1, options.max_width)
        text = Text(" " * max(0, width - len(self.values)))
        for value in self.values[-width:]:
            if value is None:
                text.append(" ")
            else:
                level = min(1.0, max(0.0, value / 100))
                text.append(SPARKS[round(level * 7)], style=_level_style(level))
        yield text


class RuleLine:
    """One line of rule characters with a title at the left and a note at the right."""

    def __init__(self, left: Text, right: Text, style: str = MUTED) -> None:
        self.left = left
        self.right = right
        self.style = style

    def __rich_measure__(self, console: Console, options: ConsoleOptions) -> Measurement:
        return Measurement(min(20, options.max_width), options.max_width)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = options.max_width
        left = self.left.copy()
        right = self.right.copy()
        if right.cell_len > width // 2:
            right.truncate(width // 2, overflow="ellipsis")
        chrome = 9 if right.cell_len else 5
        room = width - right.cell_len - chrome
        if left.cell_len > room:
            left.truncate(max(0, room), overflow="ellipsis")
        line = Text.assemble((RULE * 2 + " ", self.style))
        line.append_text(left)
        if right.cell_len:
            line.append(" " + RULE * max(1, width - left.cell_len - right.cell_len - 8) + " ", style=self.style)
            line.append_text(right)
            line.append(" " + RULE * 2, style=self.style)
        else:
            line.append(" " + RULE * max(1, width - left.cell_len - 4), style=self.style)
        yield line


def render_dashboard(
    states: dict[str, HostState], settings: Settings, now: float | None = None, width: int | None = None
) -> RenderableType:
    """The whole screen for the current states, laid out for a terminal ``width`` columns wide."""
    now = time.time() if now is None else now
    threshold = HISTORY_MIN_WIDTH_DETAIL if settings.detail else HISTORY_MIN_WIDTH
    wide = width is None or width >= threshold
    show_history = wide and any(_samples(state) >= 2 for state in states.values())
    parts: list[RenderableType] = [_header(states, settings, now)]
    for spec in settings.hosts:
        parts.append(Text(""))
        parts.append(render_host(spec, states.get(spec.name, HostState()), settings, now, show_history))
    return Group(*parts)


def render_host(spec: HostSpec, state: HostState, settings: Settings, now: float, show_history: bool = False) -> RenderableType:
    """One host block; after a failed poll the last good snapshot stays visible under a yellow rule."""
    snap, latest = state.good, state.latest
    if snap is None:
        if latest is None:
            return RuleLine(_name(spec), Text("waiting for the first probe", style=MUTED))
        return RuleLine(_name(spec), Text(latest.error or "no data", style="red"), "red")
    error = state.error
    if error is not None:
        right = Text(f"{error}, showing data from {_age(now - snap.taken_at)} ago", style="red")
        rule = RuleLine(_title(spec, snap), right, "yellow")
    else:
        rule = RuleLine(_title(spec, snap), _status(snap, now, settings.detail))
    return Group(rule, _gpu_table(snap, state, settings, show_history))


def _header(states: dict[str, HostState], settings: Settings, now: float) -> Text:
    """Program name, fleet totals, poll interval and the clock."""
    snaps = [state.good for state in states.values() if state.good is not None]
    responding = sum(1 for state in states.values() if state.latest is not None and state.latest.ok)
    gpus = sum(len(snap.gpus) for snap in snaps)
    in_use = sum(1 for snap in snaps for gpu in snap.gpus if snap.in_use(gpu))
    text = Text()
    text.append("mnvitop", style=ACCENT)
    text.append(f"  {len(settings.hosts)} hosts, {responding} responding", style=MUTED)
    text.append(f"{SEP}{gpus} GPUs, {in_use} in use", style=MUTED)
    text.append(f"{SEP}every {settings.interval:g}s", style=MUTED)
    text.append(SEP + time.strftime("%H:%M:%S", time.localtime(now)), style=MUTED)
    return text


def _name(spec: HostSpec) -> Text:
    """The host's display name."""
    return Text(spec.name, style=ACCENT)


def _title(spec: HostSpec, snap: Snapshot) -> Text:
    """Rule title: name, GPU model and count, GPUs in use, GPU memory, load and RAM."""
    text = _name(spec)
    parts: list[str] = []
    names = {gpu.name for gpu in snap.gpus}
    if len(names) == 1:
        parts.append(f"{len(snap.gpus)}x {_short_name(names.pop())}")
    elif snap.gpus:
        parts.append(f"{len(snap.gpus)} GPUs")
    if snap.gpus:
        parts.append(f"{sum(1 for gpu in snap.gpus if snap.in_use(gpu))} in use")
        total = sum(gpu.mem_total or 0 for gpu in snap.gpus)
        if total:
            parts.append(f"gpu mem {sum(gpu.mem_used or 0 for gpu in snap.gpus) / 1024:.0f}/{total / 1024:.0f}G")
    if snap.load is not None:
        cpus = f"/{snap.ncpu}" if snap.ncpu else ""
        parts.append(f"load {snap.load[0]:.1f}{cpus}")
    if snap.mem_total and snap.mem_avail is not None:
        parts.append(f"ram {(snap.mem_total - snap.mem_avail) / 2**20:.0f}/{snap.mem_total / 2**20:.0f}G")
    for part in parts:
        text.append(SEP + part, style="default")
    return text


def _status(snap: Snapshot, now: float, detail: bool) -> Text:
    """Rule note: sample age, plus hostname and probe latency in the detail layout."""
    parts = [f"{_age(now - snap.taken_at)} ago"]
    if detail:
        parts.insert(0, f"probe {snap.latency:.1f}s")
        if snap.hostname:
            parts.insert(0, snap.hostname)
    return Text(SEP.join(parts), style=MUTED)


def _gpu_table(snap: Snapshot, state: HostState, settings: Settings, show_history: bool) -> Table:
    """One row per GPU with utilisation and memory bars, history, power and the processes on it."""
    detail = settings.detail
    uniform = len({gpu.name for gpu in snap.gpus}) == 1
    widths = _proc_widths(snap)
    table = Table(box=None, expand=True, show_edge=False, pad_edge=False, padding=(0, 1), header_style=MUTED)
    table.add_column("GPU", justify="right", width=3, no_wrap=True)
    if not uniform:
        table.add_column("Name", no_wrap=True, max_width=22, overflow="ellipsis")
    table.add_column("Util", width=11, ratio=2, no_wrap=True)
    table.add_column("Memory", width=19, ratio=3, no_wrap=True)
    if show_history:
        table.add_column("History", width=8, ratio=2, no_wrap=True)
    if detail:
        table.add_column("Temp", justify="right", width=4, no_wrap=True)
    table.add_column("Power", justify="right", width=8, no_wrap=True)
    if settings.show_procs:
        table.add_column(_proc_header(widths, detail), width=30, ratio=7, no_wrap=True, overflow="ellipsis")
    if not snap.gpus:
        table.add_row("-", Text("no GPUs reported", style=MUTED))
    for gpu in snap.gpus:
        cells: list[RenderableType] = [str(gpu.index)]
        if not uniform:
            cells.append(_short_name(gpu.name))
        cells += [
            Bar(_fraction(gpu.util, 100), _pct(gpu.util), 4),
            Bar(_fraction(gpu.mem_used, gpu.mem_total), _mem(gpu.mem_used, gpu.mem_total), 12),
        ]
        if show_history:
            cells.append(Sparkline(state.history.get(gpu.uuid, ())))
        if detail:
            cells.append(_temp(gpu.temp))
        cells.append(_power(gpu.power, gpu.power_limit))
        if settings.show_procs:
            cells.append(_procs_cell(snap.procs_on(gpu.index), settings, widths))
        table.add_row(*cells)
    return table


def _proc_widths(snap: Snapshot) -> dict[str, int]:
    """Column widths for the process lines of one host, so every GPU row aligns."""
    users = [(proc.user or "?")[:12] for proc in snap.procs]
    return {
        "user": max([4, *(len(user) for user in users)]),
        "pid": max([5, *(len(str(proc.pid)) for proc in snap.procs)]),
    }


def _proc_header(widths: dict[str, int], detail: bool) -> str:
    """Column header aligned with the process lines."""
    head = f"{'user':<{widths['user']}}  {'pid':>{widths['pid']}}  "
    if detail:
        head += f"{'mem':>6}  {'cpu':>4}  "
    return head + f"{'time':>6}  command"


def _procs_cell(procs: list[Proc], settings: Settings, widths: dict[str, int]) -> Text:
    """The largest process on a GPU, or up to ``max_procs`` of them in the detail layout."""
    if not procs:
        return Text("idle", style=MUTED)
    pattern = _pattern(settings.highlight)
    shown = procs[: settings.procs_shown]
    text = Text()
    for i, proc in enumerate(shown):
        if i:
            text.append("\n")
        text.append_text(_proc_line(proc, pattern, widths, settings.detail))
    hidden = len(procs) - len(shown)
    if hidden > 0:
        text.append(f"\n+{hidden} more" if settings.detail else f"  +{hidden}", style=MUTED)
    return text


def _proc_line(proc: Proc, pattern: re.Pattern | None, widths: dict[str, int], detail: bool) -> Text:
    """user pid [mem cpu] time command, emphasised when the command matches the highlight pattern."""
    command = proc.command or "?"
    hit = pattern is not None and pattern.search(command) is not None
    text = Text()
    text.append(f"{(proc.user or '?')[:12]:<{widths['user']}}  ", style="cyan")
    text.append(f"{proc.pid:>{widths['pid']}}  ", style="bold" if hit else "")
    if detail:
        text.append(f"{_gib(proc.gpu_mem):>6}  ", style="magenta")
        text.append(f"{_cpu(proc.cpu):>4}  ", style=MUTED)
    text.append(f"{_age(proc.etimes):>6}  ", style=MUTED)
    text.append(command, style="bold green" if hit else "")
    return text


@lru_cache(maxsize=8)
def _pattern(highlight: str | None) -> re.Pattern | None:
    """Compiled highlight regex, cached across renders."""
    return re.compile(highlight) if highlight else None


def _samples(state: HostState) -> int:
    """Longest utilisation history recorded for a host."""
    return max((len(series) for series in state.history.values()), default=0)


def _fraction(value: int | None, total: int | None) -> float | None:
    """value/total, or None when either side is unknown."""
    if value is None or not total:
        return None
    return value / total


def _level_style(fraction: float) -> str:
    """Green below half, yellow below 85 percent, red above."""
    if fraction < 0.5:
        return "green"
    if fraction < 0.85:
        return "yellow"
    return "red"


def _short_name(name: str) -> str:
    """GPU name without the vendor prefix and edition suffix."""
    name = name.removeprefix("NVIDIA ").strip()
    for suffix in (" Server Edition", " Workstation Edition"):
        name = name.removesuffix(suffix)
    return name


def _pct(value: int | None) -> str:
    """Utilisation label."""
    return "n/a" if value is None else f"{value}%"


def _mem(used: int | None, total: int | None) -> str:
    """used/total in GiB from MiB values."""
    if used is None or not total:
        return "n/a"
    return f"{used / 1024:.1f}/{total / 1024:.1f}G"


def _gib(mib: int | None) -> str:
    """GiB label from a MiB value."""
    return "n/a" if mib is None else f"{mib / 1024:.1f}G"


def _cpu(value: float | None) -> str:
    """CPU percentage label."""
    return "-" if value is None else f"{value:.0f}%"


def _temp(value: int | None) -> Text:
    """Temperature, coloured from 70 C."""
    if value is None:
        return Text("n/a", style=MUTED)
    style = "red" if value >= 80 else "yellow" if value >= 70 else ""
    return Text(f"{value}C", style=style)


def _power(draw: float | None, limit: float | None) -> Text:
    """draw/limit in watts."""
    if draw is None:
        return Text("n/a", style=MUTED)
    if limit:
        return Text(f"{draw:.0f}/{limit:.0f}W", style=_level_style(draw / limit))
    return Text(f"{draw:.0f}W")


def _age(seconds: float | None) -> str:
    """Compact duration such as 45s, 12m05s, 1h23m or 2d03h."""
    if seconds is None:
        return "-"
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    if s < 86400:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    return f"{s // 86400}d{(s % 86400) // 3600:02d}h"
