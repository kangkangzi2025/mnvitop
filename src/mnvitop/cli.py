"""Command line entry point: argument parsing and the two polling loops."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live

from . import __version__
from .config import Settings, build_settings, default_config_path, load_config
from .model import HostState
from .poller import Poller
from .render import render_dashboard
from .ssh import default_control_dir, run_probe


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser."""
    parser = argparse.ArgumentParser(
        prog="mnvitop",
        description="Watch the GPUs and GPU processes of several ssh hosts from one terminal.",
    )
    parser.add_argument(
        "hosts",
        nargs="*",
        metavar="[NAME=]TARGET",
        help="ssh destination (an alias from ~/.ssh/config or user@host) with an optional display name; "
        "the target 'local' probes this machine; default: the hosts in the config file",
    )
    parser.add_argument("-c", "--config", type=Path, default=default_config_path(), help="TOML config (default: %(default)s)")
    parser.add_argument("-i", "--interval", type=float, help="seconds between GPU samples (default 0.5, minimum 0.2)")
    parser.add_argument("-p", "--proc-interval", type=float, help="seconds between process list refreshes while streaming (default 3)")
    parser.add_argument("-t", "--timeout", type=float, help="seconds before a probe is abandoned (default 20)")
    parser.add_argument("-1", "--once", action="store_true", help="poll once, print, exit with status 2 if a host failed")
    parser.add_argument("--json", action="store_true", help="print one JSON line per poll round instead of the dashboard")
    parser.add_argument("--no-stream", action="store_true", help="spawn a full probe per interval instead of streaming nvidia-smi over one session")
    parser.add_argument("-a", "--all", action="store_true", help="detail layout: temperature, process cpu and memory, hostname, and several processes per GPU")
    parser.add_argument("--no-procs", action="store_true", help="hide the process column")
    parser.add_argument("--max-procs", type=int, help="processes shown per GPU (default 1, or 4 with --all)")
    parser.add_argument("--highlight", metavar="REGEX", help="emphasise processes whose command matches")
    parser.add_argument("--no-mux", action="store_true", help="open a fresh ssh connection for every poll")
    parser.add_argument("-w", "--width", type=int, help="render width in columns (default: the terminal width, or $COLUMNS)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the interactive or the line-oriented loop."""
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = build_settings(args, load_config(args.config))
    if not settings.hosts:
        parser.error(f"no hosts given and none listed in {args.config}")
    control_dir = None if args.no_mux else default_control_dir()
    interactive = sys.stdout.isatty() and not args.once and not args.json
    try:
        if interactive:
            return asyncio.run(watch(settings, control_dir, args.width))
        return asyncio.run(rounds(settings, control_dir, once=args.once, as_json=args.json, width=args.width))
    except KeyboardInterrupt:
        return 0


async def watch(settings: Settings, control_dir: Path | None, width: int | None = None) -> int:
    """Interactive mode: the poller keeps every host fresh while the screen redraws at the display interval."""
    poller = Poller(settings, control_dir, stream=settings.stream)
    poller.start()
    redraw = max(0.2, min(1.0, settings.interval))
    try:
        with Live(render_dashboard(poller.states, settings), console=Console(width=width), screen=True, auto_refresh=False) as live:
            while True:
                await asyncio.sleep(redraw)
                live.update(render_dashboard(poller.states, settings, width=live.console.width), refresh=True)
    finally:
        await poller.close()


async def rounds(settings: Settings, control_dir: Path | None, once: bool, as_json: bool, width: int | None = None) -> int:
    """Line-oriented mode: poll all hosts together, print, sleep, repeat."""
    console = Console(width=width)
    states = {spec.name: HostState() for spec in settings.hosts}
    while True:
        started = time.monotonic()
        snaps = await asyncio.gather(*(run_probe(spec, settings.timeout, control_dir) for spec in settings.hosts))
        for snap in snaps:
            states[snap.host].update(snap)
        if as_json:
            print(json.dumps({"time": time.time(), "hosts": [snap.to_dict() for snap in snaps]}), flush=True)
        else:
            console.print(render_dashboard(states, settings, width=console.width))
        if once:
            return 2 if any(not snap.ok for snap in snaps) else 0
        await asyncio.sleep(max(0.0, settings.interval - (time.monotonic() - started)))
