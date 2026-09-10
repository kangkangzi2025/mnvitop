"""MCP server that lets any agent ask the fleet two questions: what is free, and where can this run.

The tools return compact digests with a ``summary`` line first, so an agent
can answer from one field and only read the structure when it needs to.
Hosts come from the same hosts.toml the CLI uses (``MNVITOP_CONFIG`` overrides
the path, ``MNVITOP_HOSTS`` a comma separated list of ssh targets).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from .config import HostSpec, Settings, default_config_path, hosts_from_config, load_config, parse_host_arg
from .digest import find_gpus as _find_gpus
from .digest import fleet_digest
from .model import Snapshot
from .ssh import default_control_dir, run_probe

INSTRUCTIONS = (
    "GPU fleet monitor. Call find_gpus to pick a host and GPUs for a job, fleet_status for an overview "
    "of what is running where, list_hosts to see the fleet. Read the summary field first. "
    "These machines are shared: never stop processes you did not start."
)


def load_settings() -> Settings:
    """Hosts and timeouts from the config file or environment."""
    path = Path(os.environ.get("MNVITOP_CONFIG") or default_config_path())
    config = load_config(path)
    settings = Settings(hosts=hosts_from_config(config))
    env_hosts = os.environ.get("MNVITOP_HOSTS")
    if env_hosts:
        settings.hosts = [parse_host_arg(arg.strip()) for arg in env_hosts.split(",") if arg.strip()]
    settings.timeout = float(config.get("timeout", settings.timeout))
    return settings


async def probe_hosts(settings: Settings, names: list[str] | None = None) -> list[Snapshot]:
    """Probe the named hosts (all when None) concurrently; unknown names become error snapshots."""
    specs: list[HostSpec] = []
    unknown: list[Snapshot] = []
    wanted = list(settings.hosts) if not names else []
    if names:
        by_name = {spec.name: spec for spec in settings.hosts}
        for name in names:
            if name in by_name:
                wanted.append(by_name[name])
            else:
                unknown.append(Snapshot(host=name, error="not a configured host; see list_hosts"))
    specs = wanted
    snaps = await asyncio.gather(*(run_probe(spec, settings.timeout, default_control_dir()) for spec in specs))
    return list(snaps) + unknown


NO_HOSTS = "no hosts configured: write ~/.config/mnvitop/hosts.toml or set MNVITOP_HOSTS=alias1,alias2"


async def list_hosts() -> dict:
    """List the configured GPU hosts: display name and ssh target. No remote calls; use it to learn the names the other tools accept."""
    settings = load_settings()
    if not settings.hosts:
        return {"hosts": [], "summary": NO_HOSTS}
    return {"hosts": [{"name": spec.name, "ssh": "local" if spec.local else spec.ssh} for spec in settings.hosts]}


async def fleet_status(hosts: list[str] | None = None, processes: bool = True) -> dict:
    """Live GPU status of every configured host (or just the named ones), probed now over ssh; takes 1-5 s.

    Returns a fleet summary line, then per host: free and busy GPU counts, free VRAM, load, and per GPU
    the utilisation, VRAM used and free, temperature and (with processes=true) the largest processes with
    user, pid, VRAM, age and command. A GPU is busy when a process holds it or it has more than five percent
    of its memory taken. Hosts that do not answer come back with ok=false and an error instead of raising.
    """
    settings = load_settings()
    if not settings.hosts:
        return {"summary": NO_HOSTS, "hosts": []}
    snaps = await probe_hosts(settings, hosts)
    return fleet_digest(snaps, processes=processes)


async def find_gpus(count: int = 1, min_free_gb: float = 0, hosts: list[str] | None = None) -> dict:
    """Find where a job that needs ``count`` idle GPUs with at least ``min_free_gb`` free VRAM each can run right now.

    Call this first for any placement question. Returns found, a summary line, placements best first (the
    least loaded host that has enough idle GPUs, with the GPU indices to use), and when nothing fits,
    alternatives: hosts with fewer idle GPUs and busy GPUs that still have the VRAM (shared with someone
    else's job, so only for small experiments).
    """
    settings = load_settings()
    if not settings.hosts:
        return {"found": False, "summary": NO_HOSTS, "placements": [], "alternatives": []}
    snaps = await probe_hosts(settings, hosts)
    return _find_gpus(snaps, count=count, min_free_gb=min_free_gb)


def build_server():
    """The MCP server with the three tools registered; works with mcp 1.x (FastMCP) and 2.x (MCPServer)."""
    try:
        try:
            from mcp.server.mcpserver import MCPServer as Server
        except ImportError:
            from mcp.server.fastmcp import FastMCP as Server
    except ImportError as exc:
        raise SystemExit("the MCP server needs the 'mcp' package: pip install 'mnvitop[mcp]'") from exc
    server = Server("mnvitop", instructions=INSTRUCTIONS)
    server.tool()(list_hosts)
    server.tool()(fleet_status)
    server.tool()(find_gpus)
    return server


def main() -> None:
    """Run the server over stdio."""
    build_server().run()


if __name__ == "__main__":
    main()
