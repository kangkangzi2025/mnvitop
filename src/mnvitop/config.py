"""Settings: the host list from the command line or hosts.toml, plus polling options."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class HostSpec:
    """One monitored host: a display name and an ssh destination."""

    name: str
    ssh: str = ""
    ssh_args: tuple[str, ...] = ()
    local: bool = False


@dataclass
class Settings:
    """Everything the poller and the renderer need."""

    hosts: list[HostSpec] = field(default_factory=list)
    interval: float = 0.5
    timeout: float = 20.0
    max_procs: int | None = None
    show_procs: bool = True
    detail: bool = False
    highlight: str | None = None
    proc_interval: float = 3.0
    stream: bool = True

    @property
    def procs_shown(self) -> int:
        """Processes per GPU: the largest one, or four in the detail layout, unless max_procs says otherwise."""
        if self.max_procs is not None:
            return max(1, self.max_procs)
        return 4 if self.detail else 1


def default_config_path() -> Path:
    """$XDG_CONFIG_HOME/mnvitop/hosts.toml, falling back to ~/.config."""
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "mnvitop" / "hosts.toml"


def parse_host_arg(arg: str) -> HostSpec:
    """Turn ``[NAME=]TARGET`` into a HostSpec; the target ``local`` probes this machine."""
    name, sep, target = arg.partition("=")
    if not sep:
        name, target = arg, arg
    if target == "local":
        return HostSpec(name=name, local=True)
    return HostSpec(name=name, ssh=target)


def load_config(path: Path) -> dict:
    """Read a hosts.toml file; a missing file is an empty config."""
    if not path.is_file():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def hosts_from_config(data: dict) -> list[HostSpec]:
    """Build HostSpecs from ``hosts``, which holds strings or tables with name/ssh/ssh_args/local."""
    specs: list[HostSpec] = []
    for entry in data.get("hosts", []):
        if isinstance(entry, str):
            specs.append(parse_host_arg(entry))
            continue
        ssh = str(entry.get("ssh", entry.get("name", "")))
        name = str(entry.get("name", ssh))
        specs.append(
            HostSpec(
                name=name,
                ssh=ssh,
                ssh_args=tuple(str(arg) for arg in entry.get("ssh_args", [])),
                local=bool(entry.get("local", False)),
            )
        )
    return specs


def build_settings(args, config: dict) -> Settings:
    """Merge command line arguments over the config file over the defaults."""
    settings = Settings()
    settings.hosts = [parse_host_arg(arg) for arg in args.hosts] or hosts_from_config(config)
    for key in ("interval", "timeout", "max_procs", "highlight", "proc_interval"):
        value = getattr(args, key, None)
        if value is None:
            value = config.get(key)
        if value is not None:
            setattr(settings, key, value)
    settings.interval = max(0.2, float(settings.interval))
    settings.timeout = max(1.0, float(settings.timeout))
    settings.proc_interval = max(settings.interval, float(settings.proc_interval))
    settings.stream = bool(config.get("stream", True)) and not getattr(args, "no_stream", False)
    settings.detail = bool(config.get("detail", False)) or bool(getattr(args, "all", False))
    settings.show_procs = bool(config.get("show_procs", True)) and not getattr(args, "no_procs", False)
    return settings
