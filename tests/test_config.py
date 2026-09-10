"""Config merging tests."""

import argparse

from mnvitop.config import HostSpec, build_settings, hosts_from_config, parse_host_arg


def _args(**overrides):
    base = dict(hosts=[], interval=None, timeout=None, max_procs=None, highlight=None, no_procs=False, proc_interval=None, no_stream=False, all=False)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_parse_host_arg_forms():
    assert parse_host_arg("h20") == HostSpec(name="h20", ssh="h20")
    assert parse_host_arg("box=root@1.2.3.4") == HostSpec(name="box", ssh="root@1.2.3.4")
    assert parse_host_arg("here=local") == HostSpec(name="here", local=True)


def test_hosts_from_config_tables_and_strings():
    data = {"hosts": ["h20", {"name": "semi", "ssh": "root@x", "ssh_args": ["-p", 8022]}, {"name": "me", "local": True}]}
    specs = hosts_from_config(data)
    assert specs[0] == HostSpec(name="h20", ssh="h20")
    assert specs[1] == HostSpec(name="semi", ssh="root@x", ssh_args=("-p", "8022"))
    assert specs[2].local


def test_cli_hosts_override_config_and_values_merge():
    config = {"hosts": ["a", "b"], "interval": 30, "highlight": "ray::"}
    settings = build_settings(_args(hosts=["c"], timeout=3), config)
    assert [h.name for h in settings.hosts] == ["c"]
    assert settings.interval == 30
    assert settings.timeout == 3
    assert settings.highlight == "ray::"
    assert settings.show_procs
    settings = build_settings(_args(no_procs=True), config)
    assert [h.name for h in settings.hosts] == ["a", "b"]
    assert not settings.show_procs


def test_stream_and_interval_floors():
    settings = build_settings(_args(interval=0.05, proc_interval=0.01), {})
    assert settings.interval == 0.2
    assert settings.proc_interval == 0.2
    assert settings.stream
    settings = build_settings(_args(no_stream=True), {"stream": True})
    assert not settings.stream
    settings = build_settings(_args(), {"stream": False})
    assert not settings.stream


def test_detail_and_procs_shown():
    assert build_settings(_args(), {}).procs_shown == 1
    assert build_settings(_args(all=True), {}).procs_shown == 4
    assert build_settings(_args(max_procs=2), {}).procs_shown == 2
    assert build_settings(_args(), {"detail": True}).detail
