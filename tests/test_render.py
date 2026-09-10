"""Rendering smoke tests: the dashboard must render every state without raising."""

import io

from rich.console import Console

from mnvitop.config import HostSpec, Settings
from mnvitop.model import HostState, Snapshot
from mnvitop.probe import parse_probe_output
from mnvitop.render import render_dashboard
from tests.test_probe import SAMPLE


def _render(states, settings, width=200):
    console = Console(file=io.StringIO(), width=width, force_terminal=False)
    console.print(render_dashboard(states, settings, now=1_800_000_000.0, width=width))
    return console.file.getvalue()


def test_renders_good_stale_failed_and_pending_hosts():
    good = parse_probe_output("a", SAMPLE)
    good.taken_at = 1_800_000_000.0 - 35
    settings = Settings(hosts=[HostSpec("a", "a"), HostSpec("b", "b"), HostSpec("c", "c"), HostSpec("d", "d")], highlight="ray::")
    states = {
        "a": HostState(latest=good, good=good),
        "b": HostState(latest=Snapshot(host="b", error="timeout after 20s"), good=good),
        "c": HostState(latest=Snapshot(host="c", error="ssh: Could not resolve hostname c")),
        "d": HostState(),
    }
    out = _render(states, settings)
    assert "node-a" not in out
    assert "RTX PRO 6000" in out
    assert "MegatronTrainRayActor" in out
    assert "timeout after 20s, showing data from 35s ago" in out
    assert "Could not resolve hostname" in out
    assert "waiting for the first probe" in out
    assert all(len(line) <= 200 for line in out.splitlines())
    settings.detail = True
    assert "node-a" in _render(states, settings)


def test_no_procs_and_no_gpus():
    empty = Snapshot(host="a", hostname="x")
    settings = Settings(hosts=[HostSpec("a", "a")], show_procs=False)
    out = _render({"a": HostState(latest=empty, good=empty)}, settings)
    assert "no GPUs reported" in out
    assert "command" not in out


def _history_state(samples):
    state = HostState()
    for util in samples:
        snap = parse_probe_output("a", SAMPLE)
        snap.gpus[0].util = util
        state.update(snap)
    return state


def test_history_column_needs_samples_and_width():
    settings = Settings(hosts=[HostSpec("a", "a")])
    assert "History" not in _render({"a": _history_state([10])}, settings)
    wide = _render({"a": _history_state([0, 50, 100])}, settings)
    assert "History" in wide
    assert "\u2581\u2585\u2588" in wide
    assert "History" not in _render({"a": _history_state([0, 50, 100])}, settings, width=80)
    detail = Settings(hosts=[HostSpec("a", "a")], detail=True)
    assert "History" not in _render({"a": _history_state([0, 50, 100])}, detail, width=120)
    assert "History" in _render({"a": _history_state([0, 50, 100])}, detail, width=160)


def test_failed_polls_keep_history_and_good_snapshot():
    state = _history_state([10, 20])
    state.update(Snapshot(host="a", error="timeout after 20s"))
    assert state.good is not None
    assert list(state.history.values())[0] == [10, 20] or list(list(state.history.values())[0]) == [10, 20]


def test_bars_scale_with_width():
    good = parse_probe_output("a", SAMPLE)
    settings = Settings(hosts=[HostSpec("a", "a")], show_procs=False)
    narrow = _render({"a": HostState(latest=good, good=good)}, settings, width=100)
    wide = _render({"a": HostState(latest=good, good=good)}, settings, width=220)
    assert narrow.count("\u2504") < wide.count("\u2504")


def test_meta_error_shows_as_stale():
    good = parse_probe_output("a", SAMPLE)
    state = HostState(latest=good, good=good, meta_error="timeout after 20s")
    out = _render({"a": state}, Settings(hosts=[HostSpec("a", "a")]))
    assert "timeout after 20s, showing data from" in out


def test_lean_shows_largest_process_and_detail_shows_more():
    good = parse_probe_output("a", SAMPLE)
    state = HostState(latest=good, good=good)
    lean = _render({"a": state}, Settings(hosts=[HostSpec("a", "a")]))
    assert "420938" in lean and "416091" not in lean
    assert "+1" in lean
    assert "Temp" not in lean and "cpu" not in lean
    detail = _render({"a": state}, Settings(hosts=[HostSpec("a", "a")], detail=True))
    assert "420938" in detail and "416091" in detail
    assert "Temp" in detail and "cpu" in detail and "node-a" in detail


def test_rule_line_never_wraps():
    long = parse_probe_output("a", SAMPLE)
    long.gpus[0].name = long.gpus[1].name = long.gpus[2].name = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    long.hostname = "a-very-long-hostname-that-goes-on-and-on-and-on"
    state = HostState(latest=long, good=long)
    for width in (60, 80, 100, 130):
        for detail in (False, True):
            out = _render({"a": state}, Settings(hosts=[HostSpec("a", "a")], detail=detail), width=width)
            rule = [line for line in out.splitlines() if line.startswith("\u2500\u2500 ")][0]
            assert len(rule) == width, (width, detail, rule)
            assert "Server Edition" not in rule
