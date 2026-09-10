"""HostState merge semantics for streaming."""

from mnvitop.model import HostState, Snapshot
from mnvitop.probe import parse_probe_output, probe_script
from tests.test_probe import SAMPLE

META_ONLY = "\n".join(line for line in SAMPLE.splitlines() if not line.startswith("GPU|"))


def test_refresh_meta_keeps_streamed_gpus_and_history_but_takes_processes():
    state = HostState()
    full = parse_probe_output("a", SAMPLE)
    state.update(full)
    streamed = parse_probe_output("a", SAMPLE)
    streamed.gpus[0].util = 77
    streamed.procs = []
    state.update(streamed)
    assert state.good.procs == []
    meta = parse_probe_output("a", META_ONLY)
    assert meta.gpus == []
    assert meta.procs and all(proc.gpu_index is None for proc in meta.procs)
    state.refresh_meta(meta)
    assert state.good.gpus[0].util == 77
    assert {proc.pid for proc in state.good.procs} == {416091, 420938, 999}
    assert state.good.procs_on(0)[0].pid == 420938
    assert list(state.history["GPU-aaaa"]) == [0, 77]
    assert state.error is None


def test_meta_failure_marks_stale_without_losing_data():
    state = HostState()
    state.update(parse_probe_output("a", SAMPLE))
    state.refresh_meta(Snapshot(host="a", error="timeout after 20s"))
    assert state.error == "timeout after 20s"
    assert state.good is not None and state.good.gpus
    state.refresh_meta(parse_probe_output("a", META_ONLY))
    assert state.error is None


def test_mark_error_and_recovery():
    state = HostState()
    state.update(parse_probe_output("a", SAMPLE))
    state.mark_error("a", "stream stalled for 20s")
    assert state.error == "stream stalled for 20s"
    state.update(parse_probe_output("a", SAMPLE))
    assert state.error is None


def test_probe_script_variants():
    assert "--query-gpu" in probe_script(True)
    assert "--query-gpu" not in probe_script(False)
    assert "--query-compute-apps" in probe_script(False)
