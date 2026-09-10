"""Agent digest tests."""

from mnvitop.digest import find_gpus, fleet_digest, host_digest
from mnvitop.model import Snapshot
from mnvitop.probe import parse_probe_output
from tests.test_probe import SAMPLE


def _snap(name="a"):
    return parse_probe_output(name, SAMPLE)


def test_host_digest_counts_and_summary():
    digest = host_digest(_snap())
    assert digest["ok"] and digest["gpus"] == 3
    assert digest["busy"] == 2 and digest["free"] == 1
    assert digest["gpu_model"] == "mixed"
    assert digest["summary"].startswith("a: 1/3 mixed free")
    gpu0 = digest["gpu"][0]
    assert gpu0["busy"] and gpu0["procs"][0]["pid"] == 420938
    assert gpu0["procs"][0]["cmd"].startswith("ray::MegatronTrainRayActor")
    assert gpu0["vram_free_gb"] == round((97871 - 20837) / 1024, 1)


def test_host_digest_without_processes_is_smaller():
    with_procs = host_digest(_snap(), processes=True)
    without = host_digest(_snap(), processes=False)
    assert "procs" in with_procs["gpu"][0] and "procs" not in without["gpu"][0]


def test_fleet_digest_totals_and_unreachable():
    digest = fleet_digest([_snap("a"), Snapshot(host="b", error="timeout after 20s")])
    assert digest["summary"].startswith("1/2 hosts up; 1/3 GPUs free")
    assert "unreachable: b" in digest["summary"]
    assert digest["hosts"][1] == {"host": "b", "ok": False, "error": "timeout after 20s", "summary": "b: unreachable (timeout after 20s)"}


def test_find_gpus_places_on_idle_gpu_and_reports_alternatives():
    result = find_gpus([_snap("a")], count=1)
    assert result["found"]
    assert result["placements"][0]["host"] == "a" and result["placements"][0]["gpus"] == [2]
    result = find_gpus([_snap("a")], count=2)
    assert not result["found"]
    assert result["alternatives"][0]["idle_gpus"] == [2]
    assert "no host has 2 idle GPU" in result["summary"]


def test_find_gpus_prefers_less_loaded_host():
    busy = _snap("busy")
    calm = _snap("calm")
    calm.load = (0.5, 0.5, 0.5)
    result = find_gpus([busy, calm], count=1)
    assert [item["host"] for item in result["placements"]] == ["calm", "busy"]


def test_find_gpus_respects_min_free_vram():
    result = find_gpus([_snap("a")], count=1, min_free_gb=200)
    assert not result["found"] and result["placements"] == []
