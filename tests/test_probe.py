"""Parser tests against captured probe output."""

from mnvitop.probe import parse_probe_output

SAMPLE = """
HOST|node-a|16.67 17.39 17.70|192|2113408280|1814123804
GPU|0, GPU-aaaa, NVIDIA H20, 0, 20837, 97871, 34, 118.96, 500.00
GPU|1, GPU-bbbb, NVIDIA H20, 99, 86652, 97887, 51, 316.15, 600.00
GPU|2, GPU-cccc, NVIDIA RTX PRO 6000 Blackwell Server Edition, [N/A], [N/A], 97887, 27, [N/A], [N/A]
APP|416091, GPU-aaaa, 2148
APP|420938, GPU-aaaa, 18648
APP|420938, GPU-bbbb, 16516
APP|999, GPU-bbbb, [N/A]
PS| 416091 root                                 121 98.8 2217464 sglang::scheduler
PS| 420938 wenfukang                            100 28.0 17730200 ray::MegatronTrainRayActor.train --flag a b
"""


def test_host_fields():
    snap = parse_probe_output("a", SAMPLE)
    assert snap.ok
    assert snap.hostname == "node-a"
    assert snap.load == (16.67, 17.39, 17.70)
    assert snap.ncpu == 192
    assert snap.mem_total == 2113408280
    assert snap.mem_avail == 1814123804


def test_gpus_and_na_fields():
    snap = parse_probe_output("a", SAMPLE)
    assert [g.index for g in snap.gpus] == [0, 1, 2]
    assert snap.gpus[0].mem_used == 20837
    assert snap.gpus[1].power == 316.15
    assert snap.gpus[2].util is None
    assert snap.gpus[2].power_limit is None
    assert snap.gpus[2].name.startswith("NVIDIA RTX PRO 6000")


def test_processes_merge_ps_and_map_gpus():
    snap = parse_probe_output("a", SAMPLE)
    on0 = snap.procs_on(0)
    assert [p.pid for p in on0] == [420938, 416091]
    assert on0[0].user == "wenfukang"
    assert on0[0].command == "ray::MegatronTrainRayActor.train --flag a b"
    assert on0[0].etimes == 100
    assert on0[0].cpu == 28.0
    on1 = snap.procs_on(1)
    assert {p.pid for p in on1} == {420938, 999}
    unresolved = next(p for p in on1 if p.pid == 999)
    assert unresolved.user is None and unresolved.command is None and unresolved.gpu_mem is None


def test_error_without_gpus():
    snap = parse_probe_output("a", "HOST|x|0.1 0.2 0.3|4|1|1\nERR|nvidia-smi not found in PATH\n")
    assert not snap.ok
    assert "nvidia-smi" in snap.error


def test_nvml_failure_line_becomes_error():
    snap = parse_probe_output("a", "HOST|x|||\nGPU|Failed to initialize NVML: Driver/library version mismatch\n")
    assert snap.error == "Failed to initialize NVML: Driver/library version mismatch"


def test_empty_output():
    snap = parse_probe_output("a", "")
    assert snap.error == "no output from probe"


def test_to_dict_is_json_friendly():
    import json

    snap = parse_probe_output("a", SAMPLE)
    json.dumps(snap.to_dict())
