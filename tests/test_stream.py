"""Streaming parser tests."""

from mnvitop.stream import BlockParser, stream_script

ROW = "{i}, GPU-{i}, NVIDIA H20, {u}, 100, 97871, 40, 100.0, 500.0"


def test_blocks_end_when_index_wraps():
    parser = BlockParser()
    assert parser.feed(ROW.format(i=0, u=10)) is None
    assert parser.feed(ROW.format(i=1, u=20)) is None
    sample = parser.feed(ROW.format(i=0, u=30))
    assert [gpu.util for gpu in sample] == [10, 20]
    assert parser.feed(ROW.format(i=1, u=40)) is None
    sample = parser.feed(ROW.format(i=0, u=50))
    assert [gpu.util for gpu in sample] == [30, 40]


def test_single_gpu_host_yields_every_line_after_the_first():
    parser = BlockParser()
    assert parser.feed(ROW.format(i=0, u=1)) is None
    assert [gpu.util for gpu in parser.feed(ROW.format(i=0, u=2))] == [1]
    assert [gpu.util for gpu in parser.feed(ROW.format(i=0, u=3))] == [2]


def test_garbage_lines_are_ignored():
    parser = BlockParser()
    assert parser.feed("Failed to initialize NVML: Driver/library version mismatch") is None
    assert parser.feed("") is None
    assert parser.feed(ROW.format(i=0, u=5)) is None


def test_stream_script_uses_lms_and_floors_interval():
    assert "-lms 1000" in stream_script(1000)
    assert "-lms 100" in stream_script(10)
    assert "stdbuf -oL" in stream_script(500)
