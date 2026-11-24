"""Application exceptions remain distinct from generator completion."""

import pytest

from pydumpling import DumpOptions
from pydumpling.snapshot import SnapshotContext, _capture


@pytest.mark.parametrize("base", [object, list])
@pytest.mark.parametrize("raise_stop", [True, False])
def test_application_stop_iteration_in_mapping(base, raise_stop):
    if _capture.pydumpling_native is None:
        pytest.skip("requires the native capture extension")

    stop = StopIteration("application iteration ended")
    calls = []

    class Name(str):
        def __radd__(self, prefix):
            calls.append(self)
            if raise_stop:
                raise stop
            # Application code can raise the exact message and cause that PEP
            # 479 would supply. A text/cause check would misclassify this.
            raise RuntimeError("generator raised StopIteration") from stop

    class Record(base):
        pass

    Record.__qualname__ = Name("Record")

    def roots():
        try:
            source = {"first": 1, "record": Record(), "after": 2}
            return (yield "mapping", source, 0, (), False, dict)
        except StopIteration as error:
            return error

    for run in (_capture.run_python, _capture.run):
        ctx = SnapshotContext(DumpOptions())
        result = run(ctx, roots())
        if raise_stop:
            assert result is stop
            assert ctx._truncation == {}
        else:
            assert result == {"first": 1}
            assert ctx._truncation == {"concurrent_mutation": 1}
    assert calls == ["Record", "Record"]


@pytest.mark.parametrize("mode", ["stack", "exception"])
def test_frame_callback_stopiteration_uses_minimal_snapshot(tmp_path, mode):
    from pydumpling import dump_current_traceback, load_dumpling, save_dumping

    calls = []

    class Name(str):
        def __radd__(self, prefix):
            calls.append(self)
            raise StopIteration("application callback")

    class Record:
        pass

    Record.__qualname__ = Name("Record")
    path = tmp_path / "minimal.dump"
    options = DumpOptions(max_frames=1)

    def capture(record):
        if mode == "stack":
            dump_current_traceback(path, options=options, strict=True)
        else:
            try:
                raise ValueError("original error")
            except ValueError:
                save_dumping(path, options=options, strict=True)

    capture(Record())
    metadata = load_dumpling(path)["snapshot_meta"]
    assert calls == ["Record"]
    assert metadata["attempts"] == [{"stage": "full", "error": "StopIteration"}]
    assert metadata["truncation"]["minimal_snapshot"] == 1
