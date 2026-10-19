import hashlib
import importlib.util
import io
import json
import re
import tarfile
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "download_sim_v1.py"
_SPEC = importlib.util.spec_from_file_location("download_sim_v1", _PATH)
dl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dl)


def _write_package(root: Path, files: dict[str, bytes], monkeypatch) -> Path:
    """sim-v1 tree with a data_report.json listing ``files``; pins its SHA in the module."""
    sim = root / "sim-v1"
    entries = {}
    for relative, content in files.items():
        path = sim / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        entries[relative] = {"sha256": hashlib.sha256(content).hexdigest(),
                             "bytes": len(content)}
    report = json.dumps({"tree_sha256": "t" * 64, "files": entries}).encode()
    (sim / "data_report.json").write_bytes(report)
    monkeypatch.setattr(dl, "DATA_REPORT_SHA256", hashlib.sha256(report).hexdigest())
    monkeypatch.setattr(dl, "TREE_SHA256", "t" * 64)
    return sim


def test_verify_accepts_matching_files_of_selected_parts(tmp_path, monkeypatch):
    sim = _write_package(tmp_path, {"food/samples.jsonl": b"{}\n", "ads/x.parquet": b"ads"},
                         monkeypatch)
    (sim / "ads" / "x.parquet").unlink()  # ads not selected: its absence is fine
    assert dl.verify(sim, ["food"]) == []


def test_verify_reports_missing_and_changed_files(tmp_path, monkeypatch):
    sim = _write_package(tmp_path, {"food/a.jsonl": b"a", "food/b.jsonl": b"b"}, monkeypatch)
    (sim / "food" / "a.jsonl").write_bytes(b"changed")
    (sim / "food" / "b.jsonl").unlink()
    assert dl.verify(sim, ["food"]) == ["SHA-256 or size mismatch: food/a.jsonl",
                                        "missing food/b.jsonl"]


def test_verify_rejects_a_report_that_is_not_the_pinned_one(tmp_path, monkeypatch):
    sim = _write_package(tmp_path, {"food/a.jsonl": b"a"}, monkeypatch)
    monkeypatch.setattr(dl, "DATA_REPORT_SHA256", "0" * 64)
    expected = f"{sim / 'data_report.json'}: SHA-256 differs from the pinned sim-v1 report"
    assert dl.verify(sim, ["food"]) == [expected]


def test_file_pattern_selects_report_and_parts_only():
    pattern = re.compile(dl.file_pattern(["food", "pretrain"]))
    names = ["sim-v1/data_report.json", "sim-v1/food/samples.jsonl", "sim-v1-pretrain.tar",
             "sim-v1/ads/events.parquet", "sim-v1/baseline/food.json", "run_meta.json"]
    assert [n for n in names if pattern.search(n)] == names[:3]


def _tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def test_extract_pretrain_unpacks_and_removes_the_tar(tmp_path):
    _tar(tmp_path / dl.PRETRAIN_TAR, {"sim-v1/pretrain/manifest.json": b"{}"})
    dl.extract_pretrain(tmp_path, keep_tar=False)
    assert (tmp_path / "sim-v1" / "pretrain" / "manifest.json").read_bytes() == b"{}"
    assert not (tmp_path / dl.PRETRAIN_TAR).exists()


@pytest.mark.parametrize("name", ["../evil.txt", "sim-v1/food/x.json", "/abs/pretrain/x"])
def test_extract_pretrain_rejects_members_outside_the_pretrain_folder(tmp_path, name):
    _tar(tmp_path / dl.PRETRAIN_TAR, {name: b"x"})
    with pytest.raises(ValueError, match="unexpected member"):
        dl.extract_pretrain(tmp_path, keep_tar=True)
