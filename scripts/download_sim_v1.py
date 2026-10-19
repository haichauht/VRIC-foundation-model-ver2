"""Download the simulated super-app dataset ``sim-v1`` from Kaggle and verify every file.

Source: the output of the Kaggle kernel ``hunhthhichu/vric-sim-generate`` (public). Needs the
Kaggle CLI (``pip install kaggle``) and an API
token (``~/.kaggle/kaggle.json`` or ``KAGGLE_USERNAME`` / ``KAGGLE_KEY``). Description of the data:
``docs/sim_v1_data.md``.

    python scripts/download_sim_v1.py --out data_sim                      # everything, ~5.3 GB
    python scripts/download_sim_v1.py --out data_sim --parts food ads pretrain
    python scripts/download_sim_v1.py --out data_sim --parts food --verify-only

Result: ``<out>/sim-v1/{pretrain,food,ads,baseline,oracle}/`` and ``<out>/sim-v1/data_report.json``.
``data_report.json`` is checked against the SHA-256 pinned below, then every downloaded file
against the SHA-256 that ``data_report.json`` lists for it. Exit code 0 = all verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

KERNEL = "hunhthhichu/vric-sim-generate"
DATA_REPORT_SHA256 = "00facb49b9134949b2665d291dba62b906d5c77a3563f8a1136505eb2240cc3f"
TREE_SHA256 = "2fa327e89d3d9f29b832f26c966d4a0dcb3e3b08691642328e92ebf6fad2d399"
PARTS = ("pretrain", "food", "ads", "baseline", "oracle")
PRETRAIN_TAR = "sim-v1-pretrain.tar"
PATTERNS = {
    "food": r"^sim-v1/food/",
    "ads": r"^sim-v1/ads/",
    "baseline": r"^sim-v1/baseline/",
    "oracle": r"^sim-v1/oracle/",
    "pretrain": r"^sim-v1-pretrain\.tar$",
}
REPORT = r"^sim-v1/data_report\.json$"


def file_pattern(parts: list[str]) -> str:
    """One regex for ``kaggle kernels output --file-pattern``: the report plus the parts."""
    return "|".join([REPORT, *(PATTERNS[part] for part in parts)])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_pretrain(out: Path, *, keep_tar: bool) -> None:
    """Unpack ``sim-v1-pretrain.tar`` into ``<out>/sim-v1/pretrain`` (members are checked)."""
    tar_path = out / PRETRAIN_TAR
    with tarfile.open(tar_path) as archive:
        members = archive.getmembers()
        for member in members:
            name = Path(member.name)
            if (name.is_absolute() or ".." in name.parts or name.parts[:2] != ("sim-v1", "pretrain")
                    or not (member.isfile() or member.isdir())):
                raise ValueError(f"unexpected member in {tar_path.name}: {member.name!r}")
        archive.extractall(out, members=members)
    if not keep_tar:
        tar_path.unlink()


def verify(root: Path, parts: list[str]) -> list[str]:
    """Problems found under ``root`` (= ``<out>/sim-v1``) for the given parts; [] = verified."""
    report_path = root / "data_report.json"
    if not report_path.is_file():
        return [f"missing {report_path}"]
    if sha256(report_path) != DATA_REPORT_SHA256:
        return [f"{report_path}: SHA-256 differs from the pinned sim-v1 report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["tree_sha256"] != TREE_SHA256:
        return [f"{report_path}: tree_sha256 differs from the pinned value"]
    problems = []
    for relative, expected in sorted(report["files"].items()):
        if relative.split("/", 1)[0] not in parts:
            continue
        path = root / relative
        if not path.is_file():
            problems.append(f"missing {relative}")
        elif path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            problems.append(f"SHA-256 or size mismatch: {relative}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, type=Path, help="target directory")
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    parser.add_argument("--verify-only", action="store_true", help="do not download")
    parser.add_argument("--keep-tar", action="store_true", help="keep sim-v1-pretrain.tar")
    parser.add_argument("--kaggle", default="kaggle", help="path to the Kaggle CLI")
    args = parser.parse_args(argv)
    out: Path = args.out
    if not args.verify_only:
        if shutil.which(args.kaggle) is None and not Path(args.kaggle).is_file():
            print(f"Kaggle CLI not found ({args.kaggle!r}); pip install kaggle", file=sys.stderr)
            return 2
        out.mkdir(parents=True, exist_ok=True)
        command = [args.kaggle, "kernels", "output", KERNEL, "-p", str(out),
                   "--file-pattern", file_pattern(args.parts)]
        print("$", " ".join(command), flush=True)
        if subprocess.call(command) != 0:
            print("download failed (Kaggle API token missing? see docs/sim_v1_data.md)",
                  file=sys.stderr)
            return 3
        if "pretrain" in args.parts and (out / PRETRAIN_TAR).is_file():
            print("extracting", PRETRAIN_TAR, flush=True)
            extract_pretrain(out, keep_tar=args.keep_tar)
    print("verifying (SHA-256 of every file; a few minutes for baseline/ and pretrain/)",
          flush=True)
    problems = verify(out / "sim-v1", args.parts)
    for problem in problems[:50]:
        print("  -", problem, file=sys.stderr)
    if problems:
        print(f"FAILED: {len(problems)} problem(s)", file=sys.stderr)
        return 1
    print(f"OK: sim-v1 {', '.join(args.parts)} verified in {out / 'sim-v1'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
