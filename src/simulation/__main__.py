"""``python -m src.simulation``: simulate, export, check and report one dataset (spec §7).

Usage::

    python -m src.simulation --preset smoke --out experiments/sim
    python -m src.simulation --config configs/simulation/full.yaml --out DIR [--seed N]
                             [--skip-pretrain]

Writes ``<out>/<dataset_version>/{pretrain,food,ads,baseline,oracle}/`` and then
``<out>/<dataset_version>/data_report.json`` (spec §6.6); prints the report path.

RNG consumption order (fixed, part of the data contract): ONE
``numpy.random.default_rng(config.seed)`` is consumed by ``build_world`` -> ``simulate`` ->
``build_food_requests`` -> ``build_ad_requests``, in that order. The exports, the checks and the
difficulty report draw nothing, so ``--skip-pretrain`` changes no other file. Same seed and config
-> same bytes in every file except ``data_report.json`` (run time, timings and the library /
platform versions in ``environment``).

``data_report.json`` is written whenever the output directory was created, with ``status``:
``complete``; ``failed`` (a check failed, or the checker or the difficulty report crashed;
``failed_step`` and ``error`` say which); ``incomplete`` (generation or export crashed: the
packages are partial and must not be used). The CLI never overwrites a non-empty directory.

Exit codes: 0 complete; 2 usage error or non-empty ``<out>/<dataset_version>``; 3 status
``failed``; 4 status ``incomplete``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import traceback
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow as pa

from .behavior import simulate
from .calibrate import difficulty_report
from .checks import (
    REPORT_NAME,
    CheckError,
    file_hashes,
    run_checks,
    simulation_summary,
    tree_sha256,
    write_data_report,
)
from .config import GENERATOR_VERSION, PRESETS, config_sha256, load_config
from .export_ads import build_ad_requests, export_ads
from .export_baseline import export_baseline
from .export_food import build_food_requests, export_food
from .export_oracle import export_oracle
from .export_pretrain import export_pretrain
from .world import build_world

RNG_ORDER = ("build_world", "simulate", "build_food_requests", "build_ad_requests")
EXIT_OK, EXIT_USAGE, EXIT_FAILED, EXIT_INCOMPLETE = 0, 2, 3, 4


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.simulation",
                                     description="Generate the simulated super-app dataset.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--preset", choices=sorted(PRESETS), help="built-in preset")
    source.add_argument("--config", type=Path, help="YAML with 'preset:' plus overrides")
    parser.add_argument("--out", type=Path, required=True,
                        help="parent directory; files go to <out>/<dataset_version>/")
    parser.add_argument("--seed", type=int, default=None, help="override config.seed")
    parser.add_argument("--skip-pretrain", action="store_true",
                        help="do not write the pretrain package (faster calibration runs)")
    return parser


def _environment() -> dict[str, str]:
    """Library and platform provenance (ruling R13): recorded in ``data_report.json`` only,
    never in a hashed manifest, so the tree hash does not depend on the machine."""
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pyarrow": pa.__version__, "platform": platform.platform()}


def _log(message: str) -> None:
    print(f"[sim] {message}", file=sys.stderr, flush=True)


def _describe(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _existing_status(root: Path) -> str:
    """Status recorded in an existing ``data_report.json`` (for the refusal message)."""
    try:
        status = json.loads((root / REPORT_NAME).read_text(encoding="utf-8")).get("status")
    except (OSError, ValueError, AttributeError):
        return "no readable data_report.json"
    return f"data_report.json status: {status}"


class _Steps:
    """Runs named steps, records their durations and the step that is running."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self.current: str | None = None

    def __call__(self, name: str, fn, *args, **kwargs):
        self.current = name
        t0 = perf_counter()
        result = fn(*args, **kwargs)
        self.timings[name] = round(perf_counter() - t0, 3)
        _log(f"{name}: {self.timings[name]:.1f} s")
        self.current = None
        return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = PRESETS[args.preset] if args.preset else load_config(args.config)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    root = args.out / config.dataset_version
    if root.exists() and any(root.iterdir()):
        print(f"error: output directory {root} is not empty ({_existing_status(root)}); "
              "delete it to regenerate", file=sys.stderr)
        return EXIT_USAGE
    root.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    step = _Steps()
    report = {
        "status": "incomplete",
        "failed_step": None,
        "error": None,
        "dataset_version": config.dataset_version,
        "generator_version": GENERATOR_VERSION,
        "seed": config.seed,
        "sim_config_sha256": config_sha256(config),
        "config": asdict(config),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": _environment(),
        "command": {"preset": args.preset,
                    "config": None if args.config is None else str(args.config),
                    "seed_override": args.seed, "skip_pretrain": args.skip_pretrain},
        "rng_order": list(RNG_ORDER),
        "packages": [],
        "skipped": ["pretrain"] if args.skip_pretrain else [],
        "counts": None, "ctr": None, "personas": None, "intents": None,
        "timings_seconds": step.timings,
        "total_seconds": None,
        "checks": None,
        "difficulty": None,
        "tree_sha256": None,
        "files": None,
    }

    def finish(status: str, error: BaseException | str | None = None) -> int:
        if error is not None:
            report["failed_step"] = report["failed_step"] or step.current
            report["error"] = error if isinstance(error, str) else _describe(error)
            _log(f"{status}: {report['failed_step']}: {report['error']}")
        report["status"] = status
        report["total_seconds"] = round(perf_counter() - started, 3)
        if report["files"] is None:
            try:
                report["files"] = file_hashes(root)
                report["tree_sha256"] = tree_sha256(report["files"])
            except OSError as hash_error:
                _log(f"could not hash the output files: {_describe(hash_error)}")
        print(write_data_report(root, report))
        if status == "incomplete":
            _log(f"run incomplete: the packages under {root} are partial; delete it to rerun")
        return {"complete": EXIT_OK, "failed": EXIT_FAILED}.get(status, EXIT_INCOMPLETE)

    try:
        rng = np.random.default_rng(config.seed)
        world = step("build_world", build_world, config, rng)
        sim = step("simulate", simulate, world, rng)
        food_requests = step("build_food_requests", build_food_requests, sim, rng)
        ad_requests = step("build_ad_requests", build_ad_requests, sim, rng)
        exports = [("food", export_food, (sim, food_requests, root)),
                   ("ads", export_ads, (sim, ad_requests, root)),
                   ("baseline", export_baseline, (sim, food_requests, ad_requests, root)),
                   ("oracle", export_oracle, (sim, food_requests, ad_requests, root))]
        if not args.skip_pretrain:
            exports.insert(0, ("pretrain", export_pretrain, (sim, root)))
        exported: dict[str, dict] = {}
        for package, fn, fn_args in exports:
            exported[package] = step(f"export_{package}", fn, *fn_args)
            report["packages"].append(package)
        summary = step("summary", simulation_summary, sim, food_requests, ad_requests)
    except Exception as error:  # noqa: BLE001 - recorded in data_report.json, exit 4
        traceback.print_exc()
        return finish("incomplete", error)
    del world, sim, food_requests, ad_requests  # the checks re-read the files
    report.update(counts={**summary["counts"], "pretrain": exported.get("pretrain")},
                  ctr=summary["ctr"], personas=summary["personas"], intents=summary["intents"])

    try:
        checks = step("checks", run_checks, root, config,
                      expect_pretrain=not args.skip_pretrain)
    except CheckError as failure:
        report["checks"] = {"status": "failed", "results": failure.report.get("results"),
                            "error": str(failure)}
        report["files"] = failure.report.get("files") or None
        report["tree_sha256"] = failure.report.get("tree_sha256")
        return finish("failed", str(failure))
    except Exception as error:  # noqa: BLE001 - a checker crash fails the run, exit 3
        traceback.print_exc()
        report["checks"] = {"status": "failed", "results": None, "error": _describe(error)}
        return finish("failed", error)
    report.update(checks={"status": checks["status"], "results": checks["results"]},
                  files=checks["files"], tree_sha256=checks["tree_sha256"])

    try:
        report["difficulty"] = step("difficulty", difficulty_report, root)
    except Exception as error:  # noqa: BLE001 - recorded in data_report.json, exit 3
        traceback.print_exc()
        return finish("failed", error)
    return finish("complete")


if __name__ == "__main__":
    raise SystemExit(main())
