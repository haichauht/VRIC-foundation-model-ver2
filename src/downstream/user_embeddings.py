"""Per-request foundation user embeddings for a sim_v1 Food or Ads package.

Run: python -m src.downstream.user_embeddings --task food --config configs/food_recommendation_sim.yaml
         --out experiments/embeddings [--init pretrained random] [--device auto]

Requests are tokenized by the downstream readers (``load_food_inputs`` / ``load_ad_inputs``) with
the checkpoint's frozen preprocessing, so every embedding sees only history strictly before the
request cutoff, exactly as the downstream heads do. ``--init random`` is the control arm (the
checkpoint's config and vocabulary with fresh weights seeded by the config ``seed``).

Writes ``{task}_{init}.npz`` (request_id, split, long_term, short_term; one row per request) and a
``{task}_{init}.json`` provenance sidecar. B1 reads them through ``foundation_features``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from src.contracts.output import sha256_file
from src.downstream.foundation_init import reinitialize_foundation
from src.downstream.head_training import embed_users_on_device, encoder_device

TASKS = ("food", "ads")
INITS = ("pretrained", "random")


def _load(task, config):
    """Foundation and inputs exactly as the downstream run of ``task`` loads them (sim_v1 only)."""
    if config["data"].get("format") != "sim_v1":
        raise ValueError("user embeddings are exported for data.format sim_v1 packages only")
    root, version = config["data"]["root"], config["data"]["expected_version"]
    if task == "food":
        from src.downstream.food_inputs import load_food_foundation, load_food_inputs
        from src.downstream.food_samples import read_sim_food_manifest

        manifest, _ = read_sim_food_manifest(root, version)
        foundation = load_food_foundation(
            config["checkpoint"]["path"], expected_version=version,
            expected_sim_config_sha256=manifest["sim_config_sha256"])
        return foundation, load_food_inputs(config, foundation)
    from src.downstream.ad_inputs import load_ad_foundation, load_ad_inputs
    from src.downstream.ad_optimization import _sim_config_sha256

    foundation = load_ad_foundation(config["checkpoint"]["path"], expected_version=version,
                                    expected_sim_config_sha256=_sim_config_sha256(root))
    return foundation, load_ad_inputs(config, foundation)


def export_user_embeddings(config, *, task, out_dir, inits=INITS, device="cpu"):
    """Encode every request once per foundation arm; return the written ``.npz`` paths."""
    if task not in TASKS or not inits or not set(inits) <= set(INITS):
        raise ValueError(f"task must be one of {list(TASKS)} and init a subset of {list(INITS)}")
    target = encoder_device(device)
    foundation, inputs = _load(task, config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for init in dict.fromkeys(inits):
        arm = foundation if init == "pretrained" else reinitialize_foundation(foundation, config["seed"])
        arm.to(target).eval()
        requests, splits, long_term, short_term = [], [], [], []
        with torch.no_grad():
            for split, batches in inputs.batches.items():
                for batch in batches:
                    values = embed_users_on_device(arm, batch.foundation)
                    long_term.append(values["long_term"].float().numpy())
                    short_term.append(values["short_term"].float().numpy())
                    requests.extend(rows[0]["request_id"] for rows in batch.rows)
                    splits.extend(split for _ in batch.rows)
        arm.to("cpu")
        if len(set(requests)) != len(requests):
            raise ValueError("a request was encoded twice")
        path = out / f"{task}_{init}.npz"
        np.savez(path, request_id=np.array(requests), split=np.array(splits),
                 long_term=np.concatenate(long_term), short_term=np.concatenate(short_term))
        path.with_suffix(".json").write_text(json.dumps({
            "format_version": 1, "task": task, "init": init, "seed": config["seed"],
            "checkpoint_sha256": sha256_file(config["checkpoint"]["path"]),
            "data_version": config["data"]["expected_version"],
            "data_manifest_sha256": inputs.data_manifest_sha256,
            "preprocess_fingerprint": inputs.preprocess_fingerprint,
            "d_model": inputs.d_model, "request_count": len(requests),
        }, indent=2), encoding="utf-8")
        written.append(path)
        print(json.dumps({"task": task, "init": init, "requests": len(requests),
                          "path": str(path)}), flush=True)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--config", required=True, help="the task's downstream sim_v1 config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--init", nargs="+", choices=INITS, default=list(INITS))
    parser.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    args = parser.parse_args(argv)
    try:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        export_user_embeddings(config, task=args.task, out_dir=args.out, inits=args.init,
                               device=args.device)
        return 0
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError) as error:
        print(f"user_embeddings task={args.task}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
