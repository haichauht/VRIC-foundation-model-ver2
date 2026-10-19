"""Run local baselines: python -m src.baselines {validate,run} --config path."""

import argparse
import json
import sys

from .contracts import ContractError
from .runner import run


def main() -> int:
    parser = argparse.ArgumentParser(description="Local B0/B1 handoff runner")
    parser.add_argument("command", choices=["validate", "run"])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        result = run(args.config, validate_only=args.command == "validate")
    except (ContractError, ValueError, TypeError, KeyError, OSError) as error:
        print(f"baseline failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
