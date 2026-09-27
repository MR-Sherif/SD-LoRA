"""Compatibility launcher for the shared SD-LoRA entry point."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRESETS = ROOT / "configs" / "best_hyperparameters.json"


def build_command(dataset, shots, data_root, wandb_mode="disabled"):
    presets = json.loads(PRESETS.read_text(encoding="utf-8"))
    config_name = presets["datasets"][dataset][str(shots)]["config"]
    config_path = ROOT / "configs" / config_name
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    return [
        sys.executable, str(ROOT / "main.py"),
        "--config", str(config_path),
        "--shots", str(shots),
        "--root_path", str(data_root),
        "--wandb_mode", wandb_mode,
    ]


def main():
    presets = json.loads(PRESETS.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(presets["datasets"]))
    parser.add_argument("--shots", required=True, type=int, choices=(1, 2, 4, 8, 16))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--wandb-mode", default="disabled", choices=("disabled", "offline", "online"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    command = build_command(args.dataset, args.shots, args.data_root, args.wandb_mode)
    print(subprocess.list2cmdline(command), flush=True)
    if args.dry_run:
        return
    if not args.data_root.is_dir():
        parser.error(f"Dataset root does not exist: {args.data_root}")
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=os.environ.copy()))


if __name__ == "__main__":
    main()
