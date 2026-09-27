"""Select ViT-B/16 experiment arguments from a private W&B audit cache.

Usage: python scripts/select_presets.py /path/to/wandb_audit_private.json
The generated presets contain only argument values, a score, and a run URL.
"""

import json
import math
import re
import sys
from pathlib import Path


PROJECTS = {
    "imagenet": ("imagenet.yaml", ["ICML_Imagenet"]),
    "caltech101": ("caltech101.yaml", ["ICML_Caltech"]),
    "oxford_pets": ("oxford_pets.yaml", ["ICML_OxfordPets"]),
    "stanford_cars": ("stanford_cars.yaml", ["ICML_Cars", "ICML_StanfordCars"]),
    "oxford_flowers": ("oxford_flowers.yaml", ["ICML_Flowers"]),
    "food101": ("food101.yaml", ["ICML_Food"]),
    "aircrafts": ("fgvc.yaml", ["ICML_FGVCAircraft"]),
    "sun397": ("sun397.yaml", ["ICML_SUN"]),
    "dtd": ("dtd.yaml", ["ICML_DTD", "ICML_DTD_Sweep", "ICML_DTD_Sweep_4shot"]),
    "eurosat": ("eurosat.yaml", ["ICML_EuroSAT"]),
    "ucf101": ("ucf101.yaml", ["ICML_UCF", "UCF_1shot_ICML_Sweep"]),
}

ARGUMENTS = {
    "lr", "weight_decay", "batch_size", "hgt_num_layers", "hgt_num_heads",
    "transformer_num_layers", "transformer_nhead", "transformer_ff_multiplier",
    "pooling_ratio", "dropout_rate", "train_epoch", "init_beta", "init_alpha",
    "init_gamma", "lambda_con", "focal_loss_gamma", "lora_r", "lora_alpha",
    "lora_dropout", "patience", "num_workers",
}

# The historical run logs used HGT names for the RGT layer controls.
ARGUMENT_RENAMES = {
    "hgt_num_layers": "rgt_num_layers",
    "hgt_num_heads": "rgt_num_heads",
}

METRIC = "final_test_accuracy_after_search"


def candidates(runs, projects, config_name, shots):
    regular = re.compile(rf"^{shots}_shot_b16$")
    for run in runs:
        if run["project"] not in projects or run["state"] != "finished":
            continue
        cfg = run["config"]
        if cfg.get("shots") != shots or Path(cfg.get("config", "")).name != config_name:
            continue
        if not regular.fullmatch(run["name"]):
            if run["project"] not in ("ICML_DTD_Sweep", "ICML_DTD_Sweep_4shot", "UCF_1shot_ICML_Sweep"):
                continue
        score = run["summary"].get(METRIC)
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            continue
        yield run


def select(runs):
    output = {
        "schema_version": 3,
        "backbone": "ViT-B/16",
        "selection": "Highest final_test_accuracy_after_search among completed compatible ViT-B/16 runs with the original unseeded run name, plus the DTD and UCF sweep projects; excludes named ablations, seeded variants, and other backbones.",
        "metric_note": "This is a historical test-set metric from the supplied scripts. Presets are archival best observed settings, not validation-only hyperparameter selections or independent reproductions of paper averages.",
        "datasets": {},
    }
    for dataset, (config_name, projects) in PROJECTS.items():
        entries = {}
        for shots in (1, 2, 4, 8, 16):
            options = list(candidates(runs, projects, config_name, shots))
            if not options:
                raise RuntimeError(f"No compatible completed run: {dataset}, {shots} shots")
            best = sorted(options, key=lambda r: (-r["summary"][METRIC], r["project"], r["id"]))[0]
            entries[str(shots)] = {
                "config": config_name,
                "args": {ARGUMENT_RENAMES.get(key, key): best["config"][key]
                         for key in sorted(ARGUMENTS & best["config"].keys())},
                "source": {
                    "url": f"https://wandb.ai/scholarsherif-ehu/{best['project']}/runs/{best['id']}",
                    "project": best["project"],
                    "run_id": best["id"],
                    "name": best["name"],
                    "metric": METRIC,
                    "score": best["summary"][METRIC],
                    "candidate_count": len(options),
                },
            }
        output["datasets"][dataset] = entries
    return output


def main():
    runs = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    result = select(runs)
    target = Path(__file__).resolve().parents[1] / "configs" / "best_hyperparameters.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {sum(len(x) for x in result['datasets'].values())} presets to {target}")


if __name__ == "__main__":
    main()
