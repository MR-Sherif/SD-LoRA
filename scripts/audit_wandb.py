"""Read-only W&B audit; requires WANDB_API_KEY in the environment.

Run this locally to refresh the private analysis cache. The cache is excluded
from Git and does not contain the API key.
"""

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


ENTITY = "scholarsherif-ehu"
PROJECTS = [
    "ICML_Imagenet", "ICML_Cars", "ICML_DTD", "ICML_SUN", "ICML_Caltech",
    "ICML_Food", "ICML_Flowers", "ICML_EuroSAT", "ICML_StanfordCars",
    "ICML_OxfordPets", "ICML_FGVCAircraft", "ICML_UCF", "ICML_Pets",
    "ICML_DTD_Sweep", "ICML_DTD_Sweep_4shot", "UCF_1shot_ICML_Sweep",
]


def request(query, variables):
    key = os.environ["WANDB_API_KEY"]
    authorization = base64.b64encode(("api:" + key).encode()).decode()
    payload = json.dumps({"query": query, "variables": variables}).encode()
    for attempt in range(4):
        req = urllib.request.Request(
            "https://api.wandb.ai/graphql", data=payload,
            headers={"Content-Type": "application/json", "Authorization": "Basic " + authorization},
        )
        try:
            result = json.load(urllib.request.urlopen(req, timeout=45))
            if "errors" in result:
                raise RuntimeError(result["errors"])
            return result["data"]
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


QUERY = """query ($entity: String!, $project: String!, $after: String) {
  model(name: $project, entityName: $entity) {
    buckets(first: 100, after: $after) {
      edges { node { name displayName state config summaryMetrics } }
      pageInfo { hasNextPage endCursor }
    }
  }
}"""


def unpack(value):
    raw = json.loads(value or "{}")
    return {key: item.get("value", item) if isinstance(item, dict) else item
            for key, item in raw.items() if key != "_wandb"}


def main():
    all_runs = []
    for project in PROJECTS:
        cursor = None
        count = 0
        while True:
            result = request(QUERY, {"entity": ENTITY, "project": project, "after": cursor})
            model = result["model"]
            if not model:
                break
            page = model["buckets"]
            for edge in page["edges"]:
                run = edge["node"]
                all_runs.append({
                    "project": project,
                    "id": run["name"],
                    "name": run["displayName"],
                    "state": run["state"],
                    "config": unpack(run["config"]),
                    "summary": json.loads(run["summaryMetrics"] or "{}"),
                })
                count += 1
            if not page["pageInfo"]["hasNextPage"]:
                break
            cursor = page["pageInfo"]["endCursor"]
        print(f"{project}: {count} runs", file=sys.stderr)

    cache = Path(os.environ.get("WANDB_AUDIT_CACHE", "wandb_audit_private.json"))
    cache.write_text(json.dumps(all_runs, indent=2), encoding="utf-8")
    summary = defaultdict(lambda: {"shots": Counter(), "metrics": Counter(), "states": Counter()})
    for run in all_runs:
        part = summary[run["project"]]
        part["shots"][str(run["config"].get("shots"))] += 1
        part["metrics"].update(run["summary"].keys())
        part["states"][run["state"]] += 1
    for project, part in summary.items():
        print(project, "shots", dict(part["shots"]), "states", dict(part["states"]))
        print("  metrics", dict(part["metrics"].most_common(25)))
    print("Private cache:", cache)


if __name__ == "__main__":
    main()
