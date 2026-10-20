#!/usr/bin/env python3

import json
from pathlib import Path

import local_experiments.run_avengerspro_cached as cached


OriginalRouter = cached.CachedSimpleClusterRouter
_original_run_routing = OriginalRouter.run_routing


def run_routing_and_export_details(self):
    results = _original_run_routing(self)

    seed = getattr(self.config, "seed", "unknown")

    out_dir = Path("local_experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = (
        out_dir /
        f"avengerspro_pair_seed{seed}_routing_details.json"
    )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            results.get("routing_details", []),
            f,
            indent=2,
        )

    print()
    print(
        f"[routing-details] Saved "
        f"{len(results.get('routing_details', []))} records to:"
    )
    print(out_path)

    return results


OriginalRouter.run_routing = run_routing_and_export_details

cached.official.SimpleClusterRouter = OriginalRouter


if __name__ == "__main__":
    cached.official.main()
