"""Extract capability-profile-v1 features for the full frozen-fold pool.

Writes diagnostics/capability_profile/capability_features_v1.csv: one row per
query, keyed by (dataset, index, query_sha1) for joining ONLY -- those key
columns are never router features. No outcome / label column is written here;
the extractor is run from raw query text alone.

Also prints feature prevalence (label-free) as a sanity check of the heuristics.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capability_profile_v1 as cp  # noqa: E402
import run_cv_eval as cv  # noqa: E402

OUT_DIR = cv.REPO / "diagnostics" / "capability_profile"
OUT_CSV = OUT_DIR / "capability_features_v1.csv"


def main():
    pool = cv.load_full_pool_with_folds()
    folds = pd.read_csv(cv.OUT / "fold_assignments.csv")[["dataset", "index", "query_sha1"]]
    pool = pool.merge(folds, on=["dataset", "index"], how="inner")
    assert len(pool) == len(folds)

    tok = cp.load_tokenizer()
    feats = pd.DataFrame(cp.extract_many(pool["query"].tolist(), tok))[cp.FEATURES]
    out = pd.concat([pool[["dataset", "index", "query_sha1"]].reset_index(drop=True), feats], axis=1)
    out.insert(3, "extractor_version", cp.VERSION)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assert not OUT_CSV.exists(), f"refusing to overwrite {OUT_CSV}"
    out.to_csv(OUT_CSV, index=False)

    print(f"rows={len(out)}  wrote {OUT_CSV}")
    print("binary prevalence:")
    print(out[cp.BINARY_FEATURES].mean().round(3).to_string())
    print("continuous summary:")
    print(out[cp.CONTINUOUS_FEATURES].describe().T[["mean", "50%", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
