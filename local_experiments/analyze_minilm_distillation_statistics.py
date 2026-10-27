from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


OUT = Path("local_experiments/results")

METHODS = {
    "hard_nontie": (
        OUT / "minilm_nonlinear_hard_label_multiseed.csv"
    ),
    "soft_nontie": (
        OUT / "minilm_nonlinear_soft_nontie_multiseed.csv"
    ),
    "neutral_tie": (
        OUT / "minilm_nonlinear_neutral_tie_multiseed.csv"
    ),
    "soft_all": (
        OUT
        / "minilm_nonlinear_score_distillation_multiseed.csv"
    ),
}

# Pre-specified comparisons.
#
# Each comparison asks one concrete research question.
COMPARISONS = [
    (
        "soft_nontie_vs_hard_nontie",
        "hard_nontie",
        "soft_nontie",
        "Does the GTE soft target help when training data are held constant?",
    ),
    (
        "soft_all_vs_soft_nontie",
        "soft_nontie",
        "soft_all",
        "Do GTE-scored tie queries add value beyond non-tie distillation?",
    ),
    (
        "soft_all_vs_neutral_tie",
        "neutral_tie",
        "soft_all",
        "Are GTE scores better than simply labeling ties as neutral?",
    ),
    (
        "soft_all_vs_hard_nontie",
        "hard_nontie",
        "soft_all",
        "What is the total gain of the full distilled method?",
    ),
]


def load_method(name, path):
    df = pd.read_csv(path)

    required = {
        "seed",
        "router_macro_f1",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{name}: missing columns {sorted(missing)}"
        )

    df = (
        df[["seed", "router_macro_f1"]]
        .drop_duplicates(subset=["seed"])
        .sort_values("seed")
        .reset_index(drop=True)
    )

    if len(df) != 10:
        raise ValueError(
            f"{name}: expected 10 seeds, found {len(df)}"
        )

    return df.rename(
        columns={
            "router_macro_f1": name,
        }
    )


def holm_adjust(p_values):
    """
    Holm multiple-comparison correction.
    """
    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    m = len(p_values)

    order = np.argsort(p_values)

    adjusted_sorted = np.empty(
        m,
        dtype=float,
    )

    running_max = 0.0

    for rank, index in enumerate(order):
        multiplier = m - rank

        candidate = min(
            1.0,
            multiplier * p_values[index],
        )

        running_max = max(
            running_max,
            candidate,
        )

        adjusted_sorted[rank] = running_max

    adjusted = np.empty(
        m,
        dtype=float,
    )

    for rank, index in enumerate(order):
        adjusted[index] = adjusted_sorted[rank]

    return adjusted


def main():
    frames = []

    for name, path in METHODS.items():
        print(f"Loading {name}: {path}")

        frames.append(
            load_method(
                name,
                path,
            )
        )

    merged = frames[0]

    for frame in frames[1:]:
        merged = merged.merge(
            frame,
            on="seed",
            how="inner",
            validate="one_to_one",
        )

    expected_seeds = list(
        range(42, 52)
    )

    actual_seeds = (
        merged["seed"]
        .astype(int)
        .tolist()
    )

    if actual_seeds != expected_seeds:
        raise ValueError(
            "Expected paired seeds 42-51, got "
            f"{actual_seeds}"
        )

    print()
    print("=" * 88)
    print("METHOD MEANS")
    print("=" * 88)

    method_rows = []

    for name in METHODS:
        values = merged[
            name
        ].to_numpy(
            dtype=float
        )

        mean = values.mean()
        sd = values.std(
            ddof=1
        )

        method_rows.append({
            "method": name,
            "n_seeds": len(values),
            "mean_macro_f1": mean,
            "sd_macro_f1": sd,
            "min_macro_f1": values.min(),
            "max_macro_f1": values.max(),
        })

        print(
            f"{name:18s} "
            f"mean={mean:.6f}  "
            f"sd={sd:.6f}"
        )

    comparison_rows = []

    for (
        comparison_name,
        baseline_name,
        improved_name,
        question,
    ) in COMPARISONS:

        baseline = merged[
            baseline_name
        ].to_numpy(
            dtype=float
        )

        improved = merged[
            improved_name
        ].to_numpy(
            dtype=float
        )

        diff = (
            improved
            - baseline
        )

        n = len(diff)

        mean_diff = (
            diff.mean()
        )

        sd_diff = (
            diff.std(
                ddof=1
            )
        )

        se_diff = (
            sd_diff
            / np.sqrt(n)
        )

        t_critical = stats.t.ppf(
            0.975,
            df=n - 1,
        )

        ci_low = (
            mean_diff
            - t_critical * se_diff
        )

        ci_high = (
            mean_diff
            + t_critical * se_diff
        )

        t_result = stats.ttest_rel(
            improved,
            baseline,
        )

        try:
            wilcoxon_result = (
                stats.wilcoxon(
                    improved,
                    baseline,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
            )

            wilcoxon_stat = (
                wilcoxon_result.statistic
            )

            wilcoxon_p = (
                wilcoxon_result.pvalue
            )

        except ValueError:
            wilcoxon_stat = np.nan
            wilcoxon_p = np.nan

        tolerance = 1e-12

        wins = int(
            np.sum(
                diff > tolerance
            )
        )

        losses = int(
            np.sum(
                diff < -tolerance
            )
        )

        ties = (
            n
            - wins
            - losses
        )

        sign_n = (
            wins
            + losses
        )

        if sign_n:
            sign_p = stats.binomtest(
                wins,
                n=sign_n,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        else:
            sign_p = 1.0

        # Paired standardized effect size.
        if sd_diff > 0:
            cohen_dz = (
                mean_diff
                / sd_diff
            )
        else:
            cohen_dz = np.nan

        comparison_rows.append({
            "comparison": comparison_name,
            "question": question,
            "baseline": baseline_name,
            "improved": improved_name,
            "n_seeds": n,
            "baseline_mean_f1":
                baseline.mean(),
            "improved_mean_f1":
                improved.mean(),
            "mean_f1_difference":
                mean_diff,
            "difference_sd":
                sd_diff,
            "ci95_low":
                ci_low,
            "ci95_high":
                ci_high,
            "wins":
                wins,
            "losses":
                losses,
            "ties":
                ties,
            "cohen_dz":
                cohen_dz,
            "paired_t_stat":
                t_result.statistic,
            "paired_t_p":
                t_result.pvalue,
            "wilcoxon_stat":
                wilcoxon_stat,
            "wilcoxon_p":
                wilcoxon_p,
            "sign_test_p":
                sign_p,
        })

    comparisons = pd.DataFrame(
        comparison_rows
    )

    comparisons[
        "paired_t_p_holm"
    ] = holm_adjust(
        comparisons[
            "paired_t_p"
        ].to_numpy()
    )

    comparisons[
        "wilcoxon_p_holm"
    ] = holm_adjust(
        comparisons[
            "wilcoxon_p"
        ]
        .fillna(1.0)
        .to_numpy()
    )

    methods = pd.DataFrame(
        method_rows
    )

    print()
    print("=" * 88)
    print("PAIRED COMPARISONS")
    print("=" * 88)

    for _, row in (
        comparisons.iterrows()
    ):
        print()
        print(row["comparison"])
        print(
            f"  {row['question']}"
        )
        print(
            f"  Mean F1 difference : "
            f"{row['mean_f1_difference']:+.6f}"
        )
        print(
            f"  95% CI             : "
            f"[{row['ci95_low']:+.6f}, "
            f"{row['ci95_high']:+.6f}]"
        )
        print(
            f"  Wins/Losses/Ties   : "
            f"{int(row['wins'])}/"
            f"{int(row['losses'])}/"
            f"{int(row['ties'])}"
        )
        print(
            f"  Cohen dz           : "
            f"{row['cohen_dz']:.4f}"
        )
        print(
            f"  Paired t p         : "
            f"{row['paired_t_p']:.6f}"
        )
        print(
            f"  Paired t Holm p    : "
            f"{row['paired_t_p_holm']:.6f}"
        )
        print(
            f"  Wilcoxon p         : "
            f"{row['wilcoxon_p']:.6f}"
        )
        print(
            f"  Wilcoxon Holm p    : "
            f"{row['wilcoxon_p_holm']:.6f}"
        )
        print(
            f"  Sign-test p        : "
            f"{row['sign_test_p']:.6f}"
        )

    method_path = (
        OUT
        / "minilm_distillation_method_statistics.csv"
    )

    comparison_path = (
        OUT
        / "minilm_distillation_paired_comparisons.csv"
    )

    per_seed_path = (
        OUT
        / "minilm_distillation_paired_seed_values.csv"
    )

    methods.to_csv(
        method_path,
        index=False,
    )

    comparisons.to_csv(
        comparison_path,
        index=False,
    )

    merged.to_csv(
        per_seed_path,
        index=False,
    )

    print()
    print("=" * 88)
    print("SAVED")
    print("=" * 88)
    print(method_path)
    print(comparison_path)
    print(per_seed_path)


if __name__ == "__main__":
    main()
