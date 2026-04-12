"""
Publication-quality figure generation for Section IV of the tokenomics research paper.

Generates IEEE-style plots (Figures 5-8) for:
  - Figure 5: Allocation distribution across proposals
  - Figure 6: Circulating supply growth trajectories
  - Figure 7: Fairness drift (insider share + Gini over time)
  - Figure 8: Stress test scenario outcomes

All figures are publication-ready with:
  - Colorblind-friendly palettes (Okabe-Ito)
  - High resolution (300 DPI)
  - Consistent styling (seaborn + manual refinements)
  - IEEE figure caption format in filenames
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns


OKABE_ITO_PALETTE = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#999999",
]

STRESS_COLORS = {
    "Bull": "#009E73",
    "Neutral": "#56B4E9",
    "Bear": "#D55E00",
    "Unlock Shock": "#F0E442",
    "Liquidity Pressure": "#CC79A7",
}

FIGURE_SIZE_SINGLE = (8, 5)
FIGURE_SIZE_DOUBLE = (10, 5)
DPI = 300
FONT_SIZE_LABEL = 11
FONT_SIZE_TICK = 10
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette(OKABE_ITO_PALETTE)


def _ensure_output_dir(output_dir: str) -> Path:
    """Create output directory if needed."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _save_figure(fig, output_dir: Path, filename_base: str, caption: str = "") -> None:
    """Save figure as PNG and PDF with IEEE-style naming."""
    filename_base = filename_base.replace(" ", "_")

    for ext in ["png", "pdf"]:
        if caption:
            full_name = f"{filename_base}.{ext}"
        else:
            full_name = f"{filename_base}.{ext}"
        filepath = output_dir / full_name

        if ext == "png":
            fig.savefig(filepath, dpi=DPI, bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(filepath, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {filepath}")


def plot_allocation_distribution(results_csv: str, output_dir: str) -> None:
    """
    Boxplot/violin plot showing distribution of allocation shares.

    Reads allocation data from CSV with columns:
      - proposal_id, team_pct, investor_pct, insider_pct, distributed_pct

    Creates violin plots with overlaid box plots for each category.
    """
    output_path = _ensure_output_dir(output_dir)


    df = pd.read_csv(results_csv)
    categories = ["team_pct", "investor_pct", "insider_pct", "distributed_pct"]

    if not all(col in df.columns for col in categories):
        print(f"  Warning: Missing allocation columns in {results_csv}")
        return


    data_melted = df.melt(
        value_vars=categories,
        var_name="Category",
        value_name="Percentage"
    )
    data_melted["Category"] = data_melted["Category"].str.replace("_pct", "").str.title()


    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)


    parts = ax.violinplot(
        [df[cat].dropna().values for cat in categories],
        positions=range(len(categories)),
        widths=0.7,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )


    for pc in parts["bodies"]:
        pc.set_facecolor(OKABE_ITO_PALETTE[0])
        pc.set_alpha(0.6)
        pc.set_edgecolor("black")
        pc.set_linewidth(1.5)


    bp = ax.boxplot(
        [df[cat].dropna().values for cat in categories],
        positions=range(len(categories)),
        widths=0.3,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="red", linewidth=2),
        boxprops=dict(facecolor=OKABE_ITO_PALETTE[1], alpha=0.7),
        whiskerprops=dict(color="black", linewidth=1.5),
        capprops=dict(color="black", linewidth=1.5),
    )


    category_labels = [cat.replace("_pct", "").title() for cat in categories]
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(category_labels, fontsize=FONT_SIZE_TICK)
    ax.set_ylabel("Allocation Share (%)", fontsize=FONT_SIZE_LABEL)
    ax.set_xlabel("Allocation Category", fontsize=FONT_SIZE_LABEL)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 105)


    for i, cat in enumerate(categories):
        mean_val = df[cat].mean()
        ax.text(i, 102, f"μ={mean_val:.1f}", ha="center", fontsize=9)

    plt.tight_layout()
    _save_figure(fig, output_path, "Fig05_allocation_distribution",
                 "Token allocation distribution across proposals")
    plt.close(fig)


def plot_circulating_supply_growth(simulation_results: Union[str, List[Dict]], output_dir: str) -> None:
    """
    Line chart showing median circulating supply growth over 60 months.

    Input: List of SimulationReport dicts with supply_release.circulating_supply
    or JSON file path containing results.

    Shows: Median curve with IQR shaded band.
    """
    output_path = _ensure_output_dir(output_dir)


    if isinstance(simulation_results, str):
        with open(simulation_results, "r") as f:
            results_list = json.load(f)
    else:
        results_list = simulation_results


    supply_curves = []
    for result in results_list:
        if isinstance(result, dict):
            if "supply_release" in result:
                supply_data = result["supply_release"]
                if "circulating_supply" in supply_data:
                    curve = supply_data["circulating_supply"]
                    if len(curve) > 1:
                        supply_curves.append(curve)

    if not supply_curves:
        print(f"  Warning: No valid supply curves found in {simulation_results}")
        return


    supply_pct = []
    for curve in supply_curves:
        max_supply = max(curve) if max(curve) > 0 else 1
        pct_curve = [100 * v / max_supply for v in curve]
        supply_pct.append(pct_curve)


    months = np.arange(0, 61)
    aligned_curves = []
    for curve in supply_pct:
        if len(curve) >= 61:
            aligned_curves.append(curve[:61])
        else:

            padded = curve + [curve[-1]] * (61 - len(curve))
            aligned_curves.append(padded)

    aligned_curves = np.array(aligned_curves)


    median_curve = np.median(aligned_curves, axis=0)
    p25_curve = np.percentile(aligned_curves, 25, axis=0)
    p75_curve = np.percentile(aligned_curves, 75, axis=0)
    p5_curve = np.percentile(aligned_curves, 5, axis=0)
    p95_curve = np.percentile(aligned_curves, 95, axis=0)


    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)


    ax.fill_between(months, p5_curve, p95_curve, alpha=0.15, color=OKABE_ITO_PALETTE[1],
                    label="90% CI")
    ax.fill_between(months, p25_curve, p75_curve, alpha=0.3, color=OKABE_ITO_PALETTE[0],
                    label="IQR")


    ax.plot(months, median_curve, color=OKABE_ITO_PALETTE[0], linewidth=2.5, label="Median")


    ax.set_xlabel("Time (months)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Circulating Supply (%)", fontsize=FONT_SIZE_LABEL)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=FONT_SIZE_TICK, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0, 12, 24, 36, 48, 60])


    for month in [0, 12, 24, 60]:
        idx = month
        if idx < len(median_curve):
            ax.axvline(month, color="gray", linestyle="--", alpha=0.3, linewidth=1)

    plt.tight_layout()
    _save_figure(fig, output_path, "Fig06_circulating_supply_growth",
                 "Median circulating supply trajectory with 90% CI and IQR")
    plt.close(fig)


def plot_fairness_drift(simulation_results: Union[str, List[Dict]], output_dir: str) -> None:
    """
    Two-panel subplot showing:
    (a) Median insider share over time at checkpoints (0, 12, 24, full unlock)
    (b) Median Gini over time at same checkpoints

    Input: List of SimulationReport dicts with fairness_evaluation data.
    """
    output_path = _ensure_output_dir(output_dir)


    if isinstance(simulation_results, str):
        with open(simulation_results, "r") as f:
            results_list = json.load(f)
    else:
        results_list = simulation_results


    # Collect snapshots per proposal, bucketing the last snapshot as "full_unlock"
    # Standard checkpoints are 0, 12, 24; the 4th snapshot (if present) varies
    # in actual month (45, 46, 58, 59) but represents the same concept: full unlock.
    # We bucket all post-24 snapshots into a canonical "full_unlock" label at x=36
    # for clean plotting, since only 9/100 proposals have this 4th point.
    CANONICAL_CHECKPOINTS = [0, 12, 24, 36]  # 36 = display position for "full unlock"
    CHECKPOINT_LABELS = ["0", "12", "24", "Full\nUnlock"]

    insider_by_checkpoint = {c: [] for c in CANONICAL_CHECKPOINTS}
    gini_by_checkpoint = {c: [] for c in CANONICAL_CHECKPOINTS}

    for result in results_list:
        if not isinstance(result, dict):
            continue

        if "fairness_evaluation" in result:
            fair_eval = result["fairness_evaluation"]
            if "snapshots" in fair_eval:
                snaps = fair_eval["snapshots"]
                for snap in snaps:
                    month = snap["month"]
                    insider = snap.get("insider_share", 0)
                    gini = snap.get("gini", 0)

                    if month in [0, 12, 24]:
                        insider_by_checkpoint[month].append(insider)
                        gini_by_checkpoint[month].append(gini)
                    elif month > 24:
                        # Bucket as "full unlock" at canonical position 36
                        insider_by_checkpoint[36].append(insider)
                        gini_by_checkpoint[36].append(gini)

        elif result.get("status") == "Success" and "t0_gini" in result:
            for month, gini_key, insider_key in [
                (0, "t0_gini", "insider_share_t0"),
                (12, "gini_12m", None),
                (24, "gini_24m", None),
            ]:
                gini_val = result.get(gini_key)
                if gini_val is not None:
                    gini_by_checkpoint[month].append(gini_val)
                    insider_val = result.get(insider_key, 0) if insider_key else 0
                    insider_by_checkpoint[month].append(insider_val)

    if not any(insider_by_checkpoint.values()):
        print(f"  Warning: No valid fairness data found in {simulation_results}")
        return

    # For proposals without a full_unlock snapshot (91/100 have last snap at month 24),
    # use their month-24 values as the full_unlock value (no change after 24 = flat line)
    if len(insider_by_checkpoint[36]) < len(insider_by_checkpoint[0]):
        proposals_with_full = len(insider_by_checkpoint[36])
        proposals_without = len(insider_by_checkpoint[24]) - proposals_with_full
        # These proposals have no vesting past 24 months, so their month-24 value persists
        # We include them so full_unlock has all 100 proposals for a fair comparison
        # Sort month-24 values; the first `proposals_with_full` already contributed to full_unlock
        # Append the remaining month-24 values
        all_24_insider = insider_by_checkpoint[24].copy()
        all_24_gini = gini_by_checkpoint[24].copy()
        # Add month-24 values for proposals that don't have a separate full_unlock snapshot
        insider_by_checkpoint[36].extend(all_24_insider[proposals_with_full:])
        gini_by_checkpoint[36].extend(all_24_gini[proposals_with_full:])

    # Compute statistics for each checkpoint
    checkpoints = CANONICAL_CHECKPOINTS
    median_insider = [np.median(insider_by_checkpoint[c]) if insider_by_checkpoint[c] else 0 for c in checkpoints]
    q25_insider = [np.percentile(insider_by_checkpoint[c], 25) if insider_by_checkpoint[c] else 0 for c in checkpoints]
    q75_insider = [np.percentile(insider_by_checkpoint[c], 75) if insider_by_checkpoint[c] else 0 for c in checkpoints]

    median_gini = [np.median(gini_by_checkpoint[c]) if gini_by_checkpoint[c] else 0 for c in checkpoints]
    q25_gini = [np.percentile(gini_by_checkpoint[c], 25) if gini_by_checkpoint[c] else 0 for c in checkpoints]
    q75_gini = [np.percentile(gini_by_checkpoint[c], 75) if gini_by_checkpoint[c] else 0 for c in checkpoints]

    # Print data point counts for verification
    for c, label in zip(checkpoints, CHECKPOINT_LABELS):
        n_insider = len(insider_by_checkpoint[c])
        n_gini = len(gini_by_checkpoint[c])
        print(f"    Checkpoint {label.replace(chr(10), ' ')}: {n_insider} insider points, {n_gini} gini points")


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE_DOUBLE)


    ax1.fill_between(checkpoints, q25_insider, q75_insider, alpha=0.3,
                     color=OKABE_ITO_PALETTE[0], label="IQR")
    ax1.plot(checkpoints, median_insider, "o-", color=OKABE_ITO_PALETTE[0],
             linewidth=2, markersize=6, label="Median")
    ax1.set_xlabel("Checkpoint", fontsize=FONT_SIZE_LABEL)
    ax1.set_ylabel("Insider Share", fontsize=FONT_SIZE_LABEL)
    ax1.set_title("(a) Insider Concentration", fontsize=FONT_SIZE_LABEL, weight="bold")
    ax1.set_xticks(checkpoints)
    ax1.set_xticklabels(CHECKPOINT_LABELS, fontsize=FONT_SIZE_TICK)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=FONT_SIZE_TICK)
    ax1.set_ylim(0, 1)


    ax2.fill_between(checkpoints, q25_gini, q75_gini, alpha=0.3,
                     color=OKABE_ITO_PALETTE[1], label="IQR")
    ax2.plot(checkpoints, median_gini, "s-", color=OKABE_ITO_PALETTE[1],
             linewidth=2, markersize=6, label="Median")
    ax2.set_xlabel("Checkpoint", fontsize=FONT_SIZE_LABEL)
    ax2.set_ylabel("Gini Coefficient", fontsize=FONT_SIZE_LABEL)
    ax2.set_title("(b) Distribution Inequality", fontsize=FONT_SIZE_LABEL, weight="bold")
    ax2.set_xticks(checkpoints)
    ax2.set_xticklabels(CHECKPOINT_LABELS, fontsize=FONT_SIZE_TICK)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=FONT_SIZE_TICK)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    _save_figure(fig, output_path, "Fig07_fairness_drift",
                 "Fairness metrics over time: (a) insider concentration, (b) Gini coefficient")
    plt.close(fig)


def plot_stress_test_outcomes(simulation_results: Union[str, List[Dict]], output_dir: str) -> None:
    """
    Bar chart showing scenario pass rates across valid proposals.

    Shows: 5 scenarios (bull, neutral, bear, unlock_shock, liquidity_pressure)
    with pass rates as bar heights and error bands.
    """
    output_path = _ensure_output_dir(output_dir)


    if isinstance(simulation_results, str):
        with open(simulation_results, "r") as f:
            results_list = json.load(f)
    else:
        results_list = simulation_results


    scenario_outcomes = {
        "Bull": [],
        "Neutral": [],
        "Bear": [],
        "Unlock Shock": [],
        "Liquidity Pressure": []
    }

    for result in results_list:
        if not isinstance(result, dict):
            continue


        if "stress_test" in result:
            stress = result["stress_test"]
            if "scenarios" in stress:
                for scenario in stress["scenarios"]:
                    name = scenario.get("name", "")
                    viable = scenario.get("viable", False)
                    if name in scenario_outcomes:
                        scenario_outcomes[name].append(1 if viable else 0)


        elif "scenario_details" in result:
            for name, details in result["scenario_details"].items():
                if name in scenario_outcomes:
                    viable = details.get("viable", False)
                    scenario_outcomes[name].append(1 if viable else 0)


    scenarios = list(scenario_outcomes.keys())
    pass_rates = []
    ci_lower = []
    ci_upper = []

    for scenario in scenarios:
        outcomes = scenario_outcomes[scenario]
        if outcomes:
            rate = np.mean(outcomes)
            n = len(outcomes)
            se = np.sqrt(rate * (1 - rate) / n) if n > 0 else 0
            ci_l = max(0, rate - 1.96 * se)
            ci_h = min(1, rate + 1.96 * se)
            pass_rates.append(rate)
            ci_lower.append(ci_l)
            ci_upper.append(ci_h)
        else:
            pass_rates.append(0)
            ci_lower.append(0)
            ci_upper.append(0)


    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)

    x_pos = np.arange(len(scenarios))
    colors = [STRESS_COLORS.get(s, OKABE_ITO_PALETTE[0]) for s in scenarios]


    errors = [
        [pr - cl for pr, cl in zip(pass_rates, ci_lower)],
        [ch - pr for pr, ch in zip(pass_rates, ci_upper)]
    ]

    bars = ax.bar(x_pos, pass_rates, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5,
                  yerr=errors, capsize=8, error_kw={"elinewidth": 2, "ecolor": "black"})


    scenario_labels = scenarios
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenario_labels, fontsize=FONT_SIZE_TICK, rotation=15, ha="right")
    ax.set_ylabel("Pass Rate", fontsize=FONT_SIZE_LABEL)
    ax.set_xlabel("Stress Scenario", fontsize=FONT_SIZE_LABEL)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.grid(True, axis="y", alpha=0.3)


    for i, (bar, rate) in enumerate(zip(bars, pass_rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f"{rate*100:.0f}%", ha="center", va="bottom", fontsize=10, weight="bold")


    ax.axhline(0.7, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Viability threshold (70%)")
    ax.legend(fontsize=FONT_SIZE_TICK, loc="lower right")

    plt.tight_layout()
    _save_figure(fig, output_path, "Fig08_stress_test_outcomes",
                 "Pass rates across 5 stress test scenarios with 95% CI")
    plt.close(fig)


def generate_all_figures(data_dir: str, output_dir: str) -> None:
    """
    Master function that loads all result files and generates all figures.

    Expected data files in data_dir:
      - allocation_results.csv (for Fig 5)
      - simulation_results.json (for Figs 6, 7)
      - stress_test_results.json (for Fig 8)
      - batch_results.csv (for per-project metrics)
    """
    data_path = Path(data_dir)
    output_path = _ensure_output_dir(output_dir)

    print(f"Generating publication-quality figures from data in {data_dir}")
    print(f"Output directory: {output_dir}\n")


    alloc_file = data_path / "allocation_results.csv"

    if not alloc_file.exists():
        batch_csv = data_path / "batch_results.csv"
        if batch_csv.exists():
            try:
                df = pd.read_csv(batch_csv)
                if all(col in df.columns for col in ["team_pct", "investor_pct", "insider_pct", "distributed_pct"]):
                    alloc_file = batch_csv
            except Exception:
                pass

    if alloc_file.exists():
        print("Figure 5: Allocation Distribution...")
        try:
            plot_allocation_distribution(str(alloc_file), str(output_path))
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print(f"  Skipping Fig 5 (no allocation data found)")


    sim_file = data_path / "simulation_results.json"

    if not sim_file.exists():
        checkpoint_file = data_path / "checkpoint_results.json"
        if checkpoint_file.exists():
            sim_file = checkpoint_file

    if sim_file.exists():
        print("Figure 6: Circulating Supply Growth...")
        try:
            plot_circulating_supply_growth(str(sim_file), str(output_path))
        except Exception as e:
            print(f"  Error: {e}")

        print("Figure 7: Fairness Drift...")
        try:
            plot_fairness_drift(str(sim_file), str(output_path))
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print(f"  Skipping simulation figures (no simulation data found)")


    stress_file = data_path / "simulation_results.json"
    if not stress_file.exists():
        stress_file = data_path / "checkpoint_results.json"
    if not stress_file.exists():
        stress_file = data_path / "batch_results_full.json"

    if stress_file.exists():
        print("Figure 8: Stress Test Outcomes...")
        try:
            plot_stress_test_outcomes(str(stress_file), str(output_path))
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print(f"  Skipping Fig 8 (no stress test data found)")

    print(f"\nAll available figures generated in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for tokenomics research paper Section IV"
    )

    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default="./experiment_results",
        help="Directory containing result CSV/JSON files (default: ./experiment_results)"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./figures",
        help="Directory to save figures (default: ./figures)"
    )

    parser.add_argument(
        "--fig5", action="store_true",
        help="Generate only Figure 5 (allocation distribution)"
    )

    parser.add_argument(
        "--fig6", action="store_true",
        help="Generate only Figure 6 (circulating supply growth)"
    )

    parser.add_argument(
        "--fig7", action="store_true",
        help="Generate only Figure 7 (fairness drift)"
    )

    parser.add_argument(
        "--fig8", action="store_true",
        help="Generate only Figure 8 (stress test outcomes)"
    )

    parser.add_argument(
        "--allocation-csv", type=str,
        help="Path to allocation results CSV (for Fig 5)"
    )

    parser.add_argument(
        "--simulation-json", type=str,
        help="Path to simulation results JSON (for Figs 6, 7)"
    )

    parser.add_argument(
        "--stress-json", type=str,
        help="Path to stress test results JSON (for Fig 8)"
    )

    args = parser.parse_args()


    if any([args.fig5, args.fig6, args.fig7, args.fig8]):

        if args.fig5 and args.allocation_csv:
            plot_allocation_distribution(args.allocation_csv, args.output_dir)

        if args.fig6 and args.simulation_json:
            plot_circulating_supply_growth(args.simulation_json, args.output_dir)

        if args.fig7 and args.simulation_json:
            plot_fairness_drift(args.simulation_json, args.output_dir)

        if args.fig8 and args.stress_json:
            plot_stress_test_outcomes(args.stress_json, args.output_dir)
    else:

        generate_all_figures(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()

