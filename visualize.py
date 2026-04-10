"""
Publication-quality figure generation for Section IV of the tokenomics research paper.

Generates IEEE-style plots (Figures 5-8) for:
  - Figure 5: Allocation distribution across proposals
  - Figure 6: Circulating supply growth trajectories
  - Figure 7: Fairness drift (insider share + Gini over time)
  - Figure 8: Stress test scenario outcomes
  - Supplementary: Ablation comparison

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


# ── Global style configuration ──────────────────────────────

OKABE_ITO_PALETTE = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # blueish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#999999",  # gray
]

STRESS_COLORS = {
    "bull": "#009E73",
    "neutral": "#56B4E9",
    "bear": "#D55E00",
    "unlock_shock": "#F0E442",
    "liquidity_pressure": "#CC79A7",
}

FIGURE_SIZE_SINGLE = (8, 5)
FIGURE_SIZE_DOUBLE = (10, 5)
DPI = 300
FONT_SIZE_LABEL = 11
FONT_SIZE_TICK = 10
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette(OKABE_ITO_PALETTE)


# ── Helper utilities ────────────────────────────────────────

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



# ────────────────────────────────────────────────────────────
# Figure 5: Allocation Distribution
# ────────────────────────────────────────────────────────────

def plot_allocation_distribution(results_csv: str, output_dir: str) -> None:
    """
    Boxplot/violin plot showing distribution of allocation shares.

    Reads allocation data from CSV with columns:
      - proposal_id, team_pct, investor_pct, insider_pct, distributed_pct

    Creates violin plots with overlaid box plots for each category.
    """
    output_path = _ensure_output_dir(output_dir)

    # Load data
    df = pd.read_csv(results_csv)
    categories = ["team_pct", "investor_pct", "insider_pct", "distributed_pct"]

    if not all(col in df.columns for col in categories):
        print(f"  Warning: Missing allocation columns in {results_csv}")
        return

    # Prepare data for violin plot (long format)
    data_melted = df.melt(
        value_vars=categories,
        var_name="Category",
        value_name="Percentage"
    )
    data_melted["Category"] = data_melted["Category"].str.replace("_pct", "").str.title()

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)

    # Violin plot with box overlay
    parts = ax.violinplot(
        [df[cat].dropna().values for cat in categories],
        positions=range(len(categories)),
        widths=0.7,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    # Style violin plot
    for pc in parts["bodies"]:
        pc.set_facecolor(OKABE_ITO_PALETTE[0])
        pc.set_alpha(0.6)
        pc.set_edgecolor("black")
        pc.set_linewidth(1.5)

    # Overlay box plot
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

    # Labels and formatting
    category_labels = [cat.replace("_pct", "").title() for cat in categories]
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(category_labels, fontsize=FONT_SIZE_TICK)
    ax.set_ylabel("Allocation Share (%)", fontsize=FONT_SIZE_LABEL)
    ax.set_xlabel("Allocation Category", fontsize=FONT_SIZE_LABEL)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 105)

    # Add mean values as text
    for i, cat in enumerate(categories):
        mean_val = df[cat].mean()
        ax.text(i, 102, f"μ={mean_val:.1f}", ha="center", fontsize=9)

    plt.tight_layout()
    _save_figure(fig, output_path, "Fig05_allocation_distribution",
                 "Token allocation distribution across proposals")
    plt.close(fig)


# ────────────────────────────────────────────────────────────
# Figure 6: Circulating Supply Growth
# ────────────────────────────────────────────────────────────

def plot_circulating_supply_growth(simulation_results: Union[str, List[Dict]], output_dir: str) -> None:
    """
    Line chart showing median circulating supply growth over 60 months.

    Input: List of SimulationReport dicts with supply_release.circulating_supply
    or JSON file path containing results.

    Shows: Median curve with IQR shaded band.
    """
    output_path = _ensure_output_dir(output_dir)

    # Load data
    if isinstance(simulation_results, str):
        with open(simulation_results, "r") as f:
            results_list = json.load(f)
    else:
        results_list = simulation_results

    # Extract circulating supply timeseries from each proposal
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

    # Normalize curves to percentages (0-100)
    supply_pct = []
    for curve in supply_curves:
        max_supply = max(curve) if max(curve) > 0 else 1
        pct_curve = [100 * v / max_supply for v in curve]
        supply_pct.append(pct_curve)

    # Align all curves to month 0-60
    months = np.arange(0, 61)
    aligned_curves = []
    for curve in supply_pct:
        if len(curve) >= 61:
            aligned_curves.append(curve[:61])
        else:
            # Pad with last value if needed
            padded = curve + [curve[-1]] * (61 - len(curve))
            aligned_curves.append(padded)

    aligned_curves = np.array(aligned_curves)

    # Calculate statistics
    median_curve = np.median(aligned_curves, axis=0)
    p25_curve = np.percentile(aligned_curves, 25, axis=0)
    p75_curve = np.percentile(aligned_curves, 75, axis=0)
    p5_curve = np.percentile(aligned_curves, 5, axis=0)
    p95_curve = np.percentile(aligned_curves, 95, axis=0)

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)

    # Plot percentile bands
    ax.fill_between(months, p5_curve, p95_curve, alpha=0.15, color=OKABE_ITO_PALETTE[1],
                    label="90% CI")
    ax.fill_between(months, p25_curve, p75_curve, alpha=0.3, color=OKABE_ITO_PALETTE[0],
                    label="IQR")

    # Plot median
    ax.plot(months, median_curve, color=OKABE_ITO_PALETTE[0], linewidth=2.5, label="Median")

    # Formatting
    ax.set_xlabel("Time (months)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Circulating Supply (%)", fontsize=FONT_SIZE_LABEL)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=FONT_SIZE_TICK, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0, 12, 24, 36, 48, 60])

    # Add checkpoint annotations
    for month in [0, 12, 24, 60]:
        idx = month
        if idx < len(median_curve):
            ax.axvline(month, color="gray", linestyle="--", alpha=0.3, linewidth=1)

    plt.tight_layout()
    _save_figure(fig, output_path, "Fig06_circulating_supply_growth",
                 "Median circulating supply trajectory with 90% CI and IQR")
    plt.close(fig)


# ────────────────────────────────────────────────────────────
# Figure 7: Fairness Drift
# ────────────────────────────────────────────────────────────

def plot_fairness_drift(simulation_results: Union[str, List[Dict]], output_dir: str) -> None:
    """
    Two-panel subplot showing:
    (a) Median insider share over time at checkpoints (0, 12, 24, full unlock)
    (b) Median Gini over time at same checkpoints

    Input: List of SimulationReport dicts with fairness_evaluation data.
    """
    output_path = _ensure_output_dir(output_dir)

    # Load data
    if isinstance(simulation_results, str):
        with open(simulation_results, "r") as f:
            results_list = json.load(f)
    else:
        results_list = simulation_results

    # Extract fairness snapshots from all proposals
    insider_shares = []
    gini_values = []
    checkpoint_months = set()

    for result in results_list:
        if isinstance(result, dict) and "fairness_evaluation" in result:
            fair_eval = result["fairness_evaluation"]
            if "snapshots" in fair_eval:
                for snap in fair_eval["snapshots"]:
                    checkpoint_months.add(snap["month"])
                    if isinstance(snap, dict):
                        insider_shares.append((snap["month"], snap.get("insider_share", 0)))
                        gini_values.append((snap["month"], snap.get("gini", 0)))

    if not insider_shares or not gini_values:
        print(f"  Warning: No valid fairness data found in {simulation_results}")
        return

    # Organize by checkpoint
    checkpoint_months = sorted(list(checkpoint_months))

    # Group by month and compute stats
    insider_by_month = {}
    gini_by_month = {}

    for month, insider in insider_shares:
        if month not in insider_by_month:
            insider_by_month[month] = []
        insider_by_month[month].append(insider)

    for month, gini in gini_values:
        if month not in gini_by_month:
            gini_by_month[month] = []
        gini_by_month[month].append(gini)

    # Compute medians and quartiles
    months_sorted = sorted(insider_by_month.keys())
    median_insider = [np.median(insider_by_month[m]) for m in months_sorted]
    q25_insider = [np.percentile(insider_by_month[m], 25) for m in months_sorted]
    q75_insider = [np.percentile(insider_by_month[m], 75) for m in months_sorted]

    median_gini = [np.median(gini_by_month[m]) for m in months_sorted]
    q25_gini = [np.percentile(gini_by_month[m], 25) for m in months_sorted]
    q75_gini = [np.percentile(gini_by_month[m], 75) for m in months_sorted]

    # Create two-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE_DOUBLE)

    # Panel A: Insider Share
    ax1.fill_between(months_sorted, q25_insider, q75_insider, alpha=0.3,
                     color=OKABE_ITO_PALETTE[0], label="IQR")
    ax1.plot(months_sorted, median_insider, "o-", color=OKABE_ITO_PALETTE[0],
             linewidth=2, markersize=6, label="Median")
    ax1.set_xlabel("Time (months)", fontsize=FONT_SIZE_LABEL)
    ax1.set_ylabel("Insider Share (%)", fontsize=FONT_SIZE_LABEL)
    ax1.set_title("(a) Insider Concentration", fontsize=FONT_SIZE_LABEL, weight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=FONT_SIZE_TICK)
    ax1.set_ylim(0, 100)

    # Panel B: Gini Coefficient
    ax2.fill_between(months_sorted, q25_gini, q75_gini, alpha=0.3,
                     color=OKABE_ITO_PALETTE[1], label="IQR")
    ax2.plot(months_sorted, median_gini, "s-", color=OKABE_ITO_PALETTE[1],
             linewidth=2, markersize=6, label="Median")
    ax2.set_xlabel("Time (months)", fontsize=FONT_SIZE_LABEL)
    ax2.set_ylabel("Gini Coefficient", fontsize=FONT_SIZE_LABEL)
    ax2.set_title("(b) Distribution Inequality", fontsize=FONT_SIZE_LABEL, weight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=FONT_SIZE_TICK)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    _save_figure(fig, output_path, "Fig07_fairness_drift",
                 "Fairness metrics over time: (a) insider concentration, (b) Gini coefficient")
    plt.close(fig)


# ────────────────────────────────────────────────────────────
# Figure 8: Stress Test Outcomes
# ────────────────────────────────────────────────────────────

def plot_stress_test_outcomes(simulation_results: Union[str, List[Dict]], output_dir: str) -> None:
    """
    Bar chart showing scenario pass rates across valid proposals.

    Shows: 5 scenarios (bull, neutral, bear, unlock_shock, liquidity_pressure)
    with pass rates as bar heights and error bands.
    """
    output_path = _ensure_output_dir(output_dir)

    # Load data
    if isinstance(simulation_results, str):
        with open(simulation_results, "r") as f:
            results_list = json.load(f)
    else:
        results_list = simulation_results

    # Extract stress test results
    scenario_outcomes = {
        "bull": [],
        "neutral": [],
        "bear": [],
        "unlock_shock": [],
        "liquidity_pressure": []
    }

    for result in results_list:
        if isinstance(result, dict) and "stress_test" in result:
            stress = result["stress_test"]
            if "scenarios" in stress:
                for scenario in stress["scenarios"]:
                    name = scenario.get("name", "").lower()
                    viable = scenario.get("viable", False)
                    if name in scenario_outcomes:
                        scenario_outcomes[name].append(1 if viable else 0)

    # Calculate pass rates and confidence intervals
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

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)

    x_pos = np.arange(len(scenarios))
    colors = [STRESS_COLORS.get(s, OKABE_ITO_PALETTE[0]) for s in scenarios]

    # Bars with error bars
    errors = [
        [pr - cl for pr, cl in zip(pass_rates, ci_lower)],
        [ch - pr for pr, ch in zip(pass_rates, ci_upper)]
    ]

    bars = ax.bar(x_pos, pass_rates, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5,
                  yerr=errors, capsize=8, error_kw={"elinewidth": 2, "ecolor": "black"})

    # Labels and formatting
    scenario_labels = [s.replace("_", " ").title() for s in scenarios]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenario_labels, fontsize=FONT_SIZE_TICK, rotation=15, ha="right")
    ax.set_ylabel("Pass Rate (%)", fontsize=FONT_SIZE_LABEL)
    ax.set_xlabel("Stress Scenario", fontsize=FONT_SIZE_LABEL)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)

    # Add percentage labels on bars
    for i, (bar, rate) in enumerate(zip(bars, pass_rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f"{rate*100:.0f}%", ha="center", va="bottom", fontsize=10, weight="bold")

    # Add 70% viability threshold line
    ax.axhline(0.7, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Viability threshold (70%)")
    ax.legend(fontsize=FONT_SIZE_TICK, loc="lower right")

    plt.tight_layout()
    _save_figure(fig, output_path, "Fig08_stress_test_outcomes",
                 "Pass rates across 5 stress test scenarios with 95% CI")
    plt.close(fig)


# ────────────────────────────────────────────────────────────
# Master generation function
# ────────────────────────────────────────────────────────────

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

    # Figure 5: Allocation Distribution
    alloc_file = data_path / "allocation_results.csv"
    if alloc_file.exists():
        print("Figure 5: Allocation Distribution...")
        try:
            plot_allocation_distribution(str(alloc_file), str(output_path))
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print(f"  Skipping (file not found: {alloc_file})")

    # Figures 6, 7: Simulation results
    sim_file = data_path / "simulation_results.json"
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
        print(f"  Skipping simulation figures (file not found: {sim_file})")

    # Figure 8: Stress Test Results
    stress_file = data_path / "stress_test_results.json"
    if stress_file.exists():
        print("Figure 8: Stress Test Outcomes...")
        try:
            plot_stress_test_outcomes(str(stress_file), str(output_path))
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print(f"  Skipping (file not found: {stress_file})")

    print(f"\nAll available figures generated in {output_dir}")


# ────────────────────────────────────────────────────────────
# CLI Entry Point
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for tokenomics research paper Section IV"
    )

    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default="./results",
        help="Directory containing result CSV/JSON files (default: ./results)"
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

    # Determine which figures to generate
    if any([args.fig5, args.fig6, args.fig7, args.fig8]):
        # Selective generation
        if args.fig5 and args.allocation_csv:
            plot_allocation_distribution(args.allocation_csv, args.output_dir)

        if args.fig6 and args.simulation_json:
            plot_circulating_supply_growth(args.simulation_json, args.output_dir)

        if args.fig7 and args.simulation_json:
            plot_fairness_drift(args.simulation_json, args.output_dir)

        if args.fig8 and args.stress_json:
            plot_stress_test_outcomes(args.stress_json, args.output_dir)
    else:
        # Generate all available figures
        generate_all_figures(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
