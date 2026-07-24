"""
Publication-quality figure generation for Section IV of the tokenomics research paper.

Generates IEEE-style plots (Figures 3-7) for:
  - Figure 3: Control layer finding frequencies across proposals
  - Figure 4: Allocation distribution across proposals
  - Figure 5: Circulating supply growth trajectories
  - Figure 6: Fairness drift (insider share + Gini over time)
  - Figure 7: Stress test scenario outcomes

All figures are publication-ready with:
  - Colorblind-friendly palettes (Okabe-Ito)
  - High resolution (300 DPI)
  - Consistent styling (seaborn + manual refinements)
  - IEEE figure caption format in filenames
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    "Unlock Shock": "#B07D00",
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


def _save_figure(fig, output_dir: Path, filename_base: str) -> None:
    """Save figure as PNG and PDF with IEEE-style naming."""
    filename_base = filename_base.replace(" ", "_")

    for ext in ["png", "pdf"]:
        filepath = output_dir / f"{filename_base}.{ext}"
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


    ax.boxplot(
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

    # Control threshold reference lines
    thresholds = {
        "team_pct": (35, "Team ≤35%"),
        "investor_pct": (25, "Investor ≤25%"),
        "distributed_pct": (20, "Dist. ≥20%"),
    }
    for i, cat in enumerate(categories):
        if cat in thresholds:
            val, lbl = thresholds[cat]
            ax.hlines(val, i - 0.4, i + 0.4, colors="#D55E00", linewidths=1.5,
                      linestyles="--", zorder=6)
            ax.text(i + 0.43, val, lbl, va="center", fontsize=7.5,
                    color="#D55E00", fontstyle="italic")

    plt.tight_layout()
    _save_figure(fig, output_path, "Fig03_allocation_distribution")
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


    for month in [12, 24, 60]:
        idx = month
        if idx < len(median_curve):
            ax.axvline(month, color="gray", linestyle="--", alpha=0.3, linewidth=1)
            val = median_curve[idx]
            ax.annotate(f"median:\n{val:.1f}%",
                        xy=(month, val), xytext=(month + 1.5, val - 10),
                        fontsize=8, color="#555555",
                        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=1))

    plt.tight_layout()
    _save_figure(fig, output_path, "Fig04_circulating_supply_growth")
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

    # Collect per-proposal snapshots first, then aggregate — avoids index-position
    # assumptions when backfilling full_unlock from month-24 values.
    per_proposal: List[Dict] = []

    for result in results_list:
        if not isinstance(result, dict):
            continue

        proposal: Dict = {}
        if "fairness_evaluation" in result:
            fair_eval = result["fairness_evaluation"]
            if "snapshots" in fair_eval:
                for snap in fair_eval["snapshots"]:
                    month = snap["month"]
                    insider = snap.get("insider_share", 0)
                    gini = snap.get("gini", 0)
                    if month in [0, 12, 24]:
                        proposal[month] = (insider, gini)
                    elif month > 24:
                        proposal[36] = (insider, gini)
        elif result.get("status") == "Success" and "t0_gini" in result:
            for month, gini_key, insider_key in [
                (0, "t0_gini", "insider_share_t0"),
                (12, "gini_12m", None),
                (24, "gini_24m", None),
            ]:
                gini_val = result.get(gini_key)
                if gini_val is not None:
                    insider_val = result.get(insider_key, 0) if insider_key else 0
                    proposal[month] = (insider_val, gini_val)

        if proposal:
            per_proposal.append(proposal)

    if not per_proposal:
        print(f"  Warning: No valid fairness data found in {simulation_results}")
        return

    insider_by_checkpoint = {c: [] for c in CANONICAL_CHECKPOINTS}
    gini_by_checkpoint = {c: [] for c in CANONICAL_CHECKPOINTS}

    for proposal in per_proposal:
        for cp in [0, 12, 24]:
            if cp in proposal:
                insider_by_checkpoint[cp].append(proposal[cp][0])
                gini_by_checkpoint[cp].append(proposal[cp][1])
        # full_unlock: use post-24 snapshot if available, else use month-24 (flat)
        if 36 in proposal:
            insider_by_checkpoint[36].append(proposal[36][0])
            gini_by_checkpoint[36].append(proposal[36][1])
        elif 24 in proposal:
            insider_by_checkpoint[36].append(proposal[24][0])
            gini_by_checkpoint[36].append(proposal[24][1])

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
    ax2.axhline(0.60, color="#D55E00", linestyle="--", linewidth=1.5, alpha=0.8,
                label="Control threshold (0.60)")
    ax2.legend(fontsize=FONT_SIZE_TICK)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    _save_figure(fig, output_path, "Fig05_fairness_drift")
    plt.close(fig)




def plot_alpha_sensitivity(sensitivity_json: str, output_dir: str) -> None:
    """
    Line chart of scenario-level and overall stress-test pass rates as a
    function of the SDR demand-baseline parameter alpha (Fig. 6).

    Input: experiment_results/sensitivity_alpha.json, produced by
    `python analyze_results.py --sensitivity`.
    """
    output_path = _ensure_output_dir(output_dir)

    with open(sensitivity_json, "r") as f:
        sweep = json.load(f)
    rows = sweep["sweep"]
    n = sweep["n_projects"]
    alphas = [r["alpha"] for r in rows]
    scenarios = ["Bull", "Neutral", "Bear", "Unlock Shock", "Liquidity Pressure"]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)

    markers = {"Bull": "o", "Neutral": "s", "Bear": "^",
               "Unlock Shock": "D", "Liquidity Pressure": "v"}
    for s in scenarios:
        rates = [r["scenario_pass_counts"][s] / n for r in rows]
        ax.plot(alphas, rates, marker=markers[s], linewidth=1.8, markersize=5,
                color=STRESS_COLORS[s], label=s)

    overall = [r["overall_pass_count"] / n for r in rows]
    ax.plot(alphas, overall, "k--", marker="*", linewidth=2.2, markersize=9,
            label="Overall (≥ 4/5 scenarios)")

    default_alpha = sweep.get("default_alpha", 0.15)
    ax.axvline(default_alpha, color="gray", linestyle=":", linewidth=1.5, alpha=0.9)
    ax.annotate(f"α = {default_alpha}", xy=(default_alpha, 0.05),
                xytext=(default_alpha + 0.012, 0.05), fontsize=9, color="#555555")

    ax.set_xlabel("Demand-baseline parameter α  (D(0) = max(C(0), αS))",
                  fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Pass Rate", fontsize=FONT_SIZE_LABEL)
    ax.set_ylim(-0.03, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.set_xlim(0, max(alphas) * 1.02)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    _save_figure(fig, output_path, "Fig06_alpha_sensitivity")
    plt.close(fig)


def generate_all_figures(data_dir: str, output_dir: str) -> None:
    """
    Load all result files and generate the four figures used in the paper:
      Fig. 3 allocation distribution, Fig. 4 circulating supply growth,
      Fig. 5 fairness (temporal concentration change), Fig. 6 alpha sensitivity.
    """
    data_path = Path(data_dir)
    output_path = _ensure_output_dir(output_dir)

    print("Generating publication-quality figures from data in " + str(data_dir))
    print("Output directory: " + str(output_dir) + "\n")

    # -- Fig. 3: Allocation Distribution --
    alloc_file = data_path / "allocation_results.csv"
    if not alloc_file.exists():
        batch_csv = data_path / "batch_results.csv"
        if batch_csv.exists():
            try:
                df = pd.read_csv(batch_csv)
                if all(c in df.columns for c in ["team_pct", "investor_pct", "insider_pct", "distributed_pct"]):
                    alloc_file = batch_csv
            except Exception:
                pass
    if alloc_file.exists():
        print("Figure 3: Allocation Distribution...")
        try:
            plot_allocation_distribution(str(alloc_file), str(output_path))
        except Exception as e:
            print("  Error: " + str(e))
    else:
        print("  Skipping Fig 3 (no allocation data found)")

    # -- Figs. 4-5: Supply Growth + Fairness --
    sim_file = data_path / "simulation_results.json"
    if not sim_file.exists():
        checkpoint_file = data_path / "checkpoint_results.json"
        if checkpoint_file.exists():
            sim_file = checkpoint_file
    if sim_file.exists():
        print("Figure 4: Circulating Supply Growth...")
        try:
            plot_circulating_supply_growth(str(sim_file), str(output_path))
        except Exception as e:
            print("  Error: " + str(e))
        print("Figure 5: Fairness (temporal concentration change)...")
        try:
            plot_fairness_drift(str(sim_file), str(output_path))
        except Exception as e:
            print("  Error: " + str(e))
    else:
        print("  Skipping simulation figures (no simulation data found)")

    # -- Fig. 6: Alpha Sensitivity --
    sensitivity_file = data_path / "sensitivity_alpha.json"
    if sensitivity_file.exists():
        print("Figure 6: Demand-Baseline Alpha Sensitivity...")
        try:
            plot_alpha_sensitivity(str(sensitivity_file), str(output_path))
        except Exception as e:
            print("  Error: " + str(e))
    else:
        print("  Skipping Fig 6 (run: python analyze_results.py --sensitivity)")

    print("All figures generated in " + str(output_dir))


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for the tokenomics paper (Figs. 3-6)"
    )
    parser.add_argument("--data-dir", "-d", type=str, default="./experiment_results",
                        help="Directory containing result CSV/JSON files")
    parser.add_argument("--output-dir", "-o", type=str, default="./figures",
                        help="Directory to save figures")
    parser.add_argument("--fig3", action="store_true", help="Fig. 3 (allocation distribution)")
    parser.add_argument("--fig4", action="store_true", help="Fig. 4 (circulating supply growth)")
    parser.add_argument("--fig5", action="store_true", help="Fig. 5 (fairness / temporal concentration)")
    parser.add_argument("--fig6", action="store_true", help="Fig. 6 (demand-baseline alpha sensitivity)")
    parser.add_argument("--allocation-csv", type=str, help="Allocation results CSV (Fig. 3)")
    parser.add_argument("--simulation-json", type=str, help="Simulation results JSON (Figs. 4-5)")
    parser.add_argument("--sensitivity-json", type=str,
                        default="./experiment_results/sensitivity_alpha.json",
                        help="Alpha sensitivity JSON (Fig. 6)")
    args = parser.parse_args()

    if any([args.fig3, args.fig4, args.fig5, args.fig6]):
        if args.fig3 and args.allocation_csv:
            plot_allocation_distribution(args.allocation_csv, args.output_dir)
        if args.fig4 and args.simulation_json:
            plot_circulating_supply_growth(args.simulation_json, args.output_dir)
        if args.fig5 and args.simulation_json:
            plot_fairness_drift(args.simulation_json, args.output_dir)
        if args.fig6:
            plot_alpha_sensitivity(args.sensitivity_json, args.output_dir)
    else:
        generate_all_figures(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()

