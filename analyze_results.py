"""
Post-Experiment Analysis Script.

Reads the experiment results from experiment_results/ and generates:
  1. experiment_summary.json — aggregate statistics
  2. Formatted console output of all tables for Section IV
  3. Per-category, per-input-type, and per-scenario breakdowns

Run after run_full_experiment.py completes:
    python3 analyze_results.py
"""

import json
import os
import sys
from collections import Counter
from typing import Dict, List

import numpy as np

RESULTS_DIR = "experiment_results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "checkpoint_results.json")


def load_results():
    if not os.path.exists(RESULTS_FILE):
        print(f"ERROR: {RESULTS_FILE} not found. Run run_full_experiment.py first.")
        sys.exit(1)
    with open(RESULTS_FILE) as f:
        return json.load(f)


def safe_mean(vals):
    vals = [v for v in vals if v is not None]
    return round(float(np.mean(vals)), 4) if vals else None

def safe_std(vals):
    vals = [v for v in vals if v is not None]
    return round(float(np.std(vals)), 4) if len(vals) > 1 else None



def analyze():
    results = load_results()
    success = [r for r in results if r.get("status") == "Success"]
    failed = [r for r in results if r.get("status") != "Success"]

    print("=" * 80)
    print("EXPERIMENT ANALYSIS — SECTION IV DATA")
    print("=" * 80)

    # ── TABLE V: Batch Generation Performance ──────────────────
    print("\n" + "=" * 80)
    print("TABLE V. Batch Generation Performance")
    print("=" * 80)
    total = len(results)
    n_success = len(success)
    n_failed = len(failed)
    parse_rate = n_success / total if total > 0 else 0

    first_pass_valid = sum(1 for r in success if r.get("passed_filter"))
    first_pass_rate = first_pass_valid / n_success if n_success > 0 else 0

    print(f"  Total inputs processed:     {total}")
    print(f"  Successful generation:      {n_success} ({parse_rate:.0%})")
    print(f"  Failed generation:          {n_failed}")
    print(f"  First-pass filter validity: {first_pass_valid}/{n_success} ({first_pass_rate:.0%})")

    if failed:
        print(f"\n  Failed projects:")
        for r in failed:
            print(f"    - {r.get('project_name', '?')}: {r.get('error', r.get('status', '?'))[:80]}")

    # ── TABLE VI: Control & Filter Issues ──────────────────────
    print("\n" + "=" * 80)
    print("TABLE VI. Common Control and Filter Issues")
    print("=" * 80)

    control_freq = Counter()
    filter_freq = Counter()
    for r in success:
        for f in r.get("control_findings", []):
            control_freq[f] += 1
        for f in r.get("filter_issues", []):
            filter_freq[f] += 1

    print(f"\n  Control Layer Issues (out of {n_success} proposals):")
    print(f"  {'Check':<40} {'Count':>6} {'Rate':>8}")
    print(f"  {'-'*54}")
    for check, count in control_freq.most_common():
        print(f"  {check:<40} {count:>6} {count/n_success:>7.0%}")

    print(f"\n  Filter Layer Issues (out of {n_success} proposals):")
    print(f"  {'Check':<40} {'Count':>6} {'Rate':>8}")
    print(f"  {'-'*54}")
    for check, count in filter_freq.most_common():
        print(f"  {check:<40} {count:>6} {count/n_success:>7.0%}")

    # ── TABLE VII: Fairness Evaluation ─────────────────────────
    print("\n" + "=" * 80)
    print("TABLE VII. Fairness Evaluation Summary")
    print("=" * 80)

    gini_t0 = [r["t0_gini"] for r in success if r.get("t0_gini") is not None]
    gini_12 = [r["gini_12m"] for r in success if r.get("gini_12m") is not None]
    gini_24 = [r["gini_24m"] for r in success if r.get("gini_24m") is not None]
    gini_full = [r["gini_full"] for r in success if r.get("gini_full") is not None]
    insider_t0 = [r["insider_share_t0"] for r in success if r.get("insider_share_t0") is not None]
    drift = [r["fairness_drift"] for r in success if r.get("fairness_drift") is not None]

    print(f"\n  {'Metric':<30} {'Mean':>8} {'Std':>8} {'Median':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*70}")
    for label, vals in [
        ("Gini (t=0)", gini_t0),
        ("Gini (12m)", gini_12),
        ("Gini (24m)", gini_24),
        ("Gini (full unlock)", gini_full),
        ("Insider share (t=0)", insider_t0),
        ("Fairness drift", drift),
    ]:
        if vals:
            print(f"  {label:<30} {np.mean(vals):>8.4f} {np.std(vals):>8.4f} "
                  f"{np.median(vals):>8.4f} {min(vals):>8.4f} {max(vals):>8.4f}")

    # ── TABLE VIII: Supply Release ─────────────────────────────
    print("\n" + "=" * 80)
    print("TABLE VIII. Supply Release Metrics")
    print("=" * 80)
    y1_inf = [r["year1_inflation_proxy"] for r in success if r.get("year1_inflation_proxy") is not None]
    if y1_inf:
        print(f"  Year-1 Inflation Proxy: mean={np.mean(y1_inf):.1f}%, "
              f"std={np.std(y1_inf):.1f}%, median={np.median(y1_inf):.1f}%, "
              f"min={min(y1_inf):.1f}%, max={max(y1_inf):.1f}%")

    # ── TABLE IX: Stress Test Results ──────────────────────────
    print("\n" + "=" * 80)
    print("TABLE IX. Sustainability Stress Test Results")
    print("=" * 80)

    scenarios = ["Bull", "Neutral", "Bear", "Unlock Shock", "Liquidity Pressure"]
    print(f"\n  {'Scenario':<22} {'Pass':>6} {'Fail':>6} {'Rate':>8} {'Avg Recovery':>14}")
    print(f"  {'-'*60}")

    for sname in scenarios:
        viable = sum(1 for r in success
                     if r.get("scenario_details", {}).get(sname, {}).get("viable", False))
        not_viable = n_success - viable
        rate = viable / n_success if n_success > 0 else 0
        recovery = [r.get("scenario_details", {}).get(sname, {}).get("recovery_months")
                    for r in success]
        recovery = [v for v in recovery if v is not None]
        avg_rec = f"{np.mean(recovery):.1f}m" if recovery else "--"
        print(f"  {sname:<22} {viable:>6} {not_viable:>6} {rate:>7.0%} {avg_rec:>14}")

    overall_pass = sum(1 for r in success if r.get("stress_passed"))
    print(f"\n  Overall stress pass (>=70% scenarios): {overall_pass}/{n_success} "
          f"({overall_pass/n_success:.0%})" if n_success else "")

    # ── BY CATEGORY ────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RESULTS BY CATEGORY")
    print("=" * 80)
    cats = sorted(set(r.get("category", "Unknown") for r in success))
    print(f"\n  {'Category':<22} {'N':>3} {'Filter%':>8} {'Gini24':>8} {'Drift':>8} "
          f"{'Stress%':>8} {'Y1Infl':>8}")
    print(f"  {'-'*72}")
    for cat in cats:
        group = [r for r in success if r.get("category") == cat]
        n = len(group)
        fp = sum(1 for r in group if r.get("passed_filter")) / n
        g24 = safe_mean([r.get("gini_24m") for r in group])
        dr = safe_mean([r.get("fairness_drift") for r in group])
        sp = sum(1 for r in group if r.get("stress_passed")) / n
        y1 = safe_mean([r.get("year1_inflation_proxy") for r in group])
        print(f"  {cat:<22} {n:>3} {fp*100:>7.1f}% {g24 or 0:>8.4f} "
              f"{dr or 0:>8.4f} {sp*100:>7.1f}% {y1 or 0:>8.1f}")

    # ── BY INPUT TYPE ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RESULTS BY INPUT TYPE")
    print("=" * 80)
    itypes = sorted(set(r.get("input_type", "unknown") for r in success))
    print(f"\n  {'Input Type':<25} {'N':>3} {'Filter%':>8} {'Gini24':>8} {'Drift':>8} "
          f"{'Stress%':>8} {'Y1Infl':>8}")
    print(f"  {'-'*72}")
    for itype in itypes:
        group = [r for r in success if r.get("input_type") == itype]
        n = len(group)
        fp = sum(1 for r in group if r.get("passed_filter")) / n
        g24 = safe_mean([r.get("gini_24m") for r in group])
        dr = safe_mean([r.get("fairness_drift") for r in group])
        sp = sum(1 for r in group if r.get("stress_passed")) / n
        y1 = safe_mean([r.get("year1_inflation_proxy") for r in group])
        print(f"  {itype:<25} {n:>3} {fp*100:>7.1f}% {g24 or 0:>8.4f} "
              f"{dr or 0:>8.4f} {sp*100:>7.1f}% {y1 or 0:>8.1f}")

    # ── PER-SCENARIO BY CATEGORY ──────────────────────────────
    print("\n" + "=" * 80)
    print("STRESS SCENARIO PASS RATES BY CATEGORY")
    print("=" * 80)
    header = f"  {'Category':<22}"
    for s in scenarios:
        header += f" {s[:8]:>8}"
    print(header)
    print(f"  {'-'*62}")
    for cat in cats:
        group = [r for r in success if r.get("category") == cat]
        n = len(group)
        line = f"  {cat:<22}"
        for sname in scenarios:
            viable = sum(1 for r in group
                         if r.get("scenario_details", {}).get(sname, {}).get("viable", False))
            line += f" {viable/n*100:>7.0f}%"
        print(line)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    # Save structured summary
    summary = {
        "total_inputs": total,
        "successful": n_success,
        "failed": n_failed,
        "parse_rate": parse_rate,
        "first_pass_filter_rate": first_pass_rate,
        "control_issues": dict(control_freq.most_common()),
        "filter_issues": dict(filter_freq.most_common()),
        "fairness": {
            "gini_t0": {"mean": safe_mean(gini_t0), "std": safe_std(gini_t0)},
            "gini_12m": {"mean": safe_mean(gini_12), "std": safe_std(gini_12)},
            "gini_24m": {"mean": safe_mean(gini_24), "std": safe_std(gini_24)},
            "gini_full": {"mean": safe_mean(gini_full), "std": safe_std(gini_full)},
            "insider_share_t0": {"mean": safe_mean(insider_t0), "std": safe_std(insider_t0)},
            "fairness_drift": {"mean": safe_mean(drift), "std": safe_std(drift)},
        },
        "supply_release": {
            "year1_inflation": {"mean": safe_mean(y1_inf), "std": safe_std(y1_inf)},
        },
        "stress_test": {},
    }
    for sname in scenarios:
        viable = sum(1 for r in success
                     if r.get("scenario_details", {}).get(sname, {}).get("viable", False))
        summary["stress_test"][sname] = {
            "pass_count": viable,
            "total": n_success,
            "pass_rate": round(viable / n_success, 4) if n_success else 0,
        }

    out_path = os.path.join(RESULTS_DIR, "experiment_analysis.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nStructured analysis saved to: {out_path}")


if __name__ == "__main__":
    analyze()
