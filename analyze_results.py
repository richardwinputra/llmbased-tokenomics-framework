"""
Post-experiment analysis.

Default prints aggregate tables. Flags: --replay (offline re-run from saved
responses), --sensitivity / --control-grid / --stress-grid (parameter sweeps),
--kb-thresholds / --kb-screen (knowledge-base analyses), --dir (analyze one run).
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

RESULTS_DIR = "experiment_results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "checkpoint_results.json")
REPLAY_SOURCE = os.path.join(RESULTS_DIR, "batch_results_full.json")
REPLAY_FILE = os.path.join(RESULTS_DIR, "replay_results.json")
SENSITIVITY_FILE = os.path.join(RESULTS_DIR, "sensitivity_alpha.json")
BATCH_INPUTS = "batch_inputs.json"


def set_results_dir(d):
    """Point all analyses at another run directory (multi-LLM conditions live
    under experiment_results/runs/<model>_<variant>[_rN]/)."""
    global RESULTS_DIR, RESULTS_FILE, REPLAY_SOURCE, REPLAY_FILE, SENSITIVITY_FILE
    RESULTS_DIR = d
    RESULTS_FILE = os.path.join(d, "checkpoint_results.json")
    REPLAY_SOURCE = os.path.join(d, "batch_results_full.json")
    REPLAY_FILE = os.path.join(d, "replay_results.json")
    SENSITIVITY_FILE = os.path.join(d, "sensitivity_alpha.json")

SCENARIOS = ["Bull", "Neutral", "Bear", "Unlock Shock", "Liquidity Pressure"]


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

def safe_median(vals):
    vals = [v for v in vals if v is not None]
    return round(float(np.median(vals)), 4) if vals else None

def safe_min(vals):
    vals = [v for v in vals if v is not None]
    return round(float(min(vals)), 4) if vals else None

def safe_max(vals):
    vals = [v for v in vals if v is not None]
    return round(float(max(vals)), 4) if vals else None

def safe_ci95(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None
    se = float(np.std(vals, ddof=1)) / (len(vals) ** 0.5)
    margin = 1.96 * se
    mean = float(np.mean(vals))
    return {"lower": round(mean - margin, 4), "upper": round(mean + margin, 4)}

def stats_dict(vals):
    return {
        "mean": safe_mean(vals),
        "std": safe_std(vals),
        "median": safe_median(vals),
        "min": safe_min(vals),
        "max": safe_max(vals),
        "ci95": safe_ci95(vals),
    }


def analyze():
    results = load_results()
    success = [r for r in results if r.get("status") == "Success"]
    failed = [r for r in results if r.get("status") != "Success"]

    print("=" * 80)
    print("EXPERIMENT ANALYSIS — SECTION IV DATA")
    print("=" * 80)


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
        print("\n  Failed projects:")
        for r in failed:
            print(f"    - {r.get('project_name', '?')}: {r.get('error', r.get('status', '?'))[:80]}")


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


    print("\n" + "=" * 80)
    print("TABLE VIII. Supply Release Metrics")
    print("=" * 80)
    y1_inf = [r["year1_inflation_proxy"] for r in success if r.get("year1_inflation_proxy") is not None]
    if y1_inf:
        print(f"  Year-1 Inflation Proxy: mean={np.mean(y1_inf):.1f}%, "
              f"std={np.std(y1_inf):.1f}%, median={np.median(y1_inf):.1f}%, "
              f"min={min(y1_inf):.1f}%, max={max(y1_inf):.1f}%")


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


    # ── Allocation statistics ──
    team_vals = [r.get("team_pct") for r in success if r.get("team_pct") is not None]
    investor_vals = [r.get("investor_pct") for r in success if r.get("investor_pct") is not None]
    insider_vals = [r.get("insider_pct") for r in success if r.get("insider_pct") is not None]
    distributed_vals = [r.get("distributed_pct") for r in success if r.get("distributed_pct") is not None]

    # ── Supply checkpoint statistics (circulating % at key months) ──
    supply_at_month: dict = {0: [], 12: [], 24: [], 60: []}
    for r in success:
        sr = r.get("supply_release", {})
        cs = sr.get("circulating_supply", [])
        if cs:
            total_s = max(cs) if max(cs) > 0 else 1
            for m in [0, 12, 24, 60]:
                if m < len(cs):
                    supply_at_month[m].append(round(100.0 * cs[m] / total_s, 4))

    # ── Category × scenario breakdown ──
    category_stress: dict = {}
    for cat in cats:
        group = [r for r in success if r.get("category") == cat]
        n_cat = len(group)
        category_stress[cat] = {"n": n_cat}
        for sname in scenarios:
            viable = sum(1 for r in group
                         if r.get("scenario_details", {}).get(sname, {}).get("viable", False))
            category_stress[cat][sname] = {
                "pass_count": viable,
                "pass_rate": round(viable / n_cat, 4) if n_cat else 0,
            }

    summary = {
        "metadata": {
            "stress_pass_threshold": 0.7,
            "gini_control_threshold": 0.60,
            "insider_control_threshold": 0.40,
            "team_control_threshold": 0.35,
            "investor_control_threshold": 0.25,
        },
        "total_inputs": total,
        "successful": n_success,
        "failed": n_failed,
        "parse_rate": parse_rate,
        "first_pass_filter_rate": first_pass_rate,
        "control_issues": dict(control_freq.most_common()),
        "filter_issues": dict(filter_freq.most_common()),
        "allocation_stats": {
            "team_pct": stats_dict(team_vals),
            "investor_pct": stats_dict(investor_vals),
            "insider_pct": stats_dict(insider_vals),
            "distributed_pct": stats_dict(distributed_vals),
        },
        "fairness": {
            "gini_t0": stats_dict(gini_t0),
            "gini_12m": stats_dict(gini_12),
            "gini_24m": stats_dict(gini_24),
            "gini_full": stats_dict(gini_full),
            "insider_share_t0": stats_dict(insider_t0),
            "fairness_drift": stats_dict(drift),
        },
        "supply_release": {
            "year1_inflation": stats_dict(y1_inf),
            "circulating_pct_at_month": {
                str(m): stats_dict(supply_at_month[m]) for m in [0, 12, 24, 60]
            },
        },
        "stress_test": {},
        "category_stress_breakdown": category_stress,
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


# Offline replay from saved raw responses (no API calls): reports parser
# correction-step frequencies and re-evaluates stress tests.

def _replay_pipeline():
    """Yield (saved_record, parse_trace, proposal, filtered_proposal, supply_release)."""
    from llm_engine import generate_tokenomics_proposal
    from control_filter import run_filter_layer
    from simulation import simulate_supply_release
    from utils import get_last_parse_trace

    if not os.path.exists(REPLAY_SOURCE):
        print(f"ERROR: {REPLAY_SOURCE} not found. Run run_full_experiment.py first.")
        sys.exit(1)
    with open(REPLAY_SOURCE) as f:
        saved = json.load(f)
    with open(BATCH_INPUTS) as f:
        inputs = {x["project_name"]: x for x in json.load(f)}

    for rec in saved:
        if rec.get("status") != "Success":
            continue
        proposal, _ = generate_tokenomics_proposal(
            inputs[rec["project_name"]], rec["raw_llm_response"])
        trace = get_last_parse_trace()
        filtered = run_filter_layer(proposal).adjusted_proposal
        yield rec, trace, proposal, filtered, simulate_supply_release(filtered, 60)


def replay():
    """Re-evaluate all saved proposals with the current pipeline code."""
    from simulation import run_stress_testing

    per_project = []
    for rec, trace, _, proposal, supply_release in _replay_pipeline():
        stress = run_stress_testing(proposal, supply_release)
        per_project.append({
            "project_name": rec["project_name"],
            "category": rec.get("category"),
            "input_type": rec.get("input_type"),
            "parse_trace": trace,
            "old_stress_passed": rec["stress_test"]["passed"],
            "stress_passed": stress.passed,
            # same shape run_full_experiment.py exports
            "stress_test": {
                "pass_rate": stress.pass_rate,
                "passed": stress.passed,
                "scenarios": [
                    {"name": s.name, "viable": s.viable,
                     "sufficient_reserves": s.sufficient_reserves,
                     "recovery_months": s.recovery_months, "notes": s.notes}
                    for s in stress.scenarios
                ],
            },
            "old_scenarios": {s["name"]: s["viable"]
                              for s in rec["stress_test"]["scenarios"]},
        })

    print("=" * 80)
    print("OFFLINE REPLAY — PARSER CORRECTION-STEP TRIGGERS (raw vs repaired)")
    print("=" * 80)
    repair = {
        "allocation_source": Counter(p["parse_trace"]["allocation_source"] for p in per_project),
        "supply_source": Counter(p["parse_trace"]["supply_source"] for p in per_project),
        "vesting_source": Counter(p["parse_trace"]["vesting_source"] for p in per_project),
        "normalization_triggered": sum(
            1 for p in per_project if p["parse_trace"]["normalization_triggered"]),
    }
    for k, v in repair.items():
        print(f"  {k:<26} {dict(v) if isinstance(v, Counter) else v}")

    print("\n" + "=" * 80)
    print("OFFLINE REPLAY — STRESS SCENARIO PASS COUNTS (saved -> current code)")
    print("=" * 80)
    for sname in SCENARIOS:
        old = sum(1 for p in per_project if p["old_scenarios"].get(sname))
        new = sum(1 for p in per_project
                  if any(s["viable"] and s["name"] == sname
                         for s in p["stress_test"]["scenarios"]))
        print(f"  {sname:<22} {old:>4} -> {new:<4}")
    old_all = sum(1 for p in per_project if p["old_stress_passed"])
    new_all = sum(1 for p in per_project if p["stress_passed"])
    print(f"  {'Overall (>=4/5)':<22} {old_all:>4} -> {new_all:<4}")

    print("\n  Overall pass by category (saved -> current):")
    by_cat = defaultdict(lambda: [0, 0, 0])
    for p in per_project:
        c = p["category"] or "Unknown"
        by_cat[c][0] += 1
        by_cat[c][1] += int(p["old_stress_passed"])
        by_cat[c][2] += int(p["stress_passed"])
    for c in sorted(by_cat):
        cn, co, cnew = by_cat[c]
        print(f"    {c:<22} n={cn:<3} {co:>3} -> {cnew:<3}")

    with open(REPLAY_FILE, "w") as f:
        json.dump(per_project, f, indent=1)
    print(f"\nReplay results saved to: {REPLAY_FILE}")
    print("Regenerate figures with:  python visualize.py")


def sensitivity(alphas=None):
    """Sweep the SDR demand-baseline parameter alpha (D(0) = max(C(0), alpha*S))."""
    import simulation
    from simulation import run_stress_testing

    alphas = alphas or [0.005, 0.01, 0.02, 0.05, 0.10, 0.125,
                        0.15, 0.175, 0.20, 0.25, 0.30, 0.50]
    cached = [(rec.get("category"), proposal, supply_release)
              for rec, _, _, proposal, supply_release in _replay_pipeline()]
    n = len(cached)
    default_alpha = simulation.DEMAND_BASELINE_ALPHA

    print("=" * 80)
    print("SENSITIVITY — DEMAND-BASELINE ALPHA SWEEP")
    print("=" * 80)
    print(f"  {'alpha':>6} | " + " ".join(f"{s[:5]:>5}" for s in SCENARIOS) + " | overall")
    rows = []
    for alpha in alphas:
        simulation.DEMAND_BASELINE_ALPHA = alpha
        scen = {s: 0 for s in SCENARIOS}
        by_cat = defaultdict(lambda: {"n": 0, "passed": 0})
        overall = 0
        for cat, proposal, supply_release in cached:
            st = run_stress_testing(proposal, supply_release)
            for s in st.scenarios:
                scen[s.name] += int(s.viable)
            overall += int(st.passed)
            by_cat[cat]["n"] += 1
            by_cat[cat]["passed"] += int(st.passed)
        rows.append({"alpha": alpha, "scenario_pass_counts": scen,
                     "overall_pass_count": overall, "by_category": dict(by_cat)})
        print(f"  {alpha:>6} | " + " ".join(f"{scen[s]:>5}" for s in SCENARIOS)
              + f" | {overall}")
    simulation.DEMAND_BASELINE_ALPHA = default_alpha

    with open(SENSITIVITY_FILE, "w") as f:
        json.dump({"n_projects": n, "default_alpha": default_alpha, "sweep": rows}, f, indent=1)
    print(f"\nSensitivity sweep saved to: {SENSITIVITY_FILE}")
    print("Generate the sensitivity figure with:  python visualize.py --fig6")


# Knowledge-base analyses: threshold percentiles and retrospective screening.

KB_FILE = "TokenomicsKnowledge.json"


def _parse_vesting_text(text):
    """Extract (cliff_months, vesting_months) from a free-text vesting
    description. Returns (0, None) when no duration is stated."""
    import re as _re
    t = str(text).lower()
    cliff = 0
    m = (_re.search(r'(\d+)\s*[- ]?(?:month|mo)\w*\s+cliff', t)
         or _re.search(r'cliff\s*(?:of|:)?\s*(\d+)\s*[- ]?(?:month|mo)', t))
    y = (_re.search(r'(\d+(?:\.\d+)?)\s*[- ]?(?:year|yr)\w*\s+cliff', t)
         or _re.search(r'cliff\s*(?:of|:)?\s*(\d+(?:\.\d+)?)\s*[- ]?(?:year|yr)', t))
    if m:
        cliff = int(m.group(1))
    elif y:
        cliff = int(float(y.group(1)) * 12)

    months = [int(x) for x in _re.findall(r'(\d+)\s*[- ]?(?:month|mo)\b\w*', t)]
    years = [float(x) for x in _re.findall(r'(\d+(?:\.\d+)?)\s*[- ]?(?:year|yr)', t)]
    candidates = months + [int(v * 12) for v in years]
    candidates = [c for c in candidates if c != cliff or candidates.count(c) > 1]
    duration = max(candidates) if candidates else None
    if duration is not None and cliff >= duration:
        cliff = 0
    return cliff, duration


def _kb_proposals():
    """Build GeneratedTokenomics objects for the real knowledge-base projects.
    KB allocations are fractions (sum to 1.0) and vesting entries are free
    text; both are converted to the pipeline's structured form."""
    from models import (GeneratedTokenomics, ProjectMetadata,
                        TokenomicsParameters, VestingDetail)
    from utils import parse_design_thinking

    with open(KB_FILE, encoding="utf-8") as f:
        kb = json.load(f)

    out = []
    for entry in kb:
        tok = entry.get("tokenomics") or {}
        alloc_raw = tok.get("allocation") or {}
        if not alloc_raw:
            continue
        allocation = {k: float(v) * 100.0 for k, v in alloc_raw.items()}

        vesting = {}
        n_vesting_texts = 0
        n_parsed = 0
        for k, txt in (tok.get("vesting") or {}).items():
            n_vesting_texts += 1
            cliff, duration = _parse_vesting_text(txt)
            if duration is not None:
                n_parsed += 1
                vesting[k] = VestingDetail(cliff_months=cliff,
                                           vesting_months=duration)

        burn_raw = tok.get("burn")
        burn = (burn_raw if burn_raw and
                str(burn_raw).strip().lower() not in {"none", "n/a", "no", ""}
                else None)

        proposal = GeneratedTokenomics(
            project_metadata=ProjectMetadata(
                project=entry.get("project", "Unknown"),
                token=entry.get("token", "TKN"),
                category=entry.get("category"),
            ),
            token_design_thinking=parse_design_thinking({}),
            tokenomics_parameters=TokenomicsParameters(
                total_supply=tok.get("total_supply") or 0,
                allocation=allocation,
                vesting=vesting,
                emissions=tok.get("emissions"),
                burn=burn,
            ),
            references=entry.get("sources", {}),
        )
        out.append({
            "project": entry.get("project", "Unknown"),
            "proposal": proposal,
            "n_vesting_texts": n_vesting_texts,
            "n_vesting_parsed": n_parsed,
        })
    return out


def kb_thresholds():
    """Derive the empirical percentile of each Table II cutoff in the KB."""
    from utils import (compute_team_pct, compute_investor_pct,
                       compute_insider_pct, compute_distributed_pct,
                       calculate_gini, is_insider_category)
    import control_filter as cf

    items = _kb_proposals()
    dists = {"team_pct": [], "investor_pct": [], "insider_pct": [],
             "distributed_pct": [], "gini_t0": [], "insider_vesting_months": []}
    for it in items:
        alloc = it["proposal"].tokenomics_parameters.allocation
        dists["team_pct"].append(compute_team_pct(alloc))
        dists["investor_pct"].append(compute_investor_pct(alloc))
        dists["insider_pct"].append(compute_insider_pct(alloc))
        dists["distributed_pct"].append(compute_distributed_pct(alloc))
        dists["gini_t0"].append(calculate_gini(list(alloc.values())))
        for k, v in it["proposal"].tokenomics_parameters.vesting.items():
            if is_insider_category(k) and v.vesting_months:
                dists["insider_vesting_months"].append(v.vesting_months)

    def pct_of(values, cutoff):
        values = sorted(values)
        return round(100.0 * sum(1 for v in values if v <= cutoff) / len(values), 1)

    cutoffs = [
        ("Distributed floor",  "distributed_pct", cf.DISTRIBUTED_FLOOR),
        ("Team max",           "team_pct",        cf.TEAM_MAX),
        ("Investor preferred", "investor_pct",    cf.INVESTOR_PREFERRED),
        ("Investor high-risk", "investor_pct",    cf.INVESTOR_HIGH),
        ("Insider preferred",  "insider_pct",     cf.INSIDER_PREFERRED),
        ("Insider high-risk",  "insider_pct",     cf.INSIDER_HIGH),
        ("Insider min vesting", "insider_vesting_months", cf.INSIDER_MIN_VESTING),
        ("Gini max",           "gini_t0",         cf.GINI_MAX),
    ]

    print("=" * 80)
    print(f"TABLE II CUTOFFS vs KB DISTRIBUTIONS (n = {len(items)} projects)")
    print("=" * 80)
    print(f"  {'Cutoff':<20} {'Value':>7} | {'KB pctl':>7} | "
          f"{'KB mean':>8} {'KB p25':>7} {'KB p50':>7} {'KB p75':>7} {'n':>4}")
    rows = []
    for label, key, cutoff in cutoffs:
        vals = dists[key]
        row = {
            "cutoff": label, "value": cutoff,
            "kb_percentile": pct_of(vals, cutoff),
            "kb_mean": round(float(np.mean(vals)), 2),
            "kb_p25": round(float(np.percentile(vals, 25)), 2),
            "kb_p50": round(float(np.percentile(vals, 50)), 2),
            "kb_p75": round(float(np.percentile(vals, 75)), 2),
            "n": len(vals),
        }
        rows.append(row)
        print(f"  {label:<20} {cutoff:>7} | {row['kb_percentile']:>6.1f}% | "
              f"{row['kb_mean']:>8} {row['kb_p25']:>7} {row['kb_p50']:>7} "
              f"{row['kb_p75']:>7} {row['n']:>4}")

    out_path = os.path.join(RESULTS_DIR, "kb_threshold_percentiles.json")
    with open(out_path, "w") as f:
        json.dump({"n_projects": len(items), "cutoffs": rows,
                   "distributions": {k: sorted(round(x, 3) for x in v)
                                     for k, v in dists.items()}}, f, indent=1)
    print(f"\nSaved: {out_path}")


def control_grid():
    """One-at-a-time sweep of Table II thresholds over the generated proposals."""
    from control_filter import run_control_layer
    import control_filter as cf

    proposals = [(rec["project_name"], proposal)
                 for rec, _, proposal, _, _ in _replay_pipeline()]
    n = len(proposals)

    grid = {
        "DISTRIBUTED_FLOOR":  [10.0, 15.0, 20.0, 25.0, 30.0],
        "TEAM_MAX":           [25.0, 30.0, 35.0, 40.0, 45.0],
        "INVESTOR_PREFERRED": [15.0, 20.0, 25.0, 30.0, 35.0],
        "INSIDER_PREFERRED":  [30.0, 35.0, 40.0, 45.0, 50.0],
        "INSIDER_MIN_VESTING": [6, 9, 12, 18, 24],
        "GINI_MAX":           [0.50, 0.55, 0.60, 0.65, 0.70],
    }

    print("=" * 80)
    print(f"CONTROL-LAYER THRESHOLD SWEEP (one-at-a-time, n = {n} proposals)")
    print("=" * 80)
    results = {}
    for param, values in grid.items():
        default = getattr(cf, param)
        rows = []
        for v in values:
            setattr(cf, param, v)
            flagged = 0
            findings_by_check = Counter()
            for _, proposal in proposals:
                res = run_control_layer(proposal)
                if res.findings:
                    flagged += 1
                for fnd in res.findings:
                    findings_by_check[fnd.check] += 1
            rows.append({"value": v, "proposals_flagged": flagged,
                         "findings_by_check": dict(findings_by_check)})
        setattr(cf, param, default)
        results[param] = {"default": default, "sweep": rows}
        line = " ".join(f"{r['value']}:{r['proposals_flagged']}" for r in rows)
        print(f"  {param:<20} flagged@value -> {line}")

    out_path = os.path.join(RESULTS_DIR, "sensitivity_control_thresholds.json")
    with open(out_path, "w") as f:
        json.dump({"n_proposals": n, "grid": results}, f, indent=1)
    print(f"\nSaved: {out_path}")


def stress_grid():
    """One-at-a-time sweep of stress-test parameters over the saved curves."""
    import simulation
    from simulation import run_stress_testing

    cached = [(proposal, supply_release)
              for _, _, _, proposal, supply_release in _replay_pipeline()]
    n = len(cached)

    grid = {
        "SDR_THRESHOLD_BASE":      [1.0, 1.25, 1.5, 1.75, 2.0],
        "SDR_THRESHOLD_BEAR":      [1.5, 1.75, 2.0, 2.25, 2.5],
        "SDR_THRESHOLD_SHOCK":     [1.5, 1.75, 2.0, 2.25, 2.5],
        "SDR_THRESHOLD_LIQUIDITY": [3.5, 4.0, 4.5, 5.0, 5.5],
        "SPIKE_THRESHOLD":         [0.05, 0.10, 0.15, 0.20, 0.30],
        "GROWTH_BULL":             [0.025, 0.0375, 0.05, 0.0625, 0.075],
        "GROWTH_NEUTRAL":          [0.005, 0.0075, 0.01, 0.015, 0.02],
        "GROWTH_BEAR":             [-0.03, -0.025, -0.02, -0.015, -0.01],
        "GROWTH_LIQUIDITY":        [0.0025, 0.005, 0.0075, 0.01],
        "SELL_PRESSURE_MULTIPLIER": [2.0, 2.5, 3.0, 3.5, 4.0],
    }

    print("=" * 80)
    print(f"STRESS-PARAMETER SWEEP (one-at-a-time, n = {n} proposals)")
    print("=" * 80)
    results = {}
    for param, values in grid.items():
        default = getattr(simulation, param)
        rows = []
        for v in values:
            setattr(simulation, param, v)
            scen = {s: 0 for s in SCENARIOS}
            overall = 0
            for proposal, supply_release in cached:
                st = run_stress_testing(proposal, supply_release)
                for s in st.scenarios:
                    scen[s.name] += int(s.viable)
                overall += int(st.passed)
            rows.append({"value": v, "scenario_pass_counts": scen,
                         "overall_pass_count": overall})
        setattr(simulation, param, default)
        results[param] = {"default": default, "sweep": rows}
        line = " ".join(f"{r['value']}:{r['overall_pass_count']}" for r in rows)
        print(f"  {param:<24} overall@value -> {line}")

    out_path = os.path.join(RESULTS_DIR, "sensitivity_stress_params.json")
    with open(out_path, "w") as f:
        json.dump({"n_proposals": n, "grid": results}, f, indent=1)
    print(f"\nSaved: {out_path}")


def kb_screen():
    """Apply the control, filter, and simulation modules to the real KB projects."""
    from control_filter import run_control_layer, run_filter_layer
    from simulation import run_simulation_module

    items = _kb_proposals()
    per_project = []
    for it in items:
        control = run_control_layer(it["proposal"])
        filt = run_filter_layer(it["proposal"])
        sim = run_simulation_module(filt.adjusted_proposal)
        stress = sim.stress_test
        per_project.append({
            "project": it["project"],
            "n_vesting_texts": it["n_vesting_texts"],
            "n_vesting_parsed": it["n_vesting_parsed"],
            "control_findings": [f.check for f in control.findings],
            "control_requires_iteration": control.requires_iteration,
            "insider_pct": round(control.insider_pct, 2),
            "passed_filter": filt.passed,
            "initial_circulating_pct": round(
                sim.supply_release.initial_circulating_pct, 1),
            "stress_passed": stress.passed,
            "scenarios": {s.name: s.viable for s in stress.scenarios},
        })

    n = len(per_project)
    n_vest_all = sum(p["n_vesting_texts"] for p in per_project)
    n_vest_ok = sum(p["n_vesting_parsed"] for p in per_project)

    print("=" * 80)
    print(f"RETROSPECTIVE SCREENING — {n} REAL KB PROJECTS")
    print("=" * 80)
    print(f"  Vesting text entries parsed to months: {n_vest_ok}/{n_vest_all}")
    print(f"  Projects with >=1 control finding:     "
          f"{sum(1 for p in per_project if p['control_findings'])}/{n}")
    print(f"  Projects passing filter first-pass:    "
          f"{sum(1 for p in per_project if p['passed_filter'])}/{n}")
    ctrl = Counter(c for p in per_project for c in p["control_findings"])
    for check, cnt in ctrl.most_common():
        print(f"    {check:<38} {cnt}")
    print("\n  Stress scenario pass counts:")
    for s in SCENARIOS:
        print(f"    {s:<22} {sum(1 for p in per_project if p['scenarios'].get(s))}/{n}")
    n_pass = sum(1 for p in per_project if p["stress_passed"])
    print(f"    {'Overall (>=4/5)':<22} {n_pass}/{n}")
    flagged = sorted(p["project"] for p in per_project if not p["stress_passed"])
    print(f"\n  Projects flagged by the sustainability screen ({len(flagged)}):")
    for i in range(0, len(flagged), 6):
        print("    " + ", ".join(flagged[i:i+6]))

    out_path = os.path.join(RESULTS_DIR, "kb_screening.json")
    with open(out_path, "w") as f:
        json.dump({"n_projects": n,
                   "vesting_parse_coverage": f"{n_vest_ok}/{n_vest_all}",
                   "overall_pass_count": n_pass,
                   "per_project": per_project}, f, indent=1)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-experiment analysis")
    parser.add_argument("--replay", action="store_true",
                        help="Re-run pipeline offline from saved raw LLM responses")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Sweep the SDR demand-baseline alpha parameter")
    parser.add_argument("--kb-thresholds", action="store_true",
                        help="Derive Table II cutoff percentiles from the KB")
    parser.add_argument("--control-grid", action="store_true",
                        help="Sweep control-layer thresholds (one-at-a-time)")
    parser.add_argument("--stress-grid", action="store_true",
                        help="Sweep stress-test parameters (one-at-a-time)")
    parser.add_argument("--kb-screen", action="store_true",
                        help="Run the screening pipeline on the real KB projects")
    parser.add_argument("--dir", default=None,
                        help="Analyze another run directory, e.g. "
                             "experiment_results/runs/<model>_<variant>")
    args = parser.parse_args()

    if args.dir:
        set_results_dir(args.dir)

    ran_any = False
    for flag, fn in [(args.replay, replay), (args.sensitivity, sensitivity),
                     (args.kb_thresholds, kb_thresholds),
                     (args.control_grid, control_grid),
                     (args.stress_grid, stress_grid),
                     (args.kb_screen, kb_screen)]:
        if flag:
            fn()
            ran_any = True
    if not ran_any:
        analyze()

