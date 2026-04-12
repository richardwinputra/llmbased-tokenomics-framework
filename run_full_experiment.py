"""
Full Experiment Runner — 100 structured inputs through the complete pipeline.

Features:
  - All inputs follow the structured Token Design Thinking format
  - Saves intermediate results after each project (checkpoint resume)
  - Captures detailed per-scenario stress test data
  - Exports comprehensive CSV + JSON for Section IV analysis
"""

import json
import os
import re
import time
import random
import traceback
from datetime import datetime
from dataclasses import asdict

import numpy as np
from dotenv import load_dotenv

load_dotenv(override=True)

from llm_engine import (
    create_structured_prompt, ask_openai_enhanced,
    generate_tokenomics_proposal,
)
from control_filter import run_control_layer, run_filter_layer
from simulation import run_simulation_module
from utils import summarize_all_projects, calculate_gini


BATCH_FILE = "batch_inputs.json"
KB_FILE = "TokenomicsKnowledge.json"
OUTPUT_DIR = "experiment_results"
RAW_LLM_DIR = os.path.join(OUTPUT_DIR, "raw_llm_responses")
SEED = 42
MODEL = "gpt-5.4-mini-2026-03-17"


def load_data():
    with open(BATCH_FILE) as f:
        batch_inputs = json.load(f)
    with open(KB_FILE, encoding="utf-8") as f:
        knowledge_base = json.load(f)
    return batch_inputs, knowledge_base


def run_single(user_input, knowledge_base, dataset, project_summaries, seed):
    """Run a single evaluation through the full pipeline and return detailed metrics."""
    np.random.seed(seed)
    random.seed(seed)

    summaries = project_summaries


    prompt = create_structured_prompt(user_input, summaries)


    result_text = ask_openai_enhanced(
        prompt, "structured", model_override=MODEL,
    )

    if not result_text or result_text.startswith("Error"):
        return {"status": "LLM_Failed", "error": result_text[:200]}


    proposal, context = generate_tokenomics_proposal(user_input, result_text)


    control_result = run_control_layer(proposal, context)


    filter_result = run_filter_layer(proposal, context)


    sim_report = run_simulation_module(
        filter_result.adjusted_proposal, context, knowledge_base, dataset,
    )


    alloc = filter_result.adjusted_proposal.tokenomics_parameters.allocation
    overall_gini = calculate_gini(list(alloc.values()))

    fairness = sim_report.fairness_evaluation
    snapshots = fairness.snapshots
    t0 = snapshots[0] if snapshots else None
    m12 = next((s for s in snapshots if s.month == 12), None)
    m24 = next((s for s in snapshots if s.month == 24), None)
    full = next((s for s in snapshots if s.month > 24), None)


    stress = sim_report.stress_test
    scenario_details = {}
    for s in stress.scenarios:
        scenario_details[s.name] = {
            "viable": s.viable,
            "sufficient_reserves": s.sufficient_reserves,
            "recovery_months": s.recovery_months,
            "notes": s.notes,
        }


    supply_release_data = {
        "circulating_supply": sim_report.supply_release.circulating_supply,
        "initial_circulating_pct": sim_report.supply_release.initial_circulating_pct,
        "year1_circulating_pct": sim_report.supply_release.year1_circulating_pct,
        "year2_circulating_pct": sim_report.supply_release.year2_circulating_pct,
        "full_unlock_month": sim_report.supply_release.full_unlock_month,
        "year1_inflation_proxy": sim_report.supply_release.year1_inflation_proxy,
    }
    fairness_data = {
        "fairness_drift": fairness.fairness_drift,
        "snapshots": [
            {"month": s.month, "insider_share": s.insider_share,
             "distributed_share": s.distributed_share, "gini": s.gini}
            for s in snapshots
        ],
    }
    stress_data = {
        "pass_rate": stress.pass_rate,
        "passed": stress.passed,
        "scenarios": [
            {"name": s.name, "viable": s.viable,
             "sufficient_reserves": s.sufficient_reserves,
             "recovery_months": s.recovery_months, "notes": s.notes}
            for s in stress.scenarios
        ],
    }

    return {
        "status": "Success",
        "passed_filter": filter_result.passed,
        "control_aligned": control_result.aligned,
        "control_requires_iteration": control_result.requires_iteration,
        "n_control_findings": len(control_result.findings),
        "control_findings": [f.check for f in control_result.findings],
        "n_filter_issues": sum(1 for c in filter_result.checks if not c.passed),
        "filter_issues": [c.rule for c in filter_result.checks if not c.passed],
        "overall_gini": overall_gini,
        "t0_gini": t0.gini if t0 else None,
        "t0_insider_share": t0.insider_share if t0 else None,
        "t0_distributed_share": t0.distributed_share if t0 else None,
        "gini_12m": m12.gini if m12 else None,
        "gini_24m": m24.gini if m24 else None,
        "gini_full": full.gini if full else None,
        "insider_share_t0": t0.insider_share if t0 else None,
        "fairness_drift": fairness.fairness_drift,
        "stress_pass_rate": stress.pass_rate,
        "stress_passed": stress.passed,
        "year1_inflation_proxy": sim_report.supply_release.year1_inflation_proxy,
        "scenario_details": scenario_details,
        "n_allocations": len(alloc),
        "allocation": dict(alloc),
        "recommendations": sim_report.recommendations,

        "supply_release": supply_release_data,
        "fairness_evaluation": fairness_data,
        "stress_test": stress_data,

        "insider_pct": control_result.insider_pct,
        "distributed_pct": control_result.distributed_pct,
        "team_pct": control_result.team_pct,
        "investor_pct": control_result.investor_pct,

        "raw_llm_response": result_text,
        "generated_tokenomics_payload": asdict(proposal),
    }


def _save_raw_llm_response(project_name: str, raw_text: str) -> None:
    """Save raw LLM response to an individual file immediately after API call.

    This is a safety net separate from the bulk checkpoint: if checkpoint_results.json
    is ever corrupted, individual raw files let you re-parse without re-calling the API.
    """
    os.makedirs(RAW_LLM_DIR, exist_ok=True)
    safe_name = re.sub(r'[^\w\-]', '_', project_name)
    filepath = os.path.join(RAW_LLM_DIR, f"{safe_name}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(raw_text)


def run_experiment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint_results = os.path.join(OUTPUT_DIR, "checkpoint_results.json")

    print("=" * 70)
    print("FULL EXPERIMENT: 100 inputs through Full Pipeline")
    print("=" * 70)
    start_time = time.time()


    print("\n[1/3] Loading data...")
    batch_inputs, knowledge_base = load_data()
    dataset = knowledge_base
    print(f"  {len(batch_inputs)} inputs, {len(knowledge_base)} KB entries")


    resolved_inputs = batch_inputs


    print("\n[2/3] Computing RAG summaries...")
    project_summaries = summarize_all_projects(knowledge_base)
    print(f"  Summary length: {len(project_summaries)} chars")


    all_results = []
    completed_keys = set()
    if os.path.exists(checkpoint_results):
        with open(checkpoint_results) as f:
            all_results = json.load(f)
        completed_keys = {r["project_name"] for r in all_results}
        print(f"  Resuming: {len(all_results)} results already done")


    print(f"\n[3/3] Running Full Pipeline experiment...")
    total_runs = len(resolved_inputs)
    run_count = len(all_results)

    for i, user_input in enumerate(resolved_inputs):
        project_name = user_input.get("project_name", f"Project_{i}")
        input_type = "structured"

        if project_name in completed_keys:
            continue

        run_count += 1
        print(f"\n  [{run_count}/{total_runs}] {project_name} ({input_type})")

        try:
            metrics = run_single(
                user_input=user_input,
                knowledge_base=knowledge_base,
                dataset=dataset,
                project_summaries=project_summaries,
                seed=SEED + i,
            )
        except Exception as e:
            traceback.print_exc()
            metrics = {"status": "Error", "error": str(e)[:300]}

        metrics["project_name"] = project_name
        metrics["input_type"] = input_type
        metrics["category"] = user_input.get("category",
                                   batch_inputs[i].get("category", "Unknown"))

        # Save raw LLM response to an individual file immediately — independent of
        # checkpoint so a corrupted checkpoint never loses raw API outputs.
        raw_text = metrics.get("raw_llm_response", "")
        if raw_text:
            _save_raw_llm_response(project_name, raw_text)

        all_results.append(metrics)
        completed_keys.add(project_name)


        if metrics.get("status") == "Success":
            g24 = metrics.get("gini_24m")
            spr = metrics.get("stress_pass_rate")
            flt = "PASS" if metrics.get("passed_filter") else "FAIL"
            if g24 is not None and spr is not None:
                print(f"    Filter={flt} | Gini24m={g24:.3f} | StressPass={spr:.0%}")
            else:
                print(f"    {metrics['status']} (missing metrics)")
        else:
            print(f"    {metrics['status']}: {metrics.get('error', '')[:80]}")

        with open(checkpoint_results, "w") as f:
            json.dump(all_results, f, indent=2, default=str)


        time.sleep(0.5)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT COMPLETE: {len(all_results)} results in {elapsed/60:.1f} minutes")
    print(f"{'=' * 70}")


    export_results(all_results)
    return all_results


def export_results(all_results):
    """Export results to CSV, detailed JSON, and visualization-ready JSON."""
    import csv


    csv_path = os.path.join(OUTPUT_DIR, "batch_results.csv")
    flat_fields = [
        "project_name", "input_type", "category",
        "status",
        "passed_filter", "control_aligned", "control_requires_iteration",
        "n_control_findings", "n_filter_issues",
        "overall_gini", "t0_gini", "gini_12m", "gini_24m", "gini_full",
        "insider_share_t0", "t0_distributed_share", "fairness_drift",
        "stress_pass_rate", "stress_passed", "year1_inflation_proxy",
        "n_allocations",
        "insider_pct", "distributed_pct", "team_pct", "investor_pct",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"  CSV exported: {csv_path}")


    json_path = os.path.join(OUTPUT_DIR, "batch_results_full.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  JSON exported: {json_path}")


    sim_results = []
    for r in all_results:
        if r.get("status") != "Success":
            continue
        sim_results.append({
            "project_name": r.get("project_name", "Unknown"),
            "supply_release": r.get("supply_release", {}),
            "fairness_evaluation": r.get("fairness_evaluation", {}),
            "stress_test": r.get("stress_test", {}),
        })
    sim_path = os.path.join(OUTPUT_DIR, "simulation_results.json")
    with open(sim_path, "w") as f:
        json.dump(sim_results, f, indent=2, default=str)
    print(f"  Simulation results exported: {sim_path}")


    alloc_rows = []
    for r in all_results:
        if r.get("status") != "Success":
            continue
        alloc_rows.append({
            "project_name": r.get("project_name", "Unknown"),
            "team_pct": r.get("team_pct", 0),
            "investor_pct": r.get("investor_pct", 0),
            "insider_pct": r.get("insider_pct", 0),
            "distributed_pct": r.get("distributed_pct", 0),
        })
    alloc_path = os.path.join(OUTPUT_DIR, "allocation_results.csv")
    if alloc_rows:
        with open(alloc_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=alloc_rows[0].keys())
            writer.writeheader()
            writer.writerows(alloc_rows)
        print(f"  Allocation results exported: {alloc_path}")


    summary = compute_summary(all_results)
    summary_path = os.path.join(OUTPUT_DIR, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary exported: {summary_path}")


    print_summary_table(summary)


def compute_summary(all_results):
    """Compute aggregate statistics per condition and per category."""
    summary = {"conditions": {}, "categories": {}, "input_types": {},
               "timestamp": datetime.now().isoformat(),
               "total_results": len(all_results)}

    success = [r for r in all_results if r.get("status") == "Success"]

    def stats_for_group(results):
        if not results:
            return {"n": 0}
        n = len(results)

        def safe_mean(key):
            vals = [r[key] for r in results if r.get(key) is not None]
            return round(float(np.mean(vals)), 4) if vals else None

        def safe_std(key):
            vals = [r[key] for r in results if r.get(key) is not None]
            return round(float(np.std(vals)), 4) if len(vals) > 1 else None

        filter_pass = sum(1 for r in results if r.get("passed_filter")) / n
        stress_pass = sum(1 for r in results if r.get("stress_passed")) / n
        control_aligned = sum(1 for r in results if r.get("control_aligned")) / n


        scenario_pass = {}
        for sname in ["Bull", "Neutral", "Bear", "Unlock Shock", "Liquidity Pressure"]:
            viable_count = sum(
                1 for r in results
                if r.get("scenario_details", {}).get(sname, {}).get("viable", False)
            )
            scenario_pass[sname] = round(viable_count / n, 4)

        return {
            "n": n,
            "filter_pass_rate": round(filter_pass, 4),
            "stress_overall_pass_rate": round(stress_pass, 4),
            "control_aligned_rate": round(control_aligned, 4),
            "overall_gini_mean": safe_mean("overall_gini"),
            "overall_gini_std": safe_std("overall_gini"),
            "t0_gini_mean": safe_mean("t0_gini"),
            "gini_12m_mean": safe_mean("gini_12m"),
            "gini_24m_mean": safe_mean("gini_24m"),
            "gini_24m_std": safe_std("gini_24m"),
            "gini_full_mean": safe_mean("gini_full"),
            "insider_share_t0_mean": safe_mean("insider_share_t0"),
            "insider_share_t0_std": safe_std("insider_share_t0"),
            "fairness_drift_mean": safe_mean("fairness_drift"),
            "fairness_drift_std": safe_std("fairness_drift"),
            "stress_pass_rate_mean": safe_mean("stress_pass_rate"),
            "year1_inflation_mean": safe_mean("year1_inflation_proxy"),
            "year1_inflation_std": safe_std("year1_inflation_proxy"),
            "scenario_pass_rates": scenario_pass,
        }


    cats = set(r.get("category", "Unknown") for r in success)
    for cat in sorted(cats):
        group = [r for r in success if r.get("category") == cat]
        summary["categories"][cat] = stats_for_group(group)


    for itype in ["structured"]:
        group = [r for r in success if r.get("input_type") == itype]
        summary["input_types"][itype] = stats_for_group(group)


    control_freq = {}
    filter_freq = {}
    for r in success:
        for finding in r.get("control_findings", []):
            control_freq[finding] = control_freq.get(finding, 0) + 1
        for issue in r.get("filter_issues", []):
            filter_freq[issue] = filter_freq.get(issue, 0) + 1
    summary["control_issue_frequency"] = dict(sorted(control_freq.items(), key=lambda x: -x[1]))
    summary["filter_issue_frequency"] = dict(sorted(filter_freq.items(), key=lambda x: -x[1]))

    return summary


def print_summary_table(summary):
    """Print formatted summary table."""
    print(f"{'='*80}")
    print("FULL PIPELINE EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"{'Category':<22} {'N':>3} {'Filter%':>8} {'Gini24':>7} {'Drift':>7} {'Stress%':>8} {'Y1Infl':>7}")
    print(f"{'-'*80}")

    for cat, stats in summary["categories"].items():
        if stats["n"] == 0:
            print(f"{cat:<22} {'--':>3}")
            continue
        print(
            f"{cat:<22} {stats['n']:>3} "
            f"{stats['filter_pass_rate']*100:>7.1f}% "
            f"{stats['gini_24m_mean'] or 0:>7.3f} "
            f"{stats['fairness_drift_mean'] or 0:>7.3f} "
            f"{stats['stress_overall_pass_rate']*100:>7.1f}% "
            f"{stats['year1_inflation_mean'] or 0:>7.0f}"
        )

    print(f"{'='*80}")
    print("PER-SCENARIO STRESS PASS RATES")
    print(f"{'='*80}")
    for cat_stats in summary["categories"].values():
        for scenario, rate in cat_stats.get("scenario_pass_rates", {}).items():
            print(f"  {scenario:<22} {rate*100:>6.1f}%")
        break

    print(f"{'='*80}")
    print("BY INPUT TYPE")
    print(f"{'='*80}")
    for itype, stats in summary["input_types"].items():
        print(
            f"  {itype:<25} N={stats['n']:>2} "
            f"Filter={stats['filter_pass_rate']*100:.0f}% "
            f"Gini24={stats['gini_24m_mean'] or 0:.3f} "
            f"Stress={stats['stress_overall_pass_rate']*100:.0f}%"
        )

    print()


if __name__ == "__main__":
    run_experiment()

