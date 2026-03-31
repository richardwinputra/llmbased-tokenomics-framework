"""
Batch evaluation harness for ablation experiments.

Refactored to use direct imports (no subprocess) for structured metrics capture.
Supports ablation conditions:
  - With/without RAG (knowledge base)
  - With/without Filter layer
  - Different LLMs (model override)
  - With/without ABM (simulation layer)

Outputs a proper comparison table for the paper.
"""

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from models import (
    GeneratedTokenomics, ProjectContext, FilterLayerResult,
    SimulationReport, FairnessMetrics,
)
from utils import (
    get_initial_supply, normalize_allocation_key,
    summarize_all_projects, calculate_gini,
)
from llm_engine import (
    create_structured_prompt, create_generic_prompt,
    ask_openai_enhanced, generate_tokenomics_proposal,
)
from control_filter import (
    run_control_layer, run_filter_layer,
    calculate_temporal_fairness,
)
from simulation import (
    run_simulations_layer,
    estimate_drawdown_from_prices, estimate_inflation_from_supply,
)
from scenario_generator import compute_price_stability_coefficient


# ── Single evaluation run (direct import, no subprocess) ──────

def run_single_evaluation(
    user_input: Dict,
    knowledge_base: List[Dict],
    dataset: List[Dict],
    disable_rag: bool = False,
    disable_filter: bool = False,
    disable_abm: bool = False,
    model_override: Optional[str] = None,
    seed: int = 42,
    mc_replications: int = 100,
) -> Dict:
    """
    Run a single project through the full pipeline and return structured metrics.

    Returns a dict with all metrics needed for ablation comparison.
    """
    np.random.seed(seed)
    random.seed(seed)

    try:
        # RAG
        if disable_rag:
            project_summaries = ""
        else:
            project_summaries = summarize_all_projects(knowledge_base)

        # LLM generation
        if user_input.get('input_type') == 'structured':
            prompt = create_structured_prompt(user_input, project_summaries)
        else:
            prompt = create_generic_prompt(user_input, project_summaries)

        result_text = ask_openai_enhanced(
            prompt,
            user_input.get('input_type', 'structured'),
            model_override=model_override,
        )

        # Proposal
        proposal, context = generate_tokenomics_proposal(user_input, result_text)

        # Control Layer
        control_result = run_control_layer(proposal, context)

        # Filter Layer
        if disable_filter:
            filter_result = FilterLayerResult(
                passed=True, adjusted_proposal=proposal,
                allocation_issues=[], vesting_issues=[],
                economic_issues=[], governance_issues=[],
                fairness_metrics=None, governance_risk=None,
            )
        else:
            filter_result = run_filter_layer(proposal, context)

        # Simulation Layer
        if disable_abm:
            # Run without ABM: only category-level metrics
            fairness = calculate_temporal_fairness(filter_result.adjusted_proposal)
            alloc = filter_result.adjusted_proposal.tokenomics_parameters.allocation
            overall_gini = calculate_gini(list(alloc.values()))
            return {
                "status": "Success",
                "passed_filter": filter_result.passed,
                "control_aligned": control_result.aligned,
                "control_requires_iteration": control_result.requires_iteration,
                "n_control_findings": (len(control_result.stakeholder_findings) +
                                       len(control_result.compliance_findings) +
                                       len(control_result.feasibility_findings)),
                "n_filter_issues": (len(filter_result.allocation_issues) +
                                    len(filter_result.vesting_issues) +
                                    len(filter_result.economic_issues) +
                                    len(filter_result.governance_issues)),
                "overall_gini": overall_gini,
                "t0_gini": fairness.t0_gini,
                "gini_12m": fairness.gini_12m,
                "gini_24m": fairness.gini_24m,
                "governance_influence": fairness.governance_influence_index,
                "agent_gini_final": None,
                "agent_theil_final": None,
                "agent_atkinson_final": None,
                "agent_gini_ci_low": None,
                "agent_gini_ci_high": None,
                "mean_max_drawdown": None,
                "mean_price_volatility": None,
                "price_stability": None,
                "abm_enabled": False,
            }

        sim_report = run_simulations_layer(
            filter_result.adjusted_proposal, context, knowledge_base, dataset,
            mc_replications=mc_replications,
        )

        # Price stability from ABM
        price_stability = None
        if sim_report.monte_carlo_result and sim_report.monte_carlo_result.mean_price_path:
            price_stability = compute_price_stability_coefficient(
                sim_report.monte_carlo_result.mean_price_path
            )

        alloc = filter_result.adjusted_proposal.tokenomics_parameters.allocation
        overall_gini = calculate_gini(list(alloc.values()))

        return {
            "status": "Success",
            "passed_filter": filter_result.passed,
            "control_aligned": control_result.aligned,
            "control_requires_iteration": control_result.requires_iteration,
            "n_control_findings": (len(control_result.stakeholder_findings) +
                                   len(control_result.compliance_findings) +
                                   len(control_result.feasibility_findings)),
            "n_filter_issues": (len(filter_result.allocation_issues) +
                                len(filter_result.vesting_issues) +
                                len(filter_result.economic_issues) +
                                len(filter_result.governance_issues)),
            "overall_gini": overall_gini,
            "t0_gini": sim_report.fairness_metrics.t0_gini,
            "gini_12m": sim_report.fairness_metrics.gini_12m,
            "gini_24m": sim_report.fairness_metrics.gini_24m,
            "governance_influence": sim_report.fairness_metrics.governance_influence_index,
            "agent_gini_final": sim_report.fairness_metrics.agent_gini_final,
            "agent_theil_final": sim_report.fairness_metrics.agent_theil_final,
            "agent_atkinson_final": sim_report.fairness_metrics.agent_atkinson_final,
            "agent_gini_ci_low": sim_report.fairness_metrics.agent_gini_ci_low,
            "agent_gini_ci_high": sim_report.fairness_metrics.agent_gini_ci_high,
            "mean_max_drawdown": (sim_report.monte_carlo_result.mean_max_drawdown
                                  if sim_report.monte_carlo_result else None),
            "mean_price_volatility": (sim_report.monte_carlo_result.mean_price_volatility
                                      if sim_report.monte_carlo_result else None),
            "price_stability": price_stability,
            "abm_enabled": True,
        }

    except Exception as e:
        return {"status": "Failed", "error": str(e)[:200]}


# ── Ablation experiment runner ────────────────────────────────

ABLATION_CONDITIONS = [
    {"label": "Full Pipeline",       "disable_rag": False, "disable_filter": False, "disable_abm": False, "model": None},
    {"label": "No RAG",              "disable_rag": True,  "disable_filter": False, "disable_abm": False, "model": None},
    {"label": "No Filter",           "disable_rag": False, "disable_filter": True,  "disable_abm": False, "model": None},
    {"label": "No ABM",              "disable_rag": False, "disable_filter": False, "disable_abm": True,  "model": None},
    {"label": "No RAG + No Filter",  "disable_rag": True,  "disable_filter": True,  "disable_abm": False, "model": None},
]


def run_ablation_experiment(
    batch_inputs: List[Dict],
    knowledge_base: List[Dict],
    dataset: List[Dict],
    conditions: Optional[List[Dict]] = None,
    model_override: Optional[str] = None,
    seed: int = 42,
    mc_replications: int = 30,
    verbose: bool = True,
) -> Tuple[List[Dict], str]:
    """
    Run ablation experiments: each project × each condition.

    Returns:
      - List of result dicts (one per project × condition)
      - Formatted comparison table string for the paper
    """
    if conditions is None:
        conditions = ABLATION_CONDITIONS

    # If model override provided, add it as an extra condition
    if model_override:
        conditions = conditions + [
            {"label": f"Alt Model ({model_override})",
             "disable_rag": False, "disable_filter": False, "disable_abm": False,
             "model": model_override},
        ]

    all_results = []

    for i, project_input in enumerate(batch_inputs):
        project_name = project_input.get("project_name", f"Project_{i}")

        for condition in conditions:
            label = condition["label"]
            if verbose:
                print(f"\n  [{i+1}/{len(batch_inputs)}] {project_name} — {label}")

            # Unique seed per project × condition so different projects
            # get genuinely different random sequences.
            project_seed = seed + i * 1000 + conditions.index(condition)

            metrics = run_single_evaluation(
                user_input=project_input,
                knowledge_base=knowledge_base,
                dataset=dataset,
                disable_rag=condition["disable_rag"],
                disable_filter=condition["disable_filter"],
                disable_abm=condition.get("disable_abm", False),
                model_override=condition.get("model") or model_override,
                seed=project_seed,
                mc_replications=mc_replications,
            )

            metrics["project_name"] = project_name
            metrics["condition"] = label
            metrics["rag_enabled"] = not condition["disable_rag"]
            metrics["filter_enabled"] = not condition["disable_filter"]
            metrics["model"] = condition.get("model") or model_override or "default"
            all_results.append(metrics)

            if verbose and metrics.get("status") == "Success":
                gini_str = f"Gini={metrics.get('agent_gini_final', metrics.get('gini_24m', 'N/A'))}"
                dd_str = f"DD={metrics.get('mean_max_drawdown', 'N/A')}"
                print(f"    {gini_str}, {dd_str}, Filter={'PASS' if metrics.get('passed_filter') else 'FAIL'}")

    # Generate comparison table
    table = _format_comparison_table(all_results, conditions)

    return all_results, table


# ── Comparison table formatter ────────────────────────────────

def _format_comparison_table(results: List[Dict], conditions: List[Dict]) -> str:
    """
    Format ablation results as a structured comparison table.
    Rows = conditions, Columns = aggregated metrics across all projects.
    """
    condition_labels = [c["label"] for c in conditions]

    # Aggregate metrics per condition
    condition_metrics = {}
    for label in condition_labels:
        cond_results = [r for r in results if r.get("condition") == label and r.get("status") == "Success"]
        if not cond_results:
            condition_metrics[label] = None
            continue

        def _safe_mean(key):
            vals = [r[key] for r in cond_results if r.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        def _safe_std(key):
            vals = [r[key] for r in cond_results if r.get(key) is not None]
            return float(np.std(vals)) if vals else None

        condition_metrics[label] = {
            "n_projects": len(cond_results),
            "pass_rate": sum(1 for r in cond_results if r.get("passed_filter")) / len(cond_results),
            "gini_24m_mean": _safe_mean("gini_24m"),
            "gini_24m_std": _safe_std("gini_24m"),
            "agent_gini_mean": _safe_mean("agent_gini_final"),
            "agent_gini_std": _safe_std("agent_gini_final"),
            "theil_mean": _safe_mean("agent_theil_final"),
            "atkinson_mean": _safe_mean("agent_atkinson_final"),
            "drawdown_mean": _safe_mean("mean_max_drawdown"),
            "volatility_mean": _safe_mean("mean_price_volatility"),
            "stability_mean": _safe_mean("price_stability"),
            "gov_influence_mean": _safe_mean("governance_influence"),
            "n_findings_mean": _safe_mean("n_control_findings"),
        }

    # Format table
    lines = []
    lines.append("=" * 110)
    lines.append("ABLATION COMPARISON TABLE")
    lines.append("=" * 110)
    lines.append(f"{'Condition':<25} {'N':>3} {'Pass%':>6} {'Gini24m':>8} {'AgentGini':>10} "
                 f"{'Theil':>7} {'Atkinson':>9} {'MaxDD':>7} {'Stab':>6} {'GovInfl':>8}")
    lines.append("-" * 110)

    for label in condition_labels:
        m = condition_metrics.get(label)
        if m is None:
            lines.append(f"{label:<25} {'—':>3} {'—':>6} {'—':>8} {'—':>10} "
                         f"{'—':>7} {'—':>9} {'—':>7} {'—':>6} {'—':>8}")
            continue

        def _fmt(val, fmt=".3f"):
            return f"{val:{fmt}}" if val is not None else "—"

        lines.append(
            f"{label:<25} {m['n_projects']:>3} "
            f"{m['pass_rate']*100:>5.0f}% "
            f"{_fmt(m['gini_24m_mean']):>8} "
            f"{_fmt(m['agent_gini_mean']):>10} "
            f"{_fmt(m['theil_mean']):>7} "
            f"{_fmt(m['atkinson_mean']):>9} "
            f"{_fmt(m['drawdown_mean']):>7} "
            f"{_fmt(m['stability_mean']):>6} "
            f"{_fmt(m['gov_influence_mean']):>8}"
        )

    lines.append("=" * 110)
    lines.append("Notes: Gini24m = category-level Gini at 24 months; AgentGini = agent-level from ABM;")
    lines.append("       MaxDD = mean max drawdown; Stab = price stability coefficient;")
    lines.append("       GovInfl = governance influence index. All values are means across projects.")
    lines.append("")

    return "\n".join(lines)


# ── CSV export ────────────────────────────────────────────────

def export_ablation_csv(results: List[Dict], filepath: str) -> None:
    """Export detailed ablation results to CSV."""
    if not results:
        return

    fieldnames = [
        "project_name", "condition", "rag_enabled", "filter_enabled", "model",
        "status", "error", "passed_filter", "control_aligned", "control_requires_iteration",
        "n_control_findings", "n_filter_issues",
        "overall_gini", "t0_gini", "gini_12m", "gini_24m", "governance_influence",
        "agent_gini_final", "agent_theil_final", "agent_atkinson_final",
        "agent_gini_ci_low", "agent_gini_ci_high",
        "mean_max_drawdown", "mean_price_volatility", "price_stability",
        "abm_enabled",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"Ablation results exported to {filepath}")


# ── CLI entry point ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ablation Evaluation Harness (direct import)")
    parser.add_argument("--batch-file", required=True,
                        help="JSON file containing list of project inputs")
    parser.add_argument("--output-csv", default="ablation_results.csv",
                        help="CSV output for detailed results")
    parser.add_argument("--output-table", default="ablation_table.txt",
                        help="Text file for comparison table")
    parser.add_argument("--model-override", type=str, default=None,
                        help="Additional LLM model for ablation (e.g. gpt-3.5-turbo)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mc-replications", type=int, default=30,
                        help="Monte Carlo replications per evaluation (default 30)")
    parser.add_argument("--kb-path", type=str, default="TokenomicsKnowledge.json",
                        help="Path to knowledge base JSON")
    args = parser.parse_args()

    # Load inputs
    with open(args.batch_file, "r") as f:
        batch_inputs = json.load(f)

    with open(args.kb_path, "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)

    dataset = knowledge_base  # KB doubles as dataset

    print(f"Loaded {len(batch_inputs)} project inputs, {len(knowledge_base)} KB entries.")
    print(f"Running ablation with {len(ABLATION_CONDITIONS)} conditions...")

    results, table = run_ablation_experiment(
        batch_inputs=batch_inputs,
        knowledge_base=knowledge_base,
        dataset=dataset,
        model_override=args.model_override,
        seed=args.seed,
        mc_replications=args.mc_replications,
    )

    # Print table
    print("\n" + table)

    # Export
    export_ablation_csv(results, args.output_csv)
    with open(args.output_table, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"Comparison table saved to {args.output_table}")


if __name__ == "__main__":
    main()
