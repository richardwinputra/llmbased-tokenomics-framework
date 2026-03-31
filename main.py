"""
Tokenomics Pipeline – Main entry point.

Modular architecture:
  models.py         – Data structures (dataclasses)
  utils.py          – Parsing, normalization, Gini, helpers
  llm_engine.py     – Prompt engineering, OpenAI API, proposal generation
  control_filter.py – Control layer + Filter layer
  simulation.py     – All simulation sub-modules + backtesting helpers
  advanced_simulations.py – Agent-based market simulation (ABM)
"""

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from models import (
    GeneratedTokenomics, ProjectContext, FilterLayerResult,
    SimulationReport,
)
from utils import (
    get_initial_supply, normalize_allocation_key,
    extract_allocations_from_knowledge_base,
    extract_allocation_enhanced, summarize_all_projects,
)
from llm_engine import (
    get_input_mode, get_structured_input, get_generic_input,
    create_structured_prompt, create_generic_prompt,
    ask_openai_enhanced, generate_tokenomics_proposal,
    proposal_from_dataset_entry, proposal_from_entry_via_llm,
)
from control_filter import (
    run_control_layer, run_filter_layer,
    calculate_temporal_fairness,
)
from simulation import (
    run_simulations_layer,
    dataset_allocation_statistics, dataset_outcome_overview,
    estimate_drawdown_from_prices, estimate_inflation_from_supply,
)


# ── Configuration ─────────────────────────────────────────────

DEFAULT_HISTORICAL_DATASET_PATH = "TokenomicsKnowledge.json"


def load_knowledge_base(path: str = "TokenomicsKnowledge.json") -> List[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Knowledge base file not found.")
        sys.exit(1)


def load_historical_dataset(path: str) -> List[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Historical dataset not found at {path}. Continuing without it.")
        return []


def seed_random_generators(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


# ── CLI ───────────────────────────────────────────────────────

def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenomics pipeline controller")
    parser.add_argument("--historical-dataset", dest="historical_dataset",
                        default=DEFAULT_HISTORICAL_DATASET_PATH,
                        help="Path to JSON dataset of historical project outcomes.")
    parser.add_argument("--dataset-report", dest="dataset_report", action="store_true",
                        help="Skip LLM flow and print dataset validation summary.")
    parser.add_argument("--seed", dest="seed", type=int, default=42,
                        help="Random seed for deterministic simulations.")
    parser.add_argument("--input-file", dest="input_file", type=str, default=None,
                        help="Path to JSON input file (bypasses interactive mode).")
    parser.add_argument("--disable-rag", dest="disable_rag", action="store_true",
                        help="Disable RAG (knowledge base) for ablation testing.")
    parser.add_argument("--disable-filter", dest="disable_filter", action="store_true",
                        help="Disable the Filter layer for ablation testing.")
    parser.add_argument("--model-override", dest="model_override", type=str, default=None,
                        help="Override the OpenAI model (e.g. gpt-3.5-turbo).")
    parser.add_argument("--run-scenarios", dest="run_scenarios", action="store_true",
                        help="Run 100+ scenario analysis with sensitivity sweeps (no LLM needed).")
    parser.add_argument("--n-scenarios", dest="n_scenarios", type=int, default=100,
                        help="Number of LHS scenarios to generate (default: 100).")
    parser.add_argument("--scenario-output-dir", dest="scenario_output_dir",
                        default="scenario_exports",
                        help="Output directory for scenario analysis results.")
    return parser.parse_args()


# ── Dataset validation CLI ────────────────────────────────────

def run_dataset_validation_cli(dataset: List[Dict], knowledge_base: List[Dict]) -> None:
    if not dataset:
        print("No entries available in the historical dataset.")
        return

    print("Historical Dataset Validation")
    print(f"Total reference projects: {len(dataset)}")

    allocation_stats = dataset_allocation_statistics(dataset)
    if allocation_stats:
        print("\nAllocation aggregates (mean, min-max):")
        for key, stats in allocation_stats.items():
            print(f" - {key}: mean {stats['mean']:.1f}% (min {stats['min']:.1f}%, max {stats['max']:.1f}%)")

    allocations = [entry.get("allocation", {}) for entry in dataset]
    community_shares = [a.get("Community", 0.0) for a in allocations if isinstance(a, dict)]
    insider_shares = [(a.get("Team", 0.0) + a.get("Investors", 0.0)) for a in allocations if isinstance(a, dict)]

    if community_shares:
        print(f"Avg community share: {np.mean(community_shares):.1f}%")
    if insider_shares:
        print(f"Avg insider share (team+investors): {np.mean(insider_shares):.1f}%")

    overview = dataset_outcome_overview(dataset)
    if overview.get("drawdowns"):
        print(f"Median max drawdown: {overview['drawdowns']['median']:.2f}")
    if overview.get("inflation"):
        print(f"Median supply inflation: {overview['inflation']['median']:.2f}")
    incident_rate = overview.get("incident_rate")
    if incident_rate is not None:
        print(f"Governance incidents recorded in {incident_rate*100:.1f}% of projects")

    print(f"\nReference knowledge base entries available: {len(knowledge_base)}")


# ── Pipeline output ───────────────────────────────────────────

def save_pipeline_outputs(output_dir: str,
                          tokenomics: GeneratedTokenomics,
                          context: ProjectContext,
                          control_result, filter_result,
                          simulation_report: Optional[SimulationReport] = None) -> None:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(os.path.join(output_dir, f"proposal_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(tokenomics), f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, f"context_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(context), f, ensure_ascii=False, indent=2)

    control_payload = {
        "aligned": control_result.aligned,
        "requires_iteration": control_result.requires_iteration,
        "stakeholder_findings": [asdict(f) for f in control_result.stakeholder_findings],
        "compliance_findings": [asdict(f) for f in control_result.compliance_findings],
        "feasibility_findings": [asdict(f) for f in control_result.feasibility_findings],
    }
    with open(os.path.join(output_dir, f"control_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(control_payload, f, ensure_ascii=False, indent=2)

    alloc = filter_result.adjusted_proposal.tokenomics_parameters.allocation
    filter_payload = {
        "passed": filter_result.passed,
        "allocation_issues": filter_result.allocation_issues,
        "vesting_issues": filter_result.vesting_issues,
        "economic_issues": filter_result.economic_issues,
        "governance_issues": filter_result.governance_issues,
        "adjusted_allocations": alloc,
    }
    with open(os.path.join(output_dir, f"filter_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(filter_payload, f, ensure_ascii=False, indent=2)

    if simulation_report:
        sim_payload = {
            "scenarios": [asdict(s) for s in simulation_report.scenarios],
            "gini_layers": asdict(simulation_report.gini_layers),
            "supply_dynamics": asdict(simulation_report.supply_dynamics),
            "baseline_comparison": asdict(simulation_report.baseline_comparison) if simulation_report.baseline_comparison else None,
            "fairness_report": simulation_report.fairness_report,
            "validated_model_summary": simulation_report.validated_model_summary,
            "recommendations": simulation_report.recommendations,
        }
        with open(os.path.join(output_dir, f"simulation_{timestamp}.json"), "w", encoding="utf-8") as f:
            json.dump(sim_payload, f, ensure_ascii=False, indent=2)


# ── Backtesting ───────────────────────────────────────────────

def run_backtest(dataset: List[Dict], knowledge_base: List[Dict],
                 include_kb: bool, seed: int, use_llm: bool) -> None:
    seed_random_generators(seed)
    entries = list(dataset)
    if include_kb:
        entries += knowledge_base

    if not entries:
        print("No entries available for backtesting.")
        return

    os.makedirs("backtest_exports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join("backtest_exports", f"backtest_report_{timestamp}.csv")

    project_summaries = summarize_all_projects(knowledge_base)
    rows = []
    for entry in entries:
        if use_llm:
            proposal, context = proposal_from_entry_via_llm(entry, project_summaries)
        else:
            proposal, context = proposal_from_dataset_entry(entry)
        control_result = run_control_layer(proposal, context)
        filter_result = run_filter_layer(proposal, context)
        sim_report = run_simulations_layer(filter_result.adjusted_proposal, context, knowledge_base, dataset)

        outcomes = entry.get("outcomes", {}) or {}
        predicted_drawdown = estimate_drawdown_from_prices(
            sim_report.agent_market_detail.price_path if sim_report.agent_market_detail else [])
        initial_supply = get_initial_supply(proposal)
        predicted_inflation = estimate_inflation_from_supply(sim_report.supply_dynamics, initial_supply)
        predicted_incident = 1 if sim_report.governance_risk.capture_risk_score >= 2.5 else 0

        rows.append({
            "project": proposal.project_metadata.project,
            "token": proposal.project_metadata.token,
            "control_requires_iteration": control_result.requires_iteration,
            "filter_passed": filter_result.passed,
            "t0_gini": sim_report.fairness_metrics.t0_gini,
            "gini_12m": sim_report.fairness_metrics.gini_12m,
            "gini_24m": sim_report.fairness_metrics.gini_24m,
            "gov_influence_index": sim_report.governance_risk.governance_influence_index,
            "baseline": sim_report.baseline_comparison.baseline_name if sim_report.baseline_comparison else "",
            "pred_drawdown": predicted_drawdown,
            "actual_drawdown": outcomes.get("max_drawdown"),
            "drawdown_error": (predicted_drawdown - outcomes.get("max_drawdown")) if (predicted_drawdown is not None and outcomes.get("max_drawdown") is not None) else None,
            "pred_inflation": predicted_inflation,
            "actual_inflation": outcomes.get("supply_inflation"),
            "inflation_error": (predicted_inflation - outcomes.get("supply_inflation")) if (predicted_inflation is not None and outcomes.get("supply_inflation") is not None) else None,
            "pred_incident_flag": predicted_incident,
            "actual_incidents": outcomes.get("governance_incidents"),
        })

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Backtest completed on {len(rows)} entries. Report saved to {csv_path}")


# ── Main pipeline ─────────────────────────────────────────────

def main():
    args = parse_cli_args()

    knowledge_base = load_knowledge_base()
    dataset = load_historical_dataset(args.historical_dataset)

    if args.dataset_report:
        run_dataset_validation_cli(dataset, knowledge_base)
        return

    if args.run_scenarios:
        from scenario_generator import (
            run_full_scenario_analysis,
            export_scenario_results_csv,
            export_sensitivity_results_csv,
        )
        seed_random_generators(args.seed)
        os.makedirs(args.scenario_output_dir, exist_ok=True)
        report = run_full_scenario_analysis(
            n_scenarios=args.n_scenarios, seed=args.seed,
            run_sensitivity=True, verbose=True,
        )
        print("\n" + "=" * 60)
        print("SCENARIO ANALYSIS SUMMARY")
        print("=" * 60)
        for key, val in report.summary_statistics.items():
            print(f"  {key}: {val:.4f}")
        for note in report.notes:
            print(f"  - {note}")
        export_scenario_results_csv(
            report.scenario_results,
            os.path.join(args.scenario_output_dir, "scenario_results.csv"),
        )
        if report.sensitivity_results:
            export_sensitivity_results_csv(
                report.sensitivity_results,
                os.path.join(args.scenario_output_dir, "sensitivity_results.csv"),
            )
        return

    seed_random_generators(args.seed)
    os.makedirs("exported_charts", exist_ok=True)

    # --- Input ---
    if args.input_file:
        print(f"Loading input from {args.input_file}...")
        try:
            with open(args.input_file, "r") as f:
                user_input = json.load(f)
            if "input_type" not in user_input:
                user_input["input_type"] = "structured"
        except Exception as e:
            print(f"Error reading input file: {e}")
            return
    else:
        input_mode = get_input_mode()
        user_input = get_structured_input() if input_mode == '1' else get_generic_input()

    # --- RAG ---
    if args.disable_rag:
        print("\n[ABLATION] RAG is disabled. Using empty project summaries.")
        project_summaries = ""
    else:
        project_summaries = summarize_all_projects(knowledge_base)

    # --- LLM generation ---
    if user_input.get('input_type') == 'structured':
        prompt = create_structured_prompt(user_input, project_summaries)
    else:
        prompt = create_generic_prompt(user_input, project_summaries)

    print("\nGenerating token design...")
    result = ask_openai_enhanced(prompt, user_input.get('input_type', 'structured'),
                                 model_override=args.model_override)
    print("Tokenomics Design")
    print(result)

    token_symbol = user_input.get('token_symbol', user_input.get('project_name', 'TOKEN'))
    labels, values = extract_allocation_enhanced(result)
    if len(values) <= 1:
        print("Could not extract proper token allocation from the recommendation.")
        return

    # --- Proposal ---
    proposal, context = generate_tokenomics_proposal(user_input, result)

    # --- Control Layer ---
    control_result = run_control_layer(proposal, context)
    for finding in control_result.stakeholder_findings:
        print(f"[{finding.severity.upper()}] {finding.stakeholder}: {finding.message}")
    for finding in control_result.compliance_findings:
        print(f"[{finding.severity.upper()}] Compliance: {finding.message}")
    for finding in control_result.feasibility_findings:
        print(f"[{finding.severity.upper()}] Feasibility: {finding.message}")
    if control_result.aligned:
        print("Control Layer: proposal aligns with stated goals.")

    # --- Filter Layer ---
    if args.disable_filter:
        print("\n[ABLATION] Filter layer is disabled. Proceeding with raw proposal.")
        filter_result = FilterLayerResult(
            passed=True, adjusted_proposal=proposal,
            allocation_issues=[], vesting_issues=[],
            economic_issues=[], governance_issues=[],
            fairness_metrics=None, governance_risk=None,
        )
    else:
        filter_result = run_filter_layer(proposal, context)
        for label, issues in [("Allocation", filter_result.allocation_issues),
                              ("Vesting", filter_result.vesting_issues),
                              ("Economic", filter_result.economic_issues),
                              ("Governance", filter_result.governance_issues)]:
            if issues:
                print(f"{label} Issues:")
                for issue in issues:
                    print(f" - {issue}")
        if filter_result.passed:
            print("Filter Layer: Passed hard constraints check.")
        else:
            print("Filter Layer: Critical issues detected.")

    if control_result.requires_iteration or not filter_result.passed:
        print("\n[INFO] Critical findings detected, continuing to simulations for insight-only run.")

    # --- Simulation Layer ---
    sim_report = run_simulations_layer(filter_result.adjusted_proposal, context, knowledge_base, dataset)

    for scenario in sim_report.scenarios:
        print(f"\nScenario: {scenario.name}")
        print(f"  {scenario.description}")
        print(f"  Supply trajectory: {[int(x) for x in scenario.supply_over_time]}")
        for note in scenario.notes:
            print(f"   - {note}")

    print("\nFairness Report:")
    print(sim_report.fairness_report)

    if sim_report.agent_market_detail:
        detail = sim_report.agent_market_detail
        print("Agent-based Market Simulation:")
        print(f"  Avg price path sample: {detail.price_path[:5]} ...")
        print(f"  Liquidity levels sample: {detail.liquidity_levels[:5]} ...")
        for note in detail.notes:
            print(f"   - {note}")

    print("\nBaseline Comparison:")
    if sim_report.baseline_comparison:
        print(f"Closest baseline: {sim_report.baseline_comparison.baseline_name}")
        for item in sim_report.baseline_comparison.similarities:
            print(f"  - {item}")
        for item in sim_report.baseline_comparison.differences:
            print(f"  - {item}")

    print("\nValidated Model Summary:")
    print(sim_report.validated_model_summary)

    print("\nRecommendations:")
    for rec in sim_report.recommendations:
        print(f" - {rec}")

    # --- Save ---
    save_pipeline_outputs(
        output_dir="pipeline_exports", tokenomics=proposal, context=context,
        control_result=control_result, filter_result=filter_result,
        simulation_report=sim_report,
    )

    project_name = user_input.get('project_name', 'your project')
    print(f"\nTokenomics design for {project_name} completed!")


if __name__ == "__main__":
    main()
