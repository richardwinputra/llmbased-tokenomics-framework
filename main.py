"""Single-proposal pipeline: generation, control and filter, and simulation."""

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List

import numpy as np

from models import GeneratedTokenomics, ProjectContext, SimulationReport
from utils import summarize_all_projects, extract_allocation_enhanced, get_initial_supply
from llm_engine import (
    get_structured_input,
    create_structured_prompt,
    ask_openai_enhanced, generate_tokenomics_proposal,
)
from control_filter import run_control_layer, run_filter_layer
from simulation import run_simulation_module


DEFAULT_KB_PATH = "TokenomicsKnowledge.json"


def load_knowledge_base(path: str = DEFAULT_KB_PATH) -> List[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Knowledge base file not found at {path}.")
        sys.exit(1)


def seed_random_generators(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-Based Tokenomics Screening Framework"
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic simulations.")
    parser.add_argument("--input-file", type=str, default=None,
                        help="Path to JSON input file (bypasses interactive mode).")
    parser.add_argument("--model-override", type=str, default=None,
                        help="Override the OpenAI model (e.g. gpt-5.4).")
    parser.add_argument("--output-dir", type=str, default="pipeline_exports",
                        help="Output directory for pipeline results.")
    return parser.parse_args()


def save_pipeline_outputs(
    output_dir: str,
    tokenomics: GeneratedTokenomics,
    context: ProjectContext,
    control_result,
    filter_result,
    sim_report: SimulationReport,
    raw_llm_response: str = "",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    with open(os.path.join(output_dir, f"proposal_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(tokenomics), f, ensure_ascii=False, indent=2)


    if raw_llm_response:
        with open(os.path.join(output_dir, f"llm_response_{timestamp}.txt"), "w", encoding="utf-8") as f:
            f.write(raw_llm_response)


    with open(os.path.join(output_dir, f"context_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(context), f, ensure_ascii=False, indent=2)


    control_payload = {
        "aligned": control_result.aligned,
        "requires_iteration": control_result.requires_iteration,
        "insider_pct": control_result.insider_pct,
        "distributed_pct": control_result.distributed_pct,
        "team_pct": control_result.team_pct,
        "investor_pct": control_result.investor_pct,
        "gini_t0": control_result.gini_t0,
        "findings": [asdict(f) for f in control_result.findings],
    }
    with open(os.path.join(output_dir, f"control_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(control_payload, f, ensure_ascii=False, indent=2)


    filter_payload = {
        "passed": filter_result.passed,
        "checks": [asdict(c) for c in filter_result.checks],
        "adjusted_allocations": filter_result.adjusted_proposal.tokenomics_parameters.allocation,
    }
    with open(os.path.join(output_dir, f"filter_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(filter_payload, f, ensure_ascii=False, indent=2)


    sim_payload = {
        "supply_release": {
            "total_supply": get_initial_supply(filter_result.adjusted_proposal),
            "initial_circulating_pct": sim_report.supply_release.initial_circulating_pct,
            "year1_circulating_pct": sim_report.supply_release.year1_circulating_pct,
            "year2_circulating_pct": sim_report.supply_release.year2_circulating_pct,
            "full_unlock_month": sim_report.supply_release.full_unlock_month,
            "year1_inflation_proxy": sim_report.supply_release.year1_inflation_proxy,
        },
        "fairness_evaluation": {
            "snapshots": [asdict(s) for s in sim_report.fairness_evaluation.snapshots],
            "fairness_drift": sim_report.fairness_evaluation.fairness_drift,
        },
        "stress_test": {
            "scenarios": [asdict(s) for s in sim_report.stress_test.scenarios],
            "pass_rate": sim_report.stress_test.pass_rate,
            "passed": sim_report.stress_test.passed,
        },
        "recommendations": sim_report.recommendations,
    }

    with open(os.path.join(output_dir, f"simulation_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(sim_payload, f, ensure_ascii=False, indent=2)


def print_control_results(control_result) -> None:
    print("\n--- Control Layer (Table II) ---")
    print(f"  Insider allocation:     {control_result.insider_pct:.1f}%")
    print(f"  Distributed allocation: {control_result.distributed_pct:.1f}%")
    print(f"  Team allocation:        {control_result.team_pct:.1f}%")
    print(f"  Investor allocation:    {control_result.investor_pct:.1f}%")
    print(f"  Initial Gini:           {control_result.gini_t0:.3f}")

    if control_result.findings:
        for f in control_result.findings:
            print(f"  [{f.severity.upper()}] {f.check}: {f.message}")
    if control_result.aligned:
        print("  Result: Proposal aligns with design criteria.")
    elif control_result.requires_iteration:
        print("  Result: High-risk findings detected; iteration recommended.")
    else:
        print("  Result: Warnings detected; review recommended.")


def print_filter_results(filter_result) -> None:
    print("\n--- Filter Layer (Table III) ---")
    for c in filter_result.checks:
        status = "PASS" if c.passed else "FAIL"
        print(f"  [{status}] {c.rule}: {c.message}")
    print(f"  Result: {'PASSED' if filter_result.passed else 'FAILED'}")


def print_simulation_results(sim_report: SimulationReport) -> None:
    sr = sim_report.supply_release
    print("\n--- Supply Release Simulation ---")
    print(f"  Initial circulating:  {sr.initial_circulating_pct:.1f}%")
    print(f"  Year 1 circulating:   {sr.year1_circulating_pct:.1f}%")
    print(f"  Year 2 circulating:   {sr.year2_circulating_pct:.1f}%")
    if sr.full_unlock_month:
        print(f"  Full unlock month:    {sr.full_unlock_month}")
    if sr.year1_inflation_proxy is not None:
        print(f"  Year 1 inflation proxy: {sr.year1_inflation_proxy:.1f}%")

    print("\n--- Fairness Evaluation ---")
    for snap in sim_report.fairness_evaluation.snapshots:
        print(f"  Month {snap.month:>3}: Insider {snap.insider_share:.3f}, "
              f"Distributed {snap.distributed_share:.3f}, Gini {snap.gini:.3f}")
    print(f"  Fairness drift: {sim_report.fairness_evaluation.fairness_drift:.3f}")

    print("\n--- Sustainability Stress Testing ---")
    for s in sim_report.stress_test.scenarios:
        viable = "VIABLE" if s.viable else "FAILED"
        recovery = f", recovery {s.recovery_months}m" if s.recovery_months else ""
        print(f"  {s.name:<20} [{viable}]{recovery}")
        for note in s.notes:
            print(f"    - {note}")
    print(f"  Pass rate: {sim_report.stress_test.pass_rate:.0%} "
          f"({'PASSED' if sim_report.stress_test.passed else 'FAILED'} >= 70% threshold)")

    print("\n--- Recommendations ---")
    for rec in sim_report.recommendations:
        print(f"  - {rec}")


def main():
    args = parse_cli_args()
    knowledge_base = load_knowledge_base()
    seed_random_generators(args.seed)


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
        user_input = get_structured_input()


    project_summaries = summarize_all_projects(knowledge_base)


    prompt = create_structured_prompt(user_input, project_summaries)

    print("\nGenerating tokenomics design...")
    result = ask_openai_enhanced(prompt, model_override=args.model_override)
    print("\n--- LLM Response ---")
    print(result)

    labels, values = extract_allocation_enhanced(result)
    if len(values) <= 1:
        print("Could not extract proper token allocation from the LLM response.")
        return


    proposal, context = generate_tokenomics_proposal(user_input, result)


    control_result = run_control_layer(proposal)
    print_control_results(control_result)


    filter_result = run_filter_layer(proposal)
    print_filter_results(filter_result)

    if control_result.requires_iteration or (hasattr(filter_result, 'passed') and not filter_result.passed):
        print("\n[INFO] Issues detected; continuing to simulation for diagnostic insight.")


    print("\nRunning simulation and evaluation module...")
    sim_report = run_simulation_module(filter_result.adjusted_proposal)
    print_simulation_results(sim_report)


    save_pipeline_outputs(
        output_dir=args.output_dir,
        tokenomics=proposal, context=context,
        control_result=control_result, filter_result=filter_result,
        sim_report=sim_report,
        raw_llm_response=result,
    )

    project_name = user_input.get('project_name', 'your project')
    print(f"\nTokenomics screening for {project_name} completed!")
    print(f"Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

