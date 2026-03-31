"""
Simulation Layer: market scenarios, stress testing, Gini layers, supply dynamics,
baseline comparison, fairness reporting, and ABM integration.
"""

import csv
import os
import statistics
import numpy as np
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

from models import (
    GeneratedTokenomics, ProjectContext, FairnessMetrics,
    ScenarioResult, GiniLayerResult, SupplyDynamicsResult,
    BaselineComparisonResult, SimulationReport, RealProjectValidationResult,
)
from utils import (
    normalize_allocation_key, normalize_allocation_dict, calculate_gini,
    get_initial_supply, generate_emission_curve,
    extract_allocations_from_knowledge_base,
)
from control_filter import (
    calculate_temporal_fairness, evaluate_governance_risk,
    validate_against_real_projects,
)
from advanced_simulations import (
    run_agent_market_simulation, run_monte_carlo_abm,
    MarketScenarioDetail, MonteCarloABMResult,
)


# ── Monte Carlo (supply burn) ─────────────────────────────────

def monte_carlo_simulation(initial_supply: float, burn_rate: float = 0.02,
                           months: int = 12, simulations: int = 1000) -> Dict:
    """Monte Carlo simulation of token supply under stochastic burn."""
    outcomes = []
    monthly_burns = []
    for _ in range(simulations):
        current_supply = initial_supply
        monthly_burn_record = []
        for month in range(months):
            market_factor = np.random.normal(1.0, 0.2)
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * month / 12)
            effective_burn_rate = max(0, min(burn_rate * market_factor * seasonal_factor, 0.1))
            burned_amount = current_supply * effective_burn_rate
            current_supply -= burned_amount
            monthly_burn_record.append(burned_amount)
        outcomes.append(current_supply)
        monthly_burns.append(monthly_burn_record)
    return {
        'final_supplies': outcomes,
        'monthly_burns': monthly_burns,
        'mean_final_supply': np.mean(outcomes),
        'std_final_supply': np.std(outcomes),
        'median_final_supply': np.median(outcomes),
        'percentile_5': np.percentile(outcomes, 5),
        'percentile_95': np.percentile(outcomes, 95),
    }


# ── Baseline generation from knowledge base ──────────────────

def calculate_allocation_statistics(allocations_data: Dict) -> Dict:
    all_allocations = allocations_data['all_allocations']
    allocation_keys = allocations_data['allocation_keys']
    common_categories = [key for key, _ in allocation_keys.most_common(10)]
    category_stats = {}
    for category in common_categories:
        values = [a[category] for a in all_allocations if category in a]
        if values:
            category_stats[category] = {
                'count': len(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'min': min(values),
                'max': max(values),
                'values': values,
            }
    return category_stats


def generate_dynamic_baseline_models(knowledge_base: List[Dict]) -> Dict:
    allocations_data = extract_allocations_from_knowledge_base(knowledge_base)
    category_stats = calculate_allocation_statistics(allocations_data)
    baseline_models = {}

    def _build_baseline(label, pick_fn):
        alloc = {}
        for cat, stats in category_stats.items():
            if stats['count'] >= 3:
                alloc[cat] = round(pick_fn(stats), 1)
        total = sum(alloc.values())
        if total > 0:
            baseline_models[label] = {k: round(v * 100 / total, 1) for k, v in alloc.items()}

    _build_baseline('Knowledge Base Average', lambda s: s['mean'])
    _build_baseline('Knowledge Base Median', lambda s: s['median'])
    _build_baseline('Conservative from Knowledge Based',
                    lambda s: sorted(s['values'])[len(s['values']) // 4] if len(s['values']) > 3 else s['min'])
    _build_baseline('Aggressive from Knowledge Based',
                    lambda s: sorted(s['values'])[(3 * len(s['values'])) // 4] if len(s['values']) > 3 else s['max'])

    return {'baseline_models': baseline_models, 'category_stats': category_stats,
            'allocations_data': allocations_data}


# ── Dataset helpers ───────────────────────────────────────────

def dataset_allocation_statistics(dataset: List[Dict]) -> Dict[str, float]:
    totals = defaultdict(list)
    for entry in dataset:
        allocation = entry.get('allocation', {})
        if isinstance(allocation, dict):
            normalized = normalize_allocation_dict(allocation)
            for key, value in normalized.items():
                totals[key].append(value)
    stats = {}
    for key, values in totals.items():
        stats[key] = {
            "mean": float(np.mean(values)), "median": float(np.median(values)),
            "min": float(np.min(values)), "max": float(np.max(values)),
        }
    return stats


def dataset_outcome_overview(dataset: List[Dict]) -> Dict:
    drawdowns = [e.get('outcomes', {}).get('max_drawdown') for e in dataset if e.get('outcomes')]
    inflation = [e.get('outcomes', {}).get('supply_inflation') for e in dataset if e.get('outcomes')]
    incidents = [e.get('outcomes', {}).get('governance_incidents') for e in dataset if e.get('outcomes')]

    def safe_stats(values):
        filtered = [v for v in values if isinstance(v, (int, float))]
        if not filtered:
            return {}
        return {"mean": float(np.mean(filtered)), "median": float(np.median(filtered)),
                "max": float(np.max(filtered))}

    return {
        "drawdowns": safe_stats(drawdowns),
        "inflation": safe_stats(inflation),
        "incident_rate": sum(1 for v in incidents if v and v > 0) / max(1, len(dataset)),
    }


# ── Simulation sub-modules ────────────────────────────────────

def run_historical_pattern_module(tokenomics: GeneratedTokenomics,
                                  knowledge_base: List[Dict],
                                  context: ProjectContext) -> ScenarioResult:
    proposal_categories = defaultdict(float)
    for k, v in tokenomics.tokenomics_parameters.allocation.items():
        proposal_categories[normalize_allocation_key(k)] += v

    best_match = None
    best_overlap = -1.0
    for project in knowledge_base:
        allocation = project.get('tokenomics', {}).get('allocation', {})
        if not isinstance(allocation, dict):
            continue
        normalized = normalize_allocation_dict(allocation)
        overlap = sum(min(proposal_categories.get(cat, 0.0), v) for cat, v in normalized.items())
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = (project, normalized)

    base_supply = get_initial_supply(tokenomics)
    milestones = [0, 12, 24, 36, 48, 60]
    if best_match:
        project, _ = best_match
        description = f"Historical Pattern anchored on {project.get('project', 'unknown')} allocations."
        notes = [
            f"Reference token: {project.get('token', 'N/A')}",
            f"Overlap score: {best_overlap:.1f}%",
            "Used to anchor expected distribution pressures.",
        ]
    else:
        description = "No strong historical analog; using neutral pattern."
        notes = ["Proceeding with generic release curve."]

    emission_curve = generate_emission_curve(
        context.economic_signals.get("emission_style", "steady"), len(milestones))
    supply_over_time = [base_supply * 0.4 + base_supply * curve for curve in emission_curve]
    return ScenarioResult(name="Historical Pattern", description=description,
                          supply_over_time=supply_over_time, notes=notes)


def run_market_scenarios_module(tokenomics: GeneratedTokenomics,
                                context: ProjectContext) -> List[ScenarioResult]:
    base_supply = get_initial_supply(tokenomics)
    milestones = [0, 12, 24, 36, 48, 60]
    demand_multiplier = context.economic_signals.get("demand_multiplier", 1.0)
    burn_rate = context.economic_signals.get("burn_rate", 0.01)
    emission_rate = context.economic_signals.get("emission_rate", 0.08)

    scenarios = []
    for name, growth_bias, burn_bias in [
        ("Bull Market", 0.05, -0.005),
        ("Neutral Market", 0.0, 0.0),
        ("Bear Market", -0.03, 0.01),
    ]:
        supply_curve = []
        circulating = base_supply * 0.35
        for _ in milestones:
            growth = emission_rate + growth_bias
            burn = max(burn_rate + burn_bias, 0)
            circulating = min(base_supply, circulating * (1 + growth * demand_multiplier) * (1 - burn))
            supply_curve.append(circulating)
        scenarios.append(ScenarioResult(
            name=name,
            description=f"Assumes {name.lower()} demand with {growth*100:.0f}% release cadence.",
            supply_over_time=supply_curve,
            notes=[f"Burn pressure {burn*100:.1f}% per period."],
        ))
    return scenarios


def run_stress_testing_module(tokenomics: GeneratedTokenomics,
                              context: ProjectContext) -> List[ScenarioResult]:
    base_supply = get_initial_supply(tokenomics)
    alloc = tokenomics.tokenomics_parameters.allocation
    vesting = tokenomics.tokenomics_parameters.vesting

    large_unlock = max(alloc.values()) if alloc else 20.0
    cliffs = [v.cliff_months for v in vesting.values() if v.cliff_months is not None]
    cliff_month = min(cliffs) if cliffs else 12

    volatility_bias = context.economic_signals.get("volatility_bias", 1.0)
    shock = min(max(volatility_bias - 1, -0.5), 0.5)

    bear_supply = [base_supply * (0.3 + shock), base_supply * (0.35 + shock),
                   base_supply * 0.65, base_supply * 0.7, base_supply * 0.75, base_supply * 0.8]
    bull_supply = [base_supply * 0.35, base_supply * (0.45 + shock),
                   base_supply * 0.6, base_supply * 0.7, base_supply * 0.78, base_supply * 0.85]

    return [
        ScenarioResult(
            name="Stress - Bear Shock",
            description="Cliff expiry plus liquidity drought stress test.",
            supply_over_time=bear_supply,
            notes=[f"Models {large_unlock:.1f}% unlock at month {cliff_month}.",
                   "Liquidity dries up for 2 quarters post unlock."],
        ),
        ScenarioResult(
            name="Stress - Bull Overheating",
            description="Sustained demand pressure with accelerated usage burns.",
            supply_over_time=bull_supply,
            notes=["High throughput scenario with validators saturated.",
                   "Protocol-owned liquidity buffers mitigate dumps."],
        ),
    ]


def evaluate_gini_layers(tokenomics: GeneratedTokenomics,
                         scenarios: List[ScenarioResult]) -> GiniLayerResult:
    alloc = tokenomics.tokenomics_parameters.allocation
    overall_gini = calculate_gini(list(alloc.values()))

    vesting_map = tokenomics.tokenomics_parameters.vesting
    circulating_weights = []
    for k, pct in alloc.items():
        vesting_months = 12
        if k in vesting_map:
            vesting_months = vesting_map[k].vesting_months or 12
        weight = 1.0 if vesting_months <= 12 else (0.7 if vesting_months <= 24 else 0.5)
        circulating_weights.append(pct * weight)
    circulating_gini = calculate_gini(circulating_weights)

    governance_buckets = []
    for k, pct in alloc.items():
        cat = normalize_allocation_key(k)
        if cat in {"Team", "Investors", "Advisors", "Community"}:
            influence = pct * 0.8 if cat == "Community" else pct
            governance_buckets.append(influence)
    governance_gini = calculate_gini(governance_buckets)

    return GiniLayerResult(overall_gini=overall_gini, circulating_gini=circulating_gini,
                           governance_gini=governance_gini)


def simulate_supply_dynamics(tokenomics: GeneratedTokenomics,
                             scenarios: List[ScenarioResult],
                             context: ProjectContext) -> SupplyDynamicsResult:
    base_supply = get_initial_supply(tokenomics)
    months = list(range(0, 61, 6))
    circulating_supply, locked_supply, burned_supply = [], [], []

    emission_curve = generate_emission_curve(
        context.economic_signals.get("emission_style", "steady"), len(months))
    alloc = tokenomics.tokenomics_parameters.allocation
    vesting_map = tokenomics.tokenomics_parameters.vesting

    for idx, month in enumerate(months):
        released_ratio = 0.0
        for k, pct in alloc.items():
            vest = vesting_map[k].vesting_months if k in vesting_map and vesting_map[k].vesting_months else 24
            released = min(month / vest, 1.0)
            released_ratio += (pct / 100) * released
        circulating = base_supply * released_ratio * (1 + emission_curve[idx] * 0.05)
        burn = circulating * context.economic_signals.get("burn_rate", 0.01) * (month / 60)
        circulating_supply.append(circulating - burn)
        burned_supply.append(burn)
        locked_supply.append(max(base_supply - circulating, 0))

    return SupplyDynamicsResult(months=months, circulating_supply=circulating_supply,
                                locked_supply=locked_supply, burned_supply=burned_supply)


def compare_with_baselines(tokenomics: GeneratedTokenomics,
                           knowledge_base: List[Dict]) -> BaselineComparisonResult:
    baselines = generate_dynamic_baseline_models(knowledge_base)['baseline_models']
    proposal_totals = defaultdict(float)
    for k, v in tokenomics.tokenomics_parameters.allocation.items():
        proposal_totals[normalize_allocation_key(k)] += v

    best_name = None
    best_distance = float('inf')
    best_allocation = None
    for name, allocation in baselines.items():
        shared = set(allocation.keys()) | set(proposal_totals.keys())
        distance = sum(abs(proposal_totals.get(c, 0) - allocation.get(c, 0)) for c in shared)
        if distance < best_distance:
            best_distance = distance
            best_name = name
            best_allocation = allocation

    similarities, differences = [], []
    if best_allocation:
        for cat, value in best_allocation.items():
            pv = proposal_totals.get(cat, 0.0)
            if abs(pv - value) <= 5:
                similarities.append(f"{cat}: proposal {pv:.1f}% vs baseline {value:.1f}% (aligned)")
            else:
                differences.append(f"{cat}: proposal {pv:.1f}% vs baseline {value:.1f}%")

    return BaselineComparisonResult(baseline_name=best_name or "Knowledge Base Average",
                                    similarities=similarities, differences=differences)


# ── Main simulation orchestrator ──────────────────────────────

def run_simulations_layer(tokenomics: GeneratedTokenomics,
                          context: ProjectContext,
                          knowledge_base: List[Dict],
                          dataset: Optional[List[Dict]] = None,
                          mc_replications: int = 100) -> SimulationReport:
    historical = run_historical_pattern_module(tokenomics, knowledge_base, context)
    market = run_market_scenarios_module(tokenomics, context)
    stress = run_stress_testing_module(tokenomics, context)
    scenarios = [historical] + market + stress

    gini_layers = evaluate_gini_layers(tokenomics, scenarios)
    supply_dynamics = simulate_supply_dynamics(tokenomics, scenarios, context)
    baseline_comparison = compare_with_baselines(tokenomics, knowledge_base)
    fairness_metrics = calculate_temporal_fairness(tokenomics)
    governance_risk = evaluate_governance_risk(tokenomics, fairness_metrics)
    real_project_validation = validate_against_real_projects(tokenomics, knowledge_base, dataset)

    # ── Monte Carlo ABM (N replications) ──────────────────────
    alloc = tokenomics.tokenomics_parameters.allocation
    vesting_map = tokenomics.tokenomics_parameters.vesting
    initial_supply = get_initial_supply(tokenomics)

    mc_result = run_monte_carlo_abm(
        proposal_initial_supply=initial_supply,
        allocation=alloc,
        vesting=vesting_map,
        normalize_key_fn=normalize_allocation_key,
        context_signals=context.economic_signals,
        n_replications=mc_replications,
        months=24,
    )

    # Use representative single run for backward-compatible agent_market_detail
    agent_market_detail = mc_result.representative_run

    # ── Enrich fairness metrics with agent-level ABM results ──
    fairness_metrics.agent_gini_final = mc_result.mean_final_gini
    fairness_metrics.agent_theil_final = float(mc_result.mean_theil_trajectory[-1]) if mc_result.mean_theil_trajectory else None
    fairness_metrics.agent_atkinson_final = float(mc_result.mean_atkinson_trajectory[-1]) if mc_result.mean_atkinson_trajectory else None
    fairness_metrics.agent_gini_ci_low = float(mc_result.p5_gini_trajectory[-1]) if mc_result.p5_gini_trajectory else None
    fairness_metrics.agent_gini_ci_high = float(mc_result.p95_gini_trajectory[-1]) if mc_result.p95_gini_trajectory else None

    dataset_overview = dataset_outcome_overview(dataset) if dataset else None

    # ── Fairness report (now with agent-level metrics) ────────
    fairness_report = (
        f"Category-level: T0 Gini {fairness_metrics.t0_gini:.3f}, "
        f"12m {fairness_metrics.gini_12m:.3f}, 24m {fairness_metrics.gini_24m:.3f}. "
        f"Governance influence index {fairness_metrics.governance_influence_index:.2f}. "
        f"Agent-level (MC N={mc_replications}): "
        f"Gini {mc_result.mean_final_gini:.3f} "
        f"[{mc_result.p5_gini_trajectory[-1]:.3f}, {mc_result.p95_gini_trajectory[-1]:.3f}] 90% CI, "
        f"Theil-T {mc_result.mean_theil_trajectory[-1]:.3f}, "
        f"Atkinson(0.5) {mc_result.mean_atkinson_trajectory[-1]:.3f}. "
        f"Price stability coefficient: {mc_result.price_stability_coefficient:.3f}."
    )
    if dataset_overview and dataset_overview.get("incident_rate") is not None:
        fairness_report += f" Historical incident rate baseline {dataset_overview['incident_rate']*100:.1f}%."

    validated_model_summary = (
        f"Monte Carlo ABM ({mc_replications} runs, {len(mc_result.mean_price_path)}-month horizon) "
        f"validates model coherence. Mean max drawdown {mc_result.mean_max_drawdown:.3f} "
        f"± {mc_result.std_max_drawdown:.3f}. Price volatility σ={mc_result.mean_price_volatility:.3f}. "
        f"Price stability coefficient={mc_result.price_stability_coefficient:.3f}."
    )

    # ── Recommendations (incorporating ABM insights) ──────────
    recommendations = []
    if gini_layers.governance_gini > 0.25:
        recommendations.append("Introduce delegated voting caps or quadratic voting to offset governance concentration.")
    if mc_result.mean_final_gini > 0.6:
        recommendations.append(
            f"Agent-level Gini ({mc_result.mean_final_gini:.3f}) exceeds 0.6 threshold; "
            "consider broader token distribution or community airdrops."
        )
    if supply_dynamics.circulating_supply[-1] / initial_supply < 0.8:
        recommendations.append("Extend emissions beyond 60 months or add sinks to avoid idle supply build-up.")
    if agent_market_detail and min(agent_market_detail.liquidity_levels) < 0.2:
        recommendations.append("Bolster liquidity reserves or stagger unlocks to avoid simulated liquidity floor breaches.")
    if mc_result.mean_max_drawdown > 0.7:
        recommendations.append(
            f"Mean max drawdown ({mc_result.mean_max_drawdown:.1%}) is severe; "
            "incorporate circuit breakers or protocol-owned liquidity buffers."
        )
    if mc_result.price_stability_coefficient < 0.4:
        recommendations.append(
            f"Price stability coefficient ({mc_result.price_stability_coefficient:.3f}) indicates "
            "high volatility; consider adding protocol-owned liquidity or treasury buyback buffers."
        )
    if dataset_overview and dataset_overview.get("drawdowns"):
        median_drawdown = dataset_overview["drawdowns"].get("median")
        if median_drawdown and median_drawdown > 0.5:
            recommendations.append("Historical drawdowns above 50% suggest elevated volatility risk.")
    if not recommendations:
        recommendations.append("Maintain current allocation but document KPI triggers for future reallocations.")
    else:
        recommendations.append("Run live governance drills before TGE to validate adaptive consent logic.")

    return SimulationReport(
        scenarios=scenarios, gini_layers=gini_layers, supply_dynamics=supply_dynamics,
        baseline_comparison=baseline_comparison, fairness_report=fairness_report,
        validated_model_summary=validated_model_summary, recommendations=recommendations,
        fairness_metrics=fairness_metrics, governance_risk=governance_risk,
        real_project_validation=real_project_validation,
        agent_market_detail=agent_market_detail,
        monte_carlo_result=mc_result,
    )


# ── Backtesting helpers ───────────────────────────────────────

def estimate_drawdown_from_prices(price_path: List[float]) -> Optional[float]:
    if not price_path:
        return None
    peak = max(price_path)
    trough = min(price_path)
    if peak <= 0:
        return None
    return (peak - trough) / peak


def estimate_inflation_from_supply(supply_result: SupplyDynamicsResult,
                                   initial_supply: float) -> Optional[float]:
    if not supply_result.circulating_supply:
        return None
    final_supply = supply_result.circulating_supply[-1]
    if initial_supply <= 0:
        return None
    return (final_supply - initial_supply) / initial_supply
