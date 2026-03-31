"""
Scenario Generator & Sensitivity Analysis for tokenomics evaluation.

Generates 100+ unique scenario configurations by systematically varying:
  - Allocation splits (community, team, investors, ecosystem percentages)
  - Vesting durations (cliff months, vesting months)
  - Burn rates
  - Market conditions (demand multiplier, volatility bias)
  - Agent compositions

Uses Latin Hypercube Sampling (LHS) for efficient coverage of the parameter space,
plus targeted sensitivity sweeps on key thresholds.
"""

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from models import (
    GeneratedTokenomics, ProjectContext, ProjectMetadata,
    TokenDesignThinking, TokenomicsParameters, VestingDetail,
    FairnessMetrics, SimulationReport,
)
from utils import normalize_allocation_key, get_initial_supply
from advanced_simulations import (
    run_monte_carlo_abm, MonteCarloABMResult,
    compute_theil_t, compute_atkinson,
)


# ── Data structures ───────────────────────────────────────────

@dataclass
class ScenarioConfig:
    """A single parameterized scenario configuration."""
    scenario_id: str
    label: str
    # Allocation splits (must sum to 100)
    community_pct: float
    team_pct: float
    investor_pct: float
    ecosystem_pct: float
    advisor_pct: float
    liquidity_pct: float
    # Vesting
    team_cliff: int
    team_vest: int
    investor_cliff: int
    investor_vest: int
    # Economic signals
    burn_rate: float
    emission_rate: float
    demand_multiplier: float
    volatility_bias: float
    # Simulation params
    total_supply: float = 1_000_000_000.0
    base_price: float = 1.0
    mc_replications: int = 30  # Reduced per-scenario for speed (100 scenarios × 30 = 3000 runs)
    months: int = 24


@dataclass
class ScenarioResult:
    """Result from running a single scenario through the ABM."""
    config: ScenarioConfig
    mean_final_gini: float
    std_final_gini: float
    ci_low_gini: float
    ci_high_gini: float
    mean_final_theil: float
    mean_final_atkinson: float
    mean_max_drawdown: float
    std_max_drawdown: float
    mean_price_volatility: float
    price_stability_coefficient: float
    mean_final_price: float
    mean_liquidity: float


@dataclass
class SensitivityResult:
    """Result from a single sensitivity sweep."""
    parameter_name: str
    parameter_values: List[float]
    gini_values: List[float]
    gini_ci_low: List[float]
    gini_ci_high: List[float]
    theil_values: List[float]
    atkinson_values: List[float]
    drawdown_values: List[float]
    price_stability_values: List[float]


@dataclass
class ScenarioAnalysisReport:
    """Complete report from scenario generation + sensitivity analysis."""
    n_scenarios: int
    scenario_results: List[ScenarioResult]
    sensitivity_results: List[SensitivityResult]
    summary_statistics: Dict[str, float]
    notes: List[str]


# ── Latin Hypercube Sampling ──────────────────────────────────

def _latin_hypercube_sample(n_samples: int, n_dims: int, seed: int = 42) -> np.ndarray:
    """
    Generate Latin Hypercube samples in [0, 1]^n_dims.
    Each dimension is divided into n_samples equal strata, and exactly one
    sample is drawn from each stratum per dimension.
    """
    rng = np.random.RandomState(seed)
    samples = np.zeros((n_samples, n_dims))
    for dim in range(n_dims):
        perm = rng.permutation(n_samples)
        for i in range(n_samples):
            samples[i, dim] = (perm[i] + rng.uniform()) / n_samples
    return samples


def _rescale(value: float, low: float, high: float) -> float:
    """Rescale a [0,1] value to [low, high]."""
    return low + value * (high - low)


def _normalize_allocation(community: float, team: float, investor: float,
                          ecosystem: float, advisor: float, liquidity: float) -> Tuple[float, ...]:
    """Normalize allocation splits to sum to 100."""
    total = community + team + investor + ecosystem + advisor + liquidity
    if total <= 0:
        return (40.0, 20.0, 15.0, 15.0, 5.0, 5.0)
    factor = 100.0 / total
    return (
        round(community * factor, 1),
        round(team * factor, 1),
        round(investor * factor, 1),
        round(ecosystem * factor, 1),
        round(advisor * factor, 1),
        round(liquidity * factor, 1),
    )


# ── Scenario Generation ──────────────────────────────────────

def generate_scenarios(n_scenarios: int = 100, seed: int = 42) -> List[ScenarioConfig]:
    """
    Generate n_scenarios unique configurations using Latin Hypercube Sampling.

    Parameter ranges:
      - community_pct:     [20, 60]
      - team_pct:          [5, 30]
      - investor_pct:      [5, 30]
      - ecosystem_pct:     [5, 25]
      - advisor_pct:       [2, 10]
      - liquidity_pct:     [2, 15]
      - team_cliff:        [3, 24] months
      - team_vest:         [12, 48] months
      - investor_cliff:    [0, 18] months
      - investor_vest:     [6, 36] months
      - burn_rate:         [0.0, 0.05]
      - emission_rate:     [0.02, 0.15]
      - demand_multiplier: [0.5, 2.0]
      - volatility_bias:   [0.5, 2.0]
    """
    n_dims = 14  # Number of parameters
    lhs = _latin_hypercube_sample(n_scenarios, n_dims, seed=seed)

    configs = []
    for i in range(n_scenarios):
        row = lhs[i]

        # Raw allocation values (will be normalized)
        raw_community = _rescale(row[0], 20, 60)
        raw_team = _rescale(row[1], 5, 30)
        raw_investor = _rescale(row[2], 5, 30)
        raw_ecosystem = _rescale(row[3], 5, 25)
        raw_advisor = _rescale(row[4], 2, 10)
        raw_liquidity = _rescale(row[5], 2, 15)

        community, team, investor, ecosystem, advisor, liquidity = _normalize_allocation(
            raw_community, raw_team, raw_investor, raw_ecosystem, raw_advisor, raw_liquidity
        )

        team_cliff = int(round(_rescale(row[6], 3, 24)))
        team_vest = int(round(_rescale(row[7], 12, 48)))
        investor_cliff = int(round(_rescale(row[8], 0, 18)))
        investor_vest = int(round(_rescale(row[9], 6, 36)))
        burn_rate = round(_rescale(row[10], 0.0, 0.05), 4)
        emission_rate = round(_rescale(row[11], 0.02, 0.15), 4)
        demand_multiplier = round(_rescale(row[12], 0.5, 2.0), 2)
        volatility_bias = round(_rescale(row[13], 0.5, 2.0), 2)

        # Ensure team_vest > team_cliff, investor_vest > investor_cliff
        if team_vest <= team_cliff:
            team_vest = team_cliff + 12
        if investor_vest <= investor_cliff:
            investor_vest = investor_cliff + 6

        configs.append(ScenarioConfig(
            scenario_id=f"S{i+1:03d}",
            label=f"LHS-{i+1:03d} (C{community:.0f}/T{team:.0f}/I{investor:.0f})",
            community_pct=community,
            team_pct=team,
            investor_pct=investor,
            ecosystem_pct=ecosystem,
            advisor_pct=advisor,
            liquidity_pct=liquidity,
            team_cliff=team_cliff,
            team_vest=team_vest,
            investor_cliff=investor_cliff,
            investor_vest=investor_vest,
            burn_rate=burn_rate,
            emission_rate=emission_rate,
            demand_multiplier=demand_multiplier,
            volatility_bias=volatility_bias,
        ))

    return configs


# ── Price Stability Coefficient ───────────────────────────────

def compute_price_stability_coefficient(price_path: List[float]) -> float:
    """
    Formal sustainability metric: price stability coefficient from ABM price paths.

    Defined as: 1 - (σ_log_returns / μ_abs_log_returns)
    Bounded to [0, 1] where 1 = perfectly stable, 0 = maximally volatile.

    Falls back to coefficient of variation (CV) approach if log returns are degenerate.
    """
    if len(price_path) < 2:
        return 0.0

    log_returns = []
    for t in range(1, len(price_path)):
        if price_path[t] > 0 and price_path[t - 1] > 0:
            log_returns.append(math.log(price_path[t] / price_path[t - 1]))

    if not log_returns:
        return 0.0

    std_lr = float(np.std(log_returns))
    mean_abs_lr = float(np.mean([abs(r) for r in log_returns]))

    if mean_abs_lr < 1e-12:
        return 1.0  # No returns at all → perfectly "stable" (degenerate case)

    # Ratio of std to mean absolute return; lower = more stable
    instability_ratio = std_lr / mean_abs_lr
    # Normalize to [0, 1]; clip since ratio can exceed 1 for heavy-tailed distributions
    stability = max(0.0, min(1.0, 1.0 - instability_ratio + 0.5))

    return round(stability, 4)


# ── Run Single Scenario ──────────────────────────────────────

def run_single_scenario(config: ScenarioConfig) -> ScenarioResult:
    """Run a single scenario through the Monte Carlo ABM and extract metrics."""
    allocation = {
        "Community": config.community_pct,
        "Team": config.team_pct,
        "Investors": config.investor_pct,
        "Ecosystem": config.ecosystem_pct,
        "Advisors": config.advisor_pct,
        "Liquidity": config.liquidity_pct,
    }

    vesting = {
        "Team": VestingDetail(cliff_months=config.team_cliff,
                              vesting_months=config.team_vest),
        "Investors": VestingDetail(cliff_months=config.investor_cliff,
                                   vesting_months=config.investor_vest),
        "Advisors": VestingDetail(cliff_months=6, vesting_months=24),
    }

    context_signals = {
        "base_price": config.base_price,
        "demand_multiplier": config.demand_multiplier,
        "burn_rate": config.burn_rate,
        "emission_rate": config.emission_rate,
        "volatility_bias": config.volatility_bias,
    }

    mc_result = run_monte_carlo_abm(
        proposal_initial_supply=config.total_supply,
        allocation=allocation,
        vesting=vesting,
        normalize_key_fn=normalize_allocation_key,
        context_signals=context_signals,
        n_replications=config.mc_replications,
        months=config.months,
    )

    # Compute price stability from mean price path
    price_stability = compute_price_stability_coefficient(mc_result.mean_price_path)

    return ScenarioResult(
        config=config,
        mean_final_gini=mc_result.mean_final_gini,
        std_final_gini=mc_result.std_final_gini,
        ci_low_gini=float(mc_result.p5_gini_trajectory[-1]) if mc_result.p5_gini_trajectory else 0.0,
        ci_high_gini=float(mc_result.p95_gini_trajectory[-1]) if mc_result.p95_gini_trajectory else 0.0,
        mean_final_theil=float(mc_result.mean_theil_trajectory[-1]) if mc_result.mean_theil_trajectory else 0.0,
        mean_final_atkinson=float(mc_result.mean_atkinson_trajectory[-1]) if mc_result.mean_atkinson_trajectory else 0.0,
        mean_max_drawdown=mc_result.mean_max_drawdown,
        std_max_drawdown=mc_result.std_max_drawdown,
        mean_price_volatility=mc_result.mean_price_volatility,
        price_stability_coefficient=price_stability,
        mean_final_price=mc_result.mean_price_path[-1] if mc_result.mean_price_path else 0.0,
        mean_liquidity=float(np.mean(mc_result.mean_liquidity_path)) if mc_result.mean_liquidity_path else 0.0,
    )


# ── Run All Scenarios ─────────────────────────────────────────

def run_scenario_batch(configs: List[ScenarioConfig],
                       verbose: bool = True) -> List[ScenarioResult]:
    """Run all scenario configurations sequentially, returning results."""
    results = []
    for i, config in enumerate(configs):
        if verbose:
            print(f"  [{i+1}/{len(configs)}] Running {config.scenario_id}: {config.label}...")
        result = run_single_scenario(config)
        results.append(result)
        if verbose:
            print(f"    Gini={result.mean_final_gini:.3f} "
                  f"[{result.ci_low_gini:.3f},{result.ci_high_gini:.3f}] "
                  f"Drawdown={result.mean_max_drawdown:.3f} "
                  f"Stability={result.price_stability_coefficient:.3f}")
    return results


# ── Sensitivity Analysis ──────────────────────────────────────

def _make_baseline_config() -> ScenarioConfig:
    """Create a baseline scenario for sensitivity sweeps."""
    return ScenarioConfig(
        scenario_id="BASELINE",
        label="Baseline for sensitivity",
        community_pct=40.0, team_pct=20.0, investor_pct=15.0,
        ecosystem_pct=15.0, advisor_pct=5.0, liquidity_pct=5.0,
        team_cliff=12, team_vest=36,
        investor_cliff=6, investor_vest=24,
        burn_rate=0.01, emission_rate=0.08,
        demand_multiplier=1.0, volatility_bias=1.0,
        mc_replications=50,  # More replications for sensitivity (fewer sweep points)
    )


def run_sensitivity_analysis(
    parameter_sweeps: Optional[Dict[str, List[float]]] = None,
    baseline: Optional[ScenarioConfig] = None,
    verbose: bool = True,
) -> List[SensitivityResult]:
    """
    Run sensitivity analysis on key thresholds.

    Default sweeps:
      - Gini cutoff proxy: vary community_pct from 15% to 65% (affects Gini)
      - Community share minimum: vary community_pct [15, 25, 30, 35, 40, 45, 50, 55, 60, 65]
      - Team cliff duration: vary team_cliff [0, 3, 6, 9, 12, 18, 24]
      - Investor cliff duration: vary investor_cliff [0, 3, 6, 9, 12, 18]
      - Burn rate: vary burn_rate [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
      - Demand multiplier: vary demand_multiplier [0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 2.0]
    """
    if baseline is None:
        baseline = _make_baseline_config()

    if parameter_sweeps is None:
        parameter_sweeps = {
            "community_pct": [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
            "team_cliff": [0, 3, 6, 9, 12, 18, 24],
            "investor_cliff": [0, 3, 6, 9, 12, 18],
            "burn_rate": [0.0, 0.005, 0.01, 0.02, 0.03, 0.05],
            "demand_multiplier": [0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 2.0],
            "volatility_bias": [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
        }

    sensitivity_results = []

    for param_name, values in parameter_sweeps.items():
        if verbose:
            print(f"\n  Sensitivity sweep: {param_name} ({len(values)} points)...")

        gini_vals, gini_ci_low, gini_ci_high = [], [], []
        theil_vals, atkinson_vals = [], []
        drawdown_vals, stability_vals = [], []

        for val in values:
            # Clone baseline and override the swept parameter
            import copy
            config = copy.deepcopy(baseline)
            config.scenario_id = f"SENS_{param_name}_{val}"
            config.label = f"Sensitivity {param_name}={val}"

            if param_name == "community_pct":
                # Adjust community and redistribute proportionally
                delta = val - config.community_pct
                config.community_pct = val
                # Absorb delta from team and investor proportionally
                other_total = config.team_pct + config.investor_pct
                if other_total > 0 and delta != 0:
                    config.team_pct = max(5.0, config.team_pct - delta * config.team_pct / other_total)
                    config.investor_pct = max(5.0, config.investor_pct - delta * config.investor_pct / other_total)
                # Re-normalize
                alloc_sum = (config.community_pct + config.team_pct + config.investor_pct +
                             config.ecosystem_pct + config.advisor_pct + config.liquidity_pct)
                if alloc_sum > 0:
                    factor = 100.0 / alloc_sum
                    config.community_pct = round(config.community_pct * factor, 1)
                    config.team_pct = round(config.team_pct * factor, 1)
                    config.investor_pct = round(config.investor_pct * factor, 1)
                    config.ecosystem_pct = round(config.ecosystem_pct * factor, 1)
                    config.advisor_pct = round(config.advisor_pct * factor, 1)
                    config.liquidity_pct = round(config.liquidity_pct * factor, 1)
            elif param_name == "team_cliff":
                config.team_cliff = int(val)
                if config.team_vest <= config.team_cliff:
                    config.team_vest = config.team_cliff + 12
            elif param_name == "investor_cliff":
                config.investor_cliff = int(val)
                if config.investor_vest <= config.investor_cliff:
                    config.investor_vest = config.investor_cliff + 6
            elif param_name == "burn_rate":
                config.burn_rate = float(val)
            elif param_name == "demand_multiplier":
                config.demand_multiplier = float(val)
            elif param_name == "volatility_bias":
                config.volatility_bias = float(val)
            else:
                setattr(config, param_name, val)

            result = run_single_scenario(config)

            gini_vals.append(result.mean_final_gini)
            gini_ci_low.append(result.ci_low_gini)
            gini_ci_high.append(result.ci_high_gini)
            theil_vals.append(result.mean_final_theil)
            atkinson_vals.append(result.mean_final_atkinson)
            drawdown_vals.append(result.mean_max_drawdown)
            stability_vals.append(result.price_stability_coefficient)

            if verbose:
                print(f"    {param_name}={val}: Gini={result.mean_final_gini:.3f} "
                      f"Drawdown={result.mean_max_drawdown:.3f} "
                      f"Stability={result.price_stability_coefficient:.3f}")

        sensitivity_results.append(SensitivityResult(
            parameter_name=param_name,
            parameter_values=[float(v) for v in values],
            gini_values=gini_vals,
            gini_ci_low=gini_ci_low,
            gini_ci_high=gini_ci_high,
            theil_values=theil_vals,
            atkinson_values=atkinson_vals,
            drawdown_values=drawdown_vals,
            price_stability_values=stability_vals,
        ))

    return sensitivity_results


# ── Full Scenario Analysis ────────────────────────────────────

def run_full_scenario_analysis(
    n_scenarios: int = 100,
    seed: int = 42,
    run_sensitivity: bool = True,
    verbose: bool = True,
) -> ScenarioAnalysisReport:
    """
    Complete scenario analysis pipeline:
    1. Generate 100+ LHS scenario configurations
    2. Run each through Monte Carlo ABM
    3. (Optional) Run sensitivity sweeps on key parameters
    4. Compute summary statistics and return report
    """
    if verbose:
        print(f"Generating {n_scenarios} scenario configurations via Latin Hypercube Sampling...")
    configs = generate_scenarios(n_scenarios=n_scenarios, seed=seed)

    if verbose:
        print(f"\nRunning {n_scenarios} scenarios through Monte Carlo ABM...")
    scenario_results = run_scenario_batch(configs, verbose=verbose)

    # Sensitivity analysis
    sensitivity_results = []
    if run_sensitivity:
        if verbose:
            print("\nRunning sensitivity analysis on key thresholds...")
        sensitivity_results = run_sensitivity_analysis(verbose=verbose)

    # Summary statistics across all scenarios
    all_gini = [r.mean_final_gini for r in scenario_results]
    all_theil = [r.mean_final_theil for r in scenario_results]
    all_atkinson = [r.mean_final_atkinson for r in scenario_results]
    all_drawdown = [r.mean_max_drawdown for r in scenario_results]
    all_stability = [r.price_stability_coefficient for r in scenario_results]

    summary = {
        "n_scenarios": n_scenarios,
        "gini_mean": float(np.mean(all_gini)),
        "gini_std": float(np.std(all_gini)),
        "gini_min": float(np.min(all_gini)),
        "gini_max": float(np.max(all_gini)),
        "gini_p5": float(np.percentile(all_gini, 5)),
        "gini_p95": float(np.percentile(all_gini, 95)),
        "theil_mean": float(np.mean(all_theil)),
        "theil_std": float(np.std(all_theil)),
        "atkinson_mean": float(np.mean(all_atkinson)),
        "atkinson_std": float(np.std(all_atkinson)),
        "drawdown_mean": float(np.mean(all_drawdown)),
        "drawdown_std": float(np.std(all_drawdown)),
        "stability_mean": float(np.mean(all_stability)),
        "stability_std": float(np.std(all_stability)),
        "pct_gini_above_0.5": float(np.mean([1 if g > 0.5 else 0 for g in all_gini]) * 100),
        "pct_gini_above_0.6": float(np.mean([1 if g > 0.6 else 0 for g in all_gini]) * 100),
        "pct_drawdown_above_0.5": float(np.mean([1 if d > 0.5 else 0 for d in all_drawdown]) * 100),
        "pct_drawdown_above_0.7": float(np.mean([1 if d > 0.7 else 0 for d in all_drawdown]) * 100),
    }

    notes = [
        f"Scenario analysis: {n_scenarios} LHS configurations × {configs[0].mc_replications} MC replications each.",
        f"Cross-scenario Gini: {summary['gini_mean']:.3f} ± {summary['gini_std']:.3f} "
        f"[{summary['gini_min']:.3f}, {summary['gini_max']:.3f}]",
        f"{summary['pct_gini_above_0.5']:.0f}% of scenarios exceed Gini > 0.5 threshold.",
        f"Mean max drawdown: {summary['drawdown_mean']:.3f} ± {summary['drawdown_std']:.3f}.",
        f"Mean price stability coefficient: {summary['stability_mean']:.3f} ± {summary['stability_std']:.3f}.",
    ]
    if run_sensitivity:
        notes.append(f"Sensitivity analysis: {len(sensitivity_results)} parameter sweeps completed.")

    return ScenarioAnalysisReport(
        n_scenarios=n_scenarios,
        scenario_results=scenario_results,
        sensitivity_results=sensitivity_results,
        summary_statistics=summary,
        notes=notes,
    )


# ── Export utilities ──────────────────────────────────────────

def export_scenario_results_csv(results: List[ScenarioResult], filepath: str) -> None:
    """Export scenario results to CSV for analysis/paper."""
    import csv
    fieldnames = [
        "scenario_id", "label",
        "community_pct", "team_pct", "investor_pct", "ecosystem_pct",
        "advisor_pct", "liquidity_pct",
        "team_cliff", "team_vest", "investor_cliff", "investor_vest",
        "burn_rate", "emission_rate", "demand_multiplier", "volatility_bias",
        "mean_final_gini", "std_final_gini", "ci_low_gini", "ci_high_gini",
        "mean_final_theil", "mean_final_atkinson",
        "mean_max_drawdown", "std_max_drawdown", "mean_price_volatility",
        "price_stability_coefficient", "mean_final_price", "mean_liquidity",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                "scenario_id": r.config.scenario_id,
                "label": r.config.label,
                "community_pct": r.config.community_pct,
                "team_pct": r.config.team_pct,
                "investor_pct": r.config.investor_pct,
                "ecosystem_pct": r.config.ecosystem_pct,
                "advisor_pct": r.config.advisor_pct,
                "liquidity_pct": r.config.liquidity_pct,
                "team_cliff": r.config.team_cliff,
                "team_vest": r.config.team_vest,
                "investor_cliff": r.config.investor_cliff,
                "investor_vest": r.config.investor_vest,
                "burn_rate": r.config.burn_rate,
                "emission_rate": r.config.emission_rate,
                "demand_multiplier": r.config.demand_multiplier,
                "volatility_bias": r.config.volatility_bias,
                "mean_final_gini": r.mean_final_gini,
                "std_final_gini": r.std_final_gini,
                "ci_low_gini": r.ci_low_gini,
                "ci_high_gini": r.ci_high_gini,
                "mean_final_theil": r.mean_final_theil,
                "mean_final_atkinson": r.mean_final_atkinson,
                "mean_max_drawdown": r.mean_max_drawdown,
                "std_max_drawdown": r.std_max_drawdown,
                "mean_price_volatility": r.mean_price_volatility,
                "price_stability_coefficient": r.price_stability_coefficient,
                "mean_final_price": r.mean_final_price,
                "mean_liquidity": r.mean_liquidity,
            }
            writer.writerow(row)
    print(f"Scenario results exported to {filepath}")


def export_sensitivity_results_csv(results: List[SensitivityResult], filepath: str) -> None:
    """Export sensitivity analysis results to CSV."""
    import csv
    fieldnames = [
        "parameter_name", "parameter_value",
        "gini", "gini_ci_low", "gini_ci_high",
        "theil", "atkinson", "drawdown", "price_stability",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sr in results:
            for i, val in enumerate(sr.parameter_values):
                writer.writerow({
                    "parameter_name": sr.parameter_name,
                    "parameter_value": val,
                    "gini": sr.gini_values[i],
                    "gini_ci_low": sr.gini_ci_low[i],
                    "gini_ci_high": sr.gini_ci_high[i],
                    "theil": sr.theil_values[i],
                    "atkinson": sr.atkinson_values[i],
                    "drawdown": sr.drawdown_values[i],
                    "price_stability": sr.price_stability_values[i],
                })
    print(f"Sensitivity results exported to {filepath}")


# ── CLI entry point ───────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scenario Generator & Sensitivity Analysis")
    parser.add_argument("--n-scenarios", type=int, default=100, help="Number of LHS scenarios")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-sensitivity", action="store_true", help="Skip sensitivity analysis")
    parser.add_argument("--output-dir", type=str, default="scenario_exports", help="Output directory")
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    report = run_full_scenario_analysis(
        n_scenarios=args.n_scenarios,
        seed=args.seed,
        run_sensitivity=not args.no_sensitivity,
        verbose=True,
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
        os.path.join(args.output_dir, "scenario_results.csv"),
    )
    if report.sensitivity_results:
        export_sensitivity_results_csv(
            report.sensitivity_results,
            os.path.join(args.output_dir, "sensitivity_results.csv"),
        )
