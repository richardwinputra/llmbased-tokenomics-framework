"""
Simulation and Evaluation Module (Section III.D).

Three components:
  1) Supply Release Simulation - Monthly release over 60-month horizon
     using cliff-and-linear vesting (equations from paper)
  2) Fairness Evaluation - Insider/distributed shares and Gini at checkpoints
  3) Sustainability Stress Testing - 5 scenarios (bull, neutral, bear,
     unlock shock, liquidity pressure)

"""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from models import (
    GeneratedTokenomics, ProjectContext,
    SupplyReleaseResult, FairnessSnapshot, FairnessEvaluationResult,
    StressScenarioResult, StressTestResult, SimulationReport,
)
from utils import (
    normalize_allocation_key, calculate_gini, get_initial_supply,
    is_insider_category, is_distributed_category,
    normalize_allocation_dict,
)


def simulate_supply_release(
    tokenomics: GeneratedTokenomics,
    horizon_months: int = 60,
) -> SupplyReleaseResult:
    """
    Model token release over time using cliff-and-linear vesting.

    For each category i:
      T_i = (p_i / 100) * S

    Circulating at month m:
      C_i(m) = 0                              if m < c_i
      C_i(m) = T_i * (m - c_i) / d_i          if c_i <= m < c_i + d_i
      C_i(m) = T_i                             if m >= c_i + d_i

    If no vesting, C_i(m) = T_i for all m >= 0.

    Total: C(m) = sum_i C_i(m)
    Locked: L(m) = S - C(m)
    """
    alloc = tokenomics.tokenomics_parameters.allocation
    vesting = tokenomics.tokenomics_parameters.vesting
    S = get_initial_supply(tokenomics)

    months = list(range(horizon_months + 1))
    category_circulating: Dict[str, List[float]] = {}
    circulating_supply: List[float] = []
    locked_supply: List[float] = []


    token_amounts = {}
    for cat, pct in alloc.items():
        token_amounts[cat] = (pct / 100.0) * S


    for cat, T_i in token_amounts.items():
        detail = vesting.get(cat)
        cat_circ = []

        for m in months:
            if not detail or (detail.cliff_months == 0 and detail.vesting_months == 0):

                cat_circ.append(T_i)
            else:
                c_i = detail.cliff_months
                d_i = detail.vesting_months
                if m < c_i:
                    cat_circ.append(0.0)
                elif d_i <= 0:

                    cat_circ.append(T_i)
                else:
                    progress = min((m - c_i) / d_i, 1.0)
                    cat_circ.append(T_i * progress)

        category_circulating[cat] = cat_circ


    for idx in range(len(months)):
        total_circ = sum(
            category_circulating[cat][idx] for cat in category_circulating
        )
        circulating_supply.append(total_circ)
        locked_supply.append(S - total_circ)


    initial_circ_pct = (circulating_supply[0] / S * 100) if S > 0 else 0.0
    year1_circ_pct = (circulating_supply[min(12, horizon_months)] / S * 100) if S > 0 else 0.0
    year2_circ_pct = (circulating_supply[min(24, horizon_months)] / S * 100) if S > 0 else 0.0


    full_unlock_month = None
    for m in months:
        if circulating_supply[m] >= 0.99 * S:
            full_unlock_month = m
            break


    year1_inflation = None
    c0 = circulating_supply[0]
    c12 = circulating_supply[min(12, horizon_months)]
    if c0 > 0:
        year1_inflation = ((c12 - c0) / c0) * 100
    elif c12 > 0:

        year1_inflation = 999.0

    return SupplyReleaseResult(
        months=months,
        circulating_supply=circulating_supply,
        locked_supply=locked_supply,
        category_circulating=category_circulating,
        initial_circulating_pct=initial_circ_pct,
        year1_circulating_pct=year1_circ_pct,
        year2_circulating_pct=year2_circ_pct,
        full_unlock_month=full_unlock_month,
        year1_inflation_proxy=year1_inflation,
    )


def evaluate_fairness(
    supply_release: SupplyReleaseResult,
    tokenomics: GeneratedTokenomics,
) -> FairnessEvaluationResult:
    """
    Assess fairness drift over checkpoints: month 0, 12, 24, and full unlock.

    I(m) = C_insider(m) / C(m)
    D(m) = C_distributed(m) / C(m)
    Gini(m) = Gini(C_1(m), C_2(m), ..., C_k(m))
    """
    alloc = tokenomics.tokenomics_parameters.allocation
    cat_circ = supply_release.category_circulating
    total_circ = supply_release.circulating_supply


    checkpoints = [0, 12, 24]
    full_unlock = supply_release.full_unlock_month
    if full_unlock is not None and full_unlock not in checkpoints:
        checkpoints.append(full_unlock)

    max_month = len(supply_release.months) - 1
    checkpoints = [m for m in checkpoints if m <= max_month]

    snapshots: List[FairnessSnapshot] = []

    for m in checkpoints:
        C_m = total_circ[m]
        if C_m <= 0:
            snapshots.append(FairnessSnapshot(
                month=m, insider_share=0.0, distributed_share=0.0, gini=0.0,
            ))
            continue


        insider_circ = sum(
            cat_circ[k][m] for k in cat_circ if is_insider_category(k)
        )
        distributed_circ = sum(
            cat_circ[k][m] for k in cat_circ if is_distributed_category(k)
        )

        I_m = insider_circ / C_m
        D_m = distributed_circ / C_m


        category_balances = [cat_circ[k][m] for k in cat_circ]
        gini_m = calculate_gini(category_balances)

        snapshots.append(FairnessSnapshot(
            month=m, insider_share=I_m, distributed_share=D_m, gini=gini_m,
        ))


    if len(snapshots) >= 2:
        insider_shares = [s.insider_share for s in snapshots]
        drift = max(insider_shares) - insider_shares[0]
    else:
        drift = 0.0

    return FairnessEvaluationResult(snapshots=snapshots, fairness_drift=drift)


def _compute_supply_demand_ratio(
    circulating_supply: List[float],
    growth_rate: float,
    horizon_months: int = 12,
) -> tuple[List[float], float]:
    """
    Compute Supply-Demand Ratio (SDR) over a given horizon.

    Args:
        circulating_supply: List of circulating supply at each month
        growth_rate: Monthly demand growth rate (e.g., 0.05 for 5%)
        horizon_months: Number of months to compute ratio over (default 12)

    Returns:
        (sdr_list, max_sdr): List of SDR values and maximum SDR over horizon
    """
    if not circulating_supply or circulating_supply[0] <= 0:
        return [], 0.0

    initial_supply = circulating_supply[0]
    demand = [initial_supply]
    sdr_list = [1.0]

    months_to_compute = min(horizon_months, len(circulating_supply) - 1)
    for m in range(1, months_to_compute + 1):

        demand_m = demand[m - 1] * (1.0 + growth_rate)
        demand.append(demand_m)


        sdr_m = circulating_supply[m] / demand_m if demand_m > 0 else float('inf')
        sdr_list.append(sdr_m)

    max_sdr = max(sdr_list) if sdr_list else 0.0
    return sdr_list, max_sdr


def _calculate_max_monthly_spike(
    circulating_supply: List[float],
) -> tuple[float, int]:
    """
    Calculate the largest month-over-month supply spike.

    Returns:
        (max_inflation, shock_month): Maximum monthly inflation % and the month it occurs
    """
    max_inflation = 0.0
    shock_month = 0

    for m in range(1, len(circulating_supply)):
        if circulating_supply[m - 1] > 0:
            inflation = (circulating_supply[m] - circulating_supply[m - 1]) / circulating_supply[m - 1]
            if inflation > max_inflation:
                max_inflation = inflation
                shock_month = m

    return max_inflation, shock_month


def run_stress_testing(
    tokenomics: GeneratedTokenomics,
    supply_release: SupplyReleaseResult,
    context: ProjectContext,
) -> StressTestResult:
    """
    Evaluate proposal under 5 predefined scenarios using data-driven analysis
    of simulated supply release curves.

    Enhancement A: Data-Driven Unlock Shock
      - Detects largest month-over-month supply spike
      - Fails if any single month dumps >15% new supply

    Enhancement B: Sell Pressure via Supply-Demand Ratio (SDR)
      - Models demand growth per scenario (Bull: +5%, Neutral: +1%, Bear: -2%)
      - Unlock Shock: -5% at shock month, then flat
      - Liquidity Pressure: +0.5% with 3x sell pressure multiplier
      - Applies modifiers for burn, reserves, vesting duration

    Enhancement C: No Free Passes
      - Bull/Neutral now use SDR framework instead of automatic pass
      - Bull fails if Year 1 inflation proxy > 200% (dilutive)
      - Neutral fails if cumulative supply outpaces demand > 50%

    Pass criterion: viable in >= 70% of scenarios.
    """
    alloc = tokenomics.tokenomics_parameters.allocation
    vesting = tokenomics.tokenomics_parameters.vesting
    S = get_initial_supply(tokenomics)

    circulating_supply = supply_release.circulating_supply


    reserve_pct = sum(
        v for k, v in alloc.items()
        if normalize_allocation_key(k) in {"Ecosystem", "Reserve"}
    )
    has_burn = tokenomics.tokenomics_parameters.burn is not None

    avg_insider_vesting = 0
    insider_vest_count = 0
    for k in alloc:
        if is_insider_category(k) and k in vesting:
            avg_insider_vesting += vesting[k].vesting_months
            insider_vest_count += 1
    if insider_vest_count > 0:
        avg_insider_vesting /= insider_vest_count


    max_monthly_spike, shock_month = _calculate_max_monthly_spike(circulating_supply)


    c0 = circulating_supply[0]
    c12 = circulating_supply[min(12, len(circulating_supply) - 1)]
    year1_inflation = supply_release.year1_inflation_proxy
    if year1_inflation is None:
        year1_inflation = ((c12 - c0) / max(c0, 1) * 100) if c0 > 0 else 0.0

    scenarios: List[StressScenarioResult] = []


    _, max_sdr_bull = _compute_supply_demand_ratio(
        circulating_supply, growth_rate=0.05, horizon_months=12
    )

    bull_sdr_threshold = 1.5

    if reserve_pct > 10.0:
        bull_sdr_threshold += 0.1
    if avg_insider_vesting > 24:
        bull_sdr_threshold += 0.1

    bull_sdr_viable = max_sdr_bull <= bull_sdr_threshold
    bull_dilution_viable = year1_inflation <= 200.0
    bull_viable = bull_sdr_viable and bull_dilution_viable

    bull_sufficient = reserve_pct >= 5.0 or has_burn
    bull_recovery = None
    bull_notes = ["Sustained demand growth (+5%/month), high market confidence."]

    if not bull_dilution_viable:
        bull_notes.append(
            f"Year 1 inflation proxy {year1_inflation:.0f}% exceeds 200%; "
            "supply unlocks too aggressive—dilutive even in bull conditions."
        )

    if not bull_sdr_viable:
        bull_notes.append(
            f"Max SDR {max_sdr_bull:.2f} exceeds threshold {bull_sdr_threshold:.2f}; "
            "supply outpaces demand despite growth."
        )
        bull_recovery = 3

    scenarios.append(StressScenarioResult(
        name="Bull", description="High demand with sustained market growth.",
        viable=bull_viable, sufficient_reserves=bull_sufficient,
        recovery_months=bull_recovery, notes=bull_notes,
    ))


    _, max_sdr_neutral = _compute_supply_demand_ratio(
        circulating_supply, growth_rate=0.01, horizon_months=12
    )


    cumulative_supply = sum(circulating_supply[:13]) if len(circulating_supply) > 12 else sum(circulating_supply)
    cumulative_demand = (
        c0 * sum((1.01 ** m) for m in range(13)) if c0 > 0 else 0
    )
    cumulative_outpace = (cumulative_supply - cumulative_demand) / max(cumulative_demand, 1)

    neutral_sdr_threshold = 1.5
    if reserve_pct > 10.0:
        neutral_sdr_threshold += 0.1
    if avg_insider_vesting > 24:
        neutral_sdr_threshold += 0.1

    neutral_sdr_viable = max_sdr_neutral <= neutral_sdr_threshold
    neutral_cumulative_viable = cumulative_outpace <= 0.50
    neutral_viable = neutral_sdr_viable and neutral_cumulative_viable

    neutral_sufficient = reserve_pct >= 10.0
    neutral_notes = ["Stable market conditions (+1%/month demand)."]

    if not neutral_cumulative_viable:
        neutral_notes.append(
            f"Cumulative supply outpaces demand by {cumulative_outpace*100:.1f}% over 12 months; "
            "not viable in stable conditions."
        )

    if not neutral_sdr_viable:
        neutral_notes.append(
            f"Max SDR {max_sdr_neutral:.2f} exceeds threshold {neutral_sdr_threshold:.2f}."
        )

    if not neutral_sufficient:
        neutral_notes.append(f"Reserve allocation {reserve_pct:.1f}% may be thin for sustained operations.")

    scenarios.append(StressScenarioResult(
        name="Neutral", description="Stable market with moderate participation.",
        viable=neutral_viable, sufficient_reserves=neutral_sufficient,
        recovery_months=None, notes=neutral_notes,
    ))


    _, max_sdr_bear = _compute_supply_demand_ratio(
        circulating_supply, growth_rate=-0.02, horizon_months=12
    )

    bear_sdr_threshold = 2.0
    if reserve_pct > 10.0:
        bear_sdr_threshold -= 0.1
    if avg_insider_vesting > 24:
        bear_sdr_threshold -= 0.1

    bear_viable = max_sdr_bear <= bear_sdr_threshold
    bear_sufficient = reserve_pct >= 15.0
    bear_recovery = None

    bear_notes = ["Declining demand (-2%/month), negative market sentiment."]
    if not bear_viable:
        bear_notes.append(
            f"Max SDR {max_sdr_bear:.2f} exceeds threshold {bear_sdr_threshold:.2f}; "
            "cannot sustain bear market downturn."
        )
        bear_recovery = 18
    else:
        bear_recovery = max(6, int(12 * (1 - reserve_pct / 30)))

    if avg_insider_vesting < 24:
        bear_notes.append(
            f"Short insider vesting ({avg_insider_vesting:.0f}m) amplifies sell pressure in bear market."
        )

    scenarios.append(StressScenarioResult(
        name="Bear", description="Sustained demand decline and negative sentiment.",
        viable=bear_viable, sufficient_reserves=bear_sufficient,
        recovery_months=bear_recovery, notes=bear_notes,
    ))


    sdr_shock = [1.0]
    demand_shock = [c0] if c0 > 0 else [1.0]

    for m in range(1, min(len(circulating_supply), 13)):
        if m == shock_month:

            demand_m = demand_shock[m - 1] * 0.95
        else:

            demand_m = demand_shock[m - 1]
        demand_shock.append(demand_m)

        sdr_m = circulating_supply[m] / demand_m if demand_m > 0 else float('inf')
        sdr_shock.append(sdr_m)

    max_sdr_shock = max(sdr_shock) if sdr_shock else 0.0
    shock_sdr_threshold = 2.0
    if reserve_pct > 10.0:
        shock_sdr_threshold -= 0.1


    shock_spike_viable = max_monthly_spike < 0.15
    shock_sdr_viable = max_sdr_shock <= shock_sdr_threshold
    shock_viable = shock_spike_viable and shock_sdr_viable

    shock_sufficient = reserve_pct >= 10.0
    shock_recovery = None

    shock_notes = [
        f"Largest month-over-month supply spike: {max_monthly_spike*100:.1f}% at month {shock_month}.",
    ]

    if not shock_spike_viable:
        shock_notes.append(
            f"Month {shock_month} spike of {max_monthly_spike*100:.1f}% exceeds 15% threshold; "
            "severe supply shock risk."
        )
        shock_recovery = 12

    if not shock_sdr_viable:
        shock_notes.append(
            f"Shock SDR {max_sdr_shock:.2f} exceeds threshold {shock_sdr_threshold:.2f}."
        )
        if shock_recovery is None:
            shock_recovery = 9

    if shock_viable and shock_recovery is None:
        shock_recovery = max(3, int(6 * max_monthly_spike / 0.10))

    scenarios.append(StressScenarioResult(
        name="Unlock Shock",
        description="Cliff expiry triggers concentrated token release.",
        viable=shock_viable, sufficient_reserves=shock_sufficient,
        recovery_months=shock_recovery, notes=shock_notes,
    ))


    liquidity_pct = sum(
        v for k, v in alloc.items()
        if normalize_allocation_key(k) in {"Liquidity", "Staking"}
    )


    _, max_sdr_liq = _compute_supply_demand_ratio(
        circulating_supply, growth_rate=0.005, horizon_months=12
    )


    max_sdr_liq_adjusted = max_sdr_liq * 3.0

    liq_sdr_threshold = 4.5
    if reserve_pct > 10.0:
        liq_sdr_threshold += 0.3
    if has_burn:
        liq_sdr_threshold += 0.45

    liq_sdr_viable = max_sdr_liq_adjusted <= liq_sdr_threshold
    liq_reserve_viable = liquidity_pct >= 5.0 and (has_burn or reserve_pct >= 10.0)
    liq_viable = liq_sdr_viable and liq_reserve_viable

    liq_sufficient = liquidity_pct >= 8.0
    liq_recovery = None

    liq_notes = [
        f"Dedicated liquidity allocation: {liquidity_pct:.1f}%. "
        f"Sell pressure 3x multiplier active (adjusted SDR: {max_sdr_liq_adjusted:.2f})."
    ]

    if not liq_sdr_viable:
        liq_notes.append(
            f"Adjusted SDR {max_sdr_liq_adjusted:.2f} exceeds threshold {liq_sdr_threshold:.2f}; "
            "insufficient market-making capacity."
        )
        liq_recovery = 12

    if not liq_reserve_viable:
        liq_notes.append(
            f"Insufficient liquidity ({liquidity_pct:.1f}%) and no burn/reserve buffer."
        )
        if liq_recovery is None:
            liq_recovery = 12
    elif liq_viable:
        liq_recovery = max(3, int(9 * (1 - liquidity_pct / 15)))

    scenarios.append(StressScenarioResult(
        name="Liquidity Pressure",
        description="Market liquidity dries up, widening spreads.",
        viable=liq_viable, sufficient_reserves=liq_sufficient,
        recovery_months=liq_recovery, notes=liq_notes,
    ))


    n_viable = sum(1 for s in scenarios if s.viable)
    pass_rate = n_viable / len(scenarios)

    return StressTestResult(
        scenarios=scenarios,
        pass_rate=pass_rate,
        passed=pass_rate >= 0.70,
    )


def generate_recommendations(
    fairness_eval: FairnessEvaluationResult,
    stress_test: StressTestResult,
    supply_release: SupplyReleaseResult,
) -> List[str]:
    """
    Generate actionable recommendations based on evaluation results.
    References data-driven findings from stress testing (e.g., specific shock months,
    supply-demand ratio breaches) rather than generic observations.
    """
    recommendations = []


    if fairness_eval.fairness_drift > 0.15:
        recommendations.append(
            f"Fairness drift of {fairness_eval.fairness_drift:.2f} detected. "
            "Consider staggering insider unlocks or adding community distribution events."
        )


    for snap in fairness_eval.snapshots:
        if snap.gini > 0.60:
            recommendations.append(
                f"Gini coefficient {snap.gini:.3f} at month {snap.month} exceeds 0.60. "
                "Consider broader token distribution mechanisms."
            )
            break


    failed = [s for s in stress_test.scenarios if not s.viable]
    if failed:
        names = ", ".join(s.name for s in failed)
        recommendations.append(
            f"Failed stress scenarios: {names}. "
            "Strengthen reserves, add burn mechanisms, or extend vesting."
        )


        for scenario in failed:

            if "Unlock Shock" in scenario.name:
                for note in scenario.notes:
                    if "spike" in note.lower() and "month" in note.lower():
                        recommendations.append(
                            f"  • {note} "
                            "Mitigation: extend vesting cliff or reduce single allocation size."
                        )
                        break
            elif "Bear" in scenario.name:
                if scenario.recovery_months:
                    recommendations.append(
                        f"  • Bear scenario recovery requires {scenario.recovery_months} months. "
                        "Strengthen reserve buffer or implement token burn to reduce supply pressure."
                    )
            elif "Liquidity" in scenario.name:
                recommendations.append(
                    f"  • {scenario.name} scenario fails; increase dedicated liquidity allocation "
                    "or implement dynamic burn tied to volume thresholds."
                )


    if supply_release.year1_inflation_proxy and supply_release.year1_inflation_proxy > 200:
        recommendations.append(
            f"Year 1 inflation proxy {supply_release.year1_inflation_proxy:.0f}% is very high. "
            "Consider longer vesting schedules to reduce early dilution, or stagger cliff releases."
        )


    for scenario in stress_test.scenarios:
        if scenario.name == "Unlock Shock":
            for note in scenario.notes:
                if "spike" in note.lower() and "month" in note.lower() and "%" in note:
                    if not any("spike" in r.lower() for r in recommendations):
                        recommendations.append(
                            f"Supply release curve analysis: {note.split('.')[0]}. "
                            "Recommend analyzing vesting schedules to smooth unlock curve."
                        )
                    break

    if not recommendations:
        recommendations.append(
            "Design passes all stress tests. Maintain current allocation "
            "and document KPI triggers (SDR thresholds, max monthly inflation) for future monitoring."
        )

    return recommendations


def run_simulation_module(
    tokenomics: GeneratedTokenomics,
    context: ProjectContext,
    knowledge_base: List[Dict],
    dataset: Optional[List[Dict]] = None,
) -> SimulationReport:
    """
    Run the complete simulation and evaluation module.
    Integrates supply release, fairness evaluation, and stress testing.
    """

    supply_release = simulate_supply_release(tokenomics, horizon_months=60)


    fairness_eval = evaluate_fairness(supply_release, tokenomics)


    stress_test = run_stress_testing(tokenomics, supply_release, context)


    recommendations = generate_recommendations(
        fairness_eval, stress_test, supply_release,
    )

    return SimulationReport(
        supply_release=supply_release,
        fairness_evaluation=fairness_eval,
        stress_test=stress_test,
        recommendations=recommendations,
    )


