"""
Output Control and Filter Module (Section III.C).

Control Layer (Table II): Diagnostic checks with severity levels.
  - Distributed allocation >= 20%
  - Team allocation <= 35%
  - Investor allocation <= 25% preferred; > 25% flagged; > 40% high risk
  - Combined insider allocation <= 40% preferred; > 40% flagged; > 60% high risk
  - Cliff <= vesting duration (vesting consistency)
  - Insider vesting >= 12 months
  - Gini <= 0.60

Filter Layer (Table III): Hard validity constraints (recalculate if failed).
  - 99.5% <= total allocation <= 100.5%
  - Each allocation >= 0
  - Total supply > 0
  - Insider and distributed groups identifiable
  - Each insider allocation has vesting defined
  - Initial circulating supply <= total supply
"""

from dataclasses import replace
from typing import Dict, List, Optional

from models import (
    GeneratedTokenomics,
    ControlLayerResult, ControlFinding,
    FilterLayerResult, FilterCheck,
)
from utils import (
    normalize_allocation_key, calculate_gini,
    is_insider_category, is_distributed_category,
    compute_insider_pct, compute_distributed_pct,
    compute_team_pct, compute_investor_pct,
)


def run_control_layer(tokenomics: GeneratedTokenomics) -> ControlLayerResult:
    """
    Diagnostic assessment per Table II.
    Returns findings with severity levels but does not block the pipeline.
    """
    findings: List[ControlFinding] = []
    alloc = tokenomics.tokenomics_parameters.allocation
    vesting = tokenomics.tokenomics_parameters.vesting


    distributed_pct = compute_distributed_pct(alloc)
    insider_pct = compute_insider_pct(alloc)
    team_pct = compute_team_pct(alloc)
    investor_pct = compute_investor_pct(alloc)


    if distributed_pct < 20.0:
        findings.append(ControlFinding(
            category="Distributed allocation",
            check="Minimum distribution floor",
            severity="warning",
            message=f"Distributed allocation {distributed_pct:.1f}% < 20% minimum floor. "
                    "Severely weak non-insider dispersion.",
        ))


    if team_pct > 35.0:
        findings.append(ControlFinding(
            category="Insider allocation",
            check="Team concentration",
            severity="warning",
            message=f"Team allocation {team_pct:.1f}% > 35% threshold. "
                    "Unusually high internal concentration.",
        ))


    if investor_pct > 40.0:
        findings.append(ControlFinding(
            category="Insider allocation",
            check="Investor concentration",
            severity="high_risk",
            message=f"Investor allocation {investor_pct:.1f}% > 40%. "
                    "High risk of concentrated external capital influence.",
        ))
    elif investor_pct > 25.0:
        findings.append(ControlFinding(
            category="Insider allocation",
            check="Investor concentration",
            severity="warning",
            message=f"Investor allocation {investor_pct:.1f}% > 25% preferred threshold. "
                    "Flagged for concentrated external capital influence.",
        ))


    if insider_pct > 60.0:
        findings.append(ControlFinding(
            category="Insider allocation",
            check="Combined insider concentration",
            severity="high_risk",
            message=f"Combined insider allocation {insider_pct:.1f}% > 60%. "
                    "High risk of governance concentration and coordinated market influence.",
        ))
    elif insider_pct > 40.0:
        findings.append(ControlFinding(
            category="Insider allocation",
            check="Combined insider concentration",
            severity="warning",
            message=f"Combined insider allocation {insider_pct:.1f}% > 40% preferred threshold. "
                    "Flagged for governance concentration.",
        ))


    for k in alloc.keys():
        detail = vesting.get(k)
        if detail and detail.cliff_months and detail.vesting_months:
            if detail.cliff_months > detail.vesting_months:
                findings.append(ControlFinding(
                    category="Release feasibility",
                    check="Vesting consistency",
                    severity="high_risk",
                    message=f"{k}: cliff ({detail.cliff_months}m) > vesting duration "
                            f"({detail.vesting_months}m). Logically incoherent schedule.",
                ))


    for k in alloc.keys():
        if is_insider_category(k):
            detail = vesting.get(k)
            if detail and detail.vesting_months and detail.vesting_months < 12:
                findings.append(ControlFinding(
                    category="Release feasibility",
                    check="Insider lockup horizon",
                    severity="warning",
                    message=f"{k} vesting ({detail.vesting_months}m) < 12 months. "
                            "Early synchronized insider unlock pressure.",
                ))


    gini_t0 = calculate_gini(list(alloc.values()))
    if gini_t0 > 0.60:
        findings.append(ControlFinding(
            category="Distributional inequality",
            check="Aggregate concentration",
            severity="warning",
            message=f"Initial Gini coefficient {gini_t0:.3f} > 0.60 threshold. "
                    "Concentration not fully captured by category thresholds.",
        ))

    requires_iteration = any(f.severity == "high_risk" for f in findings)
    aligned = len(findings) == 0

    return ControlLayerResult(
        aligned=aligned,
        requires_iteration=requires_iteration,
        findings=findings,
        insider_pct=insider_pct,
        distributed_pct=distributed_pct,
        team_pct=team_pct,
        investor_pct=investor_pct,
        gini_t0=gini_t0,
    )


def run_filter_layer(tokenomics: GeneratedTokenomics) -> FilterLayerResult:
    """
    Hard validity constraints per Table III.
    Designs that fail are recalculated (adjusted) before forwarding to simulation.
    """
    from models import FixedSupply, DynamicSupply, CappedSupply

    checks: List[FilterCheck] = []
    alloc = tokenomics.tokenomics_parameters.allocation
    vesting = tokenomics.tokenomics_parameters.vesting
    adjusted_alloc = alloc.copy()


    total_allocation = sum(alloc.values())
    if total_allocation <= 0:
        checks.append(FilterCheck(
            category="Allocation integrity",
            rule="Allocation completeness",
            passed=False,
            message="Allocation total is zero; cannot normalize.",
        ))
    elif abs(total_allocation - 100) > 0.5:
        factor = 100 / total_allocation
        adjusted_alloc = {k: round(v * factor, 2) for k, v in alloc.items()}
        checks.append(FilterCheck(
            category="Allocation integrity",
            rule="Allocation completeness",
            passed=False,
            message=f"Allocation total was {total_allocation:.1f}%. Normalized to 100%.",
        ))
    else:
        checks.append(FilterCheck(
            category="Allocation integrity",
            rule="Allocation completeness",
            passed=True,
            message=f"Allocation total {total_allocation:.1f}% within tolerance.",
        ))


    has_negative = any(v < 0 for v in adjusted_alloc.values())
    if has_negative:
        adjusted_alloc = {k: max(v, 0) for k, v in adjusted_alloc.items()}

        total = sum(adjusted_alloc.values())
        if total > 0:
            factor = 100 / total
            adjusted_alloc = {k: round(v * factor, 2) for k, v in adjusted_alloc.items()}
        checks.append(FilterCheck(
            category="Allocation validity",
            rule="No negative allocation share",
            passed=False,
            message="Negative allocation detected; clipped to zero and renormalized.",
        ))
    else:
        checks.append(FilterCheck(
            category="Allocation validity",
            rule="No negative allocation share",
            passed=True,
            message="All allocations non-negative.",
        ))


    supply = tokenomics.tokenomics_parameters.total_supply
    initial_supply = 0
    if isinstance(supply, int):
        initial_supply = supply
    elif isinstance(supply, (FixedSupply, DynamicSupply, CappedSupply)):
        initial_supply = supply.amount if isinstance(supply, FixedSupply) else supply.initial_supply

    if initial_supply > 0:
        checks.append(FilterCheck(
            category="Supply validity",
            rule="Positive total supply",
            passed=True,
            message=f"Total supply = {initial_supply:,}.",
        ))
    else:
        checks.append(FilterCheck(
            category="Supply validity",
            rule="Positive total supply",
            passed=False,
            message="Total supply missing or non-positive. Using default 1B.",
        ))


    insider_identified = any(is_insider_category(k) for k in adjusted_alloc.keys())
    distributed_identified = any(is_distributed_category(k) for k in adjusted_alloc.keys())
    groups_ok = insider_identified and distributed_identified

    if groups_ok:
        checks.append(FilterCheck(
            category="Category interpretability",
            rule="Required analytical groups identifiable",
            passed=True,
            message="Both insider and distributed groups identified.",
        ))
    else:
        checks.append(FilterCheck(
            category="Category interpretability",
            rule="Required analytical groups identifiable",
            passed=False,
            message=f"Insider identified: {insider_identified}, "
                    f"Distributed identified: {distributed_identified}.",
        ))


    insider_keys = [k for k in adjusted_alloc.keys() if is_insider_category(k)]
    # Normalize vesting keys for comparison — LLM may use raw names like "Core Team"
    # while vesting dict uses the same raw names but allocation lookup normalizes them.
    # Build a lookup: normalized_vesting_key -> True so we can match by canonical form.
    normalized_vesting_keys = {normalize_allocation_key(vk) for vk in vesting.keys()}

    all_insider_vesting_ok = True
    for k in insider_keys:
        normalized_k = normalize_allocation_key(k)
        # Accept vesting if found under the raw key OR the normalized key
        if k not in vesting and normalized_k not in normalized_vesting_keys:
            all_insider_vesting_ok = False

    if all_insider_vesting_ok and insider_keys:
        checks.append(FilterCheck(
            category="Release specification",
            rule="Insider vesting defined",
            passed=True,
            message="All insider allocations have vesting schedules defined.",
        ))
    elif not insider_keys:
        checks.append(FilterCheck(
            category="Release specification",
            rule="Insider vesting defined",
            passed=True,
            message="No insider allocations present (groups set to 0).",
        ))
    else:
        missing = [
            k for k in insider_keys
            if k not in vesting and normalize_allocation_key(k) not in normalized_vesting_keys
        ]
        checks.append(FilterCheck(
            category="Release specification",
            rule="Insider vesting defined",
            passed=False,
            message=f"Missing vesting for insider categories: {', '.join(missing)}.",
        ))


    initial_circulating_pct = 0.0
    for k, v in adjusted_alloc.items():
        detail = vesting.get(k)
        # Mirror simulation logic: only treat as immediately circulating if there is
        # no vesting detail at all, or both cliff AND vesting duration are zero.
        # cliff_months == 0 alone means vesting starts immediately (not fully unlocked).
        if not detail or (detail.cliff_months == 0 and detail.vesting_months == 0):
            initial_circulating_pct += v

    if initial_circulating_pct <= 100.0:
        checks.append(FilterCheck(
            category="Circulation feasibility",
            rule="Initial unlocked supply feasible",
            passed=True,
            message=f"Initial circulating supply {initial_circulating_pct:.1f}% <= 100%.",
        ))
    else:
        checks.append(FilterCheck(
            category="Circulation feasibility",
            rule="Initial unlocked supply feasible",
            passed=False,
            message=f"Initial circulating supply {initial_circulating_pct:.1f}% > total supply.",
        ))


    all_passed = all(c.passed for c in checks)
    new_params = replace(tokenomics.tokenomics_parameters, allocation=adjusted_alloc)
    adjusted_proposal = replace(tokenomics, tokenomics_parameters=new_params)

    return FilterLayerResult(
        passed=all_passed,
        adjusted_proposal=adjusted_proposal,
        checks=checks,
    )

