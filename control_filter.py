"""
Control Layer and Filter Layer: validation, constraint checking, proposal adjustment.
"""

from dataclasses import replace
from typing import Dict, List, Optional

from models import (
    GeneratedTokenomics, ProjectContext, FairnessMetrics, GovernanceRiskAssessment,
    ControlLayerResult, FilterLayerResult, RealProjectValidationResult,
    StakeholderFinding, ComplianceFinding, FeasibilityFinding,
)
from utils import normalize_allocation_key, calculate_gini


# ── Fairness & Governance metrics ─────────────────────────────

def calculate_temporal_fairness(tokenomics: GeneratedTokenomics) -> FairnessMetrics:
    alloc = tokenomics.tokenomics_parameters.allocation
    vesting = tokenomics.tokenomics_parameters.vesting

    def get_vested_share(category: str, month: int) -> float:
        share = alloc.get(category, 0.0)
        detail = vesting.get(category)
        if not detail:
            return share * min(month / 24.0, 1.0)
        cliff = detail.cliff_months
        total_vest = detail.vesting_months
        if month < cliff:
            return 0.0
        if total_vest == 0:
            return share
        progress = (month - cliff) / total_vest
        return share * min(max(progress, 0.0), 1.0)

    t0 = list(alloc.values())
    gini_0 = calculate_gini(t0)
    t12_shares = [get_vested_share(k, 12) for k in alloc]
    gini_12 = calculate_gini(t12_shares)
    t24_shares = [get_vested_share(k, 24) for k in alloc]
    gini_24 = calculate_gini(t24_shares)

    insiders = 0.0
    community = 0.0
    for k, v in alloc.items():
        cat = normalize_allocation_key(k)
        if cat in ["Team", "Investors"]:
            insiders += v
        elif cat == "Community":
            community += v
    gov_index = min(insiders / max(community, 1.0), 5.0)

    return FairnessMetrics(t0_gini=gini_0, gini_12m=gini_12, gini_24m=gini_24,
                           governance_influence_index=gov_index)


def evaluate_governance_risk(tokenomics: GeneratedTokenomics,
                             fairness: FairnessMetrics) -> GovernanceRiskAssessment:
    notes = []
    capture_score = min(fairness.governance_influence_index * 0.6 + fairness.t0_gini, 5.0)
    if capture_score > 2.5:
        notes.append("High insider balance relative to community; risk of governance capture.")
    voter_turnout = max(0.1, 1 - fairness.t0_gini)
    notes.append(f"Projected voter turnout {voter_turnout*100:.1f}% based on initial distribution.")
    return GovernanceRiskAssessment(
        capture_risk_score=capture_score,
        voter_turnout_projection=voter_turnout,
        governance_influence_index=fairness.governance_influence_index,
        notes=notes,
    )


# ── Validation ────────────────────────────────────────────────

def validate_against_real_projects(tokenomics: GeneratedTokenomics,
                                   knowledge_base: List[Dict],
                                   dataset: Optional[List[Dict]] = None) -> Optional[RealProjectValidationResult]:
    if not knowledge_base:
        return None
    notes = []
    score = 5.0
    if not tokenomics.tokenomics_parameters.vesting:
        notes.append("No vesting schedule defined - high risk compared to standard projects.")
        score -= 2.0
    alloc = tokenomics.tokenomics_parameters.allocation
    community = sum(v for k, v in alloc.items() if normalize_allocation_key(k) == "Community")
    if community < 40:
        notes.append(f"Community allocation {community:.1f}% is low compared to decentralized baselines.")
        score -= 1.0
    return RealProjectValidationResult(reference_project="Aggregate Baseline",
                                       similarity_score=score, warnings=notes)


# ── Control Layer ─────────────────────────────────────────────

def run_control_layer(tokenomics: GeneratedTokenomics,
                      context: ProjectContext) -> ControlLayerResult:
    stakeholder_findings: List[StakeholderFinding] = []
    compliance_findings: List[ComplianceFinding] = []
    feasibility_findings: List[FeasibilityFinding] = []

    alloc = tokenomics.tokenomics_parameters.allocation
    community_share = 0.0
    team_share = 0.0
    investor_share = 0.0
    for k, v in alloc.items():
        cat = normalize_allocation_key(k)
        if cat == "Community":
            community_share += v
        elif cat == "Team":
            team_share += v
        elif cat == "Investors":
            investor_share += v

    min_community = context.constraints.get("min_community_share", 20.0)
    max_team = context.constraints.get("max_team_share", 35.0)
    max_investor = context.constraints.get("max_investor_share", 40.0)

    if community_share < min_community:
        stakeholder_findings.append(StakeholderFinding(
            stakeholder="Community", severity="warning",
            message=f"Community allocation {community_share:.1f}% falls below target {min_community:.1f}%.",
        ))
    if team_share > max_team:
        stakeholder_findings.append(StakeholderFinding(
            stakeholder="Team", severity="warning",
            message=f"Team allocation {team_share:.1f}% exceeds cap of {max_team:.1f}%.",
        ))
    if investor_share > max_investor:
        stakeholder_findings.append(StakeholderFinding(
            stakeholder="Investors",
            severity="critical" if investor_share > 50 else "warning",
            message=f"Investor allocation {investor_share:.1f}% dominates the cap table.",
        ))

    concentrated_share = team_share + investor_share
    if concentrated_share > 70:
        compliance_findings.append(ComplianceFinding(
            severity="critical" if concentrated_share > 80 else "warning",
            message=f"Combined insider share at {concentrated_share:.1f}% could draw regulatory scrutiny.",
        ))

    vesting = tokenomics.tokenomics_parameters.vesting
    for k in alloc.keys():
        detail = vesting.get(k)
        if detail:
            if detail.cliff_months and detail.vesting_months and detail.cliff_months > detail.vesting_months:
                feasibility_findings.append(FeasibilityFinding(
                    severity="critical",
                    message=f"{k} cliff ({detail.cliff_months}m) exceeds vesting ({detail.vesting_months}m).",
                ))
            cat = normalize_allocation_key(k)
            if cat in {"Team", "Investors"} and detail.vesting_months and detail.vesting_months < 12:
                feasibility_findings.append(FeasibilityFinding(
                    severity="warning",
                    message=f"{k} unlocks in under 12 months, raising governance-capture risk.",
                ))

    requires_iteration = any(f.severity == "critical" for f in
                             stakeholder_findings + compliance_findings + feasibility_findings)
    aligned = not (stakeholder_findings or compliance_findings or feasibility_findings)

    fairness_metrics = calculate_temporal_fairness(tokenomics)
    governance_risk = evaluate_governance_risk(tokenomics, fairness_metrics)

    return ControlLayerResult(
        aligned=aligned, requires_iteration=requires_iteration,
        stakeholder_findings=stakeholder_findings,
        compliance_findings=compliance_findings,
        feasibility_findings=feasibility_findings,
        fairness_metrics=fairness_metrics,
        governance_risk=governance_risk,
    )


# ── Filter Layer ──────────────────────────────────────────────

def run_filter_layer(tokenomics: GeneratedTokenomics,
                     context: ProjectContext) -> FilterLayerResult:
    from models import FixedSupply, DynamicSupply, CappedSupply

    allocation_issues: List[str] = []
    vesting_issues: List[str] = []
    economic_issues: List[str] = []
    governance_issues: List[str] = []

    alloc = tokenomics.tokenomics_parameters.allocation
    vesting = tokenomics.tokenomics_parameters.vesting
    total_allocation = sum(alloc.values())
    adjusted_alloc = alloc.copy()

    if total_allocation <= 0:
        allocation_issues.append("CRITICAL: Allocation total is zero; cannot normalize.")
    elif abs(total_allocation - 100) > 0.5:
        factor = 100 / total_allocation if total_allocation > 0 else 1.0
        allocation_issues.append(f"INFO: Allocation total was {total_allocation:.1f}%. Normalized to 100%.")
        adjusted_alloc = {k: round(v * factor, 2) for k, v in alloc.items()}

    community_share = 0.0
    team_share = 0.0
    investor_share = 0.0
    for k, v in adjusted_alloc.items():
        cat = normalize_allocation_key(k)
        if cat == "Community":
            community_share += v
        elif cat == "Team":
            team_share += v
        elif cat == "Investors":
            investor_share += v

    min_community = context.constraints.get("min_community_share", 20.0)
    max_team = context.constraints.get("max_team_share", 35.0)
    max_investor = context.constraints.get("max_investor_share", 40.0)

    if community_share < min_community:
        allocation_issues.append(f"CRITICAL: Community share {community_share:.1f}% < target {min_community:.1f}%.")
    if team_share > max_team:
        allocation_issues.append(f"CRITICAL: Team share {team_share:.1f}% > cap {max_team:.1f}%.")
    if investor_share > max_investor:
        allocation_issues.append(f"CRITICAL: Investor share {investor_share:.1f}% > cap {max_investor:.1f}%.")

    for k in adjusted_alloc.keys():
        detail = vesting.get(k)
        if detail:
            if detail.cliff_months > detail.vesting_months:
                vesting_issues.append(f"CRITICAL: {k} cliff > vesting.")
            cat = normalize_allocation_key(k)
            if cat in {"Team", "Investors"} and detail.vesting_months < 12:
                vesting_issues.append(f"WARNING: {k} vesting under 12 months.")

    supply = tokenomics.tokenomics_parameters.total_supply
    initial_supply = 0
    if isinstance(supply, int):
        initial_supply = supply
    elif isinstance(supply, (FixedSupply, DynamicSupply, CappedSupply)):
        initial_supply = supply.amount if isinstance(supply, FixedSupply) else supply.initial_supply
    if initial_supply <= 0:
        economic_issues.append("WARNING: Initial supply missing or non-positive.")

    if community_share - team_share < 0:
        governance_issues.append("WARNING: Insiders control more tokens than the community at T0.")

    critical_present = any(
        issue.startswith("CRITICAL")
        for issue in allocation_issues + vesting_issues + economic_issues + governance_issues
    )

    new_params = replace(tokenomics.tokenomics_parameters, allocation=adjusted_alloc)
    adjusted_proposal = replace(tokenomics, tokenomics_parameters=new_params)

    fairness_metrics = calculate_temporal_fairness(adjusted_proposal)
    governance_risk = evaluate_governance_risk(adjusted_proposal, fairness_metrics)

    return FilterLayerResult(
        passed=not critical_present,
        adjusted_proposal=adjusted_proposal,
        allocation_issues=allocation_issues,
        vesting_issues=vesting_issues,
        economic_issues=economic_issues,
        governance_issues=governance_issues,
        fairness_metrics=fairness_metrics,
        governance_risk=governance_risk,
    )
