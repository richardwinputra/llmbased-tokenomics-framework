"""
Domain data structures for the tokenomics pipeline.
All dataclasses and type aliases live here.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Literal


# ── Supply types ──────────────────────────────────────────────

@dataclass
class FixedSupply:
    amount: int
    type: Literal["fixed"] = "fixed"

@dataclass
class CappedSupply:
    cap: int
    initial_supply: int
    type: Literal["capped"] = "capped"

@dataclass
class DynamicSupply:
    initial_supply: int
    emission_rate: float
    burn_rule: Optional[str]
    type: Literal["dynamic"] = "dynamic"

TokenSupply = Union[int, FixedSupply, CappedSupply, DynamicSupply]


# ── Core domain objects ───────────────────────────────────────

@dataclass
class VestingDetail:
    cliff_months: int
    vesting_months: int
    unlock_type: str = "linear"

@dataclass
class ProjectMetadata:
    project: str
    token: str
    category: Optional[str] = None

@dataclass
class TokenDesignThinking:
    purpose: str
    principles: List[str]
    positioning: str
    functions: List[str]
    stakeholders: List[str]
    economic_design: str
    legal_design: str
    tech_design: str
    power_structures: str
    team: Dict[str, str]

@dataclass
class TokenomicsParameters:
    total_supply: TokenSupply
    allocation: Dict[str, float]
    vesting: Dict[str, VestingDetail]
    emissions: Optional[str] = None
    burn: Optional[str] = None

@dataclass
class GeneratedTokenomics:
    project_metadata: ProjectMetadata
    token_design_thinking: TokenDesignThinking
    tokenomics_parameters: TokenomicsParameters
    references: Dict[str, List[str]]

@dataclass
class ProjectContext:
    description: str
    goals: List[str]
    priorities: List[str]
    constraints: Dict[str, float]
    legal_risk_tolerance: str
    economic_signals: Dict[str, float] = field(default_factory=dict)


# ── Fairness & Governance ─────────────────────────────────────

@dataclass
class FairnessMetrics:
    t0_gini: float
    gini_12m: float
    gini_24m: float
    governance_influence_index: float
    # Agent-level metrics (populated by ABM; None until ABM runs)
    agent_gini_final: Optional[float] = None
    agent_theil_final: Optional[float] = None
    agent_atkinson_final: Optional[float] = None
    agent_gini_ci_low: Optional[float] = None   # 5th percentile from MC
    agent_gini_ci_high: Optional[float] = None   # 95th percentile from MC

@dataclass
class GovernanceRiskAssessment:
    capture_risk_score: float
    voter_turnout_projection: float
    governance_influence_index: float
    notes: List[str]

@dataclass
class RealProjectValidationResult:
    reference_project: str
    similarity_score: float
    warnings: List[str]


# ── Control Layer ─────────────────────────────────────────────

@dataclass
class StakeholderFinding:
    stakeholder: str
    severity: str
    message: str

@dataclass
class ComplianceFinding:
    severity: str
    message: str

@dataclass
class FeasibilityFinding:
    severity: str
    message: str

@dataclass
class ControlLayerResult:
    aligned: bool
    requires_iteration: bool
    stakeholder_findings: List[StakeholderFinding]
    compliance_findings: List[ComplianceFinding]
    feasibility_findings: List[FeasibilityFinding]
    fairness_metrics: Optional[FairnessMetrics] = None
    governance_risk: Optional[GovernanceRiskAssessment] = None


# ── Filter Layer ──────────────────────────────────────────────

@dataclass
class FilterLayerResult:
    passed: bool
    adjusted_proposal: GeneratedTokenomics
    allocation_issues: List[str]
    vesting_issues: List[str]
    economic_issues: List[str]
    governance_issues: List[str]
    fairness_metrics: Optional[FairnessMetrics] = None
    governance_risk: Optional[GovernanceRiskAssessment] = None


# ── Simulation Layer ──────────────────────────────────────────

@dataclass
class ScenarioResult:
    name: str
    description: str
    supply_over_time: List[float]
    notes: List[str]

@dataclass
class GiniLayerResult:
    overall_gini: float
    circulating_gini: float
    governance_gini: float

@dataclass
class SupplyDynamicsResult:
    months: List[int]
    circulating_supply: List[float]
    locked_supply: List[float]
    burned_supply: List[float]

@dataclass
class BaselineComparisonResult:
    baseline_name: str
    similarities: List[str]
    differences: List[str]

@dataclass
class SimulationReport:
    scenarios: List[ScenarioResult]
    gini_layers: GiniLayerResult
    supply_dynamics: SupplyDynamicsResult
    baseline_comparison: Optional[BaselineComparisonResult]
    fairness_report: str
    validated_model_summary: str
    recommendations: List[str]
    fairness_metrics: FairnessMetrics
    governance_risk: GovernanceRiskAssessment
    real_project_validation: Optional[RealProjectValidationResult]
    agent_market_detail: Optional["MarketScenarioDetail"] = None
    monte_carlo_result: Optional["MonteCarloABMResult"] = None
