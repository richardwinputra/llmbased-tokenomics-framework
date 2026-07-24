"""Data structures for the tokenomics screening pipeline."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Literal


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


@dataclass
class VestingDetail:
    """Cliff-and-linear vesting rule for a single allocation category."""
    cliff_months: int
    vesting_months: int
    unlock_type: str = "linear"

@dataclass
class ProjectMetadata:
    """Project identifiers."""
    project: str
    token: str
    category: Optional[str] = None

@dataclass
class TokenDesignThinking:
    """Token Design Thinking fields."""
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
    """Tokenomics parameters (supply, allocation, vesting)."""
    total_supply: TokenSupply
    allocation: Dict[str, float]
    vesting: Dict[str, VestingDetail]
    emissions: Optional[str] = None
    burn: Optional[str] = None

@dataclass
class GeneratedTokenomics:
    """Complete LLM-generated tokenomics proposal."""
    project_metadata: ProjectMetadata
    token_design_thinking: TokenDesignThinking
    tokenomics_parameters: TokenomicsParameters
    references: Dict[str, List[str]]

@dataclass
class ProjectContext:
    """User-provided context and inferred constraints."""
    description: str
    goals: List[str]
    priorities: List[str]
    constraints: Dict[str, float]
    legal_risk_tolerance: str


@dataclass
class ControlFinding:
    """A single diagnostic finding from the control layer."""
    category: str
    check: str
    severity: str
    message: str

@dataclass
class ControlLayerResult:
    """Control layer output."""
    aligned: bool
    requires_iteration: bool
    findings: List[ControlFinding]
    insider_pct: float
    distributed_pct: float
    team_pct: float
    investor_pct: float
    gini_t0: float


@dataclass
class FilterCheck:
    """A single filter check result."""
    category: str
    rule: str
    passed: bool
    message: str

@dataclass
class FilterLayerResult:
    """Filter layer output."""
    passed: bool
    adjusted_proposal: GeneratedTokenomics
    checks: List[FilterCheck]


@dataclass
class SupplyReleaseResult:
    """Monthly supply release simulation over 60 months."""
    months: List[int]
    circulating_supply: List[float]
    locked_supply: List[float]
    category_circulating: Dict[str, List[float]]
    initial_circulating_pct: float
    year1_circulating_pct: float
    year2_circulating_pct: float
    full_unlock_month: Optional[int]
    year1_inflation_proxy: Optional[float]


@dataclass
class FairnessSnapshot:
    """Fairness metrics at a single checkpoint."""
    month: int
    insider_share: float
    distributed_share: float
    gini: float

@dataclass
class FairnessEvaluationResult:
    """Fairness trajectory across checkpoints (0, 12, 24, full unlock)."""
    snapshots: List[FairnessSnapshot]
    fairness_drift: float


@dataclass
class StressScenarioResult:
    """Result of a single stress scenario."""
    name: str
    description: str
    viable: bool
    sufficient_reserves: bool
    recovery_months: Optional[int]
    notes: List[str]

@dataclass
class StressTestResult:
    """Aggregated stress testing output."""
    scenarios: List[StressScenarioResult]
    pass_rate: float
    passed: bool


@dataclass
class SimulationReport:
    """Complete output of the simulation and evaluation module."""
    supply_release: SupplyReleaseResult
    fairness_evaluation: FairnessEvaluationResult
    stress_test: StressTestResult
    recommendations: List[str]

