"""
Domain data structures for the LLM-based tokenomics screening framework.

Implements the data model described in the paper:
  - Project metadata and token design thinking (Table I)
  - Tokenomics parameters (supply, allocation, vesting)
  - Control layer results (Table II)
  - Filter layer results (Table III)
  - Simulation and evaluation results (supply release, fairness, stress testing)
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


# ── Core domain objects (Table I) ────────────────────────────

@dataclass
class VestingDetail:
    """Cliff-and-linear vesting rule for a single allocation category."""
    cliff_months: int          # c_i in the paper
    vesting_months: int        # d_i in the paper
    unlock_type: str = "linear"

@dataclass
class ProjectMetadata:
    """Project Identifiers (Table I, row 1)."""
    project: str
    token: str
    category: Optional[str] = None

@dataclass
class TokenDesignThinking:
    """Token Design Thinking fields (Table I, rows 2-5)."""
    purpose: str               # Purpose and Principles
    principles: List[str]
    positioning: str
    functions: List[str]       # Functional Design
    stakeholders: List[str]
    economic_design: str       # System Design
    legal_design: str
    tech_design: str
    power_structures: str      # Governance and Power
    team: Dict[str, str]

@dataclass
class TokenomicsParameters:
    """Tokenomics Parameters (Table I, rows 6-7)."""
    total_supply: TokenSupply               # S in the paper
    allocation: Dict[str, float]            # p_i percentages
    vesting: Dict[str, VestingDetail]       # c_i, d_i per category
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
    economic_signals: Dict[str, float] = field(default_factory=dict)


# ── Control Layer (Table II) ─────────────────────────────────

@dataclass
class ControlFinding:
    """A single diagnostic finding from the control layer."""
    category: str       # Analytical category from Table II
    check: str          # Control check name
    severity: str       # "info", "warning", "high_risk"
    message: str

@dataclass
class ControlLayerResult:
    """Output of the control layer (Table II checks)."""
    aligned: bool
    requires_iteration: bool
    findings: List[ControlFinding]
    insider_pct: float
    distributed_pct: float
    team_pct: float
    investor_pct: float
    gini_t0: float


# ── Filter Layer (Table III) ─────────────────────────────────

@dataclass
class FilterCheck:
    """A single filter check result (Table III)."""
    category: str       # Validation category
    rule: str           # Filter rule name
    passed: bool
    message: str

@dataclass
class FilterLayerResult:
    """Output of the filter layer (Table III checks)."""
    passed: bool
    adjusted_proposal: GeneratedTokenomics
    checks: List[FilterCheck]


# ── Simulation: Supply Release ───────────────────────────────

@dataclass
class SupplyReleaseResult:
    """Monthly supply release simulation over 60 months."""
    months: List[int]                          # [0, 1, 2, ..., 60]
    circulating_supply: List[float]            # C(m) total
    locked_supply: List[float]                 # L(m) = S - C(m)
    category_circulating: Dict[str, List[float]]  # C_i(m) per category
    initial_circulating_pct: float             # C(0) / S * 100
    year1_circulating_pct: float               # C(12) / S * 100
    year2_circulating_pct: float               # C(24) / S * 100
    full_unlock_month: Optional[int]           # month where C(m) >= 0.99 * S
    year1_inflation_proxy: Optional[float]     # (C(12) - C(0)) / C(0) * 100


# ── Simulation: Fairness Evaluation ──────────────────────────

@dataclass
class FairnessSnapshot:
    """Fairness metrics at a single checkpoint."""
    month: int
    insider_share: float        # I(m) = C_insider(m) / C(m)
    distributed_share: float    # D(m) = C_distributed(m) / C(m)
    gini: float                 # Gini(C_1(m), ..., C_k(m))

@dataclass
class FairnessEvaluationResult:
    """Fairness trajectory across checkpoints (0, 12, 24, full unlock)."""
    snapshots: List[FairnessSnapshot]
    fairness_drift: float       # max insider share - initial insider share


# ── Simulation: Sustainability Stress Testing ────────────────

@dataclass
class StressScenarioResult:
    """Result of a single stress scenario."""
    name: str                   # bull, neutral, bear, unlock_shock, liquidity_pressure
    description: str
    viable: bool                # Remains above minimum viability threshold
    sufficient_reserves: bool   # Circulating + reserve conditions met
    recovery_months: Optional[int]  # Months to return to 80% pre-shock level
    notes: List[str]

@dataclass
class StressTestResult:
    """Aggregated stress testing output."""
    scenarios: List[StressScenarioResult]
    pass_rate: float            # Fraction of scenarios passed
    passed: bool                # pass_rate >= 0.70


# ── Full Simulation Report ───────────────────────────────────

@dataclass
class SimulationReport:
    """Complete output of the simulation and evaluation module."""
    supply_release: SupplyReleaseResult
    fairness_evaluation: FairnessEvaluationResult
    stress_test: StressTestResult
    recommendations: List[str]
