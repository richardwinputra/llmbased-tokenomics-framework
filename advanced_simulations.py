"""
Agent-Based Market Simulation (ABM) for tokenomics evaluation.

Heterogeneous agents trade against an AMM (x·y=k) pool.
Agent types: RetailHolder, Speculator, Whale, VestingInsider, TreasuryAgent, LiquidityProvider.

Key improvements over v1:
  - Agent populations are parameterized from actual proposal allocation & vesting schedules
  - Agent-level Gini is computed at each timestep from token_balance distributions
  - Theil-T and Atkinson indices are computed as complementary fairness measures
  - Monte Carlo wrapper runs N replications and returns confidence intervals
  - VestingInsider agents respect cliff + linear vesting from the proposal
  - TreasuryAgent deploys ecosystem funds strategically
  - LiquidityProvider adds/removes liquidity based on spread conditions
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils import calculate_gini


# ── Data structures ───────────────────────────────────────────

@dataclass
class MarketScenarioDetail:
    """Single-run ABM output."""
    name: str
    price_path: List[float]
    liquidity_levels: List[float]
    sentiment_trend: List[float]
    gini_trajectory: List[float]       # Agent-level Gini at each month
    theil_trajectory: List[float]      # Theil-T index at each month
    atkinson_trajectory: List[float]   # Atkinson index (ε=0.5) at each month
    agent_token_snapshot: List[float]  # Final token balances of all agents
    notes: List[str]


@dataclass
class MonteCarloABMResult:
    """Aggregated result from N Monte Carlo replications of the ABM."""
    n_replications: int
    # Price statistics
    mean_price_path: List[float]
    std_price_path: List[float]
    p5_price_path: List[float]
    p95_price_path: List[float]
    # Agent-level Gini statistics
    mean_gini_trajectory: List[float]
    std_gini_trajectory: List[float]
    p5_gini_trajectory: List[float]
    p95_gini_trajectory: List[float]
    # Theil-T statistics
    mean_theil_trajectory: List[float]
    std_theil_trajectory: List[float]
    # Atkinson statistics
    mean_atkinson_trajectory: List[float]
    std_atkinson_trajectory: List[float]
    # Liquidity statistics
    mean_liquidity_path: List[float]
    std_liquidity_path: List[float]
    # Sentiment statistics
    mean_sentiment_path: List[float]
    # Summary scalars
    mean_final_gini: float
    std_final_gini: float
    mean_max_drawdown: float
    std_max_drawdown: float
    mean_price_volatility: float
    # Population info
    population_summary: str
    notes: List[str]
    # Formal sustainability metric; defaults added after required fields
    price_stability_coefficient: float = 0.0
    # Keep one representative single run for detailed inspection
    representative_run: Optional[MarketScenarioDetail] = None


# ── Fairness index computation ────────────────────────────────
# Gini is imported from utils.calculate_gini (single canonical implementation).
# Alias for backward compatibility within this module:
compute_gini = calculate_gini


def compute_theil_t(values: List[float]) -> float:
    """Theil-T index (GE(1)): entropy-based inequality measure."""
    arr = [max(v, 1e-12) for v in values]  # avoid log(0)
    n = len(arr)
    if n == 0:
        return 0.0
    mu = sum(arr) / n
    if mu <= 0:
        return 0.0
    return sum((v / mu) * math.log(v / mu) for v in arr) / n


def compute_atkinson(values: List[float], epsilon: float = 0.5) -> float:
    """Atkinson index with inequality aversion parameter ε."""
    arr = [max(v, 1e-12) for v in values]
    n = len(arr)
    if n == 0:
        return 0.0
    mu = sum(arr) / n
    if mu <= 0:
        return 0.0
    if epsilon == 1.0:
        geo_mean = math.exp(sum(math.log(v) for v in arr) / n)
        return 1.0 - geo_mean / mu
    power = 1.0 - epsilon
    generalized_mean = (sum(v ** power for v in arr) / n) ** (1.0 / power)
    return 1.0 - generalized_mean / mu


# ── AMM Market Environment ────────────────────────────────────

class MarketEnvironment:
    """Constant-product AMM pool (x·y=k) for price discovery."""

    def __init__(self, initial_price: float, initial_liquidity_depth: float,
                 total_supply: float = 0.0):
        # Scale pool to a realistic fraction of total supply (2-5% typical DEX depth).
        # If total_supply is provided, use it; otherwise fall back to the old constant.
        if total_supply > 0:
            pool_fraction = 0.03  # 3% of supply seeded into AMM
            base_tokens = total_supply * pool_fraction * initial_liquidity_depth
        else:
            base_tokens = initial_liquidity_depth * 100_000
        # Enforce minimum pool size to prevent division-by-zero
        base_tokens = max(base_tokens, 1000.0)
        initial_price = max(initial_price, 1e-9)
        self.base_pool = base_tokens
        self.quote_pool = base_tokens * initial_price
        self.k = self.quote_pool * self.base_pool
        self.price = initial_price

    def trade(self, side: str, amount_usd: float) -> float:
        if amount_usd <= 0:
            return self.price
        if side == "BUY":
            new_quote = self.quote_pool + amount_usd
            new_base = self.k / max(new_quote, 1e-12)
            self.quote_pool = new_quote
            self.base_pool = new_base
        elif side == "SELL":
            tokens_to_sell = amount_usd / max(self.price, 1e-12)
            new_base = self.base_pool + tokens_to_sell
            new_quote = self.k / max(new_base, 1e-12)
            self.base_pool = new_base
            self.quote_pool = new_quote
        self.price = self.quote_pool / max(self.base_pool, 1e-12)
        return self.price

    def get_liquidity(self) -> float:
        # Normalized liquidity: ratio of current pool depth to initial depth
        initial_base = math.sqrt(self.k / max(self.price, 1e-9))
        return self.base_pool / max(initial_base, 1e-9)


# ── Agent types ───────────────────────────────────────────────

class Agent:
    """Base agent class. All agents hold capital (USD) and tokens."""

    def __init__(self, capital: float, token_balance: float, label: str = "Agent"):
        self.capital = capital
        self.token_balance = token_balance
        self.label = label

    def step(self, market: MarketEnvironment, sentiment: float,
             demand_multiplier: float, month: int) -> float:
        """Return signed USD volume: +BUY, -SELL, 0=HOLD."""
        return 0.0


class RetailHolder(Agent):
    """Community retail: holds long-term, buys dips, panics on severe downturns."""

    def __init__(self, capital: float, token_balance: float):
        super().__init__(capital, token_balance, "Retail")

    def step(self, market, sentiment, demand_multiplier, month):
        if sentiment < 0.2:
            vol = -(self.token_balance * 0.5) * market.price
            self.token_balance *= 0.5
            return vol
        elif sentiment > 0.6:
            invest = self.capital * 0.1 * demand_multiplier
            if invest > 0:
                self.capital -= invest
                self.token_balance += invest / max(market.price, 1e-9)
                return invest
        return 0.0


class Speculator(Agent):
    """Momentum trader: buys aggressively on high sentiment, dumps on low."""

    def __init__(self, capital: float, token_balance: float):
        super().__init__(capital, token_balance, "Speculator")

    def step(self, market, sentiment, demand_multiplier, month):
        if sentiment > 0.55:
            invest = self.capital * 0.3 * np.random.rand() * demand_multiplier
            if invest > 0:
                self.capital -= invest
                self.token_balance += invest / max(market.price, 1e-9)
                return invest
        elif sentiment < 0.45:
            vol = -self.token_balance * market.price
            self.token_balance = 0
            return vol
        return 0.0


class Whale(Agent):
    """Large holder (non-vesting insider or early buyer): sells slowly over time."""

    def __init__(self, capital: float, token_balance: float):
        super().__init__(capital, token_balance, "Whale")

    def step(self, market, sentiment, demand_multiplier, month):
        sell_pct = 0.05 if sentiment > 0.4 else 0.01
        sell_amount = self.token_balance * sell_pct
        if sell_amount > 0:
            vol = -sell_amount * market.price
            self.token_balance -= sell_amount
            return vol
        return 0.0


class VestingInsider(Agent):
    """
    Team/Investor agent whose tokens are locked by a cliff + linear vesting schedule.
    Tokens unlock over time; once unlocked, sells a fraction each month.
    """

    def __init__(self, total_token_grant: float, cliff_months: int,
                 vesting_months: int, sell_rate: float = 0.10):
        super().__init__(capital=0.0, token_balance=0.0, label="VestingInsider")
        self.total_grant = total_token_grant
        self.cliff = cliff_months
        self.vesting = max(vesting_months, 1)
        self.sell_rate = sell_rate  # fraction of unlocked tokens sold per month
        self.cumulative_unlocked = 0.0

    def step(self, market, sentiment, demand_multiplier, month):
        # Calculate how many tokens have unlocked by this month
        if month < self.cliff:
            unlocked_total = 0.0
        elif self.vesting == 0:
            unlocked_total = self.total_grant
        else:
            progress = min((month - self.cliff) / self.vesting, 1.0)
            unlocked_total = self.total_grant * progress

        # New tokens unlocked this period
        newly_unlocked = max(unlocked_total - self.cumulative_unlocked, 0.0)
        self.cumulative_unlocked = unlocked_total
        self.token_balance += newly_unlocked

        # Sell a fraction of available tokens; reduce sell pressure in good sentiment
        actual_sell_rate = self.sell_rate * (1.2 if sentiment < 0.4 else 0.8)
        sell_amount = self.token_balance * actual_sell_rate
        if sell_amount > 0:
            vol = -sell_amount * market.price
            self.token_balance -= sell_amount
            self.capital += sell_amount * market.price
            return vol
        return 0.0


class TreasuryAgent(Agent):
    """
    Protocol treasury: holds ecosystem/reserve tokens, deploys for buybacks
    during bear markets and strategic liquidity during bull markets.
    """

    def __init__(self, token_balance: float, usd_reserve: float):
        super().__init__(capital=usd_reserve, token_balance=token_balance, label="Treasury")

    def step(self, market, sentiment, demand_multiplier, month):
        if sentiment < 0.3 and self.capital > 0:
            # Counter-cyclical buyback: spend 5% of USD reserve
            invest = self.capital * 0.05
            self.capital -= invest
            self.token_balance += invest / max(market.price, 1e-9)
            return invest
        elif sentiment > 0.7 and self.token_balance > 0:
            # Sell small amount to build reserve during euphoria
            sell_amount = self.token_balance * 0.02
            vol = -sell_amount * market.price
            self.token_balance -= sell_amount
            self.capital += sell_amount * market.price
            return vol
        return 0.0


class LiquidityProvider(Agent):
    """
    Liquidity provider: adds liquidity when spreads are wide (low liquidity),
    removes when over-concentrated. Earns implicit fees from the AMM.
    """

    def __init__(self, capital: float, token_balance: float):
        super().__init__(capital, token_balance, label="LP")

    def step(self, market, sentiment, demand_multiplier, month):
        liq = market.get_liquidity()
        if liq < 0.5 and self.capital > 0:
            # Add liquidity: buy tokens to balance the pool
            invest = self.capital * 0.15
            self.capital -= invest
            self.token_balance += invest / max(market.price, 1e-9)
            return invest
        elif liq > 2.0 and self.token_balance > 0:
            # Remove excess liquidity
            sell_amount = self.token_balance * 0.1
            vol = -sell_amount * market.price
            self.token_balance -= sell_amount
            self.capital += sell_amount * market.price
            return vol
        return 0.0


# ── Agent population factory ──────────────────────────────────

def build_agent_population(
    proposal_initial_supply: float,
    allocation: Dict[str, float],
    vesting: Dict[str, "VestingDetailLike"],
    normalize_key_fn,
    context_signals: Dict[str, float],
) -> List[Agent]:
    """
    Build a heterogeneous agent population parameterized from the actual proposal.

    allocation: {"Community": 40, "Team": 20, "Investors": 15, ...} (percentages)
    vesting: {"Team": VestingDetail(cliff=12, vesting=36), ...}

    Population mapping:
      - Community % → RetailHolder agents (50) + Speculator agents (20)
      - Team % → VestingInsider agents (proportional to team allocation)
      - Investors % → VestingInsider agents (shorter vesting) + Whale agents
      - Ecosystem/Treasury % → TreasuryAgent (1)
      - Advisors % → VestingInsider agents (small)
      - Liquidity % → LiquidityProvider agents (5)
    """
    agents: List[Agent] = []

    # Categorize allocation
    community_pct = 0.0
    team_pct = 0.0
    investor_pct = 0.0
    ecosystem_pct = 0.0
    advisor_pct = 0.0
    liquidity_pct = 0.0
    other_pct = 0.0

    for k, v in allocation.items():
        cat = normalize_key_fn(k)
        if cat == "Community":
            community_pct += v
        elif cat == "Team":
            team_pct += v
        elif cat == "Investors":
            investor_pct += v
        elif cat in ("Ecosystem", "Reserve"):
            ecosystem_pct += v
        elif cat == "Advisors":
            advisor_pct += v
        elif cat in ("Liquidity", "Staking"):
            liquidity_pct += v
        else:
            other_pct += v

    total_pct = community_pct + team_pct + investor_pct + ecosystem_pct + advisor_pct + liquidity_pct + other_pct
    if total_pct <= 0:
        total_pct = 100.0

    # Token amounts per category
    community_tokens = proposal_initial_supply * community_pct / 100
    team_tokens = proposal_initial_supply * team_pct / 100
    investor_tokens = proposal_initial_supply * investor_pct / 100
    ecosystem_tokens = proposal_initial_supply * ecosystem_pct / 100
    advisor_tokens = proposal_initial_supply * advisor_pct / 100
    liquidity_tokens = proposal_initial_supply * liquidity_pct / 100

    # ── Community → 50 Retail + 20 Speculators ────────────────
    n_retail = 50
    n_spec = 20
    retail_tokens_each = community_tokens * 0.6 / max(n_retail, 1)
    spec_tokens_each = community_tokens * 0.4 / max(n_spec, 1)
    base_price = context_signals.get("base_price", 1.0)

    for _ in range(n_retail):
        agents.append(RetailHolder(capital=retail_tokens_each * base_price * 0.5,
                                   token_balance=retail_tokens_each))
    for _ in range(n_spec):
        agents.append(Speculator(capital=spec_tokens_each * base_price * 2.0,
                                 token_balance=0))  # Specs start with USD, buy in

    # ── Team → VestingInsider agents ──────────────────────────
    n_team = max(1, int(team_pct / 5))  # ~1 agent per 5% allocation
    team_vest = _get_vesting(vesting, allocation, "Team", normalize_key_fn,
                             default_cliff=12, default_vest=36)
    for _ in range(n_team):
        agents.append(VestingInsider(
            total_token_grant=team_tokens / n_team,
            cliff_months=team_vest[0], vesting_months=team_vest[1],
            sell_rate=0.08,
        ))

    # ── Investors → VestingInsider + Whale ────────────────────
    n_investor_vesting = max(1, int(investor_pct / 5))
    n_investor_whale = max(1, int(investor_pct / 10))
    inv_vest = _get_vesting(vesting, allocation, "Investors", normalize_key_fn,
                            default_cliff=6, default_vest=24)
    vesting_token_share = investor_tokens * 0.7
    whale_token_share = investor_tokens * 0.3

    for _ in range(n_investor_vesting):
        agents.append(VestingInsider(
            total_token_grant=vesting_token_share / n_investor_vesting,
            cliff_months=inv_vest[0], vesting_months=inv_vest[1],
            sell_rate=0.12,  # Investors sell faster than team
        ))
    for _ in range(n_investor_whale):
        agents.append(Whale(capital=0, token_balance=whale_token_share / n_investor_whale))

    # ── Advisors → VestingInsider ─────────────────────────────
    if advisor_pct > 0:
        n_advisors = max(1, int(advisor_pct / 5))
        adv_vest = _get_vesting(vesting, allocation, "Advisors", normalize_key_fn,
                                default_cliff=6, default_vest=24)
        for _ in range(n_advisors):
            agents.append(VestingInsider(
                total_token_grant=advisor_tokens / n_advisors,
                cliff_months=adv_vest[0], vesting_months=adv_vest[1],
                sell_rate=0.10,
            ))

    # ── Ecosystem/Treasury → 1 TreasuryAgent ─────────────────
    if ecosystem_pct > 0:
        agents.append(TreasuryAgent(
            token_balance=ecosystem_tokens,
            usd_reserve=ecosystem_tokens * base_price * 0.2,
        ))

    # ── Liquidity → LiquidityProvider agents ──────────────────
    n_lp = max(1, min(5, int(liquidity_pct / 3)))
    for _ in range(n_lp):
        agents.append(LiquidityProvider(
            capital=liquidity_tokens * base_price * 0.5 / n_lp,
            token_balance=liquidity_tokens / n_lp,
        ))

    return agents


def _get_vesting(vesting_map, allocation, target_cat, normalize_key_fn,
                 default_cliff=6, default_vest=24) -> Tuple[int, int]:
    """Look up cliff/vesting months for a category from the vesting map."""
    for k in allocation.keys():
        if normalize_key_fn(k) == target_cat and k in vesting_map:
            v = vesting_map[k]
            return (v.cliff_months or default_cliff, v.vesting_months or default_vest)
    return (default_cliff, default_vest)


# ── Single ABM run ────────────────────────────────────────────

def run_agent_market_simulation(
    proposal_initial_supply: float,
    allocation: Dict[str, float],
    vesting: Dict,
    normalize_key_fn,
    context_signals: Dict[str, float],
    months: int = 24,
) -> MarketScenarioDetail:
    """
    Single ABM run with heterogeneous agents parameterized from the proposal.
    """
    base_price = context_signals.get("base_price", 1.0)
    demand_multiplier = context_signals.get("demand_multiplier", 1.0)
    burn_rate = context_signals.get("burn_rate", 0.01)

    # Build agents from proposal
    agents = build_agent_population(
        proposal_initial_supply, allocation, vesting, normalize_key_fn, context_signals
    )
    n_agents = len(agents)

    # Calculate initial liquidity depth from community/insider ratio
    community_share = sum(v for k, v in allocation.items() if normalize_key_fn(k) == "Community")
    insider_share = sum(v for k, v in allocation.items() if normalize_key_fn(k) in {"Team", "Investors"})
    liquidity_depth = max(0.1, community_share / max(insider_share, 1))
    market = MarketEnvironment(initial_price=base_price, initial_liquidity_depth=liquidity_depth,
                                   total_supply=proposal_initial_supply)

    price_path = []
    liquidity_levels = []
    sentiment_path = []
    gini_trajectory = []
    theil_trajectory = []
    atkinson_trajectory = []

    current_sentiment = 0.5

    for month in range(months):
        monthly_buy_usd = 0.0
        monthly_sell_usd = 0.0

        np.random.shuffle(agents)

        # Token burn applied to AMM pool
        market.base_pool *= (1 - burn_rate)
        market.k = market.base_pool * market.quote_pool
        market.price = market.quote_pool / max(market.base_pool, 1e-9)

        # Macro sentiment shock
        macro_shock = np.random.normal(0, 0.15)
        current_sentiment = max(0.1, min(0.9, current_sentiment + macro_shock))

        for agent in agents:
            vol = agent.step(market, current_sentiment, demand_multiplier, month)
            if vol > 0:
                market.trade("BUY", vol)
                monthly_buy_usd += vol
            elif vol < 0:
                market.trade("SELL", abs(vol))
                monthly_sell_usd += abs(vol)

        price_path.append(round(market.price, 6))

        # Update sentiment from buy/sell pressure
        total_vol = monthly_buy_usd + monthly_sell_usd
        if total_vol > 0:
            buy_ratio = monthly_buy_usd / total_vol
            current_sentiment = current_sentiment * 0.7 + buy_ratio * 0.3
        sentiment_path.append(round(current_sentiment, 4))

        # Exogenous liquidity noise
        liq = market.get_liquidity()
        noise = np.random.normal(0, 0.05)
        market.quote_pool *= (1 + noise)
        market.base_pool *= (1 + noise)
        market.k = market.base_pool * market.quote_pool
        liquidity_levels.append(round(liq, 4))

        # ── Compute agent-level fairness indices ──────────────
        balances = [a.token_balance for a in agents]
        gini_trajectory.append(round(compute_gini(balances), 4))
        theil_trajectory.append(round(compute_theil_t(balances), 4))
        atkinson_trajectory.append(round(compute_atkinson(balances, epsilon=0.5), 4))

    # Final snapshot
    agent_token_snapshot = [a.token_balance for a in agents]

    # Population summary
    label_counts = {}
    for a in agents:
        label_counts[a.label] = label_counts.get(a.label, 0) + 1
    pop_str = ", ".join(f"{count} {label}" for label, count in sorted(label_counts.items()))

    notes = [
        f"Heterogeneous ABM with {n_agents} agents ({pop_str}).",
        f"AMM constant-product pricing; {months}-month horizon.",
        f"Avg price: {np.mean(price_path):.4f} vs Base: {base_price:.4f}",
        f"Final agent-level Gini: {gini_trajectory[-1]:.4f}",
        f"Final Theil-T: {theil_trajectory[-1]:.4f}, Atkinson(0.5): {atkinson_trajectory[-1]:.4f}",
        f"End liquidity factor: {liquidity_levels[-1]:.4f}",
        f"Sentiment drift: {sentiment_path[-1] - sentiment_path[0]:+.4f} over {months} months.",
    ]

    return MarketScenarioDetail(
        name="Heterogeneous ABM Market Simulation",
        price_path=price_path,
        liquidity_levels=liquidity_levels,
        sentiment_trend=sentiment_path,
        gini_trajectory=gini_trajectory,
        theil_trajectory=theil_trajectory,
        atkinson_trajectory=atkinson_trajectory,
        agent_token_snapshot=agent_token_snapshot,
        notes=notes,
    )


# ── Monte Carlo wrapper ──────────────────────────────────────

def run_monte_carlo_abm(
    proposal_initial_supply: float,
    allocation: Dict[str, float],
    vesting: Dict,
    normalize_key_fn,
    context_signals: Dict[str, float],
    n_replications: int = 100,
    months: int = 24,
) -> MonteCarloABMResult:
    """
    Run N independent replications of the ABM and aggregate statistics.
    Returns confidence intervals for price, Gini, Theil, Atkinson, liquidity.
    """
    all_prices = []
    all_gini = []
    all_theil = []
    all_atkinson = []
    all_liquidity = []
    all_sentiment = []
    all_max_drawdowns = []
    all_price_volatilities = []
    representative = None

    for i in range(n_replications):
        result = run_agent_market_simulation(
            proposal_initial_supply=proposal_initial_supply,
            allocation=allocation,
            vesting=vesting,
            normalize_key_fn=normalize_key_fn,
            context_signals=context_signals,
            months=months,
        )
        all_prices.append(result.price_path)
        all_gini.append(result.gini_trajectory)
        all_theil.append(result.theil_trajectory)
        all_atkinson.append(result.atkinson_trajectory)
        all_liquidity.append(result.liquidity_levels)
        all_sentiment.append(result.sentiment_trend)

        # Max drawdown for this run
        peak = result.price_path[0]
        max_dd = 0.0
        for p in result.price_path:
            peak = max(peak, p)
            dd = (peak - p) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        all_max_drawdowns.append(max_dd)

        # Price volatility (std of log returns)
        log_returns = [
            math.log(result.price_path[t] / result.price_path[t - 1])
            for t in range(1, len(result.price_path))
            if result.price_path[t - 1] > 0 and result.price_path[t] > 0
        ]
        all_price_volatilities.append(float(np.std(log_returns)) if log_returns else 0.0)

        if i == 0:
            representative = result

    # Stack into arrays for percentile computation
    prices_arr = np.array(all_prices)       # (N, months)
    gini_arr = np.array(all_gini)
    theil_arr = np.array(all_theil)
    atkinson_arr = np.array(all_atkinson)
    liq_arr = np.array(all_liquidity)
    sent_arr = np.array(all_sentiment)

    pop_summary = representative.notes[0] if representative else ""

    # Compute price stability coefficient from mean price path
    mean_price_path = np.mean(prices_arr, axis=0).round(6).tolist()
    from scenario_generator import compute_price_stability_coefficient
    psc = compute_price_stability_coefficient(mean_price_path)

    notes = [
        f"Monte Carlo ABM: {n_replications} replications × {months} months.",
        f"Mean final Gini: {float(np.mean(gini_arr[:, -1])):.4f} ± {float(np.std(gini_arr[:, -1])):.4f}",
        f"Mean max drawdown: {float(np.mean(all_max_drawdowns)):.4f} ± {float(np.std(all_max_drawdowns)):.4f}",
        f"Mean price volatility (σ log-returns): {float(np.mean(all_price_volatilities)):.4f}",
        f"Price stability coefficient: {psc:.4f}",
        pop_summary,
    ]

    return MonteCarloABMResult(
        n_replications=n_replications,
        mean_price_path=mean_price_path,
        std_price_path=np.std(prices_arr, axis=0).round(6).tolist(),
        p5_price_path=np.percentile(prices_arr, 5, axis=0).round(6).tolist(),
        p95_price_path=np.percentile(prices_arr, 95, axis=0).round(6).tolist(),
        mean_gini_trajectory=np.mean(gini_arr, axis=0).round(4).tolist(),
        std_gini_trajectory=np.std(gini_arr, axis=0).round(4).tolist(),
        p5_gini_trajectory=np.percentile(gini_arr, 5, axis=0).round(4).tolist(),
        p95_gini_trajectory=np.percentile(gini_arr, 95, axis=0).round(4).tolist(),
        mean_theil_trajectory=np.mean(theil_arr, axis=0).round(4).tolist(),
        std_theil_trajectory=np.std(theil_arr, axis=0).round(4).tolist(),
        mean_atkinson_trajectory=np.mean(atkinson_arr, axis=0).round(4).tolist(),
        std_atkinson_trajectory=np.std(atkinson_arr, axis=0).round(4).tolist(),
        mean_liquidity_path=np.mean(liq_arr, axis=0).round(4).tolist(),
        std_liquidity_path=np.std(liq_arr, axis=0).round(4).tolist(),
        mean_sentiment_path=np.mean(sent_arr, axis=0).round(4).tolist(),
        mean_final_gini=float(np.mean(gini_arr[:, -1])),
        std_final_gini=float(np.std(gini_arr[:, -1])),
        mean_max_drawdown=float(np.mean(all_max_drawdowns)),
        std_max_drawdown=float(np.std(all_max_drawdowns)),
        mean_price_volatility=float(np.mean(all_price_volatilities)),
        price_stability_coefficient=psc,
        population_summary=pop_summary,
        notes=notes,
        representative_run=representative,
    )
