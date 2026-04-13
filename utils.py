"""
Utility functions: parsing, normalization, Gini computation, economic signal inference.

Implements the allocation normalization (insider vs distributed categories),
Gini coefficient computation, and text extraction helpers described in the paper.
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Any

from models import (
    VestingDetail, TokenomicsParameters, TokenDesignThinking,
    GeneratedTokenomics, TokenSupply, FixedSupply, CappedSupply, DynamicSupply,
)


def calculate_gini(values: List[float]) -> float:
    """
    Compute the Gini coefficient for a list of non-negative values.
    Returns 0.0 for empty or all-zero inputs.
    """
    if not values:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    cumulative_sum = sum(sorted_values)
    if cumulative_sum == 0:
        return 0.0
    gini = (
        2 * sum((i + 1) * val for i, val in enumerate(sorted_values))
    ) / (n * cumulative_sum) - (n + 1) / n
    return max(0.0, gini)


_INSIDER_KEYS = {
    'team', 'founders', 'core_team', 'founder', 'core_contributors',
    'advisors', 'advisory',
    'investors', 'private_sale', 'seed', 'series_a', 'strategic',
}

_DISTRIBUTED_KEYS = {
    'community', 'public', 'public_sale', 'public_and_community',
    'community_rewards', 'users', 'airdrop', 'airdrops',
    'ecosystem', 'ecosystem_fund', 'ecosystem_incentives',
    'liquidity', 'liquidity_mining', 'liquidity_pool',
    'liquidity_incentives', 'community_liquidity_providers',
    'staking', 'staking_rewards', 'validator_rewards',
    'treasury', 'dao_treasury', 'reserve', 'reserves',
    'marketing', 'partnerships', 'development',
    'operations', 'operational', 'node_operators',
    'governance', 'governance_fund',
    'liquidity_providers', 'liquidity_provision',
}

_KEY_MAPPINGS = {

    'team': 'Team', 'founders': 'Team', 'core_team': 'Team', 'founder': 'Team',
    'core_contributors': 'Team',

    'advisors': 'Advisors', 'advisory': 'Advisors',

    'investors': 'Investors', 'private_sale': 'Investors', 'seed': 'Investors',
    'series_a': 'Investors', 'strategic': 'Investors',

    'community': 'Community', 'public': 'Community', 'public_sale': 'Community',
    'public_and_community': 'Community', 'community_rewards': 'Community',
    'users': 'Community', 'community_liquidity_providers': 'Community',
    'airdrop': 'Community', 'airdrops': 'Community',

    'ecosystem': 'Ecosystem', 'ecosystem_fund': 'Ecosystem',
    'ecosystem_incentives': 'Ecosystem', 'development': 'Ecosystem',
    'treasury': 'Ecosystem', 'dao_treasury': 'Ecosystem',

    'liquidity': 'Liquidity', 'liquidity_mining': 'Liquidity',
    'liquidity_pool': 'Liquidity', 'liquidity_incentives': 'Liquidity',
    'liquidity_providers': 'Liquidity', 'liquidity_provision': 'Liquidity',

    'staking': 'Staking', 'staking_rewards': 'Staking',
    'validator_rewards': 'Staking',

    'reserve': 'Reserve', 'reserves': 'Reserve', 'contingency': 'Reserve',

    'marketing': 'Marketing', 'partnerships': 'Marketing',

    'operations': 'Operations', 'operational': 'Operations',
    'node_operators': 'Operations',

    'governance': 'Community', 'governance_fund': 'Community',
    'dao': 'Community', 'dao_governance': 'Community',
}


def _strip_markdown(text: str) -> str:
    """Strip markdown formatting (bold, italic markers) from text."""

    text = re.sub(r'\*{1,2}', '', text)
    text = re.sub(r'_{1,2}', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_allocation_key(key: str) -> str:
    """Normalize an allocation label to a standard category name.

    Strips markdown formatting (e.g., **Team** -> Team) and maps to
    standard categories via _KEY_MAPPINGS lookup.
    """
    clean = _strip_markdown(key).lower().strip()

    if clean in _KEY_MAPPINGS:
        return _KEY_MAPPINGS[clean]

    underscore_key = clean.replace(' ', '_')
    if underscore_key in _KEY_MAPPINGS:
        return _KEY_MAPPINGS[underscore_key]

    for known_key, mapped_value in _KEY_MAPPINGS.items():
        if known_key in clean or clean in known_key:
            return mapped_value

    return clean.title()


def is_insider_category(key: str) -> bool:
    """Check if a normalized category is an insider allocation."""
    normalized = normalize_allocation_key(key)
    return normalized in {"Team", "Advisors", "Investors"}


def is_distributed_category(key: str) -> bool:
    """Check if a normalized category is a distributed allocation."""
    return not is_insider_category(key)


def compute_insider_pct(allocation: Dict[str, float]) -> float:
    """Sum of insider allocation percentages (team + advisors + investors)."""
    return sum(v for k, v in allocation.items() if is_insider_category(k))


def compute_distributed_pct(allocation: Dict[str, float]) -> float:
    """Sum of distributed allocation percentages."""
    return sum(v for k, v in allocation.items() if is_distributed_category(k))


def compute_team_pct(allocation: Dict[str, float]) -> float:
    """Sum of team/founder allocation percentages."""
    return sum(v for k, v in allocation.items()
               if normalize_allocation_key(k) == "Team")


def compute_investor_pct(allocation: Dict[str, float]) -> float:
    """Sum of investor allocation percentages."""
    return sum(v for k, v in allocation.items()
               if normalize_allocation_key(k) == "Investors")



def _infer_numeric_from_text(patterns: List[str], text: str) -> Optional[int]:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (IndexError, ValueError):
                continue
    return None


def infer_bucket_timing(label: str, text: str) -> Tuple[Optional[int], Optional[int]]:
    """Infer cliff and vesting months for a category from LLM response text."""
    safe_label = re.escape(label.strip()) if label else ""
    if not safe_label:
        return None, None
    cliff_patterns = [
        rf"{safe_label}[^\.\n]*?(\d+)\s*-?month\s+cliff",
        rf"{safe_label}[^\.\n]*?cliff\s*:?\s*(\d+)",
    ]
    vesting_patterns = [
        rf"{safe_label}[^\.\n]*?(\d+)\s*-?month\s+vesting",
        rf"{safe_label}[^\.\n]*?vesting\s*:?\s*(\d+)",
    ]
    return (
        _infer_numeric_from_text(cliff_patterns, text),
        _infer_numeric_from_text(vesting_patterns, text),
    )


def infer_constraints() -> Dict[str, float]:
    """Infer control/filter thresholds from user goals per paper Table II."""
    constraints: Dict[str, float] = {}

    constraints["min_distributed_share"] = 20.0
    constraints["max_team_share"] = 35.0
    constraints["max_investor_preferred"] = 25.0
    constraints["max_investor_flagged"] = 40.0
    constraints["max_insider_preferred"] = 40.0
    constraints["max_insider_flagged"] = 60.0
    constraints["min_insider_vesting_months"] = 12
    constraints["max_gini"] = 0.60
    return constraints


def infer_legal_risk_tolerance(user_input: Dict) -> str:
    legal_text = user_input.get("legal_design", "").lower()
    if any(w in legal_text for w in ["security", "regulation", "compliance"]):
        return "conservative"
    if any(w in legal_text for w in ["progressive", "experimental", "aggressive"]):
        return "aggressive"
    return "balanced"


def extract_json_payload(text: str) -> Optional[Dict]:
    """Extract a JSON object from LLM response text."""
    json_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
    for block in json_pattern.findall(text):
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_allocation_enhanced(text: str) -> Tuple[List[str], List[float]]:
    """
    Extract allocation labels and percentages from LLM text output.
    Paper Algorithm 1, lines 15-24: RegexScan, ExtractLabels, ExtractPercents.
    """
    patterns = [
        r"- ([^:]+):\s*(\d{1,3}(?:\.\d{1,2})?)%",
        r"([^:]+):\s*(\d{1,3}(?:\.\d{1,2})?)%",
        r"\u2022 ([^:]+):\s*(\d{1,3}(?:\.\d{1,2})?)%",
        r"(\w+(?:\s+\w+)*)\s*-\s*(\d{1,3}(?:\.\d{1,2})?)%",
    ]
    labels, values = [], []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for label, value in matches:
                clean_label = _strip_markdown(label).strip().title()
                try:
                    clean_value = float(value)
                    if clean_label not in labels:
                        labels.append(clean_label)
                        values.append(clean_value)
                except ValueError:
                    continue
            if len(values) > 1:
                break


    total = sum(values)
    if total > 110:
        print(f"Warning: Allocation total ({total}%) exceeds 100%")
    elif total < 90:
        print(f"Warning: Allocation total ({total}%) is less than 90%")
    return labels, values


def extract_total_supply(text: str) -> Optional[float]:
    """Extract total supply number from LLM response text."""
    patterns = [
        r'[Tt]otal\s+[Ss]upply[:\-\s]*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
        r'[Tt]otal\s+[Ss]upply[:\-\s]*(\d+(?:\.\d+)?)',
        r'[Ss]upply[:\-\s]*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*tokens?',
        r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*tokens?\s*total',
        r'[Mm]aximum\s+[Ss]upply[:\-\s]*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
        r'[Ff]ixed\s+[Ss]upply[:\-\s]*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                supply_value = float(match.group(1).replace(',', ''))
                if 1000 <= supply_value <= 1e15:
                    return supply_value
            except ValueError:
                continue
    large_numbers = re.findall(r'\b(\d{1,3}(?:,\d{3})+)\b', text)
    for num_str in large_numbers:
        try:
            num_value = float(num_str.replace(',', ''))
            if 1000000 <= num_value <= 1e15:
                return num_value
        except ValueError:
            continue
    return None


def parse_token_supply(data: Any) -> TokenSupply:
    if isinstance(data, (int, float)):
        return int(data)
    if isinstance(data, dict):
        if "cap" in data:
            return CappedSupply(cap=int(data.get("cap", 0)),
                                initial_supply=int(data.get("initial_supply", 0)))
        elif "emission_rate" in data:
            return DynamicSupply(
                initial_supply=int(data.get("initial_supply", 0)),
                emission_rate=float(data.get("emission_rate", 0.0)),
                burn_rule=data.get("burn_rule"),
            )
        elif "amount" in data:
            return FixedSupply(amount=int(data.get("amount", 0)))
    if isinstance(data, str):
        try:
            return int(float(re.sub(r'[^\d.]', '', data)))
        except ValueError:
            pass
    return 0


def parse_design_thinking(data: Dict) -> TokenDesignThinking:
    return TokenDesignThinking(
        purpose=data.get("purpose", ""),
        principles=data.get("principles", []) if isinstance(data.get("principles"), list) else [],
        positioning=data.get("positioning", ""),
        functions=data.get("functions", []) if isinstance(data.get("functions"), list) else [],
        stakeholders=data.get("stakeholders", []) if isinstance(data.get("stakeholders"), list) else [],
        economic_design=data.get("economic_design", ""),
        legal_design=data.get("legal_design", ""),
        tech_design=data.get("tech_design", ""),
        power_structures=data.get("power_structures", ""),
        team=data.get("team", {}) if isinstance(data.get("team"), dict) else {},
    )


def parse_vesting(data: Dict) -> Dict[str, VestingDetail]:
    result = {}
    if not isinstance(data, dict):
        return result
    for k, v in data.items():
        if isinstance(v, dict):
            result[k] = VestingDetail(
                cliff_months=int(v.get("cliff_months", 0)),
                vesting_months=int(v.get("vesting_months", 0)),
                unlock_type=v.get("unlock_type", "linear"),
            )
    return result


def extract_from_markdown_table(text: str) -> Tuple[Dict[str, float], Dict[str, VestingDetail]]:
    """
    Parse allocation and vesting data from a markdown table produced by GPT-5.4.

    Expects a table with at minimum a 'Category' column and a '%' column.
    Optionally reads cliff_months and vesting_months columns.

    Example row:
      | Core Team & Founders | 16% | 32,000,000 | 12 | 48 | notes |
    """
    allocation: Dict[str, float] = {}
    vesting: Dict[str, VestingDetail] = {}

    # Find all markdown table rows (lines starting with |)
    rows = [line.strip() for line in text.splitlines() if line.strip().startswith('|')]
    if not rows:
        return allocation, vesting

    # Identify header row — look for a row containing 'allocation' or '%' or 'category'
    header_idx = None
    headers = []
    for i, row in enumerate(rows):
        cells = [c.strip().lower() for c in row.strip('|').split('|')]
        if any(kw in ' '.join(cells) for kw in ('allocation', 'category', '%', 'cliff', 'vesting')):
            # Skip separator rows like |---|---|---|
            if all(re.match(r'^[-:]+$', c) for c in cells if c):
                continue
            header_idx = i
            headers = [c.strip().lower() for c in row.strip('|').split('|')]
            break

    if header_idx is None:
        return allocation, vesting

    # Map header names to column indices
    def find_col(*candidates):
        for c in candidates:
            for i, h in enumerate(headers):
                if c in h:
                    return i
        return None

    cat_col   = find_col('category', 'name', 'bucket')
    pct_col   = find_col('allocation', '%', 'share', 'percent')
    cliff_col = find_col('cliff')
    vest_col  = find_col('vesting_months', 'vesting months', 'vest')

    if cat_col is None or pct_col is None:
        return allocation, vesting

    # Parse data rows
    for row in rows[header_idx + 1:]:
        cells = [c.strip() for c in row.strip('|').split('|')]
        if not cells or all(re.match(r'^[-:]+$', c) for c in cells if c):
            continue  # skip separators
        if len(cells) <= max(filter(None, [cat_col, pct_col])):
            continue

        category = _strip_markdown(cells[cat_col]).strip()
        if not category or len(category) > 80:
            continue

        # Extract percentage — accept "50%" or "50"
        raw_pct = cells[pct_col].replace('%', '').replace(',', '').strip()
        try:
            pct = float(raw_pct)
            if not (0 < pct <= 100):
                continue
        except ValueError:
            continue

        allocation[category] = pct

        # Extract vesting if columns exist
        cliff, vest = None, None
        if cliff_col is not None and cliff_col < len(cells):
            try:
                cliff = int(cells[cliff_col].replace(',', '').strip())
            except ValueError:
                pass
        if vest_col is not None and vest_col < len(cells):
            try:
                vest = int(cells[vest_col].replace(',', '').strip())
            except ValueError:
                pass

        if cliff is not None or vest is not None:
            vesting[category] = VestingDetail(
                cliff_months=cliff or 0,
                vesting_months=vest or 0,
                unlock_type="linear",
            )

    return allocation, vesting


def extract_tokenomics_parameters(payload: Dict, raw_text: str) -> TokenomicsParameters:
    """Parse LLM output into TokenomicsParameters (Algorithm 1, line 14)."""
    params_block = payload.get("tokenomics_parameters", {})
    if not isinstance(params_block, dict):
        params_block = {}


    raw_supply = params_block.get("total_supply") or payload.get("total_supply")
    if not raw_supply:
        raw_supply = extract_total_supply(raw_text)
    total_supply = parse_token_supply(raw_supply)


    raw_alloc = (params_block.get("allocation") or
                 payload.get("allocations") or
                 payload.get("allocation"))
    allocation = {}
    vesting_from_table: Dict[str, VestingDetail] = {}

    if isinstance(raw_alloc, dict):
        # JSON block parsed successfully — use it directly
        for k, v in raw_alloc.items():
            try:
                clean_k = _strip_markdown(k).strip()
                if len(clean_k) > 80:  # guard against garbage keys
                    continue
                allocation[clean_k] = float(v)
            except (ValueError, TypeError):
                pass
    else:
        # Try markdown table first (GPT-5.4 prose output)
        allocation, vesting_from_table = extract_from_markdown_table(raw_text)
        if not allocation:
            # Last resort: regex scan — but filter out garbage long keys
            labels, values = extract_allocation_enhanced(raw_text)
            for l, v in zip(labels, values):
                if len(l) <= 80:  # skip garbled multi-sentence keys
                    allocation[l] = v


    if not allocation:
        allocation = {
            "Community": 40.0, "Team": 15.0, "Investors": 12.0,
            "Ecosystem": 18.0, "Advisors": 5.0, "Liquidity": 10.0,
        }


    raw_vesting = params_block.get("vesting") or payload.get("vesting")
    vesting = parse_vesting(raw_vesting) if isinstance(raw_vesting, dict) else {}
    # Use table-extracted vesting if JSON had none
    if not vesting and vesting_from_table:
        vesting = vesting_from_table
    # Last resort: infer from prose text
    if not vesting and allocation:
        for key in allocation.keys():
            cliff, vest = infer_bucket_timing(key, raw_text)
            if cliff is not None or vest is not None:
                vesting[key] = VestingDetail(
                    cliff_months=cliff or 0, vesting_months=vest or 0)

    return TokenomicsParameters(
        total_supply=total_supply, allocation=allocation, vesting=vesting,
        emissions=params_block.get("emissions"), burn=params_block.get("burn"),
    )


def enforce_tokenomics_constraints(tokenomics: GeneratedTokenomics) -> GeneratedTokenomics:
    """Normalize allocation to 100% and ensure minimum category diversity."""
    alloc = tokenomics.tokenomics_parameters.allocation
    total = sum(alloc.values())
    if abs(total - 100) > 0.5 and total > 0:
        factor = 100 / total
        tokenomics.tokenomics_parameters.allocation = {
            k: round(v * factor, 2) for k, v in alloc.items()
        }
    return tokenomics


def get_initial_supply(tokenomics: GeneratedTokenomics) -> float:
    """Extract the numeric initial supply value, with 1B default fallback."""
    supply = tokenomics.tokenomics_parameters.total_supply
    result = 0.0
    if isinstance(supply, (int, float)):
        result = float(supply)
    elif isinstance(supply, FixedSupply):
        result = float(supply.amount)
    elif isinstance(supply, CappedSupply):
        result = float(supply.initial_supply)
    elif isinstance(supply, DynamicSupply):
        result = float(supply.initial_supply)
    if result <= 0:
        result = 1_000_000_000.0
    return result


def summarize_all_projects(kb: List[Dict]) -> str:
    """Create text summaries of all KB projects for RAG prompt injection."""
    summaries = []
    for i, proj in enumerate(kb, 1):
        try:
            project_name = proj.get('project', 'Unknown')
            token_symbol = proj.get('token', 'N/A')
            design_thinking = proj.get('token_design_thinking', {})
            tokenomics = proj.get('tokenomics', {})
            purpose = design_thinking.get('purpose', 'N/A')
            functions = design_thinking.get('functions', [])
            total_supply = tokenomics.get('total_supply', 'N/A')
            allocation = tokenomics.get('allocation', {})
            alloc_str = "N/A"
            if isinstance(allocation, dict):
                alloc_str = ", ".join(f"{k}: {v}" for k, v in allocation.items())
            elif isinstance(allocation, str):
                alloc_str = allocation
            summaries.append(
                f"{i}. {project_name} ({token_symbol}): "
                f"Purpose: {purpose} | Functions: {', '.join(functions) if functions else 'N/A'} | "
                f"Total Supply: {total_supply} | Key Allocation: {alloc_str}"
            )
        except Exception as e:
            print(f"Error processing project {i}: {e}")
    return "\n".join(summaries)

