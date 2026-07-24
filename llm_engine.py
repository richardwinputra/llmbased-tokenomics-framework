"""
Tokenomics Generation Module (Section III.B, Algorithm 1).

Handles:
  - Interactive and file-based input collection (structured)
  - Prompt engineering with RAG from knowledge base
  - LLM API calls (OpenAI directly, or any model via OpenRouter) with fallback
  - Proposal construction: LLM text -> structured GeneratedTokenomics

Multi-LLM support (for the model-comparison experiment):
  Model names containing "/" (e.g. "anthropic/claude-sonnet-4.5",
  "google/gemini-2.5-flash") are routed through OpenRouter's OpenAI-compatible
  Chat Completions endpoint (requires OPENROUTER_API_KEY in .env).
  Plain OpenAI model names keep using the native Responses API.

  Optional env vars:
    OPENROUTER_PROVIDER  pin a specific provider for reproducibility
                         (sets provider.order + allow_fallbacks=false)
"""

import os
import time
from typing import Dict, Tuple, Optional

from openai import OpenAI, RateLimitError, APIStatusError
from dotenv import load_dotenv

from models import (
    GeneratedTokenomics, ProjectContext, ProjectMetadata,
)
from utils import (
    extract_json_payload, extract_tokenomics_parameters,
    parse_design_thinking, infer_constraints, infer_legal_risk_tolerance,
    enforce_tokenomics_constraints,
)

load_dotenv(override=True)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_clients: Dict[str, OpenAI] = {}

# Metadata of the most recent LLM call (requested/served model, provider),
# logged per project by run_full_experiment.py for exact-version records.
LAST_CALL_META: Dict = {}


def _provider_for(model_name: str) -> str:
    """OpenRouter slugs are namespaced ("vendor/model"); OpenAI names are not."""
    return "openrouter" if "/" in model_name else "openai"


def _get_client(provider: str = "openai") -> OpenAI:
    if provider not in _clients:
        if provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY not set")
            _clients[provider] = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            _clients[provider] = OpenAI(api_key=api_key)
    return _clients[provider]


def get_structured_input() -> Dict:
    """Collect structured input following the Token Design Thinking framework."""
    data = {
        "input_type": "structured",
        "project_name": input("Project name: "),
        "token_symbol": input("Token symbol: "),
        "token_purpose": input("Token Purpose: ").split(","),
        "core_principles": input("Core Principles: ").split(","),
        "token_functions": input("Token functions (comma separated): ").split(","),
        "stakeholders": input("Stakeholders (comma separated): ").split(","),
        "economic_design": input("Economic Design: "),
        "legal_design": input("Legal Design: "),
        "technical_design": input("Technical Design: "),
        "gov_structures": input("Governance Structures: "),
        "total_supply_preference": input("Total supply preference (fixed/unlimited/capped): "),
        "inflation_preference": input("Inflation type (e.g., deflationary, stable, mild inflationary): "),
        "similar_projects": input("Similar projects for reference (comma separated): ").split(","),
    }
    data["token_functions"] = [x.strip() for x in data["token_functions"] if x.strip()]
    data["stakeholders"] = [x.strip() for x in data["stakeholders"] if x.strip()]
    data["similar_projects"] = [x.strip().lower() for x in data["similar_projects"] if x.strip()]
    return data


_SYSTEM_MESSAGE = """You are a tokenomics design expert grounded in the Token Design Thinking framework.

When designing tokenomics, you consider:
1. PURPOSE - Core purpose of the project
2. PRINCIPLES & VALUES - Mission, vision, guiding principles
3. POSITIONING & BUSINESS MODEL - For-profit, non-profit, or mixed
4. SYSTEM FUNCTIONS & TOKEN FUNCTIONS - Separate system from token functions
5. STAKEHOLDERS - All key stakeholder types and their incentives
6. TOKEN TYPES AND ROLES - Number and types of tokens needed
7. ECONOMIC DESIGN - Supply, issuance, sinks, pricing, sustainability
8. LEGAL & REGULATORY DESIGN - Functional classification and constraints
9. TECHNICAL DESIGN - On-chain vs off-chain, L1 vs L2, custody
10. POWER STRUCTURES - Voting, information, market, mediation power
11. TEAM, ROADMAP & EVOLUTION - Progressive decentralization

Formatting Requirements:
- Use exact numbers for total supply
- Format allocations as "Category: XX%"
- Ensure all percentages sum to 100%
- Include vesting schedules with cliff and duration in months for each category
- IMPORTANT: Every insider category (team, founders, advisors, investors, strategic backers)
  MUST have an explicit vesting schedule with cliff_months and vesting_months. Do not omit any.
- Justify decisions using reasoning and references from similar projects

OUTPUT STRUCTURE (MANDATORY):
After your narrative explanation, you MUST end your response with a JSON block in this exact format:

```json
{
  "tokenomics_parameters": {
    "total_supply": <integer>,
    "allocation": {
      "<Category Name>": <percentage as float>
    },
    "vesting": {
      "<Category Name>": {
        "cliff_months": <integer>,
        "vesting_months": <integer>,
        "unlock_type": "linear"
      }
    }
  }
}
```

Ensure:
- allocation values sum to exactly 100.0
- Every insider category in allocation has a matching entry in vesting
- Use the same category name strings in both allocation and vesting
"""


def create_structured_prompt(user_input: Dict, project_summaries: str) -> str:
    """
    Algorithm 1, lines 3-7: Build prompt from structured input.
    Appends project overview, design constraints, similar projects,
    stakeholders, and all knowledge base summaries.
    """
    similar_projects_context = ""
    if user_input.get('similar_projects'):
        similar_projects_context = (
            f"The user referenced: {', '.join(user_input['similar_projects'])}. "
            "Leverage design patterns from these cases."
        )
    stakeholder_context = ""
    if user_input.get('stakeholders'):
        stakeholder_context = (
            f"Key stakeholders: {', '.join(user_input['stakeholders'])}. "
            "Ensure their roles and incentives are mapped."
        )

    # Prompt variant "no-kb" (knowledge-base ablation): callers pass empty
    # project_summaries, and the Reference Projects section is omitted entirely.
    reference_context = ""
    if project_summaries:
        reference_context = f"Reference Projects:\n{project_summaries}\n"

    return f"""Transform the structured data below into a coherent tokenomics model.

Project Overview:
- Project Name: {user_input['project_name']}
- Token Symbol: {user_input['token_symbol']}
- Core Principles: {', '.join(user_input.get('core_principles', []))}
- Token Purpose: {', '.join(user_input.get('token_purpose', []))}
- Core Functions: {', '.join(user_input.get('token_functions', []))}

Design Preferences:
- Supply Model: {user_input.get('total_supply_preference', 'Not specified')}
- Inflation Strategy: {user_input.get('inflation_preference', 'Not specified')}
- Economic Design: {user_input.get('economic_design', 'Not specified')}
- Legal Context: {user_input.get('legal_design', 'Not specified')}
- Technical Approach: {user_input.get('technical_design', 'Not specified')}
- Governance Structures: {user_input.get('gov_structures', 'Not specified')}

{stakeholder_context}
{similar_projects_context}

{reference_context}
Provide: token role, total supply, allocation (Category: XX%), vesting schedule
(cliff and duration in months for each category), governance design, economic model.

IMPORTANT: End your response with a ```json code block containing the structured
tokenomics_parameters object (total_supply, allocation dict, vesting dict).
Every insider category MUST appear in the vesting dict.
"""


def ask_openai_enhanced(prompt: str, model_override: Optional[str] = None) -> str:
    """
    Algorithm 1, line 13: LLM_Response_Text <- LLM_API(prompt).
    Calls the primary model with fallback. OpenAI models use the native
    Responses API; OpenRouter slugs ("vendor/model") use Chat Completions.
    """
    system_msg = str(_SYSTEM_MESSAGE)
    system_msg += "\n\nNOTE: Use the structured input as primary source of truth."

    primary = model_override or os.getenv("OPENAI_MODEL", "gpt-5.4-mini-2026-03-17")
    fallback = model_override or os.getenv("OPENAI_MODEL_FALLBACK", "gpt-5.4-mini-2026-03-17")

    def call_model(model_name: str, effort_level: str = "medium") -> str:
        provider = _provider_for(model_name)
        client = _get_client(provider)
        LAST_CALL_META.clear()
        LAST_CALL_META.update({"requested_model": model_name, "provider_route": provider})

        if provider == "openrouter":
            extra_body = {"reasoning": {"effort": effort_level}}
            pinned = os.getenv("OPENROUTER_PROVIDER")
            if pinned:
                extra_body["provider"] = {"order": [pinned], "allow_fallbacks": False}
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                extra_body=extra_body,
            )
            LAST_CALL_META["served_model"] = getattr(resp, "model", model_name)
            # OpenRouter returns the serving provider as a non-standard field
            LAST_CALL_META["served_provider"] = getattr(resp, "provider", None)
            if resp.choices:
                return resp.choices[0].message.content or ""
            return ""

        resp = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            reasoning={"effort": effort_level},
            text={"verbosity": "medium"}
        )
        LAST_CALL_META["served_model"] = getattr(resp, "model", model_name)
        return resp.output_text if hasattr(resp, "output_text") and resp.output_text else ""

    def call_with_retry(model_name: str, effort_level: str) -> str:
        """Retry on per-minute rate limits (free OpenRouter models: 20 req/min).
        Daily-cap errors are not retried — the runner's checkpoint resume picks
        the remaining projects up on the next run."""
        backoffs = [10, 30, 60]
        for attempt, wait in enumerate([0] + backoffs):
            if wait:
                print(f"    Rate limited; retrying in {wait}s "
                      f"(attempt {attempt}/{len(backoffs)})...")
                time.sleep(wait)
            try:
                return call_model(model_name, effort_level=effort_level)
            except RateLimitError as e:
                # Free-model daily cap: retrying within the run is pointless
                if "per day" in str(e).lower() or "daily" in str(e).lower():
                    raise
                last_err = e
            except APIStatusError as e:
                if e.status_code not in (429, 500, 502, 503):
                    raise
                last_err = e
        raise last_err

    try:
        content = call_with_retry(primary, effort_level="high")
        if not (content and content.strip()):
            content = call_with_retry(fallback, effort_level="medium")
        if not (content and content.strip()):
            return "Error: LLM response was empty. Please try again."
        return content
    except Exception as e:
        return f"Error generating recommendation: {e}"


def generate_tokenomics_proposal(
    user_input: Dict, result_text: str
) -> Tuple[GeneratedTokenomics, ProjectContext]:
    """
    Parse LLM response text into structured GeneratedTokenomics and ProjectContext.
    Implements Algorithm 1 lines 14-24: parse to JSON, regex scan, extract labels/values.
    """
    payload = extract_json_payload(result_text) or {}
    params = extract_tokenomics_parameters(payload, result_text)


    design_thinking_raw = payload.get("token_design_thinking", {}) or {}
    if "purpose" not in design_thinking_raw:
        design_thinking_raw["purpose"] = user_input.get("project_description", "")[:150]
    design_thinking = parse_design_thinking(design_thinking_raw)


    meta_raw = payload.get("project_metadata", {})
    metadata = ProjectMetadata(
        project=meta_raw.get("project") or user_input.get("project_name", "Unknown"),
        token=meta_raw.get("token") or user_input.get("token_symbol", "TKN"),
        category=meta_raw.get("category"),
    )

    refs = payload.get("references", {})

    gen_tokenomics = GeneratedTokenomics(
        project_metadata=metadata,
        token_design_thinking=design_thinking,
        tokenomics_parameters=params,
        references=refs,
    )


    description = "Structured intake provided via questionnaire."

    goals = [g.strip() for g in user_input.get("core_principles", []) if g.strip()]
    priorities = [p.strip() for p in user_input.get("token_purpose", []) if p.strip()]
    if design_thinking.principles and not goals:
        goals = design_thinking.principles

    constraints = infer_constraints()
    legal_risk = infer_legal_risk_tolerance(user_input)

    context = ProjectContext(
        description=description, goals=goals, priorities=priorities,
        constraints=constraints, legal_risk_tolerance=legal_risk,
    )

    gen_tokenomics = enforce_tokenomics_constraints(gen_tokenomics)
    return gen_tokenomics, context

