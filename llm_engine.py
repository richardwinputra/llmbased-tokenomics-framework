"""
Tokenomics Generation Module (Section III.B, Algorithm 1).

Handles:
  - Interactive and file-based input collection (structured)
  - Prompt engineering with RAG from knowledge base
  - OpenAI API calls with fallback
  - Proposal construction: LLM text -> structured GeneratedTokenomics
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from openai import OpenAI
from dotenv import load_dotenv

from models import (
    GeneratedTokenomics, ProjectContext, ProjectMetadata,
)
from utils import (
    extract_json_payload, extract_tokenomics_parameters,
    parse_design_thinking, infer_constraints, infer_legal_risk_tolerance,
    infer_economic_signals, enforce_tokenomics_constraints,
    normalize_allocation_dict, parse_token_supply,
)

load_dotenv(override=True)

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=api_key)
    return _client


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
- Include vesting schedules with cliff and duration in months
- Justify decisions using reasoning and references from similar projects
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

Reference Projects:
{project_summaries}

Provide: token role, total supply, allocation (Category: XX%), vesting schedule
(cliff and duration in months for each category), governance design, economic model.
"""


def ask_openai_enhanced(prompt: str, input_type: str = "structured",
                        model_override: Optional[str] = None) -> str:
    """
    Algorithm 1, line 13: LLM_Response_Text <- LLM_API(prompt).
    Calls OpenAI with primary model and fallback.
    """
    system_msg = str(_SYSTEM_MESSAGE)
    system_msg += "\n\nNOTE: Use the structured input as primary source of truth."

    client = _get_client()
    primary = model_override or os.getenv("OPENAI_MODEL", "gpt-5.4")
    fallback = model_override or os.getenv("OPENAI_MODEL_FALLBACK", "gpt-5.4")

    def call_model(model_name: str, effort_level: str = "medium") -> str:
        resp = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            reasoning={"effort": effort_level},
            text={"verbosity": "medium"}
        )
        return resp.output_text if hasattr(resp, "output_text") and resp.output_text else ""

    try:
        content = call_model(primary, effort_level="high")
        if not (content and content.strip()):
            content = call_model(fallback, effort_level="medium")
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

    constraints = infer_constraints(goals, priorities)
    legal_risk = infer_legal_risk_tolerance(user_input)
    economic_signals = infer_economic_signals(user_input, result_text)

    context = ProjectContext(
        description=description, goals=goals, priorities=priorities,
        constraints=constraints, legal_risk_tolerance=legal_risk,
        economic_signals=economic_signals,
    )

    gen_tokenomics = enforce_tokenomics_constraints(gen_tokenomics, context)
    return gen_tokenomics, context

