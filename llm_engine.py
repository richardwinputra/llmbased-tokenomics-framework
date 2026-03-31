"""
LLM interaction: prompt engineering, OpenAI API calls, proposal generation.
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


# ── Interactive input collection ──────────────────────────────

def get_input_mode() -> str:
    print("Choose your input method:")
    print("1. Structured Input - Detailed project specifications (Recommended)")
    print("2. Generic Input - Simple project description")
    while True:
        choice = input("\nSelect input mode (1 or 2): ").strip()
        if choice in ['1', '2']:
            return choice
        print("Invalid choice.")


def get_structured_input() -> Dict:
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


def get_generic_input() -> Dict:
    print("Tell us about your project in your own words.")
    project_name = input("Project name (optional): ").strip()
    print("\nDescribe your project. Type 'DONE' on a new line when finished:")
    description_lines = []
    while True:
        line = input()
        if line.strip().upper() == 'DONE':
            break
        description_lines.append(line)
    full_description = '\n'.join(description_lines).strip()
    if not full_description:
        print("Project description cannot be empty.")
        return get_generic_input()
    return {
        "input_type": "generic",
        "project_name": project_name if project_name else "Not specified",
        "project_description": full_description,
        "description_length": len(full_description.split()),
        "timestamp": datetime.now().isoformat(),
    }


# ── Prompt engineering ────────────────────────────────────────

_SYSTEM_MESSAGE = """
You are Dr. Tokenomics, a world-renowned blockchain economist and tokenomics architect with over a decade of experience designing sustainable token economies for projects across DeFi, GameFi, infrastructure protocols, DAOs, and beyond—many of which have achieved billions in market capitalization.

You ground all of your reasoning in the Token Design Thinking framework (Token Kitchen / Shermin Voshmgir) as represented in the TOKEN DESIGN TOOL. You understand that token design is *not* only about math or price action, but about socio-technical systems, governance, and power structures.

Whenever you design or critique a token model, you mentally walk through the following lenses:

1. PURPOSE - Clarify the core PURPOSE of the project.
2. PRINCIPLES & VALUES - Extract the project's mission, vision, and guiding PRINCIPLES.
3. POSITIONING & BUSINESS MODEL - Determine whether for-profit, non-profit, or mixed.
4. SYSTEM FUNCTIONS & TOKEN FUNCTIONS - Separate system functions from token functions.
5. STAKEHOLDERS & STAKEHOLDER MATRIX - Identify all key STAKEHOLDER types.
6. TOKENS: NUMBER, TYPES, AND ROLES - Decide how many token TYPES are necessary.
7. ECONOMIC DESIGN TOOLBOX - Think through supply, issuance, sinks, pricing, safety, sustainability.
8. LEGAL & REGULATORY DESIGN - Reflect on functional classification and regulatory constraints.
9. TECHNICAL DESIGN - Assess on-chain vs off-chain; L1 vs L2; custody models.
10. POWER STRUCTURES - Analyze VOTING, INFORMATION, MARKET, MEDIATION power.
11. TEAM, ROADMAP & EVOLUTION - Consider progressive decentralization.

Your design philosophy emphasizes long-term sustainability, clear utility-value relationships,
progressive decentralization, and transparent articulation of trade-offs.

Your communication style is precise, implementation-focused, and analytical.
All responses must be professional, comprehensive, and directly actionable.

Formatting Requirements:
- Use exact numbers for total supply
- Format allocations as "Category: XX%"
- Ensure all percentages sum to 100%
- Justify decisions using reasoning and references from similar projects
"""


def create_structured_prompt(user_input: Dict, project_summaries: str) -> str:
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

    return f"""
    Transform the structured data below into a coherent tokenomics model.

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

    Provide: token role, total supply, allocation (Category: XX%), vesting schedule,
    governance design, economic model, and phased roadmap.
    """


def create_generic_prompt(user_input: Dict, project_summaries: str) -> str:
    return f"""
    Given the project description below, extract key design insights and produce a tokenomics model.

    USER'S PROJECT DESCRIPTION:
    {user_input.get('project_description', '')}

    PROJECT NAME: {user_input.get('project_name', 'Extract from description')}

    Reference Projects:
    {project_summaries}

    Provide: token role, total supply, allocation (Category: XX%), vesting schedule,
    governance design, economic model, and phased roadmap. Ensure allocations sum to 100%.
    """


def ask_openai_enhanced(prompt: str, input_type: str = "structured",
                        model_override: Optional[str] = None) -> str:
    system_msg = str(_SYSTEM_MESSAGE)
    if input_type == "generic":
        system_msg += "\n\nNOTE: Infer project details from the limited description and your expertise."
    else:
        system_msg += "\n\nNOTE: Use the structured input as primary source of truth."

    client = _get_client()
    primary = model_override or os.getenv("OPENAI_MODEL", "gpt-4o")
    fallback = model_override or os.getenv("OPENAI_MODEL_FALLBACK", "gpt-4o")

    def call_model(model_name: str, max_tokens: int = 2400) -> str:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=max_tokens,
        )
        return resp.choices[0].message.content if resp.choices else ""

    try:
        content = call_model(primary, max_tokens=2400)
        if not (content and content.strip()):
            content = call_model(fallback, max_tokens=1800)
        if not (content and content.strip()):
            return "Error: LLM response was empty. Please try again."
        return content
    except Exception as e:
        return f"Error generating recommendation: {e}"


# ── Proposal construction ─────────────────────────────────────

def generate_tokenomics_proposal(user_input: Dict,
                                 result_text: str) -> Tuple[GeneratedTokenomics, ProjectContext]:
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

    if user_input.get("input_type") == "generic":
        description = user_input.get("project_description", "")
    else:
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


def proposal_from_dataset_entry(entry: Dict) -> Tuple[GeneratedTokenomics, ProjectContext]:
    allocation = {}
    raw_alloc = entry.get("allocation", {})
    if isinstance(raw_alloc, dict):
        for k, v in raw_alloc.items():
            try:
                allocation[k] = float(v)
            except:
                pass
    else:
        allocation = {"Community": 100.0}

    initial_supply = entry.get("tokenomics", {}).get("total_supply") or entry.get("total_supply")
    token_supply = parse_token_supply(initial_supply)

    notes = entry.get("notes", "Historical project context")
    from models import TokenomicsParameters, TokenDesignThinking, VestingDetail

    params = TokenomicsParameters(total_supply=token_supply, allocation=allocation, vesting={})
    design = TokenDesignThinking(
        purpose=notes, principles=[], positioning="", functions=[], stakeholders=[],
        economic_design="", legal_design="", tech_design="", power_structures="", team={},
    )
    metadata = ProjectMetadata(
        project=entry.get("project", "Historical"),
        token=entry.get("token", "TKN"),
    )
    gen = GeneratedTokenomics(
        project_metadata=metadata, token_design_thinking=design,
        tokenomics_parameters=params, references={},
    )
    context = ProjectContext(
        description=notes, goals=[], priorities=[], constraints={},
        legal_risk_tolerance="balanced", economic_signals={},
    )
    return gen, context


def proposal_from_entry_via_llm(entry: Dict,
                                project_summaries: str) -> Tuple[GeneratedTokenomics, ProjectContext]:
    description_lines = []
    if entry.get("notes"):
        description_lines.append(f"Notes: {entry['notes']}")
    if entry.get("allocation"):
        try:
            alloc_str = ", ".join(f"{k}: {v}%" for k, v in entry["allocation"].items())
            description_lines.append(f"Known historical allocation: {alloc_str}")
        except Exception:
            pass
    description = "\n".join(description_lines) or "Historical project without detailed notes."

    user_input = {
        "input_type": "generic",
        "project_name": entry.get("project", "Historical Project"),
        "project_description": description,
    }
    prompt = create_generic_prompt(user_input, project_summaries)
    llm_result = ask_openai_enhanced(prompt, input_type="generic")
    return generate_tokenomics_proposal(user_input, llm_result)
