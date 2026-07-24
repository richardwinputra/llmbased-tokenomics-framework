# LLM-Based Tokenomics Screening Framework

An LLM-assisted framework for designing, screening, and evaluating token economic models. The system uses a three-module pipeline — Generation, Control & Filter, and Simulation & Evaluation — grounded in the Token Design Thinking methodology (Voshmgir) and benchmarked against a knowledge base of 119 real-world token projects.

## Architecture

```
Input Base → Tokenomics Generation Module → Output Control & Filter Module → Simulation & Evaluation Module
```

**Tokenomics Generation Module**: Accepts structured or generic project descriptions, retrieves relevant precedents from the knowledge base (RAG), and prompts an LLM to produce a complete tokenomics proposal including allocation, vesting, and governance parameters.

**Output Control & Filter Module**: A two-layer validation gate. The Control Layer (Table II) runs diagnostic checks on allocation concentration, vesting adequacy, and fairness. The Filter Layer (Table III) enforces hard constraints — completeness, non-negativity, supply consistency — and recalculates on failure.

**Simulation & Evaluation Module**: Evaluates the screened proposal through supply release simulation (cliff-and-linear vesting over 60 months), fairness evaluation (insider share, distributed share, Gini tracking), and sustainability stress testing (5 scenarios, 70% pass criterion).

## Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation
```bash
pip install openai python-dotenv numpy
```

### Setup
Create a `.env` file:
```env
OPENAI_API_KEY=your_key_here
# For the multi-LLM comparison (model slugs containing "/"):
OPENROUTER_API_KEY=your_key_here
# Optional: pin a serving provider for reproducibility
# OPENROUTER_PROVIDER=anthropic
```

### Run
```bash
# Interactive mode
python main.py

# From JSON input file
python main.py --input-file sample_input.json

# Batch experiment runner (default condition: primary model, KB prompt)
python run_full_experiment.py

# Multi-LLM comparison via OpenRouter (model slugs with "/"; needs
# OPENROUTER_API_KEY in .env). Each condition writes to
# experiment_results/runs/<model>_<variant>[_rN]/
python run_full_experiment.py --model anthropic/claude-sonnet-4.5
python run_full_experiment.py --model google/gemini-2.5-flash --prompt-variant no-kb
python run_full_experiment.py --repeats 3   # generation consistency

# Analyze a specific condition
python analyze_results.py --dir experiment_results/runs/<tag>
```

### Free OpenRouter models

Free variants (model IDs ending in `:free`) are rate-limited: 20 requests/min
and 50 requests/day (1,000/day once ≥ $10 of credits has ever been purchased —
the credits themselves are not consumed by free models). Practical notes:

- The runner auto-sets `--sleep 4` for `:free` models (20 req/min cap) and
  retries transient 429s; daily-cap failures are not retried in-run.
- One condition = 100 requests, so at 50/day a run spans 3 days — just re-run
  the same command; the checkpoint resumes and retries failed projects.
- The KB-augmented prompt is ~11.5k tokens: pick free models with ≥ 32k
  context.
- If you get "No endpoints found matching your data policy", enable free
  endpoints under OpenRouter Settings → Privacy.

```bash

# Post-experiment analysis
python analyze_results.py

# Offline replay (no API calls): re-runs parse -> control -> filter -> simulation
# from the saved raw LLM responses. Reports parser correction-step triggers
# (raw vs repaired output) and re-evaluates stress tests with the current code.
python analyze_results.py --replay

# Sensitivity sweep of the SDR demand-baseline parameter alpha
python analyze_results.py --sensitivity

# Table II cutoffs vs the empirical KB distributions (percentile derivation)
python analyze_results.py --kb-thresholds

# One-at-a-time sensitivity sweeps (control thresholds / stress parameters)
python analyze_results.py --control-grid
python analyze_results.py --stress-grid

# Retrospective screening of the real knowledge-base projects
python analyze_results.py --kb-screen

# Generate publication-quality figures (Figs 3-8)
python visualize.py
```

## CLI Options

| Flag | Description |
|------|-------------|
| `--seed` | Random seed for reproducibility (default: 42) |
| `--input-file` | Path to JSON input (bypasses interactive mode) |
| `--model-override` | Override the OpenAI model name |
| `--output-dir` | Output directory (default: pipeline_exports) |

## Project Structure

| File | Purpose |
|------|---------|
| `models.py` | Data structures for all pipeline stages |
| `utils.py` | Parsing, normalization, Gini computation, allocation classification |
| `llm_engine.py` | Prompt construction, OpenAI API interaction, proposal generation |
| `control_filter.py` | Control Layer (Table II) and Filter Layer (Table III) |
| `simulation.py` | Supply release, fairness evaluation, stress testing |
| `main.py` | Pipeline orchestration and CLI |
| `run_full_experiment.py` | Full experiment runner (batch execution of structured inputs) |
| `analyze_results.py` | Post-experiment analysis; `--replay` (offline re-evaluation from saved raw responses) and `--sensitivity` (alpha sweep) |
| `visualize.py` | Generates figures based on experiment outputs (Figs 3–8) |
| `TokenomicsKnowledge.json` | Knowledge base of 119 token projects |

## Stress-Test Demand Baseline

The SDR stress test compares circulating supply C(m) against a demand
trajectory D(m) = D(0)·(1+g)^m. Because most generated proposals launch fully
locked (C(0) = 0), initializing demand at the launch float would leave the
ratio undefined; D(0) is therefore anchored as **D(0) = max(C(0), αS)** with
`simulation.DEMAND_BASELINE_ALPHA = 0.15` (≈ typical initial float at TGE).
Screening outcomes are sensitive to α, so headline pass rates should be
reported together with the sweep produced by
`python analyze_results.py --sensitivity` (Fig. 8):

| α | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 |
|---|------|------|------|------|------|
| Overall pass (of 100) | 1 | 12 | 57 | 97 | 100 |

Replay verification (`--replay`) reproduces the saved experiment exactly when
run against unchanged simulation code, and reports parser correction-step
trigger counts: for the 100 saved responses, all fields were extracted from
the structured JSON block (regex/table/prose fallbacks and normalization were
never triggered), so raw and repaired outputs are identical.
