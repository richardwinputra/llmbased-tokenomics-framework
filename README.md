# LLM-Based Tokenomics Screening Framework

An LLM-assisted framework for generating, screening, and evaluating token economic
models. The pipeline has three stages — Generation, Control & Filter, and Simulation
& Evaluation — grounded in the Token Design Thinking methodology (Voshmgir) and a
knowledge base of 119 real-world token projects.

## Architecture

```
Input → Generation Module → Control & Filter Module → Simulation & Evaluation Module
```

- **Generation** — retrieves precedents from the knowledge base (RAG) and prompts an
  LLM to produce a complete tokenomics proposal (allocation, vesting, supply).
- **Control & Filter** — the control layer runs diagnostic checks (Table II); the
  filter layer enforces hard validity constraints (Table III).
- **Simulation & Evaluation** — supply-release simulation (cliff-and-linear vesting
  over 60 months), fairness/Gini tracking, and a five-scenario SDR stress test.

## Setup

Requires Python 3.9+.

```bash
pip install openai python-dotenv numpy pandas matplotlib seaborn
```

Create a `.env` file:

```env
# Primary model (GPT-5.4-mini) is called through the OpenAI API
OPENAI_API_KEY=your_key_here

# Additional models are called through OpenRouter (any slug containing "/")
OPENROUTER_API_KEY=your_key_here

# Optional: pin a serving provider for reproducibility
# OPENROUTER_PROVIDER=anthropic
```

Model routing is automatic: a model name containing `/` (e.g. `anthropic/claude-sonnet-4.5`)
is sent through OpenRouter; a plain name (e.g. `gpt-5.4-mini`) uses the OpenAI API.

## Usage

```bash
# Single proposal
python main.py --input-file test_input.json

# Full experiment (primary model, KB-augmented prompt)
python run_full_experiment.py

# Other models via OpenRouter; each condition writes to
# experiment_results/runs/<model>_<variant>/
python run_full_experiment.py --model nvidia/nemotron-3-super-120b-a12b
python run_full_experiment.py --model qwen/qwen3.6-27b --prompt-variant no-kb

# Analysis
python analyze_results.py                 # aggregate tables
python analyze_results.py --replay        # re-run pipeline offline from saved responses
python analyze_results.py --sensitivity   # sweep the demand-baseline alpha
python analyze_results.py --kb-thresholds # Table II cutoffs vs KB distributions
python analyze_results.py --control-grid  # control-threshold sweep
python analyze_results.py --stress-grid   # stress-parameter sweep
python analyze_results.py --kb-screen     # screen the 119 real KB projects
python analyze_results.py --dir experiment_results/runs/<tag>   # analyze one run

# Figures (Fig. 3 allocation, Fig. 4 supply, Fig. 5 fairness, Fig. 6 alpha)
python visualize.py
```

### Free OpenRouter models

Model IDs ending in `:free` are rate-limited (20 req/min, 50 req/day; 1,000/day
once ≥ $10 of credits has been purchased). The runner slows to `--sleep 4` for
`:free` models and retries transient 429s; a full 100-input condition may span
several days at the daily cap, but re-running the same command resumes from the
checkpoint. The KB-augmented prompt is ~11.5k tokens, so use models with ≥ 32k
context, and enable free endpoints under OpenRouter Settings → Privacy if a run
reports "No endpoints found matching your data policy".

## Project Structure

| File | Purpose |
|------|---------|
| `models.py` | Data structures for the pipeline |
| `utils.py` | Parsing, normalization, Gini, allocation classification |
| `llm_engine.py` | Prompt construction and LLM calls (OpenAI / OpenRouter) |
| `control_filter.py` | Control layer (Table II) and filter layer (Table III) |
| `simulation.py` | Supply release, fairness, stress testing |
| `main.py` | Single-run pipeline and CLI |
| `run_full_experiment.py` | Batch experiment runner |
| `analyze_results.py` | Post-experiment analysis and sensitivity/replay sweeps |
| `visualize.py` | Figure generation (Figs. 3–6) |
| `TokenomicsKnowledge.json` | Knowledge base of 119 token projects |
| `batch_inputs.json` | 100 evaluation narratives |

## Demand baseline

The stress test compares circulating supply C(m) against demand
D(m) = D(0)·(1+g)^m. Because most proposals launch fully locked (C(0) = 0),
D(0) is anchored as **D(0) = max(C(0), αS)** with `DEMAND_BASELINE_ALPHA = 0.15`
in `simulation.py`. Pass rates are sensitive to α, so they should be read with
the sweep from `python analyze_results.py --sensitivity` (Fig. 6):

| α | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 |
|---|------|------|------|------|------|
| Overall pass (of 100) | 1 | 12 | 57 | 97 | 100 |
