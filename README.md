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
```

### Run
```bash
# Interactive mode
python main.py

# From JSON input file
python main.py --input-file sample_input.json

# Batch experiment runner
python run_full_experiment.py

# Post-experiment analysis
python analyze_results.py

# Generate publication-quality figures
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
| `analyze_results.py` | Post-experiment data analysis and tabular exports |
| `visualize.py` | Generates figures based on experiment outputs |
| `TokenomicsKnowledge.json` | Knowledge base of 119 token projects |
