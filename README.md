# 🪙 Tokenomics Design & Analysis Tool

An AI-powered framework for designing, simulating, and validating token economic models. This tool leverages OpenAI's GPT-4 to generate comprehensive tokenomics proposals while providing rigorous analytical validation through Monte Carlo simulations, Gini-based fairness metrics, and historical benchmarking.

---

## 🚀 Key Features

### 🤖 AI-Driven Tokenomics Engine
- **Two Input Modes**:
  - **Structured Intake**: Detailed questionnaire for precise control over project parameters (legal, technical, economic).
  - **Generic Description**: Natural language project descriptions for rapid prototyping.
- **Context-Aware Design**: Automatically infers constraints, legal risk profiles, and economic signals from inputs.

### 📊 Advanced Analytics & Simulations
- **Monte Carlo Simulations**: Predict potential token supply scenarios and distribution paths over time.
- **Agent Market Simulation**: Simulate price impact, liquidity depth, and market sentiment based on proposed allocations.
- **Fairness Metrics**: Calculate Gini coefficients at T=0, 12m, and 24m to measure wealth concentration and decentralization.
- **Governance Risk Assessment**: Automated scoring of capture risks and voter turnout projections.

### 📚 Knowledge Base & Benchmarking
- **Historical Analysis**: Built-in dataset of successful Web3 projects (Aave, Chainlink, Uniswap, etc.) for similarity matching and benchmarking.
- **Risk Flagging**: Identify potential failure modes by comparing new proposals with historical drawdown and inflation data.

---

## 🛠 Quick Start

### 1. Prerequisites
- Python 3.8+
- OpenAI API Key

### 2. Installation
```bash
git clone <repository-url>
cd llmbased-tokenomics
pip install -r requirements.txt
```
*Note: If `requirements.txt` is missing, install core dependencies:*
```bash
pip install openai python-dotenv matplotlib numpy
```

### 3. Setup
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run the Pipeline
```bash
python main.py
```

---

## 📖 Usage Guide

### Simulation & Analysis Options
When running the tool, you can select various analysis depths:
- **A/B Testing**: Benchmarks your proposal against the most similar successful project in the knowledge base.
- **Monte Carlo Simulation**: Generates probabilistic outcomes for supply inflation and distribution.
- **Full Comprehensive Analysis**: Runs all simulations, fairness checks, and risk assessments.

### CLI Arguments
- `--historical-dataset`: Path to a custom JSON dataset (defaults to `TokenomicsKnowledge.json`).
- `--dataset-report`: Prints a summary of the current historical dataset and exits.
- `--seed`: Set a random seed for deterministic simulation results.

---

## 📂 Project Structure

- `main.py`: Core logic for input handling, AI orchestration, and analysis pipeline.
- `advanced_simulations.py`: Market and agent-based simulation engines.
- `TokenomicsKnowledge.json`: Curated dataset of historical tokenomics models.
- `exported_charts/`: Directory where visualization outputs (pie charts, distribution graphs) are saved.

---

## 🤝 Contributing
Contributions are welcome! Please ensure you follow the existing code structure and add unit tests for any new simulation logic.