# LLM Clustering Framework

A comprehensive framework for evaluating LLM-enhanced clustering techniques, including embedding generation, keyphrase expansion, and clustering correction. Inspired by [Viswanathan et al. (2023)](https://arxiv.org/abs/2307.00524), this implementation extends the original work with modern LLMs and OpenAI embeddings.

<p align="center">
<img width="514" alt="image" src="https://github.com/user-attachments/assets/2253b9e6-e6c6-4f18-8a0f-aec46027700a" />
</p>

## Features

- **Multi-method Evaluation**:
  - Naive KMeans baseline
  - Keyphrase-enhanced clustering
  - LLM-based clustering correction
  - Pairwise constraints (PCKMeans) with LLM oracle

- **Integrated Workflow**:
  - Dataset loading with Hugging Face integration
  - Embedding generation (OpenAI/Instructor)
  - Clustering execution
  - Comprehensive metric evaluation

- **Technical Highlights**:
  - Configurable prompts and models
  - Automatic caching of embeddings
  - Detailed experiment logging
  - Multi-metric evaluation (Accuracy, F1, NMI, ARI)

## Project Structure

- **main/**: Experiment scripts
  - `run_naive_baseline.py`
  - `run_keyphrase.py`
  - `run_correction.py`
  - `run_pairwise.py`
- **src/**: Core logic
  - `baselines.py`, `config.py`, `data.py`, `llm_service.py`, `metrics.py` : Helper Modules.
  - `clustering_methods/`: Enhancement methods.

## Setup

1. Clone repo
2. Create and activate your Python virtual environment.
3. `pip install -r requirements.txt`
4. Create `.env` with `OPENAI_API_KEY=sk-...`
5. Run experiments:
   ```bash
   python -m main.run_naive_baseline tweet
   python -m main.run_pairwise banking77
   
