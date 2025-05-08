# LLM Clustering Project

A framework to explore and evaluate clustering methods enhanced by Large Language Models (LLMs), including embedding generation, keyphrase expansion, and clustering correction.

This is directly inspired from the followig paper: [Viswanathan et al. (2023)](https://arxiv.org/abs/2307.00524) And is an attempt at replicating its results with new LLM models and the OpenAI embedding models.


## Project Structure

- **main/**: Experiment scripts
  - `run_naive_baseline.py`
  - `run_keyphrase.py`
  - `run_correction.py`
  - `run_pairwise.py`
- **src/**: Core logic
  - `baselines.py`, `config.py`, `data.py`, `llm_service.py`, `metrics.py`
  - `clustering_methods/`: Clustering enhancement methods

## Features

- **Data Loading**: Supports Hugging Face datasets and caching
- **LLM Service**: Unified interface for OpenAI & Instructor embeddings
- **Baselines**: Standard clustering methods : Naive KMeans 
- **LLM Methods**: Keyphrase expansion, clustering correction and pairwise PCKMeans using an LLM as a pseudo-oracle 
- **Metrics**: Accuracy, Precision, Recall, F1, NMI, ARI
- **Config Management**: Via `src/config.py` for changing prompts, and setting up configuration variables, as well as changing embedding and generation models.

## Setup

1. Clone the repo
2. Install required dependencies : `pip install -r requirements.txt`
3. create a `.env` variable and define `OPENAI_API_KEY = sk-...`
4. launch the command : `source .env`
5. to launch a certain method experiment, use the following commands from the root folder of the repository : 
   1. `python -m main run_method_name dataset_name`
   2. the method experiment will be launched with visual feedback for the progress for embeddings and text generations.
   3. the metrics csv file will be populated with the new calculated metrics for the experiment.
   4. Another file with the prompts and what the LLM generated will also be created.

If you want to contribute to the repo and enhance it, feel free to clone it and open up a Pull Request with your suggested enhancements ^^.

Thank you.