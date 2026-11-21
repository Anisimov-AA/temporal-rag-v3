# Temporal RAG for a credit union website

A chatbot that answers member questions from 5 dated downloads of a credit union website. It picks the right download for "now" or "as of" questions, puts the newest facts first, cites its sources, and says "I don't know" when the site does not say.

**Experiments:** [the map of every step I tried and its results](https://anisimov-aa.github.io/temporal-rag-v3/config.html). The pipeline pages and the answers pages open from there.

The task is in [TASK.md](TASK.md).

## What it needs

- Python 3.11 and one GPU with 24 GB (I used an L4).
- [Ollama](https://ollama.com) 0.40 or newer, with two models:

      ollama pull bge-m3
      ollama pull gemma4:26b-a4b-it-q4_K_M

- No API keys.

## Install

    pip install torch --index-url https://download.pytorch.org/whl/cu128
    pip install -r requirements.txt

The reranker (BAAI/bge-reranker-v2-m3) downloads from Hugging Face on the first run.

## Run

Ollama must be running. Tell the solution where it is:

    export GCP_OLLAMA=http://localhost:11434
    export OLLAMA_HOST=http://localhost:11434

Then run the organizers' scorer:

    python harness/run_eval.py --submission solution --eval eval/public_eval.jsonl

The first run unpacks the snapshots and builds one index per download. This takes about 5 minutes. Later runs reuse the index.

## Results

On one L4, on the 40 public questions plus 80 reworded ones:

- accuracy 0.767 (0.839 without the 3 broken answer keys)
- the same answer on a repeat: 0.89
- p95 answer time 8 s
- final score 0.707 (conflicts not scored)

## Data

The data is a real credit union's public website, shared for the hackathon only. It is not in this repo.
