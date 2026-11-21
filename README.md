# Temporal RAG for a credit union website

A chatbot that answers member questions from 5 dated downloads of a credit union website. It picks the right download for "now" or "as of" questions, puts the newest facts first, cites its sources, and says "I don't know" when the site does not say.

**Experiments:** [the map of every step I tried and its results](https://anisimov-aa.github.io/temporal-rag-v3/config.html). The pipeline pages and the answers pages open from there.

The task is in [TASK.md](TASK.md).

## What it runs on

Everything runs on one NVIDIA GPU with 24 GB of memory. There are no API calls and no API keys.

- Gemma 4 26B A4B, 4-bit, in Ollama: writes the answer
- bge-m3 in Ollama: builds the search index
- bge-reranker-v2-m3 (Hugging Face, fp16): reranks the search results

## Setup

**1. A Linux machine with an NVIDIA GPU with 24 GB of memory** (I used an L4), 32 GB RAM and 60 GB of disk. It can be your own computer, a cloud VM or a cluster node. Check that the driver sees the GPU:

    nvidia-smi

**2. Ollama.** Install it from [ollama.com/download/linux](https://ollama.com/download/linux). Without sudo (for example on a cluster), use the manual install from the same page and unpack it into your home folder.

Start it with three options and pull the two models:

    OLLAMA_FLASH_ATTENTION=1 OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=-1 ollama serve &
    ollama pull bge-m3
    ollama pull gemma4:26b-a4b-it-q4_K_M

If the installer already started Ollama as a service, stop it first (`sudo systemctl stop ollama`) so the options apply.

**3. Python 3.11 and the packages.**

    git clone https://github.com/Anisimov-AA/temporal-rag-v3.git
    cd temporal-rag-v3
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv venv --python 3.11 && source .venv/bin/activate
    uv pip install torch --index-url https://download.pytorch.org/whl/cu128
    uv pip install -r requirements.txt

**4. The data.** The website downloads are not in this repo. Put the `snapshots_zip/` folder from the hackathon kit next to `snapshots/`. The scorer unpacks it on the first run.

**5. Where Ollama is.** Create a `.env` file in the top folder:

    GCP_OLLAMA=http://localhost:11434

## Run

    python harness/run_eval.py --submission solution --eval eval/public_eval.jsonl

- The first run builds one index per download. This takes about 5 minutes. Later runs reuse the index in `index/`.
- The reranker downloads from Hugging Face on the first run.
- The score and every answer are written to `results/`.

To check the 40 questions plus 80 reworded ones, use `--eval eval/full_eval.jsonl --repeat-sample 40`.

## Results

On one L4, on the 40 public questions plus 80 reworded ones:

- accuracy 0.767 (0.839 without the 3 broken answer keys)
- the same answer on a repeat: 0.89
- p95 answer time 8 s
- final score 0.707 (0.773 without the broken keys)

The final score counts accuracy, repeats and speed. The kit has no answer key for conflicts, so that part is left out.

## Data

The data is a real credit union's public website, shared for the hackathon only. It is not in this repo.
