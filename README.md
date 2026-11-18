# Temporal RAG for a credit union website

A chatbot that answers member questions from 5 dated downloads of a credit union website. It picks the right download for "now" or "as of" questions, trusts the official rates page over old posts, cites its sources, and says "I don't know" when the site does not say.

- **How it works:** [pipeline map](https://anisimov-aa.github.io/temporal-rag-v3/pipeline_map.html), every step with a real example
- **Experiments:** [experiments](https://anisimov-aa.github.io/temporal-rag-v3/configs_map.html), each setting tried and its results
- **The task:** [TASK.md](TASK.md)

## Run

The model runs in Ollama. Set `GCP_OLLAMA` to its address, then:

    python harness/run_eval.py --submission solution --eval eval/public_eval.jsonl

## Data

The data is a real credit union's public website, shared for the hackathon only. It is not in this repo.
