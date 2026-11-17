# Temporal RAG for a credit union website

A chatbot that answers member questions from 5 dated downloads of a credit union website. It picks the right download for "now" or "as of" questions, trusts the official rates page over old posts, cites its sources, and says "I don't know" when the site does not say.

- **How it works:** [project map](https://anisimov-aa.github.io/temporal-rag-v3/), every step with a real example
- **Experiments:** [results](https://anisimov-aa.github.io/temporal-rag-v3/results.html), 22 configs on 3 models
- **The task:** [TASK.md](TASK.md)

## Result

| | final score | accuracy | consistency | speed |
|---|---|---|---|---|
| starter | 0.093 | | | |
| my solution | 0.694 | 0.74 | 1.0 | 0.64 |

Measured on the 40 public questions. Conflicts are scored only on the hidden set.

## How it works

1. I load each download, clean the pages with 11 rules and split them by headings.
2. I store the chunks of each download in its own vector index (bge-m3) and a BM25 index.
3. For a question, I pick the download by date, search both indexes, join the lists and rerank them (bge-reranker-v2-m3).
4. Qwen3-30B-A3B writes the answer from the top 5 chunks.
5. Code picks the sources: every chunk that holds the value from the answer.
6. Separately, I read all rate tables and list where a rate changed or two pages disagree.

## Run

The model runs in Ollama. Set `GCP_OLLAMA` to its address, then:

    python harness/run_eval.py --submission solution --eval eval/public_eval.jsonl

## Folders

    solution/   my solution: solution.py and one file per step
    dev/        experiments: every setting in one config
    harness/    the scorer from the task
    eval/       40 practice questions

## Data

The data is a real credit union's public website, shared for the hackathon only. It is not in this repo.
