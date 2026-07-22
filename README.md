# Adaptive Memory Selection for Long-Term AI Agents

**CS6140: Machine Learning — Final Project**
Raghavendra Beri (002591668) · Harsha Prakash (002078092) — MS Data Science

Can a supervised ML classifier predict which past conversational **memories** will matter
to future turns *better than* plain similarity-based retrieval? We treat memory selection
as a **binary classification** problem (retain vs. discard), engineer interpretable
features over the [Multi-Session Chat (MSC)](https://parl.ai/projects/msc/) corpus, compare
five classifiers, and wire the best one into a local LLM (Ollama `llama3.1:8b`) to show it
improves answer quality while using far fewer context tokens than store-all or top-K retrieval.

## Pipeline

```
download_data.py  ->  parse_msc.py  ->  features.py  ->  labeling.py  ->  train.py  ->  evaluate.py
   (MSC tarball)      (memory recs)     (embeddings +     (retain/         (5 models,     (metrics.csv,
                                         features)         discard labels)   CV, tuning)    figures)
                                                                                  |
                                                              llm_integration.py  +  app/streamlit_app.py
                                                              (4 retrieval modes, LLM-as-judge, demo)
```

## Setup

```bash
# Python 3.10+ recommended. Optional but encouraged: a dedicated venv.
python3 -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt

# The LLM demo needs Ollama (https://ollama.com) with a Llama 3 model:
ollama pull llama3.1:8b        # generation
ollama pull dolphin-mistral    # independent LLM-as-judge
```

## Reproduce results

```bash
python src/download_data.py      # ~48 MB MSC tarball -> data/raw/
python src/parse_msc.py          # -> data/processed/memories.jsonl
python src/features.py           # embeddings (cached) + features -> data/processed/features.parquet
python src/labeling.py           # labels + validation subset -> features_labeled.parquet
python src/train.py              # trains/tunes 5 models -> models/, results/metrics.csv
python src/evaluate.py           # -> reports/figures/*.png
python src/llm_integration.py    # 4-condition LLM comparison (requires Ollama running)

streamlit run app/streamlit_app.py   # interactive demo
```

Or step through the notebooks in `notebooks/` (01 exploration → 02 EDA → 03 modeling → 04 LLM demo).

## Layout

| Path | Purpose |
|------|---------|
| `src/` | Pipeline modules (download, parse, features, labeling, train, evaluate, LLM) |
| `notebooks/` | Narrative EDA, modeling, and demo notebooks |
| `app/streamlit_app.py` | Interactive memory-selection demo |
| `models/` | Saved best classifier + scaler/PCA |
| `results/` | `metrics.csv` and derived tables |
| `reports/` | LaTeX source, figures, and the final `report.pdf` |
| `slides/` | Presentation deck + speaking script |

## Data

Multi-Session Chat (MSC) v0.1, released via ParlAI. Downloaded automatically by
`src/download_data.py`. Not redistributed in this repo (see `.gitignore`).

> Xu, J., Szlam, A., Weston, J. *Beyond Goldfish Memory: Long-Term Open-Domain
> Conversation.* ACL 2022.
