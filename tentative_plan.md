# Tentative Plan — PubMed RAG over NSCLC

> Project context document. Read this first when resuming work from a new device or new session.

---

## 1. What this project is and why

A deployable Retrieval-Augmented Generation (RAG) pipeline over ~1000 PubMed abstracts on **non-small cell lung cancer (NSCLC)**. Users ask clinical questions, the system retrieves relevant abstracts, and a Claude model generates an evidence-grounded answer with PMID citations.

**Portfolio target:** entry-to-mid-level (2–4 YOE) **data scientist / GenAI product developer** roles. **Tempus** is the explicit target company (their oncology footprint is heavy in NSCLC), but the signal generalizes to any biotech / health-tech / AI-product-shaped role.

**The differentiator vs. typical RAG demos:** measurement and methodology — a hand-built NSCLC gold question set, retrieval ablations with plots, citation-grounded answers, and quantitative numbers in the README. Most portfolio RAG projects ship the happy path and stop; this one earns the "engineered" label.

**Demo constraint:** the deployed app must be openable live in an interview from any laptop or phone. No local-machine dependency, no flaky cold starts, no API rate-limit surprises.

---

## 2. Confirmed decisions (the stack)

| Decision | Choice | Why (one-liner) |
|---|---|---|
| Disease scope | **NSCLC only** (configurable PubMed query) | Density at 1000 abstracts; tightest Tempus fit |
| Frontend / hosting | **Streamlit Community Cloud** (free, GitHub auto-deploy) | Right signal for DS roles; zero ops |
| Embeddings | **Voyage AI `voyage-3-large`** | Anthropic-owned, biomedical-strong, free tier covers project, no local model |
| Vector store | **Chroma in-process**, persistent dir committed to repo | No service to manage; ride-along with `git clone` |
| Hybrid retrieval | **`rank_bm25` + dense, fused via Reciprocal Rank Fusion** | Keyword wins on rare terms (T790M, KRAS G12C); semantic wins on paraphrase |
| Reranker | **Voyage Rerank API** (`rerank-2`) | No local model, lifts recall@5 |
| Generator | **Claude Sonnet 4.6** with **prompt caching** + **streaming** | Quality/cost sweet spot; caching makes repeat queries near-free |
| Eval | Hand-built gold set (~25 NSCLC Q+PMID pairs) + recall@k, MRR, LLM-judge faithfulness | The single biggest "senior" signal in the project |
| Ablations | `notebooks/ablations.ipynb` with retrieval-config comparisons + plots | DS-flavor differentiator; experimental science |
| Index lifecycle | **Pre-built locally, committed** to repo (no Entrez at runtime) | Demo reliability |
| Surface | Streamlit deployed; `api.py` in repo as engineering-taste signal, not deployed | Demonstrates pipeline-as-library separation |

**Total cost:** ~$10–20 across development, $0/month idle, ~pennies per demo session.

---

## 3. Repository structure

```
pubmed-rag/
├── README.md                       # overview + eval numbers table + demo URL
├── tentative_plan.md               # THIS FILE
├── pyproject.toml
├── config/
│   └── config.yaml                 # query, models, top-k, rerank-k
├── .env.example                    # NCBI_API_KEY, VOYAGE_API_KEY, ANTHROPIC_API_KEY
├── .streamlit/
│   └── secrets.toml.example        # for Streamlit Cloud secret config
├── data/
│   ├── raw/abstracts.jsonl         # committed (1000 NSCLC abstracts)
│   └── chroma/                     # committed persistent Chroma store
├── src/pubmed_rag/
│   ├── __init__.py
│   ├── settings.py                 # pydantic-settings loader
│   ├── ingest.py                   # Entrez fetch → jsonl (run locally, not in prod)
│   ├── chunking.py                 # no-op default (title + abstract = 1 chunk)
│   ├── embeddings.py               # Voyage embeddings client wrapper
│   ├── vectorstore.py              # Chroma persistent client
│   ├── retriever.py                # dense search + BM25 + RRF fusion + filters
│   ├── reranker.py                 # Voyage Rerank API wrapper
│   ├── generator.py                # Generator interface + ClaudeGenerator (streaming)
│   ├── prompts.py                  # citation-grounded answer prompt
│   ├── pipeline.py                 # thin orchestrator: retrieve → rerank → generate
│   ├── cli.py                      # typer: ingest / index / query / eval
│   └── api.py                      # FastAPI module (NOT deployed; in-repo only)
├── app/
│   └── streamlit_app.py            # the deployed surface
├── eval/
│   ├── gold.jsonl                  # ~25 NSCLC (question, expected_pmids) — user-authored
│   ├── metrics.py                  # recall@k, MRR, faithfulness (LLM-judge)
│   └── run_eval.py                 # writes eval/results.md
├── notebooks/
│   ├── exploration.ipynb           # corpus inspection
│   └── ablations.ipynb             # retrieval comparisons + plots
├── tests/
│   ├── test_ingest.py              # mocked Entrez
│   ├── test_retrieval.py           # in-memory index, asserts top-1 match
│   └── test_generator.py           # mocked Anthropic client
└── docs/
    ├── architecture.md             # diagram, design choices, eval methodology
    └── figures/                    # ablation plots embedded in README
```

---

## 4. Component-by-component reasoning

### 4.1 Ingest (`src/pubmed_rag/ingest.py`)

Uses Biopython `Bio.Entrez` (`esearch` + `efetch`) to fetch PMIDs and abstracts matching the configured query. Output: `data/raw/abstracts.jsonl`.

- **Why Biopython:** standard wrapper around NCBI E-utilities; handles batching/retries; signals familiarity with the bioinformatics ecosystem.
- **Why JSONL:** append-friendly, streaming-friendly, doesn't need full-file load.
- **Fields per record:** `pmid, title, abstract, authors[], journal, pub_date, mesh_terms[], url`. PMID is the immutable primary key; MeSH terms enable curated topic filtering; `url = https://pubmed.ncbi.nlm.nih.gov/{pmid}/` for clickable citations.
- **Why `NCBI_API_KEY`:** raises rate limit from 3 → 10 req/s. Free, 30-second registration.
- **Default query (in `config.yaml`):** `("non-small cell lung cancer"[Title/Abstract] OR "NSCLC"[Title/Abstract]) AND hasabstract[Filter] AND english[Language]`, capped at `n=1000`, sorted by date desc.
- **Run once locally; commit the output.** The deployed app never calls Entrez — eliminates a critical demo failure mode.

### 4.2 Chunking (`src/pubmed_rag/chunking.py`)

By default, **no chunking** — each abstract becomes a single chunk: `title + "\n\n" + abstract`.

- **Why no chunking:** abstracts are 150–300 words, well within embedding-model context. Splitting them would scatter related claims across vectors and *hurt* retrieval. Common reflex mistake — don't make it.
- **Why have a module anyway:** swap-in optionality for the ablations notebook (chunk vs. no-chunk vs. sentence-window).

### 4.3 Embeddings (`src/pubmed_rag/embeddings.py`)

Wraps the Voyage AI client. `embed_documents(texts)` and `embed_query(text)`, using Voyage's typed `input_type` parameter.

- **Why Voyage:** top biomedical retrieval benchmarks; Anthropic-owned (clean ecosystem story); zero local model = no RAM cost in the deployed app.
- **Why typed embeddings:** measurable retrieval lift from telling the model whether a string is a "document" vs. a "query."
- **Cost:** indexing 1000 abstracts ≈ 300 K tokens ≈ free-tier-trivial.

### 4.4 Vector store (`src/pubmed_rag/vectorstore.py`)

Thin wrapper around `chromadb.PersistentClient(path="data/chroma")`. Collection: `nsclc_abstracts`. Metadata: `pmid, title, journal, year, url`.

- **Why Chroma in-process:** no service to deploy; persistent across runs; native metadata filtering on the same collection.
- **Why not Qdrant/Pinecone/Weaviate:** for 1000 abstracts these add a service to manage. The "I built this myself" framing reads well for a DS portfolio.

### 4.5 Hybrid retrieval (`src/pubmed_rag/retriever.py`)

Runs dense (Chroma) and sparse (BM25) searches in parallel and fuses results via **Reciprocal Rank Fusion (RRF)**:
```
score(d) = Σ over retrievers of  1 / (60 + rank_i(d))
```

- **Why hybrid:** dense embeddings are good at semantics; BM25 wins on rare exact terms (gene/drug names — exactly what comes up in clinical questions).
- **Why RRF:** rank-based fusion that doesn't require comparable absolute scores. Constant 60 is from the Cormack et al. 2009 paper; standard, no tuning. Implementing it yourself in ~10 lines is a senior signal.
- **Why `rank_bm25`:** pure-Python, builds in milliseconds over 1000 docs at app startup, cached in `st.cache_resource`.
- **Pipeline:** dense top-50 ⊕ BM25 top-50 → RRF → top-20. Filters (`year_min`, `year_max`, `journals`) applied to both branches.

### 4.6 Reranker (`src/pubmed_rag/reranker.py`)

Voyage Rerank API. Takes top-20 hybrid candidates → returns top-5 cross-encoder-scored.

- **Why rerank:** bi-encoders (embeddings) encode query and document independently; cross-encoders see them together and produce much higher-quality scores. Standard pattern: bi-encoder narrows to 20–50, cross-encoder picks the final 5. Typically 5–15-point recall@5 lift.
- **Why API rather than local:** keeps deployment lightweight; no model download.

### 4.7 Generator (`src/pubmed_rag/generator.py` + `prompts.py`)

Calls Claude Sonnet 4.6 with the top-5 reranked abstracts and the user's question. Streams tokens.

- **Why Sonnet 4.6:** best quality/cost point for clinical synthesis over multiple abstracts. Haiku weaker on multi-doc reasoning; Opus overkill at this scale.
- **Why prompt caching:** system prompt + retrieved abstracts are large and partially repeat across queries. Cache breakpoint marks them as cacheable; subsequent queries pay ~10% input cost on the cached prefix. Critical for demo cost.
  - Cache breakpoint placement: cache `system + retrieved_abstracts`; **don't** cache the user question.
- **Why streaming:** a 4-second wait followed by a wall of text feels broken. Streaming feels alive. `st.write_stream` consumes Anthropic streams natively.
- **System prompt does three jobs:**
  1. Constrain scope: "answer only from provided abstracts; say so if insufficient."
  2. Force citations: "every clinical claim ends with `[PMID:nnnn]`."
  3. Set clinical tone: short, evidence-led, hedged appropriately.
- **Return shape:** generator yields `(token_chunk | citation_payload)` so Streamlit renders streaming text *and* populates citation cards in parallel.

### 4.8 Pipeline (`src/pubmed_rag/pipeline.py`)

~30-line orchestrator: `retrieve → rerank → generate`. Reusable across the CLI, FastAPI module, eval harness, and ablations notebook. **One pipeline, four entry points.**

### 4.9 Streamlit app (`app/streamlit_app.py`)

The deployed surface.

- Sidebar: filters (year range, journal multiselect), top-k slider, "show retrieval debug" toggle.
- Main: question input → streamed answer → citation cards (PMID badge, title, snippet, "Open in PubMed" button).
- Debug expander: top-5 with hybrid score + rerank score columns. Signals "I can interrogate my system, not just demo it."
- `st.cache_resource` loads Chroma + BM25 once on cold start.

**Why Streamlit (not Next.js / Gradio / FastAPI+React):** correct signal for DS / GenAI-product roles. Reviewers expect Streamlit/Gradio in DS portfolios. Next.js polish would be off-target and could *cost* signal.

### 4.10 CLI (`src/pubmed_rag/cli.py`)

Typer-based, four commands: `ingest`, `index`, `query`, `eval`. Build-time commands run once locally; query-time commands for headless testing.

### 4.11 FastAPI module (`src/pubmed_rag/api.py`) — in-repo, not deployed

`POST /query` and `GET /health` wrapping the pipeline. Lives in the repo to demonstrate the surface-as-thin-wrapper pattern. README explicitly mentions it: "Streamlit is the deployed surface; `api.py` is a thin FastAPI wrapper for headless use."

When asked "how would you productionize this?" → point at `api.py` + swap Chroma for managed vector DB + deploy on Modal/Cloud Run.

### 4.12 Eval (`eval/`) — the most important section

**This is where the project earns its senior signal.** Most RAG portfolios skip this and lose half their potential value.

- **`gold.jsonl`** — ~25 hand-built `(question, expected_pmids)` pairs. Author yourself; quality matters. Examples: "common acquired resistance mechanisms to first-gen EGFR TKIs?" → PMIDs covering T790M, MET amplification, SCLC transformation.
- **`metrics.py`:**
  - **Recall@k** at k=5 and k=10: of the gold PMIDs, how many appear in top-k retrieved?
  - **MRR (Mean Reciprocal Rank):** average of 1/(rank of first correct hit). Penalizes "found it but at rank 50."
  - **Faithfulness (LLM-judge):** for each generated claim, does at least one cited abstract support it? Score 0–1 via Claude Haiku 4.5 (cheap judge).
- **`run_eval.py`** writes `eval/results.md` (markdown table); paste into README.
- **Floor targets:** recall@5 ≥ 0.6, faithfulness ≥ 0.85. Below → use ablations notebook to diagnose.

### 4.13 Ablations notebook (`notebooks/ablations.ipynb`) — DS-flavor differentiator

Reuses the gold set. Four comparisons, each with a plot saved to `docs/figures/` and embedded in the README:

1. **Dense vs. BM25 vs. hybrid (RRF)** — bar chart of recall@5/@10/MRR.
2. **With vs. without Voyage Rerank** — paired bars across same metrics.
3. **Voyage embeddings vs. `all-MiniLM-L6-v2` baseline** — biomedical-embedding lift.
4. **Top-k sweep** at k = {5, 10, 20, 50} — justify the retrieve-20 → rerank-5 choice.

Why this matters: a notebook with real plots, methodology cells, and "here's what surprised me" prose reads as *experimental science* — exactly what a data scientist produces.

---

## 5. README structure (the actual interview artifact)

When a hiring manager opens the repo, they should see, in order:

1. One-paragraph framing: NSCLC RAG with grounded citations; why this matters in precision oncology.
2. **Live demo URL** + GIF of streaming-answer-with-citations.
3. Architecture diagram (from `docs/architecture.md`).
4. **Eval results table:** recall@5, recall@10, MRR, faithfulness — actual numbers.
5. **Ablations:** four embedded plots with one-line takeaways.
6. Stack rationale (3–5 bullets — why Voyage, why hybrid, why Sonnet, why no chunking).
7. Quickstart: `uv sync` → `streamlit run`.
8. **Honest limitations + future work** — undervalued; senior reviewers read this carefully.

---

## 6. Dependencies (`pyproject.toml`)

`biopython`, `chromadb`, `voyageai`, `rank-bm25`, `anthropic`, `pydantic-settings`, `pyyaml`, `typer`, `fastapi`, `streamlit`, `pytest`, `python-dotenv`, `matplotlib`, `jupyter`.

Python 3.11+. **No torch / sentence-transformers** (no local models — keeps memory footprint tiny so the app runs on Streamlit Cloud's 1 GB tier).

---

## 7. Cost story

| Phase | Cost |
|---|---|
| Ingest (one-time, local) | $0 (NCBI free) |
| Indexing 1000 abstracts (Voyage) | ~$0.03 (within free tier) |
| Eval iterations during dev | ~$5–10 |
| Ablations notebook runs | ~$1–2 |
| Live demo session (5 questions w/ caching) | ~$0.05 |
| Idle hosting | $0 |
| **Total project cost** | **~$10–20** |

To drive lower: run the LLM-judge faithfulness scoring with Haiku 4.5 instead of Sonnet (~3× cheaper, judge-quality is fine).

---

## 8. Build order

1. `pyproject.toml`, `settings.py`, `config/config.yaml`, `.env.example`, `.streamlit/secrets.toml.example`.
2. `ingest.py` + test → 1000 abstracts on disk, **commit to repo**.
3. `embeddings.py` (Voyage) + `vectorstore.py` (Chroma) + `cli index` → corpus indexed, **commit `data/chroma/`**.
4. `retriever.py` (dense + BM25 + RRF) + tests.
5. `reranker.py` (Voyage Rerank) + `pipeline.py` orchestrator + `cli query` (no LLM yet — print top-5).
6. `generator.py` + `prompts.py` with streaming + caching → end-to-end CLI working.
7. `app/streamlit_app.py` with streaming UI + citation cards + debug expander.
8. `eval/` — author the gold set, build metrics, generate `eval/results.md`.
9. `notebooks/ablations.ipynb` + plots into `docs/figures/`.
10. `api.py` (thin FastAPI wrapper, in-repo only).
11. README polish + `docs/architecture.md` + deploy to Streamlit Cloud.

---

## 9. Verification (acceptance per stage)

1. **Ingest** (local): `pubmed-rag ingest` → `data/raw/abstracts.jsonl` has 1000 lines with required fields. Spot-check 3 PMIDs against pubmed.ncbi.nlm.nih.gov.
2. **Index** (local): `pubmed-rag index` → `data/chroma/` populated; size <20 MB.
3. **Query smoke test:** `pubmed-rag query "What are common resistance mechanisms to first-line EGFR TKIs in NSCLC?"` → answer with PMID-grounded citations; manually verify one citation by reading the abstract.
4. **Tests:** `pytest` green (Entrez and Anthropic mocked, no network).
5. **Eval:** `pubmed-rag eval` → `eval/results.md` populated. Floor: recall@5 ≥ 0.6, faithfulness ≥ 0.85. Re-tune retrieval if below.
6. **Ablations notebook:** every cell runs end-to-end; plots saved to `docs/figures/`.
7. **Local Streamlit:** `streamlit run app/streamlit_app.py` → ask question → streamed answer + citations + debug expander.
8. **Deployed Streamlit:** push to GitHub → Streamlit Cloud picks up → public URL works → ask same question end-to-end.

---

## 10. Hiring-signal mapping (what each piece says about you)

| Piece | Signal |
|---|---|
| NSCLC focus + author-built gold set | Domain awareness; can do real clinical-DS work |
| Voyage + Claude + prompt caching | Current ecosystem fluency |
| Hybrid retrieval + RRF | "Knows RAG is more than embed-and-prompt" |
| Reranker | Understands bi-encoder/cross-encoder tradeoff |
| Streaming + citation UX | Builds for actual LLM-product users |
| Eval harness with quantitative numbers | Measurement discipline (the senior signal) |
| Ablations notebook with plots | Experimental science (the DS signal) |
| `api.py` in repo (not deployed) | Surface-as-thin-wrapper / engineering taste |
| Streamlit on free tier, ~$10 total cost | Pragmatic; ships things |
| Honest limitations section | Senior self-awareness |

---

## 11. Resuming work — quick checklist

When opening this repo on a new machine:

1. Read this file top-to-bottom.
2. Check progress against the **Build order** in §8 and the **Verification** in §9.
3. Set up env: `uv sync`, copy `.env.example` to `.env`, fill in `NCBI_API_KEY`, `VOYAGE_API_KEY`, `ANTHROPIC_API_KEY`.
4. If `data/raw/abstracts.jsonl` and `data/chroma/` are committed, you can skip ingest/index — just run the app.
5. Current next step: see the lowest unchecked item in §8.
