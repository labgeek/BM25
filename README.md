# BM25 Document Triage v0.1

A Python command-line tool for document triage. It scans a static document folder, ranks files against weighted search terms using BM25, sends the top results to OpenAI/Anthropic for evidence-bound analysis, and writes a structured text report.

---

## Table of Contents

- [What It Does](#what-it-does)
- [How BM25 Works](#how-bm25-works)
- [Project Layout](#project-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Search Terms](#search-terms)
- [Supported File Types](#supported-file-types)
- [Running the Tool](#running-the-tool)
- [Output](#output)
- [Limitations](#limitations)

---

## What It Does

The pipeline runs as follows:

1. Discover all files under `INCOMING_DOCS/`
2. Extract text from supported file types (PDF, TXT)
3. Tokenize and normalize document text
4. Load weighted query terms from `config/search_terms.json`
5. Build a BM25 index over the document corpus
6. Score and rank every document against the query
7. Send the top-ranked results to OpenAI for analyst-style relevance review
8. Write a report to `OUTPUT/report.txt` containing:
   - **Analyst Triage Summary** — LLM-generated evidence-bound assessment
   - **Retrieval Outcome** — BM25 ranked list with scores and matched terms
   - **Immediate Review Priority** — the top three documents to act on first

---

## How BM25 Works

BM25 (Best Match 25, also called Okapi BM25) is a probabilistic ranking function used in information retrieval. It scores documents relative to a query based on term frequency and document length, and is the algorithm behind many production search engines. The "25" refers to the 25th iteration of the BM family of weighting functions developed at City University of London and Okapi IR system during the 1990s.

### The Core Idea

BM25 builds on TF-IDF (term frequency-inverse document frequency) but adds two important corrections:

- **Term frequency saturation**: raw TF grows unboundedly — a document mentioning a term 100 times should not score 100× higher than one that mentions it once. BM25 applies a saturation curve so that additional occurrences of the same term yield diminishing returns.
- **Document length normalization**: a longer document is more likely to contain any given term simply because it has more words. BM25 penalizes documents that are longer than the corpus average, so relevance reflects density rather than volume.

### The Formula

For a query `Q` with terms `q₁, q₂, ..., qₙ` and a document `D`, the BM25 score is:

```
Score(D, Q) = Σ IDF(qᵢ) · [f(qᵢ, D) · (k₁ + 1)] / [f(qᵢ, D) + k₁ · (1 - b + b · |D| / avgdl)]
```

Where:

| Symbol | Meaning |
|--------|---------|
| `f(qᵢ, D)` | Raw term frequency — how many times term `qᵢ` appears in document `D` |
| `IDF(qᵢ)` | Inverse document frequency — how rare the term is across the whole corpus |
| `\|D\|` | Length of document `D` in tokens |
| `avgdl` | Average document length across the corpus |
| `k₁` | Term frequency saturation parameter (typically 1.2 – 2.0) |
| `b` | Length normalization parameter (0 = no normalization, 1 = full normalization; typically 0.75) |

### IDF: Rewarding Rare Terms

The IDF component gives more weight to terms that appear in fewer documents. A term like "malware" appearing in only 2 of 100 documents is a stronger signal than "the" appearing in all of them.

```
IDF(qᵢ) = log((N - df(qᵢ) + 0.5) / (df(qᵢ) + 0.5) + 1)
```

Where `N` is the total number of documents and `df(qᵢ)` is the number of documents containing `qᵢ`. The `+0.5` smoothing prevents division by zero and handles edge cases.

### TF Saturation: Diminishing Returns

Without saturation, term frequency contributes linearly — a document with 50 hits scores 50× a document with 1 hit. BM25 shapes TF with a saturation curve:

```
Saturated TF = f(qᵢ, D) · (k₁ + 1) / (f(qᵢ, D) + k₁)
```

As `f` grows, the numerator and denominator both grow, so the ratio approaches `(k₁ + 1)` asymptotically. With `k₁ = 1.5`, a document with 1 match scores about 1.0, 5 matches scores about 1.8, and 50 matches only scores about 1.97 — the same ceiling regardless of count.

### Length Normalization: Rewarding Density

A 10-page document is more likely to contain any keyword than a 1-page document by pure chance. The length normalization term `(1 - b + b · |D| / avgdl)` scales the effective TF:

- If `b = 0`, length is ignored entirely
- If `b = 1`, documents are fully normalized to their length
- At `b = 0.75` (the default in `rank-bm25`), a short document that contains a term scores higher than a long document with the same raw frequency

### How This Tool Applies BM25

This tool uses the `rank-bm25` library's `BM25Okapi` implementation with its default `k₁` and `b` parameters. The key customization is **query weighting**:

Because standard BM25 treats all query terms equally, this tool simulates term importance by **repeating tokens in the query vector** according to their configured weight. A term with weight `5` is injected into the query 5 times, so it contributes proportionally more to each document's score.

```
config/search_terms.json:
  "malware": 5  →  query tokens: ["malware", "malware", "malware", "malware", "malware"]
  "beacon":  4  →  query tokens: ["beacon",  "beacon",  "beacon",  "beacon"]
  "invoice": 1  →  query tokens: ["invoice"]
```

The final query fed to `BM25Okapi.get_scores()` is the concatenation of all weighted token lists. This approach preserves standard BM25 math while allowing analysts to encode domain knowledge about which terms are most operationally significant.

### Why BM25 for Document Triage?

BM25 is particularly well suited for this use case because:

- **No training required** — it works immediately on any new document set with no machine learning pipeline
- **Interpretable** — scores are grounded in term frequency and document length, making it easy to explain why a document ranked highly
- **Handles sparse signals well** — in security or compliance triage, a single mention of "beacon" or "registry persistence" in a 50-page document is a real signal, not noise
- **Computationally cheap** — indexing and scoring thousands of documents takes milliseconds, making it practical to re-run with different query configurations

---

## Project Layout

```text
BM25/
├── main.py                   # Application entry point and pipeline
├── parse_files.py            # PDF and text file extraction
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (not committed)
├── config/
│   └── search_terms.json     # Weighted query terms
├── INCOMING_DOCS/            # Drop documents here for processing
└── OUTPUT/
    └── report.txt            # Generated triage report
```

---

## Requirements

- Python 3.11 or later
- An OpenAI API key (optional — the BM25 ranking runs without it; the LLM analysis step is skipped if the key is absent)

---

## Installation

```bash
git clone <repo-url>
cd BM25
pip install -r requirements.txt
```

Dependencies:

| Package | Purpose |
|---------|---------|
| `pypdf >= 4.0.0` | PDF text extraction |
| `rank-bm25 >= 0.2.2` | BM25Okapi ranking algorithm |
| `openai >= 1.0.0` | OpenAI API client |
| `python-dotenv >= 1.0.1` | `.env` file loading |

---

## Configuration

Copy or edit `.env` in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5-mini
OPENAI_TOP_K=10
OPENAI_REASONING_EFFORT=medium
OPENAI_DOC_CHAR_LIMIT=4000
```

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(none)* | Required for LLM analysis. If absent, the script ranks documents with BM25 but skips the OpenAI step. |
| `OPENAI_MODEL` | `gpt-5-mini` | OpenAI model name. Swap for any model available on your account. |
| `OPENAI_TOP_K` | `10` | Number of top BM25-ranked documents passed to the LLM for review. |
| `OPENAI_REASONING_EFFORT` | `medium` | Reasoning level sent to the OpenAI Responses API (`low`, `medium`, or `high`). |
| `OPENAI_DOC_CHAR_LIMIT` | `4000` | Maximum characters of each document excerpt sent to the LLM. Larger values improve context but increase cost and latency. |

---

## Search Terms

Edit `config/search_terms.json` to define your query. Keys are terms or short phrases; values are integer weights (higher = more important to the triage mission).

```json
{
  "malware": 5,
  "beacon": 4,
  "powershell": 3,
  "registry": 2,
  "invoice": 1
}
```

Processing rules:

- All terms are tokenized with a Unicode word regex (`\w+`) and case-folded to lowercase
- Multi-word phrases are split into individual tokens (`"lateral movement"` → `"lateral"`, `"movement"`)
- Each token is repeated in the BM25 query proportional to its weight
- Minimum effective weight is `1`

Tune weights to match your priorities. For a malware triage scenario, terms like `beacon`, `c2`, `persistence`, and `exfiltration` should carry high weights; incidental terms like `network` or `log` should be low or omitted.

---

## Supported File Types

Files with these extensions are discovered during the directory crawl:

| Extension | Text Extraction |
|-----------|----------------|
| `.pdf` | Supported via `pypdf` |
| `.txt` | Supported with encoding fallback (`utf-8-sig` → `utf-8` → `cp1251` → `latin-1`) |
| `.doc` | Discovered but not yet parsed — will appear as skipped |
| `.docx` | Discovered but not yet parsed — will appear as skipped |
| `.xls` | Discovered but not yet parsed — will appear as skipped |
| `.xlsx` | Discovered but not yet parsed — will appear as skipped |

Scanned PDFs (image-only) will return no extractable text and will be skipped.

---

## Running the Tool

Place documents in `INCOMING_DOCS/`, then run from the project root:

```bash
python main.py
```

Console output during a run:

```
[2026-04-09T14:23:01] INCOMING_DOCS/report_q1.pdf
[2026-04-09T14:23:01] INCOMING_DOCS/analysis.txt
[2026-04-09T14:23:02] INCOMING_DOCS/invoice_march.pdf
...
Report written to OUTPUT/report.txt
```

Each file is printed with an ISO 8601 timestamp as it is discovered. Files that cannot be parsed appear in the report's skipped section with the reason.

---

## Output

The report is written to `OUTPUT/report.txt` as UTF-8 text. It contains three sections:

### 1. Analyst Triage Summary

The LLM-generated assessment of the top BM25 results. The model is instructed to reason strictly from the provided document excerpts — it will not speculate beyond what the text contains. If `OPENAI_API_KEY` is not configured, this section notes that LLM analysis was skipped and the BM25 ranked list should be used directly.

### 2. Retrieval Outcome

- Total documents discovered and indexed
- Configured search terms and their normalized query tokens
- Full ranked document list with BM25 scores and matched terms
- Skipped files with reasons (unparseable format, empty extraction, etc.)

### 3. Immediate Review Priority

The top three BM25-ranked documents listed as the first targets for human review.

---

## Limitations

- **Office formats not yet parsed.** `.doc`, `.docx`, `.xls`, and `.xlsx` files are discovered but produce no extracted text. Add `python-docx` and `openpyxl` and extend `extract_text()` in `main.py` to support them.
- **Scanned PDFs are invisible.** Image-based PDFs contain no machine-readable text and will be skipped. An OCR step (e.g., `pytesseract`) would be needed to handle them.
- **LLM context is bounded.** Document excerpts are truncated at `OPENAI_DOC_CHAR_LIMIT` characters. For large documents, the most relevant section may not be the beginning. A chunking strategy would improve coverage.
- **BM25 is lexical, not semantic.** It matches exact tokens. Synonyms, acronyms, and paraphrased concepts do not score unless they are explicitly listed as search terms.
- **No historical tracking.** Each run overwrites `OUTPUT/report.txt`. To retain history, rename or archive the output file before re-running.
