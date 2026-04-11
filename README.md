# BM25 Document Triage v0.1

This project is a simple Python tool for document triage. It scans a document folder, ranks files against weighted search terms with BM25, sends the top results to OpenAI for evidence-bound review, and writes a structured text report.

The current workflow is file-based and runs from the command line. Inputs come from `INCOMING_DOCS/` and `config/search_terms.json`. Output is written to `OUTPUT/report.txt`.

## What It Does

The pipeline is:

1. Discover files under `INCOMING_DOCS/`
2. Extract text from supported file types
3. Normalize and weight query terms from `config/search_terms.json`
4. Rank indexed documents with BM25
5. Send the top-ranked results to OpenAI for analyst-style relevance review
6. Write a report with:
   - `Analyst Triage Summary`
   - `Retrieval outcome`
   - `Immediate review priority`

## Project Layout

```text
BM25/
├── main.py
├── parse_files.py
├── requirements.txt
├── .env
├── config/
│   └── search_terms.json
├── INCOMING_DOCS/
└── OUTPUT/
    └── report.txt
```

## Requirements

- Python 3.11+ recommended
- Local filesystem access to the repo directories
- An OpenAI API key if you want LLM analysis enabled

Install dependencies:

```bash
pip install -r requirements.txt
```

Current dependencies:

- `pypdf`
- `rank-bm25`
- `openai`
- `python-dotenv`
- Eventually will need python libraries for .docx and .xlsx

## Configuration

The tool loads environment variables from `.env`.

Example:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5-mini
OPENAI_TOP_K=10
OPENAI_REASONING_EFFORT=medium
OPENAI_DOC_CHAR_LIMIT=4000
```

Variables used by `main.py`:

- `OPENAI_API_KEY`: enables OpenAI-based analysis. If missing, the script still runs but skips the LLM step.
- `OPENAI_MODEL`: OpenAI model name. Default: `gpt-5-mini`
- `OPENAI_TOP_K`: number of top BM25 documents passed to the LLM. Default: `10`
- `OPENAI_REASONING_EFFORT`: reasoning level sent to the OpenAI Responses API. Default: `medium`
- `OPENAI_DOC_CHAR_LIMIT`: excerpt length per document sent to the LLM. Default: `4000`

## Search Terms

Search terms are read from `config/search_terms.json` as a JSON object whose keys are terms or phrases and whose values are integer weights.

Example:

```json
{
  "malware": 5,
  "beacon": 4,
  "powershell": 3,
  "registry": 2,
  "invoice": 1
}
```

Behavior:

- Keys are tokenized with a Unicode word regex and normalized with `casefold()`
- Multi-word phrases are split into tokens
- Weights are applied by repeating tokens in the BM25 query
- Minimum effective weight is `1`

## Supported Files

The script discovers files with these extensions (but for now just does .pdf):

- `.pdf`
- `.doc`
- `.docx`
- `.xls`
- `.xlsx`
- `.txt`

Actual text extraction is currently implemented only for:

- `.pdf`
- `.txt`

Important consequence:

- `.doc`, `.docx`, `.xls`, and `.xlsx` are discovered during file crawl
- They are not parsed by `extract_text()`
- They will usually end up in the report as skipped with `No text extracted`

## How To Run

From the project root:

```bash
python main.py
```

What happens when you run it:

- The script scans `INCOMING_DOCS/`
- It prints each discovered file to the console with a timestamp
- It indexes documents that produce text and tokens
- It ranks those documents with BM25
- It optionally calls OpenAI on the top results
- It writes the final report to `OUTPUT/report.txt`

## Output

The generated report is a UTF-8 text file at:

```text
OUTPUT/report.txt
```

The report currently contains three main sections:

### 1. Analyst Triage Summary

This is the OpenAI-generated relevance assessment of the top BM25 results. If `OPENAI_API_KEY` is not set, this section states that LLM analysis was skipped.

### 2. Retrieval outcome

This section includes:

- document counts
- configured query terms
- normalized query tokens
- full ranked-document list with BM25 scores
- any skipped files affecting retrieval

### 3. Immediate review priority

This section lists the top three ranked documents as the first review targets.

## Implementation Notes

`main.py` is currently the full application entrypoint. It uses:

- `find_documents()` to crawl the corpus
- `load_search_terms()` to load weighted terms
- `build_weighted_query()` to normalize and weight tokens
- `BM25Okapi` from `rank_bm25` for ranking
- `llm_analyze_top_documents()` for OpenAI analysis
- `build_report()` to assemble the final text report

`parse_files.py` currently provides:

- PDF extraction via `pypdf`
- text file loading with fallback encodings: `utf-8-sig`, `utf-8`, `cp1251`, then `latin-1`

## Limitations

- Office formats are discovered but not parsed.
- Relevance quality depends on the extracted text quality and the excerpt limit sent to the LLM.
- The LLM is instructed to reason only from provided excerpts, so truncated excerpts can reduce confidence.
- Console and editor rendering may show garbled multilingual text if source files or environment encodings do not match.
- The output report is plain text only. There is no UI, API server, or interactive workflow yet.

Common reasons:

- file type is discovered but not actually parsed yet
- PDF extraction returned no text
- text file decoded but produced no usable tokens


