"""
Author: JD Durick
Email: labgeek@gmail.com
Date: 9 April 2026
Tool:   This is a two-phase document intelligence pipeline. It crawls a directory of files, 
        ranks them against weighted search terms using BM25 (a classical information retrieval algorithm), 
        then passes the top results to an OpenAI LLM for evidence-bound relevance analysis. 
        The final output is a structured text report.
"""

from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

from parse_files import parse_pdf, parse_text_file


TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_LLM_TOP_K = 10
DEFAULT_LLM_DOC_CHAR_LIMIT = 4000 


load_dotenv()


def safe_display(value) -> str:
    text = str(value)
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


def timestamped_path(value) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"{timestamp} {safe_display(value)}"


def find_documents(base_dir: str):
    supported_exts = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".txt"}
    base_path = Path(base_dir)

    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    found_files = []

    for file_path in base_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_exts:
            found_files.append(file_path.resolve())

    return found_files


def load_search_terms(term_file: str):
    term_path = Path(term_file)
    terms = json.loads(term_path.read_text(encoding="utf-8-sig"))
    if not isinstance(terms, dict):
        raise ValueError("search_terms.json must contain a JSON object")
    return terms


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(file_path)
    if suffix == ".txt":
        return parse_text_file(file_path)
    return ""


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_PATTERN.findall(text)]


def build_weighted_query(search_terms: dict[str, int]) -> tuple[list[str], dict[str, int]]:
    query_tokens = []
    query_weights: dict[str, int] = {}

    for term, weight in search_terms.items():
        tokens = tokenize(term)
        if not tokens:
            continue

        normalized_weight = max(int(weight), 1)
        for token in tokens:
            query_tokens.extend([token] * normalized_weight)
            query_weights[token] = query_weights.get(token, 0) + normalized_weight

    return query_tokens, query_weights


def build_llm_prompt(
    top_docs: list[dict[str, object]],
    query_terms: dict[str, int],
    doc_char_limit: int,
) -> str:
    lines = [
        "You are performing evidence-based document relevance analysis over BM25 retrieval results.",
        "Your task is to assess which retrieved files are genuinely relevant to the weighted query terms based on the provided excerpts, not merely which files scored highly.",
        "",
        "STRICT ANALYSIS RULES",
        "1. Use only the provided document excerpts as evidence.",
        "2. Do not use outside knowledge, assumptions, guessed context, or invented facts.",
        "3. Do not infer a document's purpose, intent, subject, or operational significance unless the excerpt supports it.",
        "4. Treat BM25 score as a retrieval signal, not proof of relevance.",
        "5. A file is relevant only if the excerpt shows meaningful contextual support for the weighted query terms.",
        "6. Mentions of query terms without meaningful context may indicate a weak match or false positive.",
        "7. Separate direct evidence from inference. Label uncertainty clearly.",
        "8. If evidence is weak, partial, ambiguous, contradictory, or missing, say so explicitly.",
        '9. If the excerpt does not support a conclusion, write exactly: "Insufficient evidence in provided text."',
        "10. Do not overclaim. Do not fill gaps. Do not present speculation as fact.",
        "",
        "RELEVANCE CRITERIA",
        "Judge each file using the following factors:",
        "- Presence of weighted query terms",
        "- Context around those terms",
        "- Whether the terms appear central to the document excerpt or only incidental",
        "- Whether multiple weighted terms reinforce the same topic or theme",
        "- Whether the excerpt suggests substantive topical relevance versus keyword coincidence",
        "- Whether the BM25 score aligns with the actual contextual evidence",
        "",
        "WEIGHTING GUIDANCE",
        "Higher-weight query terms matter more than lower-weight terms.",
        "A document matching fewer high-weight terms with strong contextual support may be more relevant than a document matching many low-weight terms with weak context.",
        "",
        "OUTPUT INSTRUCTIONS",
        "Write a concise but analyst-grade assessment using the exact section headings below.",
        "For every claim, anchor your reasoning in the provided excerpts.",
        "When possible, reference specific matched terms and the surrounding context.",
        "",
        "Required sections:",
        "1. Overall assessment",
        "   - Summarize the overall retrieval quality.",
        "   - State whether the top results appear strongly relevant, mixed, or mostly weak.",
        "",
        "2. Most relevant files",
        "   - Identify the strongest files in descending order of actual relevance.",
        "   - For each file, explain why it is relevant using excerpt-based evidence.",
        "   - State whether relevance is high, medium, or low.",
        "   - Distinguish direct evidence from inference where needed.",
        "",
        "3. Weak matches, ambiguous results, or likely false positives",
        "   - Identify files that appear weak, incidental, misleading, or unsupported by context.",
        "   - Explain whether the match appears term-driven rather than context-driven.",
        "",
        "4. Confidence and limitations",
        "   - State major limitations caused by excerpt length, missing context, ambiguity, or sparse evidence.",
        "   - Explicitly call out uncertainty where appropriate.",
        "",
        "Weighted query terms:",
    ]

    for term, weight in query_terms.items():
        lines.append(f"- {term}: weight {weight}")

    lines.extend(["", "Top BM25 documents:"])

    for index, document in enumerate(top_docs, start=1):
        excerpt = str(document["text"])[:doc_char_limit]
        matched_terms = ", ".join(document["matched_terms"]) if document["matched_terms"] else "none"

        lines.extend(
            [
                "",
                f"Document {index}",
                f"Path: {document['path']}",
                f"BM25 score: {document['score']:.6f}",
                f"Matched query tokens: {matched_terms}",
                "Excerpt:",
                excerpt,
            ]
        )

    lines.extend(
        [
            "",
            "Final reminder:",
            "Base all conclusions strictly on the provided excerpts.",
            "Do not reward a document for keyword overlap alone if the context does not support real relevance.",
        ]
    )

    return "\n".join(lines)

def llm_analyze_top_documents(
    top_docs: list[dict[str, object]],
    query_terms: dict[str, int],
) -> str:
    if not top_docs:
        return "No BM25-ranked documents were available for LLM analysis."

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "LLM analysis skipped: set OPENAI_API_KEY to enable OpenAI reasoning."

    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "medium")
    doc_char_limit = int(
        os.getenv("OPENAI_DOC_CHAR_LIMIT", str(DEFAULT_LLM_DOC_CHAR_LIMIT))
    )

    client = OpenAI(api_key=api_key)
    prompt = build_llm_prompt(top_docs, query_terms, doc_char_limit)

    system_message = """
You are a strict evidence-bound retrieval analyst performing document relevance assessment.

Your role is to evaluate BM25-ranked documents using only the provided excerpts.
You must not use outside knowledge, assumptions, guessed context, or invented facts.

Rules you must follow:
1. Treat BM25 score as a retrieval hint, not proof of relevance.
2. Determine actual relevance from the excerpt content, not from keyword overlap alone.
3. A document is relevant only when the excerpt provides meaningful contextual support for the weighted query terms.
4. Mentions of query terms without meaningful context may indicate a weak match or false positive.
5. Separate direct evidence from inference.
6. Explicitly label uncertainty whenever evidence is weak, incomplete, ambiguous, indirect, or conflicting.
7. Do not infer a document's purpose, intent, significance, or topic unless the excerpt supports that conclusion.
8. Do not overclaim, speculate, or fill gaps.
9. If the text does not support a conclusion, say exactly: "Insufficient evidence in provided text."
10. Prefer precision over completeness. It is better to be cautious than wrong.

Your output must be analytical, skeptical, and grounded in the provided text only.
""".strip()

    response = client.responses.create(
        model=model,
        reasoning={"effort": reasoning_effort},
        input=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    return response.output_text.strip()


def build_report(
    files: list[Path],
    indexed_documents: list[dict[str, object]],
    ranked_documents: list[dict[str, object]],
    search_terms: dict[str, int],
    query_weights: dict[str, int],
    skipped_files: dict[Path, str],
    llm_analysis: str,
) -> str:
    lines = [
        "Analyst Triage Summary",
        "======================",
        "",
        f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        llm_analysis,
        "",
        "Retrieval outcome",
        "=================",
        "",
        f"Documents discovered: {len(files)}",
        f"Documents indexed for BM25: {len(indexed_documents)}",
        f"Documents skipped due to read/parse errors: {len(skipped_files)}",
        "",
        "Configured query terms:",
    ]

    if search_terms:
        for term, weight in search_terms.items():
            lines.append(f"- {term}: weight {weight}")
    else:
        lines.append("- No search terms configured.")

    lines.extend(["", "Normalized query tokens:"])

    if query_weights:
        for token, weight in sorted(query_weights.items()):
            lines.append(f"- {token}: weight {weight}")
    else:
        lines.append("- No valid query tokens were produced from search terms.")

    lines.extend(["", "Ranked documents:"])

    if ranked_documents:
        for index, document in enumerate(ranked_documents, start=1):
            matched_terms = document["matched_terms"]
            matched_terms_text = ", ".join(matched_terms) if matched_terms else "none"
            lines.append(
                f"- {index}. Score {document['score']:.6f} | {document['path']} | Matched query tokens: {matched_terms_text}"
            )
    else:
        lines.append("- No documents were ranked.")

    lines.extend(["", "Immediate review priority", "=========================", ""])

    if ranked_documents:
        for index, document in enumerate(ranked_documents[:3], start=1):
            matched_terms = document["matched_terms"]
            matched_terms_text = ", ".join(matched_terms) if matched_terms else "none"
            lines.append(
                f"{index}. {document['path']} (score {document['score']:.6f}; matched query tokens: {matched_terms_text})"
            )
    else:
        lines.append("No immediate review targets identified because no documents were ranked.")

    if skipped_files:
        lines.extend(["", "Skipped files affecting retrieval:", ""])
        for file_path, reason in skipped_files.items():
            lines.append(f"- {file_path}: {reason}")

    return "\n".join(lines)


def main():
    doc_dir = "INCOMING_DOCS"
    output_dir = Path("OUTPUT")
    term_file = "config/search_terms.json"

    files = find_documents(doc_dir)
    search_terms = load_search_terms(term_file)
    query_tokens, query_weights = build_weighted_query(search_terms)
    skipped_files: dict[Path, str] = {}
    indexed_documents: list[dict[str, object]] = []

    print(f"Found {len(files)} documents for processing:\n")

    for file_path in files:
        print(timestamped_path(file_path))

        try:
            text = extract_text(file_path)
        except Exception as exc:
            skipped_files[file_path] = str(exc)
            continue
        

        if not text:
            skipped_files[file_path] = "No text extracted"
            continue

        tokens = tokenize(text)
        
        if not tokens:
            skipped_files[file_path] = "No tokens extracted from text"
            continue

        indexed_documents.append(
            {
                "path": file_path,
                "text": text,
                "tokens": tokens,
                "token_counts": Counter(tokens),
            }
        )

    ranked_documents: list[dict[str, object]] = []
    if indexed_documents and query_tokens:
        bm25 = BM25Okapi([doc["tokens"] for doc in indexed_documents])
        scores = bm25.get_scores(query_tokens)

        for document, score in zip(indexed_documents, scores):
            token_counts = document["token_counts"]
            matched_terms = sorted(
                token for token in query_weights if token_counts.get(token, 0) > 0
            )
            ranked_documents.append(
                {
                    "path": document["path"],
                    "score": float(score),
                    "matched_terms": matched_terms,
                    "text": document["text"],
                }
            )

        ranked_documents.sort(key=lambda item: item["score"], reverse=True)

    llm_top_k = int(os.getenv("OPENAI_TOP_K", str(DEFAULT_LLM_TOP_K)))
    llm_analysis = llm_analyze_top_documents(
        ranked_documents[:llm_top_k],
        search_terms,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.txt"
    report_path.write_text(
        build_report(
            files,
            indexed_documents,
            ranked_documents,
            search_terms,
            query_weights,
            skipped_files,
            llm_analysis,
        ),
        encoding="utf-8",
    )

    print(f"\nReport written to {safe_display(report_path.resolve())}")

if __name__ == "__main__":
    main()
