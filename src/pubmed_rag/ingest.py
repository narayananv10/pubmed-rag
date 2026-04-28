"""Fetch PubMed abstracts via Entrez E-utilities and persist as JSONL.

Run-once, output committed. The deployed app does not call NCBI at runtime.
Idempotent: re-running skips PMIDs already present in the output file.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from Bio import Entrez

PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


def configure_entrez(email: str, api_key: str = "") -> None:
    if not email:
        raise ValueError(
            "NCBI requires an email contact. Set NCBI_EMAIL in your .env file."
        )
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key


def existing_pmids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    pmids: set[str] = set()
    with path.open() as f:
        for line in f:
            if line.strip():
                pmids.add(json.loads(line)["pmid"])
    return pmids


def search_pmids(query: str, max_results: int) -> list[str]:
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="date",
    )
    result = Entrez.read(handle)
    handle.close()
    return list(result["IdList"])


def fetch_articles(pmids: list[str], batch_size: int) -> Iterator[dict[str, Any]]:
    """Yield raw PubmedArticle dicts in batches."""
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        handle = Entrez.efetch(
            db="pubmed",
            id=",".join(batch),
            rettype="medline",
            retmode="xml",
        )
        records = Entrez.read(handle)
        handle.close()
        yield from records.get("PubmedArticle", [])


def _abstract_text(article: dict[str, Any]) -> str:
    sections = article.get("Abstract", {}).get("AbstractText", [])
    parts: list[str] = []
    for section in sections:
        label = getattr(section, "attributes", {}).get("Label", "")
        text = str(section)
        parts.append(f"{label}: {text}" if label else text)
    return "\n".join(parts)


def _authors(article: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for a in article.get("AuthorList", []):
        last = a.get("LastName", "")
        initials = a.get("Initials", "")
        if last:
            out.append(f"{last} {initials}".strip())
        elif a.get("CollectiveName"):
            out.append(str(a["CollectiveName"]))
    return out


def _pub_date(article: dict[str, Any]) -> tuple[str, int | None]:
    pub_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})

    if "Year" in pub_date:
        year = str(pub_date["Year"])
        month = str(pub_date.get("Month", ""))
        day = str(pub_date.get("Day", ""))
        parts = [year]
        if month:
            parts.append(month)
        if day:
            parts.append(day)
        return "-".join(parts), int(year) if year.isdigit() else None

    if "MedlineDate" in pub_date:
        text = str(pub_date["MedlineDate"])
        for token in text.split():
            if token.isdigit() and len(token) == 4:
                return text, int(token)
        return text, None

    return "", None


def _mesh_terms(citation: dict[str, Any]) -> list[str]:
    return [str(h["DescriptorName"]) for h in citation.get("MeshHeadingList", [])]


def parse_record(pubmed_article: dict[str, Any]) -> dict[str, Any]:
    citation = pubmed_article["MedlineCitation"]
    pmid = str(citation["PMID"])
    article = citation["Article"]

    pub_date_str, year = _pub_date(article)

    return {
        "pmid": pmid,
        "title": str(article.get("ArticleTitle", "")),
        "abstract": _abstract_text(article),
        "authors": _authors(article),
        "journal": str(article.get("Journal", {}).get("Title", "")),
        "pub_date": pub_date_str,
        "year": year,
        "mesh_terms": _mesh_terms(citation),
        "url": PUBMED_URL.format(pmid=pmid),
    }


def append_jsonl(records: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def run(
    *,
    query: str,
    max_results: int,
    batch_size: int,
    output_path: Path,
    email: str,
    api_key: str = "",
    on_batch: callable | None = None,  # type: ignore[valid-type]
) -> tuple[int, int]:
    """Search → fetch → parse → append. Returns (n_new, n_skipped)."""
    configure_entrez(email, api_key)

    all_pmids = search_pmids(query, max_results)
    seen = existing_pmids(output_path)
    new_pmids = [p for p in all_pmids if p not in seen]
    n_skipped = len(all_pmids) - len(new_pmids)

    if not new_pmids:
        return 0, n_skipped

    n_written = 0
    batch_idx = 0
    n_batches = (len(new_pmids) + batch_size - 1) // batch_size
    pending: list[dict[str, Any]] = []
    for article in fetch_articles(new_pmids, batch_size):
        pending.append(parse_record(article))
        if len(pending) >= batch_size:
            n_written += append_jsonl(pending, output_path)
            batch_idx += 1
            if on_batch is not None:
                on_batch(batch_idx, n_batches, n_written)
            pending = []
    if pending:
        n_written += append_jsonl(pending, output_path)
        batch_idx += 1
        if on_batch is not None:
            on_batch(batch_idx, n_batches, n_written)

    return n_written, n_skipped
