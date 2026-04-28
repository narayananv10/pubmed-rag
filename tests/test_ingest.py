from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pubmed_rag.ingest import (
    append_jsonl,
    configure_entrez,
    existing_pmids,
    parse_record,
    run,
)


SAMPLE_ARTICLE: dict = {
    "MedlineCitation": {
        "PMID": "12345678",
        "Article": {
            "ArticleTitle": "EGFR mutations in NSCLC",
            "Abstract": {
                "AbstractText": [
                    "Background information about NSCLC.",
                    "Methods used in this study.",
                ],
            },
            "AuthorList": [
                {"LastName": "Smith", "Initials": "J"},
                {"LastName": "Doe", "Initials": "A"},
            ],
            "Journal": {
                "Title": "JAMA Oncology",
                "JournalIssue": {
                    "PubDate": {"Year": "2023", "Month": "Jun"},
                },
            },
        },
        "MeshHeadingList": [
            {"DescriptorName": "Carcinoma, Non-Small-Cell Lung"},
            {"DescriptorName": "Antineoplastic Agents"},
        ],
    },
}


def test_parse_record_basic_fields():
    record = parse_record(SAMPLE_ARTICLE)
    assert record["pmid"] == "12345678"
    assert record["title"] == "EGFR mutations in NSCLC"
    assert "Background" in record["abstract"]
    assert "Methods" in record["abstract"]
    assert record["authors"] == ["Smith J", "Doe A"]
    assert record["journal"] == "JAMA Oncology"
    assert record["pub_date"] == "2023-Jun"
    assert record["year"] == 2023
    assert "Carcinoma, Non-Small-Cell Lung" in record["mesh_terms"]
    assert record["url"] == "https://pubmed.ncbi.nlm.nih.gov/12345678/"


def test_parse_record_handles_missing_abstract():
    article = {
        "MedlineCitation": {
            "PMID": "1",
            "Article": {
                "ArticleTitle": "T",
                "Journal": {"JournalIssue": {"PubDate": {}}},
            },
        }
    }
    record = parse_record(article)
    assert record["abstract"] == ""
    assert record["authors"] == []
    assert record["year"] is None


def test_parse_record_medline_date_fallback():
    article = {
        "MedlineCitation": {
            "PMID": "2",
            "Article": {
                "ArticleTitle": "T",
                "Journal": {
                    "JournalIssue": {"PubDate": {"MedlineDate": "2021 Jan-Feb"}},
                },
            },
        }
    }
    record = parse_record(article)
    assert record["year"] == 2021
    assert "2021" in record["pub_date"]


def test_existing_pmids_reads_jsonl(tmp_path: Path):
    path = tmp_path / "abstracts.jsonl"
    path.write_text(
        json.dumps({"pmid": "1"}) + "\n" + json.dumps({"pmid": "2"}) + "\n"
    )
    assert existing_pmids(path) == {"1", "2"}


def test_existing_pmids_missing_file(tmp_path: Path):
    assert existing_pmids(tmp_path / "nope.jsonl") == set()


def test_append_jsonl_writes_lines(tmp_path: Path):
    path = tmp_path / "out.jsonl"
    n = append_jsonl([{"pmid": "1"}, {"pmid": "2"}], path)
    assert n == 2
    assert path.read_text().count("\n") == 2


def test_configure_entrez_requires_email():
    with pytest.raises(ValueError, match="NCBI_EMAIL"):
        configure_entrez(email="")


def test_run_idempotent(tmp_path: Path):
    """Re-running with the same PMIDs should be a no-op."""
    path = tmp_path / "abstracts.jsonl"

    with (
        patch("pubmed_rag.ingest.search_pmids") as mock_search,
        patch("pubmed_rag.ingest.fetch_articles") as mock_fetch,
        patch("pubmed_rag.ingest.configure_entrez"),
    ):
        mock_search.return_value = ["12345678"]
        mock_fetch.return_value = iter([SAMPLE_ARTICLE])

        n_new, n_skipped = run(
            query="...",
            max_results=1,
            batch_size=1,
            output_path=path,
            email="test@example.com",
        )
        assert n_new == 1
        assert n_skipped == 0

        mock_fetch.return_value = iter([SAMPLE_ARTICLE])
        n_new2, n_skipped2 = run(
            query="...",
            max_results=1,
            batch_size=1,
            output_path=path,
            email="test@example.com",
        )
        assert n_new2 == 0
        assert n_skipped2 == 1

        lines = [line for line in path.read_text().splitlines() if line]
        assert len(lines) == 1
