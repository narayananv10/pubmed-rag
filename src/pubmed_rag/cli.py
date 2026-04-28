"""PubMed RAG command-line interface (typer-based)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from . import ingest as ingest_module
from .settings import RAW_PATH, load_config, load_secrets

app = typer.Typer(add_completion=False, help="PubMed RAG pipeline.")


@app.callback()
def _root() -> None:
    """PubMed RAG pipeline."""


@app.command()
def ingest(
    output: Annotated[
        Path,
        typer.Option(help="Output JSONL path."),
    ] = RAW_PATH,
    limit: Annotated[
        int | None,
        typer.Option(help="Override config max_results (useful for smoke tests)."),
    ] = None,
) -> None:
    """Fetch PubMed abstracts matching the configured query → JSONL."""
    cfg = load_config()
    secrets = load_secrets()

    max_results = limit if limit is not None else cfg.pubmed.max_results
    typer.echo(f"Searching PubMed (max_results={max_results}) ...")

    def _on_batch(idx: int, total: int, n_written: int) -> None:
        typer.echo(f"  batch {idx}/{total} → {n_written} records written")

    n_new, n_skipped = ingest_module.run(
        query=cfg.pubmed.query,
        max_results=max_results,
        batch_size=cfg.pubmed.batch_size,
        output_path=output,
        email=secrets.ncbi_email,
        api_key=secrets.ncbi_api_key,
        on_batch=_on_batch,
    )
    typer.echo(
        f"Done. {n_new} new records appended; "
        f"{n_skipped} already cached. Output: {output}"
    )


if __name__ == "__main__":
    app()
