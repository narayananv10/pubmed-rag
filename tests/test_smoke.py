from pubmed_rag import __version__
from pubmed_rag.settings import load_config


def test_version():
    assert __version__ == "0.1.0"


def test_config_loads_and_validates():
    cfg = load_config()
    assert cfg.pubmed.max_results > 0
    assert cfg.retrieval.rerank_k <= cfg.retrieval.retrieve_k
    assert cfg.retrieval.retrieve_k <= cfg.retrieval.retrieve_pool
    assert cfg.models.embedding
    assert cfg.models.generator
