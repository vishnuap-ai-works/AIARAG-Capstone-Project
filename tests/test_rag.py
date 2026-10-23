"""Unit Tests for RAG Components (Chunking, Embedding, Retrieval, Reranking). Verifies algorithm correctness."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag.reranker import CrossEncoderReranker
from rag.retriever import DenseRetriever


@pytest.mark.anyio
async def test_dense_retriever_logs_similarity_scores(caplog):
    """Verify that retrieve() logs top-k chunk similarity scores in expected format."""
    mock_vector_store = MagicMock()
    mock_vector_store.search.return_value = [
        {"chunk": "Chunk 1 text", "metadata": {"file_name": "doc1.txt"}, "score": 0.92},
        {"chunk": "Chunk 2 text", "metadata": {"file_name": "doc2.txt"}, "score": 0.93},
        {"chunk": "Chunk 3 text", "metadata": {"file_name": "doc3.txt"}, "score": 0.97},
    ]

    retriever = DenseRetriever(vector_store=mock_vector_store)

    with patch(
        "rag.embeddings.ModelSelector.get_single_embedding",
        new_callable=AsyncMock,
    ) as mock_embed:
        mock_embed.return_value = [0.1, 0.2, 0.3]

        with caplog.at_level("INFO"):
            results = await retriever.retrieve("What is the policy?", top_k=3)

    assert len(results) == 3
    expected_log_pattern = "chunk 1. - 0.92, chunk 2. - 0.93, chunk 3. - 0.97"
    assert any(expected_log_pattern in record.message for record in caplog.records)
    assert any(
        "Retrieved Chunks (JSON):" in record.message for record in caplog.records
    )


@pytest.mark.anyio
async def test_dense_retriever_logs_reranked_scores(caplog):
    """Verify that retrieve() logs initial similarity scores and reranked scores."""
    mock_vector_store = MagicMock()
    mock_vector_store.search.return_value = [
        {"chunk": "Chunk 1", "score": 0.75},
        {"chunk": "Chunk 2", "score": 0.82},
    ]

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        {"chunk": "Chunk 2", "score": 0.82, "rerank_score": 0.96, "retrieval_rank": 2},
        {"chunk": "Chunk 1", "score": 0.75, "rerank_score": 0.88, "retrieval_rank": 1},
    ]

    retriever = DenseRetriever(vector_store=mock_vector_store, reranker=mock_reranker)

    with patch(
        "rag.embeddings.ModelSelector.get_single_embedding",
        new_callable=AsyncMock,
    ) as mock_embed:
        mock_embed.return_value = [0.1, 0.2, 0.3]

        with caplog.at_level("INFO"):
            results = await retriever.retrieve("Test query", top_k=2)

    assert len(results) == 2
    # Verify initial similarity score log
    assert any(
        "Initial retrieved chunk similarity scores: chunk 1. - 0.75, chunk 2. - 0.82"
        in record.message
        for record in caplog.records
    )
    # Verify reranked score log
    assert any(
        "Top 2 reranked chunk scores: chunk 1. - 0.96, chunk 2. - 0.88"
        in record.message
        for record in caplog.records
    )
    # Verify JSON log with both retriever and reranked chunks
    assert any(
        "Retriever and Reranked Chunks (JSON):" in record.message
        for record in caplog.records
    )


def test_cross_encoder_reranker_sigmoid_calculation(caplog):
    """Verify that CrossEncoder calculates normalized rerank scores and logs them."""
    with patch("sentence_transformers.CrossEncoder") as mock_ce_cls:
        mock_ce = MagicMock()
        # Raw logits from cross encoder: 2.4469 (sigmoid ~ 0.92), -1.0 (sigmoid ~ 0.27)
        mock_ce.predict.return_value = [2.4469, -1.0]
        mock_ce_cls.return_value = mock_ce

        reranker = CrossEncoderReranker(model_name="mock-model")

        docs = [{"chunk": "First chunk"}, {"chunk": "Second chunk"}]
        with caplog.at_level("INFO"):
            reranked = reranker.rerank("Query", docs, top_k=2)

        assert len(reranked) == 2
        assert reranked[0]["rerank_score"] == pytest.approx(0.9204, abs=0.01)
        assert reranked[1]["rerank_score"] == pytest.approx(0.2689, abs=0.01)

        assert any(
            "Top 2 reranked chunk scores: chunk 1. - 0.92, chunk 2. - 0.27"
            in record.message
            for record in caplog.records
        )


def test_json_vector_store_kb_version_and_tombstone(tmp_path):
    """Verify that JSONVectorStore records kb_version and properly filters tombstoned docs."""
    from rag.vector_store import JSONVectorStore
    import json

    db_file = str(tmp_path / "test_store.json")
    store = JSONVectorStore(filepath=db_file)

    # 1. Add active document with specific kb_version
    store.add_document(
        filename="handbook.md",
        file_type="md",
        chunks=["PTO policy: 20 days off."],
        embeddings=[[0.5, 0.5, 0.0]],
        kb_version="v2.0",
        is_tombstone=False,
    )

    with open(db_file, "r") as f:
        data = json.load(f)
    assert data["handbook.md"]["kb_version"] == "v2.0"
    assert data["handbook.md"]["is_tombstone"] is False
    assert data["handbook.md"]["chunks"][0]["metadata"]["kb_version"] == "v2.0"
    assert data["handbook.md"]["chunks"][0]["metadata"]["is_tombstone"] is False

    # Search should find the document
    results = store.search(query_embedding=[0.5, 0.5, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0]["chunk"] == "PTO policy: 20 days off."
    assert results[0]["metadata"]["kb_version"] == "v2.0"

    # 2. Tombstone the document
    success = store.tombstone_document("handbook.md", reason="superseded by v3.0")
    assert success is True

    with open(db_file, "r") as f:
        data_after = json.load(f)
    assert data_after["handbook.md"]["is_tombstone"] is True
    assert data_after["handbook.md"]["tombstone_reason"] == "superseded by v3.0"
    assert "tombstoned_at" in data_after["handbook.md"]

    # Search should now exclude tombstoned document
    results_after = store.search(query_embedding=[0.5, 0.5, 0.0], top_k=1)
    assert len(results_after) == 0


@pytest.mark.anyio
async def test_document_ingestion_pipeline_kb_version_and_tombstone(tmp_path):
    """Verify that DocumentIngestionPipeline passes kb_version and delegates tombstoning."""
    from pipeline.store import DocumentIngestionPipeline

    mock_vector_store = MagicMock()
    mock_vector_store.tombstone_document.return_value = True

    mock_loader = AsyncMock()
    mock_loader.load.return_value = "Sample handbook text content"

    mock_chunker = MagicMock()
    mock_chunker.chunk = AsyncMock(return_value=["Sample handbook text chunk"])

    mock_embedder = MagicMock()
    mock_embedder.get_embedded = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_embedder.get_sparse_embedded = MagicMock(return_value=None)

    pipeline = DocumentIngestionPipeline(
        chunker=mock_chunker,
        embedder=mock_embedder,
        vector_store=mock_vector_store,
    )

    test_file = tmp_path / "benefits.txt"
    test_file.write_text("Sample handbook text content")

    with patch("pipeline.store.DocumentLoadFactory.get_loader", return_value=mock_loader):
        chunks = await pipeline.ingest_file(test_file, kb_version="v4.2")

    assert chunks == ["Sample handbook text chunk"]
    mock_vector_store.add_document.assert_called_once_with(
        filename="benefits.txt",
        file_type="txt",
        chunks=["Sample handbook text chunk"],
        embeddings=[[0.1, 0.2, 0.3]],
        sparse_embeddings=None,
        kb_version="v4.2",
        is_tombstone=False,
    )

    # Test tombstone delegation
    res = pipeline.tombstone_file("benefits.txt", reason="policy sunset")
    assert res is True
    mock_vector_store.tombstone_document.assert_called_once_with(
        "benefits.txt", reason="policy sunset"
    )

