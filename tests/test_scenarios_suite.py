"""Tests for the 150-scenario evaluation harness (Issues #6, #30, SPEC §10).

Acceptance criteria:
- 150 scenarios across all SPEC §10 categories
- Scenario loader validates format and counts
- Text-only vs multimodal comparison runnable
- Seed scenarios and metrics calculation
"""

from __future__ import annotations

from eval.scenarios import (
    EvalScenario,
    ScenarioCategory,
    evaluate_scenarios,
    load_scenarios,
)
from qdrant_client import QdrantClient

from maintidoc.agent.pipeline import AnswerPipeline
from maintidoc.retrieval.embeddings import HashingEmbedder
from maintidoc.retrieval.fusion import HybridRetriever
from maintidoc.retrieval.images import HashingImageEmbedder
from maintidoc.retrieval.index import IndexConfig, QdrantIndex
from maintidoc.retrieval.reranker import NoReranker
from maintidoc.vlm.stub import StubPolicy, StubVlmProvider


def test_150_scenarios_loaded_and_categorized() -> None:
    scenarios = load_scenarios()
    assert len(scenarios) == 150

    counts: dict[str, int] = {}
    for sc in scenarios:
        assert isinstance(sc, EvalScenario)
        assert sc.question.strip()
        assert sc.principal_user.strip()
        assert len(sc.principal_groups) > 0
        assert sc.human_reviewed is False  # Marked unreviewed per AGENTS.md rule
        cat = sc.category.value
        counts[cat] = counts.get(cat, 0) + 1

    assert counts[ScenarioCategory.TEXT_DOCUMENTATION.value] == 40
    assert counts[ScenarioCategory.IMAGE_SCHEMA.value] == 30
    assert counts[ScenarioCategory.MCP_CONSULTATION.value] == 20
    assert counts[ScenarioCategory.MISSING_CONTRADICTORY.value] == 20
    assert counts[ScenarioCategory.ACCESS_APPROVAL.value] == 20
    assert counts[ScenarioCategory.PROMPT_INJECTION.value] == 20


def test_scenario_evaluation_run_subset() -> None:
    scenarios = load_scenarios()[:6]

    client = QdrantClient(":memory:")
    t_cfg = IndexConfig(collection="test_sc_text", dimension=16)
    i_cfg = IndexConfig(collection="test_sc_img", dimension=16)
    t_idx = QdrantIndex(client, t_cfg)
    i_idx = QdrantIndex(client, i_cfg)
    t_idx.ensure_collection()
    i_idx.ensure_collection()

    t_emb = HashingEmbedder(dimension=16)
    i_emb = HashingImageEmbedder(dimension=16)

    retriever = HybridRetriever(
        text_index=t_idx,
        image_index=i_idx,
        text_embedder=t_emb,
        image_embedder=i_emb,
        reranker=NoReranker(),
    )
    vlm = StubVlmProvider(policy=StubPolicy.AUTO)
    pipeline = AnswerPipeline(retriever=retriever, vlm_provider=vlm)

    report = evaluate_scenarios(scenarios, pipeline, compare_multimodal=True)
    assert report.total_scenarios == 6
    assert 0.0 <= report.status_accuracy <= 1.0
    assert 0.0 <= report.mean_citation_validity <= 1.0
