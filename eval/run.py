"""Evaluation runner for retrieval metrics (SPEC §4.2, §5, §10, Issue #15).

Measures Recall@5, nDCG@5, and MRR@10 across the seed query scenarios.
Supports running:
1. Deterministic hashing doubles (fast, torch-free).
2. Real semantic models (BGE-M3 1024-d, CLIP 512-d).
Reports the two result sets distinctly — never blended (Issue #15).
Logs runs to MLflow when available.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from eval.benchmark_data import SEED_QUERIES, EvalQuery
from eval.scenarios import evaluate_scenarios, load_scenarios
from qdrant_client import QdrantClient

from maintidoc.agent.pipeline import AnswerPipeline
from maintidoc.domain import DocumentRecord, Element
from maintidoc.ingestion.pdf import extract_pdf
from maintidoc.observability import get_logger
from maintidoc.retrieval.embeddings import BgeM3Embedder, Embedder, HashingEmbedder
from maintidoc.retrieval.fusion import HybridRetriever
from maintidoc.retrieval.images import (
    ClipImageEmbedder,
    HashingImageEmbedder,
    ImageEmbedder,
)
from maintidoc.retrieval.index import IndexConfig, QdrantIndex
from maintidoc.retrieval.metrics import compute_retrieval_metrics, log_metrics_to_mlflow
from maintidoc.retrieval.reranker import DeterministicReranker, NoReranker, Reranker
from maintidoc.schemas import ElementKind
from maintidoc.storage.objects import FilesystemStore, ObjectStoreError
from maintidoc.tracking import build_run_metadata, log_run_with_convention
from maintidoc.vlm.stub import StubPolicy, StubVlmProvider

logger = get_logger(__name__)

CORPUS_MANIFEST = Path("corpus/manifest.json")


def load_corpus_elements() -> tuple[list[Element], list[tuple[Element, bytes]]]:
    """Extract all elements and figure images from the synthetic corpus."""
    if not CORPUS_MANIFEST.exists():
        msg = f"manifest not found at {CORPUS_MANIFEST}"
        raise FileNotFoundError(msg)

    data = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    documents = [
        DocumentRecord.model_validate(
            {k: v for k, v in doc_dict.items() if k in DocumentRecord.model_fields}
        )
        for doc_dict in data.get("documents", [])
    ]

    all_elements: list[Element] = []
    figure_images: list[tuple[Element, bytes]] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        store = FilesystemStore(Path(tmp_dir))
        for doc in documents:
            pdf_path = Path("corpus") / doc.object_key
            if not pdf_path.exists():
                continue
            pdf_bytes = pdf_path.read_bytes()
            res = extract_pdf(data=pdf_bytes, document=doc, store=store)
            for elem in res.elements:
                all_elements.append(elem)
                if elem.kind == ElementKind.FIGURE and elem.image_key:
                    try:
                        img_bytes = store.get(elem.image_key)
                        figure_images.append((elem, img_bytes))
                    except ObjectStoreError as exc:
                        logger.debug("Could not read image %s: %s", elem.image_key, exc)

    return all_elements, figure_images


def evaluate_system(
    *,
    name: str,
    text_embedder: Embedder,
    image_embedder: ImageEmbedder,
    reranker: Reranker,
    elements: list[Element],
    figure_images: list[tuple[Element, bytes]],
    queries: tuple[EvalQuery, ...] = SEED_QUERIES,
) -> dict[str, float]:
    """Run retrieval evaluation for a given embedder + reranker pairing."""
    client = QdrantClient(":memory:")

    text_cfg = IndexConfig(
        collection=f"text_{int(time.time() * 1000)}",
        dimension=text_embedder.dimension,
    )
    image_cfg = IndexConfig(
        collection=f"image_{int(time.time() * 1000)}",
        dimension=image_embedder.dimension,
    )

    text_idx = QdrantIndex(client, text_cfg)
    image_idx = QdrantIndex(client, image_cfg)
    text_idx.ensure_collection()
    image_idx.ensure_collection()

    # Index text elements
    text_elements = [el for el in elements if el.text.strip()]
    if text_elements:
        vectors = text_embedder.embed_documents([el.text for el in text_elements])
        text_idx.upsert(
            text_elements,
            vectors,
            embedder_model=text_embedder.model_id,
        )

    # Index figure elements
    if figure_images:
        fig_elems = [fig[0] for fig in figure_images]
        fig_bytes = [fig[1] for fig in figure_images]
        fig_vectors = image_embedder.embed_images(fig_bytes)
        image_idx.upsert(
            fig_elems,
            fig_vectors,
            embedder_model=image_embedder.model_id,
        )

    retriever = HybridRetriever(
        text_index=text_idx,
        image_index=image_idx,
        text_embedder=text_embedder,
        image_embedder=image_embedder,
        reranker=reranker,
    )

    queries_results: list[tuple[list[str], set[str]]] = []
    for q in queries:
        hits = retriever.search(
            q.query,
            principal=q.principal,
            top_n=10,
        )
        retrieved_ids = [f"{h.document_id}:{h.element_id}" for h in hits]
        target_ids = {f"{doc}:{elem}" for doc, elem in q.relevant_targets}
        queries_results.append((retrieved_ids, target_ids))

    metrics = compute_retrieval_metrics(queries_results, k_recall=5, k_ndcg=5, k_mrr=10)

    params = {
        "suite": name,
        "text_embedder": text_embedder.model_id,
        "image_embedder": image_embedder.model_id,
        "reranker_arm": reranker.arm,
        "reranker_model": reranker.model_id,
    }
    log_metrics_to_mlflow(metrics, params=params, run_name=name)

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MaintiDoc Retrieval & Scenario Evaluation Harness"
    )
    parser.add_argument(
        "--mode",
        choices=["retrieval", "scenarios", "all"],
        default="all",
        help="Evaluation mode to execute (default: all)",
    )
    parser.add_argument(
        "--backend",
        choices=["hash", "real", "both"],
        default="both",
        help="Embedder backend to evaluate for retrieval (default: both)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("MaintiDoc Comprehensive Evaluation Harness (SPEC §4.2, §5, §10)")
    print("=" * 70)

    # 1. Retrieval Benchmarks
    if args.mode in ("retrieval", "all"):
        print("\n--- [A] Retrieval Benchmarks (Recall@5, nDCG@5, MRR@10) ---")
        print("Extracting corpus elements...")
        t0 = time.time()
        elements, figure_images = load_corpus_elements()
        print(
            f"Corpus loaded: {len(elements)} elements, {len(figure_images)} figure images "
            f"({time.time() - t0:.2f}s)"
        )

        results: dict[str, dict[str, float]] = {}

        if args.backend in ("hash", "both"):
            print("\n--- [1] Deterministic Hashing Doubles (torch-free) ---")
            hash_text = HashingEmbedder(dimension=64)
            hash_img = HashingImageEmbedder(dimension=64)

            m_none = evaluate_system(
                name="hash_doubles_reranker_none",
                text_embedder=hash_text,
                image_embedder=hash_img,
                reranker=NoReranker(),
                elements=elements,
                figure_images=figure_images,
            )
            results["Hash Doubles (No Reranker)"] = m_none

            m_det = evaluate_system(
                name="hash_doubles_reranker_deterministic",
                text_embedder=hash_text,
                image_embedder=hash_img,
                reranker=DeterministicReranker(),
                elements=elements,
                figure_images=figure_images,
            )
            results["Hash Doubles (Deterministic Reranker)"] = m_det

        if args.backend in ("real", "both"):
            print("\n--- [2] Real Embedders (BGE-M3 1024-d, CLIP Multilingual 512-d) ---")
            try:
                real_text = BgeM3Embedder(model_id="models/bge-m3")
                real_img = ClipImageEmbedder(
                    image_model="models/clip-vit-base-patch32",
                    text_model="models/clip-multilingual-text",
                )
                m_real_none = evaluate_system(
                    name="real_bgem3_clip_reranker_none",
                    text_embedder=real_text,
                    image_embedder=real_img,
                    reranker=NoReranker(),
                    elements=elements,
                    figure_images=figure_images,
                )
                results["Real BGE-M3 + CLIP (No Reranker)"] = m_real_none
            except Exception as exc:  # noqa: BLE001
                print(f"Skipping real models due to environment/load error: {exc}")

        print("\n" + "=" * 70)
        print(f"{'Configuration':<42} | {'Recall@5':<8} | {'nDCG@5':<8} | {'MRR@10':<8}")
        print("-" * 70)
        for cfg_name, m in results.items():
            r5, n5, m10 = m["recall@5"], m["ndcg@5"], m["mrr@10"]
            print(f"{cfg_name:<42} | {r5:<8.4f} | {n5:<8.4f} | {m10:<8.4f}")
        print("=" * 70)
        print("SPEC §5 Reranker Arms Note:")
        print("  - 'none': evaluated above")
        print("  - 'bge' (BAAI/bge-reranker-v2-m3): unmeasured (weights not downloaded on host)")
        print("  - 'adapter': unmeasured (fine-tuning pending in #28)")
        print("=" * 70)

    # 2. 150-Scenario Evaluation Suite (SPEC §10)
    if args.mode in ("scenarios", "all"):
        print("\n--- [B] 150-Scenario Evaluation Suite (SPEC §10) ---")
        try:
            scenarios = load_scenarios()
            print(f"Loaded {len(scenarios)} scenarios from eval/scenarios.json")

            # Setup memory pipeline for scenario execution
            client = QdrantClient(":memory:")
            t_cfg = IndexConfig(collection="eval_sc_text", dimension=32)
            i_cfg = IndexConfig(collection="eval_sc_img", dimension=32)
            t_idx = QdrantIndex(client, t_cfg)
            i_idx = QdrantIndex(client, i_cfg)
            t_idx.ensure_collection()
            i_idx.ensure_collection()

            t_emb = HashingEmbedder(dimension=32)
            i_emb = HashingImageEmbedder(dimension=32)

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

            acc_str = f"{report.status_accuracy:.2%}"
            count_str = f"{report.matched_status_count}/{report.total_scenarios}"
            print(f"\nScenario Accuracy: {acc_str} ({count_str})")
            print(f"Mean Citation Validity: {report.mean_citation_validity:.2%}")
            print(f"Multimodal vs Text-Only Quality Lift: {report.multimodal_lift_accuracy:+.2%}")
            print("\nAccuracy per SPEC §10 Category:")
            for cat, acc in report.category_accuracies.items():
                print(f"  - {cat:<24}: {acc:.2%}")

            # Log to MLflow with convention
            run_meta = build_run_metadata(
                model_id=vlm.model_id,
                provider=vlm.provider,
                prompt_versions={"answer": "v1"},
                reranker="none",
                device="cpu",
            )
            sc_metrics = {
                "scenario_accuracy": report.status_accuracy,
                "citation_validity": report.mean_citation_validity,
                "multimodal_lift": report.multimodal_lift_accuracy,
                "total_scenarios": float(report.total_scenarios),
            }
            log_run_with_convention(
                experiment_name="maintidoc_scenarios",
                run_name="eval_150_scenarios",
                metrics=sc_metrics,
                params=run_meta,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Scenario evaluation error: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
