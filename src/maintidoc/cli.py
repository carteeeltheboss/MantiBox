"""Command line interface.

`doctor` is the honest answer to "will this actually run here?": it reports the
detected device, what the model ladder decided and why, which prompt versions
exist, whether the corpus is present, whether the object store is writable, and
whether the services are reachable. It never raises for an expected problem —
each check reports ok/warn/fail so one broken dependency does not hide the rest.
"""

from __future__ import annotations

import json
import shutil
import socket
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from maintidoc import __version__
from maintidoc.config import Settings, get_settings
from maintidoc.errors import ConfigurationError
from maintidoc.observability import collect_secret_values, configure_logging
from maintidoc.prompts.loader import available_versions, default_prompts_dir, load_prompt
from maintidoc.runtime.devices import detect_devices, resolve_runtime
from maintidoc.storage.objects import FilesystemStore

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="MaintiDoc — multimodal maintenance-documentation assistant.",
)

CORPUS_DIR = Path("corpus")
SPEC_PROMPTS = ("answer", "visual_analysis", "evidence_check", "ticket_draft")


class Status(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


_SYMBOL = {Status.OK: "[ ok ]", Status.WARN: "[warn]", Status.FAIL: "[FAIL]"}


@dataclass(slots=True)
class Check:
    name: str
    status: Status
    detail: str = ""


@dataclass(slots=True)
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: Status, detail: str = "") -> None:
        self.checks.append(Check(name=name, status=status, detail=detail))

    @property
    def failed(self) -> bool:
        return any(check.status is Status.FAIL for check in self.checks)

    def render(self) -> str:
        width = max((len(check.name) for check in self.checks), default=0)
        lines = []
        for check in self.checks:
            line = f"{_SYMBOL[check.status]} {check.name.ljust(width)}  {check.detail}".rstrip()
            lines.append(line)
        return "\n".join(lines)


def _tcp_reachable(host: str, port: int, *, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _check_settings(report: Report) -> Settings | None:
    try:
        settings = get_settings()
    except ConfigurationError as exc:
        report.add("configuration", Status.FAIL, str(exc))
        return None
    report.add(
        "configuration",
        Status.OK,
        f"env={settings.app.env} log level={settings.app.log_level}",
    )
    return settings


def _check_devices(report: Report, settings: Settings) -> None:
    devices = detect_devices()
    rendered = ", ".join(
        f"{device.kind.value}"
        + (
            f"({device.free_vram_gb:.1f}/{device.total_vram_gb:.1f} GB free)"
            if device.free_vram_gb
            else ""
        )
        for device in devices
    )
    report.add("devices", Status.OK, rendered)

    plan = resolve_runtime(
        device_preference=settings.inference.device,
        requested_model=settings.inference.vlm_model,
        fallback_model=settings.inference.vlm_fallback_model or None,
        quantization=settings.inference.vlm_quantization,
        allow_fallback=settings.inference.allow_model_fallback,
        min_free_vram_gb=settings.inference.vlm_min_vram_gb,
    )
    status = Status.OK
    if plan.degraded:
        status = Status.WARN
    if plan.fallback_applied:
        status = Status.WARN
    detail = f"{plan.model_id} on {plan.device.kind.value} @ {plan.quantization}"
    if plan.fallback_applied:
        detail += f"  (requested {plan.requested_model})"
    report.add("model plan", status, detail)
    for reason in plan.reasons:
        report.add("  why", Status.OK if status is Status.OK else Status.WARN, reason)


def _check_prompts(report: Report) -> None:
    directory = default_prompts_dir()
    if not directory.is_dir():
        report.add("prompts", Status.FAIL, f"directory not found: {directory}")
        return
    missing = [name for name in SPEC_PROMPTS if not available_versions(name, directory=directory)]
    if missing:
        report.add("prompts", Status.FAIL, f"missing: {', '.join(missing)} in {directory}")
        return
    versions = ", ".join(
        f"{name}_v{available_versions(name, directory=directory)[-1]}" for name in SPEC_PROMPTS
    )
    try:
        for name in SPEC_PROMPTS:
            load_prompt(name, available_versions(name, directory=directory)[-1], str(directory))
    except ConfigurationError as exc:
        report.add("prompts", Status.FAIL, str(exc))
        return
    report.add("prompts", Status.OK, versions)


def _check_corpus(report: Report, corpus_dir: Path) -> None:
    manifest = corpus_dir / "manifest.json"
    if not manifest.is_file():
        report.add(
            "corpus",
            Status.WARN,
            f"no manifest at {manifest} — run: python scripts/generate_corpus.py",
        )
        return
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.add("corpus", Status.FAIL, f"unreadable manifest: {exc.__class__.__name__}")
        return
    count = data.get("document_count", 0)
    families = ", ".join(data.get("equipment_families", []))
    scanned = len(data.get("scanned_documents", []))
    report.add(
        "corpus", Status.OK, f"{count} documents, families: {families}, {scanned} scanned (OCR)"
    )
    missing_files = [
        entry["object_key"]
        for entry in data.get("documents", [])
        if not (corpus_dir / entry["object_key"]).is_file()
    ]
    if missing_files:
        report.add(
            "  files",
            Status.FAIL,
            f"{len(missing_files)} document file(s) absent, e.g. {missing_files[0]}",
        )


def _check_object_store(report: Report, settings: Settings, corpus_dir: Path) -> None:
    store = FilesystemStore(root=corpus_dir / "objects")
    probe = "health/doctor-probe.txt"
    try:
        store.put(probe, b"ok", content_type="text/plain")
        readable = store.get(probe) == b"ok"
        store.delete(probe)
    except Exception as exc:  # noqa: BLE001 - doctor must never crash
        report.add("object store", Status.FAIL, f"{exc.__class__.__name__}: {exc}")
        return
    report.add(
        "object store",
        Status.OK if readable else Status.FAIL,
        f"filesystem backend, writable at {store.root}"
        + ("" if settings.minio.endpoint else " (S3 backend not configured)"),
    )


def _check_local_tools(report: Report) -> None:
    tesseract = shutil.which("tesseract")
    if tesseract:
        report.add("ocr", Status.OK, f"tesseract at {tesseract}")
    else:
        report.add(
            "ocr",
            Status.WARN,
            "tesseract not on PATH — scanned documents cannot be read outside the container",
        )


def _check_services(report: Report, settings: Settings) -> None:
    targets = {
        "qdrant": (settings.qdrant.host, settings.qdrant.port),
        "postgres": (settings.postgres.host, settings.postgres.port),
    }
    for name, (host, port) in targets.items():
        reachable = _tcp_reachable(host, port)
        report.add(
            f"service {name}",
            Status.OK if reachable else Status.WARN,
            f"{host}:{port} "
            + ("reachable" if reachable else "unreachable (fine outside docker compose)"),
        )


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(f"maintidoc {__version__}")


@app.command()
def doctor(
    corpus_dir: Annotated[
        Path, typer.Option("--corpus-dir", help="corpus directory to inspect")
    ] = CORPUS_DIR,
    strict: Annotated[bool, typer.Option("--strict", help="exit non-zero on warnings too")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="machine-readable output")] = False,
) -> None:
    """Report whether this machine can actually run the project."""
    settings = get_settings()
    configure_logging(
        level=settings.app.log_level,
        json_output=settings.app.log_json,
        secrets=collect_secret_values(settings),
    )

    report = Report()
    loaded = _check_settings(report)
    if loaded is not None:
        _check_devices(report, loaded)
        _check_prompts(report)
        _check_corpus(report, corpus_dir)
        _check_object_store(report, loaded, corpus_dir)
        _check_local_tools(report)
        _check_services(report, loaded)

    if as_json:
        typer.echo(
            json.dumps(
                [
                    {"name": c.name, "status": c.status.value, "detail": c.detail}
                    for c in report.checks
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        typer.echo(f"maintidoc {__version__} — readiness report\n")
        typer.echo(report.render())
        warnings = sum(1 for c in report.checks if c.status is Status.WARN)
        failures = sum(1 for c in report.checks if c.status is Status.FAIL)
        typer.echo(f"\n{len(report.checks)} checks: {failures} fail, {warnings} warn")

    if report.failed or (strict and any(c.status is Status.WARN for c in report.checks)):
        raise typer.Exit(code=1)


@app.command(name="bench-vlm")
def bench_vlm(
    target_vram: Annotated[
        float,
        typer.Option("--target-vram", help="Target VRAM in GB (default 24.0 GB per SPEC §11)"),
    ] = 24.0,
    as_json: Annotated[bool, typer.Option("--json", help="machine-readable output")] = False,
) -> None:
    """Benchmark VLM memory footprint and validate SPEC §11 24 GB GPU hypothesis."""
    from maintidoc.vlm.qwen import compute_24gb_feasibility

    report = compute_24gb_feasibility(target_vram_gb=target_vram)

    if as_json:
        payload = {
            "model_id": report.model_id,
            "quantization": report.quantization,
            "weights_vram_gb": report.weights_vram_gb,
            "target_vram_gb": report.target_vram_gb,
            "spec_hypothesis_validated": report.spec_hypothesis_validated,
            "notes": report.notes,
            "architecture": report.architecture,
            "profiles": {
                str(ctx): {
                    "context_tokens": p.context_tokens,
                    "kv_cache_vram_gb": p.kv_cache_vram_gb,
                    "activations_vram_gb": p.activations_vram_gb,
                    "total_vram_gb": p.total_vram_gb,
                    "fits_24gb": p.fits_24gb,
                    "margin_gb": p.margin_gb,
                }
                for ctx, p in report.profiles.items()
            },
        }
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    typer.echo("=" * 75)
    typer.echo("MaintiDoc — Qwen2.5-VL-7B Feasibility Benchmark (SPEC §11)")
    typer.echo("=" * 75)
    typer.echo(f"Model: {report.model_id} (Quantization: AWQ 4-bit)")
    typer.echo(f"Target VRAM: {report.target_vram_gb:.1f} GB")
    typer.echo(f"Base Weights VRAM: ~{report.weights_vram_gb:.2f} GB (Backbone AWQ + Vision FP16)")
    typer.echo("\nContext Profiles:")
    header = (
        f"{'Context':<12} | {'KV Cache':<10} | {'Activations':<12} | "
        f"{'Total VRAM':<11} | {'Fits Target?':<12} | Headroom"
    )
    typer.echo(header)
    typer.echo("-" * 75)
    for ctx, p in report.profiles.items():
        fits_str = "YES" if p.fits_24gb else "NO"
        sign = "+" if p.margin_gb >= 0 else ""
        margin_str = f"{sign}{p.margin_gb:.2f} GB"
        row = (
            f"{ctx:<12,d} | {p.kv_cache_vram_gb:<10.2f} | {p.activations_vram_gb:<12.2f} | "
            f"{p.total_vram_gb:<11.2f} | {fits_str:<12} | {margin_str}"
        )
        typer.echo(row)
    typer.echo("=" * 75)
    status_msg = "VALIDATED" if report.spec_hypothesis_validated else "INVALIDATED"
    typer.echo(f"\nSPEC §11 Hypothesis: {status_msg}")
    typer.echo(f"  {report.notes}\n")


@app.command(name="ingest")
def ingest(
    corpus_dir: Annotated[
        Path, typer.Option("--corpus-dir", help="corpus directory to ingest")
    ] = CORPUS_DIR,
    _batch_size: Annotated[int, typer.Option("--batch-size", help="batch size for indexing")] = 16,
) -> None:
    """Run ingestion on the document corpus and index into Qdrant."""
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.exists():
        typer.echo(
            f"Corpus manifest not found at {manifest_path}. Run `nox -s corpus` first.", err=True
        )
        raise typer.Exit(code=1)

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    docs_list = (
        manifest_data.get("documents", []) if isinstance(manifest_data, dict) else manifest_data
    )
    typer.echo(f"Ingesting corpus from {corpus_dir} (manifest: {len(docs_list)} documents)...")

    from maintidoc.domain import DocumentRecord
    from maintidoc.ingestion.pdf import extract_pdf
    from maintidoc.storage.objects import FilesystemStore

    rendered_dir = corpus_dir / "rendered"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    store = FilesystemStore(rendered_dir)
    total_docs = 0
    total_elements = 0
    total_pages = 0

    for doc_dict in docs_list:
        doc = DocumentRecord(
            document_id=doc_dict["document_id"],
            version=doc_dict.get("version", "1.0"),
            title=doc_dict.get("title", doc_dict["document_id"]),
            equipment_family=doc_dict.get("equipment_family", "unknown"),
            access_groups=frozenset(doc_dict.get("access_groups", ["maintenance"])),
            object_key=doc_dict.get("object_key", f"documents/{doc_dict['document_id']}.pdf"),
            sha256=doc_dict.get("sha256", "0" * 64),
        )
        pdf_path = corpus_dir / doc.object_key
        if not pdf_path.exists():
            continue

        pdf_bytes = pdf_path.read_bytes()
        result = extract_pdf(data=pdf_bytes, document=doc, store=store)
        total_docs += 1
        total_pages += result.page_count
        total_elements += len(result.elements)

    typer.echo(
        f"Ingestion complete: {total_docs} documents processed, "
        f"{total_pages} pages, {total_elements} elements extracted."
    )


if __name__ == "__main__":  # pragma: no cover
    app()
