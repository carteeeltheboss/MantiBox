"""Tests for Qwen2.5-VL-7B provider, limits, and 24 GB feasibility benchmark (SPEC §3, §11)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from maintidoc.vlm.base import ImageInput, OutputSchema, VlmProvider
from maintidoc.vlm.qwen import (
    FeasibilityReport,
    QwenLimits,
    QwenVlProvider,
    compute_24gb_feasibility,
    format_qwen_messages,
)


def test_qwen_limits_defaults() -> None:
    limits = QwenLimits()
    assert limits.max_context_tokens == 32768
    assert limits.max_new_tokens == 512
    assert limits.max_pixels == 1024 * 1024
    assert limits.min_pixels == 256 * 256
    assert limits.temperature == 0.0


def test_qwen_limits_custom() -> None:
    limits = QwenLimits(
        max_context_tokens=4096,
        max_new_tokens=256,
        max_pixels=512 * 512,
        temperature=0.2,
    )
    assert limits.max_context_tokens == 4096
    assert limits.max_new_tokens == 256
    assert limits.max_pixels == 262144
    assert limits.temperature == 0.2


def test_feasibility_benchmark_calculation() -> None:
    report = compute_24gb_feasibility()
    assert isinstance(report, FeasibilityReport)
    assert report.model_id == "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    assert report.quantization == "awq-4bit"
    assert report.target_vram_gb == 24.0

    # Weights for AWQ 7B should be ~5.1 - 6.5 GB
    assert 4.0 <= report.weights_vram_gb <= 7.0

    # Test context profile at 4K, 32K and 128K
    vram_4k = report.profiles[4096]
    vram_32k = report.profiles[32768]
    vram_128k = report.profiles[128000]

    assert vram_4k.total_vram_gb < 10.0
    assert vram_32k.total_vram_gb < 14.0
    assert vram_128k.total_vram_gb < 20.0

    # Spec §11 hypothesis: 24 GB is plausible and sufficient for single-batch requests
    assert vram_4k.fits_24gb is True
    assert vram_32k.fits_24gb is True
    assert vram_128k.fits_24gb is True
    assert report.spec_hypothesis_validated is True


def test_format_qwen_messages_text_only() -> None:
    messages = format_qwen_messages(
        instruction="Quelle est la vitesse de rotation ?",
        images=(),
    )
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert len(content) == 1
    assert content[0] == {"type": "text", "text": "Quelle est la vitesse de rotation ?"}


def test_format_qwen_messages_with_image() -> None:
    img = ImageInput(data=b"fake-image-bytes", media_type="image/png", reference="fig_1")
    messages = format_qwen_messages(
        instruction="Analyse ce schéma",
        images=(img,),
    )
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert len(content) == 2
    assert content[0]["type"] == "image"
    assert content[0]["image"] == b"fake-image-bytes"
    assert content[1] == {"type": "text", "text": "Analyse ce schéma"}


def test_qwen_provider_protocol_conformance() -> None:
    provider = QwenVlProvider(model_path="models/qwen2.5-vl-7b-instruct-awq", device="cpu")
    assert isinstance(provider, VlmProvider)
    assert provider.model_id == "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    assert provider.device == "cpu"
    assert provider.fallback_applied is False
    provider.close()


def test_qwen_provider_cpu_awq_unavailable_raises_clear_error() -> None:
    provider = QwenVlProvider(model_path="models/qwen2.5-vl-7b-instruct-awq", device="cpu")
    # On host without CUDA and without AWQ compile tools, attempting generation
    # must fail with a clear, honest error (no fabricated output, no silent fallback).
    with pytest.raises(RuntimeError, match="AWQ|CUDA|kernel"):
        provider.generate(instruction="Test instruction", expects=OutputSchema.TEXT)


def test_qwen_provider_generate_with_mocked_model() -> None:
    provider = QwenVlProvider(model_path="models/qwen2.5-vl-7b-instruct-awq", device="cpu")

    mock_model = MagicMock()
    mock_processor = MagicMock()

    mock_input_ids = MagicMock()
    mock_input_ids.shape = (1, 10)
    mock_processor.return_value = {"input_ids": mock_input_ids}
    mock_processor.batch_decode.return_value = ["Réponse générée par Qwen2.5-VL."]

    mock_output = MagicMock()
    mock_new_ids = MagicMock()
    mock_new_ids.shape = (1, 5)
    mock_output.__getitem__.return_value = mock_new_ids
    mock_model.generate.return_value = mock_output

    with (
        patch.object(provider, "_get_model", return_value=mock_model),
        patch.object(provider, "_get_processor", return_value=mock_processor),
    ):
        res = provider.generate(instruction="Question de maintenance")
        assert res.model_id == "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
        assert res.provider == "qwen"
        assert "Réponse générée" in res.text
