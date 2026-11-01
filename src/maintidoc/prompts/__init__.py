"""Versioned prompts. Prompts live in `prompts/*.yaml`, never inline in code."""

from __future__ import annotations

from maintidoc.prompts.loader import (
    Prompt,
    available_versions,
    clear_prompt_cache,
    default_prompts_dir,
    latest_prompt,
    load_prompt,
)

__all__ = [
    "Prompt",
    "available_versions",
    "clear_prompt_cache",
    "default_prompts_dir",
    "latest_prompt",
    "load_prompt",
]
