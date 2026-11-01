"""Versioned prompt loading (SPEC §7).

AGENTS.md: prompts live ONLY in `prompts/*.yaml`, named `name_vN.yaml`, and a
prompt change means a **new version file** — never an edit in place. That is what
makes a run reproducible: the prompt version is recorded alongside the model,
reranker and index versions.

Nothing here builds prompt text in Python. If you find yourself concatenating
instructions in code, put them in a new versioned YAML instead.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.string import get_template_variables

from maintidoc.errors import ConfigurationError

_FILENAME = re.compile(r"^(?P<name>[a-z0-9_]+)_v(?P<version>\d+)\.yaml$")
_REQUIRED_KEYS = ("name", "version", "description", "input_variables", "template")


def default_prompts_dir() -> Path:
    """Locate the prompt directory.

    `MAINTIDOC_PROMPTS_DIR` wins so the Docker images can point at a copied,
    read-only location. Otherwise fall back to the repository layout.
    """
    override = os.environ.get("MAINTIDOC_PROMPTS_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "prompts"


@dataclass(frozen=True, slots=True)
class Prompt:
    """One immutable prompt version."""

    name: str
    version: int
    description: str
    input_variables: tuple[str, ...]
    #: Variables whose value is untrusted data (document text, OCR, tool output,
    #: user input). Each one must be delimited in the template so the model can
    #: be told where data begins and ends — the structural half of the defence
    #: required by AGENTS.md NEVER #4.
    untrusted_variables: tuple[str, ...]
    template: str
    path: Path

    @property
    def version_id(self) -> str:
        """Stable identifier recorded in provenance and MLflow runs."""
        return f"{self.name}_v{self.version}"

    def as_template(self) -> PromptTemplate:
        """langchain-core template (SPEC §3: LangChain provides prompting)."""
        return PromptTemplate(
            template=self.template,
            input_variables=list(self.input_variables),
            template_format="jinja2",
        )

    def format(self, **values: str) -> str:
        missing = [v for v in self.input_variables if v not in values]
        if missing:
            msg = f"{self.version_id}: missing prompt variables {missing}"
            raise ConfigurationError(msg)
        unexpected = sorted(set(values) - set(self.input_variables))
        if unexpected:
            msg = f"{self.version_id}: unexpected prompt variables {unexpected}"
            raise ConfigurationError(msg)
        return self.as_template().format(**values)


def available_versions(name: str, *, directory: Path | None = None) -> tuple[int, ...]:
    """Every version of `name` present on disk, ascending."""
    root = directory or default_prompts_dir()
    if not root.is_dir():
        return ()
    versions: list[int] = []
    for path in root.iterdir():
        match = _FILENAME.match(path.name)
        if match and match.group("name") == name:
            versions.append(int(match.group("version")))
    return tuple(sorted(versions))


def _validate(path: Path, raw: dict[str, object], name: str, version: int) -> Prompt:
    for key in _REQUIRED_KEYS:
        if key not in raw:
            msg = f"{path.name}: prompt file is missing required key '{key}'"
            raise ConfigurationError(msg)

    declared_name = raw["name"]
    declared_version = raw["version"]
    if declared_name != name:
        msg = f"{path.name}: declares name '{declared_name}', expected '{name}'"
        raise ConfigurationError(msg)
    if declared_version != version:
        msg = f"{path.name}: declares version {declared_version}, expected {version}"
        raise ConfigurationError(msg)

    template = raw["template"]
    if not isinstance(template, str) or not template.strip():
        msg = f"{path.name}: template must be a non-empty string"
        raise ConfigurationError(msg)

    variables = raw["input_variables"]
    if not isinstance(variables, list) or not all(isinstance(v, str) for v in variables):
        msg = f"{path.name}: input_variables must be a list of strings"
        raise ConfigurationError(msg)
    declared = tuple(variables)

    untrusted_raw = raw.get("untrusted_variables", [])
    if not isinstance(untrusted_raw, list) or not all(isinstance(v, str) for v in untrusted_raw):
        msg = f"{path.name}: untrusted_variables must be a list of strings"
        raise ConfigurationError(msg)
    untrusted = tuple(untrusted_raw)

    # A typo in a variable name would otherwise render as an empty string and
    # silently produce a prompt with no sources in it.
    found = tuple(sorted(get_template_variables(template, "jinja2")))
    if found != tuple(sorted(declared)):
        msg = (
            f"{path.name}: template uses {list(found)} but input_variables "
            f"declares {list(declared)}"
        )
        raise ConfigurationError(msg)

    unknown_untrusted = [v for v in untrusted if v not in declared]
    if unknown_untrusted:
        msg = f"{path.name}: untrusted_variables {unknown_untrusted} are not input_variables"
        raise ConfigurationError(msg)

    # Every untrusted variable must be wrapped in a tag the model is told about.
    for variable in untrusted:
        if f"<{variable}>" not in template or f"</{variable}>" not in template:
            msg = (
                f"{path.name}: untrusted variable '{variable}' must be delimited "
                f"by <{variable}> ... </{variable}> in the template"
            )
            raise ConfigurationError(msg)

    return Prompt(
        name=name,
        version=version,
        description=str(raw["description"]),
        input_variables=declared,
        untrusted_variables=untrusted,
        template=template,
        path=path,
    )


@lru_cache(maxsize=64)
def load_prompt(name: str, version: int, directory: str | None = None) -> Prompt:
    """Load one exact prompt version.

    Cached because versions are immutable by policy: changing a prompt means
    adding `_v(N+1).yaml`, not editing `_vN.yaml`.

    `directory` is a str rather than a Path so the cache key is hashable and
    stable across call sites.
    """
    root = Path(directory) if directory else default_prompts_dir()
    path = root / f"{name}_v{version}.yaml"
    if not path.is_file():
        known = available_versions(name, directory=root)
        msg = f"prompt '{name}_v{version}' not found in {root}; available versions: {list(known)}"
        raise ConfigurationError(msg)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"{path.name}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        msg = f"{path.name}: top level must be a mapping"
        raise ConfigurationError(msg)
    return _validate(path, raw, name, version)


@lru_cache(maxsize=32)
def latest_prompt(name: str, directory: str | None = None) -> Prompt:
    """Load the highest available version of `name`."""
    versions = available_versions(name, directory=Path(directory) if directory else None)
    if not versions:
        msg = f"no prompt named '{name}' found in {directory or default_prompts_dir()}"
        raise ConfigurationError(msg)
    return load_prompt(name, versions[-1], directory)


def clear_prompt_cache() -> None:
    """Test hook."""
    load_prompt.cache_clear()
    latest_prompt.cache_clear()


__all__ = [
    "Prompt",
    "available_versions",
    "clear_prompt_cache",
    "default_prompts_dir",
    "latest_prompt",
    "load_prompt",
]
