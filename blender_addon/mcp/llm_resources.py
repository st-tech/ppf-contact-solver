"""MCP resources for the LLM-friendly doc bundle shipped with the addon.

Every file under ``blender_addon/LLM/`` (plus the top-level
``blender_addon/LLM.md`` index) is exposed as an MCP resource with URI
``llm://index`` / ``llm://<relative-path-without-suffix>``.

The ``blender_addon/`` directory is the sole current section, so its
prefix is dropped from the URI: ``LLM/blender_addon/overview.md`` is
exposed as ``llm://overview`` rather than ``llm://blender_addon/overview``.
If the tree ever grows a second section (for example ``LLM/frontend/``)
the new section's prefix stays in the URI so the namespaces don't
collide.

An MCP client can therefore discover the doc bundle via
``resources/list`` and pull a specific topic file via ``resources/read``
without needing filesystem access or a separate HTTP fetch.
"""

from __future__ import annotations

from pathlib import Path

# Resolve once at module load. The addon root is two levels up from this
# file (`mcp/llm_resources.py` -> addon package root).
_ADDON_ROOT = Path(__file__).resolve().parent.parent
_LLM_INDEX = _ADDON_ROOT / "LLM.md"
_LLM_DIR = _ADDON_ROOT / "LLM"

_URI_PREFIX = "llm://"
_INDEX_URI = "llm://index"

# The single section whose prefix is dropped from URIs. Other sections
# (if added later) keep their prefix so they don't collide.
_DEFAULT_SECTION = "blender_addon"


def _uri_for(path: Path) -> str:
    """Return the llm:// URI for an LLM markdown file on disk."""
    if path == _LLM_INDEX:
        return _INDEX_URI
    rel = path.relative_to(_LLM_DIR).with_suffix("")
    parts = rel.parts
    if parts and parts[0] == _DEFAULT_SECTION:
        parts = parts[1:]
    return f"{_URI_PREFIX}{'/'.join(parts)}"


def _path_for(uri: str) -> Path | None:
    """Resolve an llm:// URI back to a path inside the addon tree.

    Returns None if the URI doesn't match a real file. The resolved path
    is checked against the allowed roots to prevent path traversal
    ("llm://../../etc/passwd" can't escape _LLM_DIR).
    """
    if uri == _INDEX_URI:
        return _LLM_INDEX if _LLM_INDEX.is_file() else None
    if not uri.startswith(_URI_PREFIX):
        return None
    rel = uri[len(_URI_PREFIX):]
    if not rel:
        return None
    # Bare names (no slash) belong to the default section; keep explicit
    # section prefixes (e.g. future "frontend/xyz") verbatim.
    if "/" not in rel:
        rel = f"{_DEFAULT_SECTION}/{rel}"
    candidate = (_LLM_DIR / f"{rel}.md").resolve()
    # Ensure we stay inside _LLM_DIR (block ".." escapes).
    try:
        candidate.relative_to(_LLM_DIR.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _description_for(path: Path) -> str:
    """First non-empty line after the top-level H1, as a one-line blurb."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    lines = text.splitlines()
    saw_title = False
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# ") and not saw_title:
            saw_title = True
            continue
        if line.startswith("#"):
            # Skipped past the first paragraph without finding one; fall back.
            return ""
        # First non-heading, non-empty line after the title.
        # Trim to a sensible length so resources/list doesn't ship walls of text.
        return line if len(line) <= 240 else line[:237] + "..."
    return ""


_ACRONYMS = {"mcp", "api", "ui", "ssh", "cpu", "gpu", "cuda", "pcg", "ccd"}


def _name_for(path: Path) -> str:
    """Human-readable resource name derived from the filename."""
    if path == _LLM_INDEX:
        return "LLM documentation index"
    words = []
    for word in path.stem.split("_"):
        words.append(word.upper() if word.lower() in _ACRONYMS else word.capitalize())
    return " ".join(words)


def list_llm_resources() -> list[dict]:
    """Enumerate every llm:// resource shipped with this addon install.

    Safe to call on every ``resources/list`` request: the enumeration
    walks a small bundled tree (~12 files) and runs in microseconds.
    """
    resources: list[dict] = []
    if _LLM_INDEX.is_file():
        resources.append({
            "uri": _INDEX_URI,
            "name": _name_for(_LLM_INDEX),
            "description": (
                _description_for(_LLM_INDEX)
                or "Top-level router listing every other llm:// resource."
            ),
            "mimeType": "text/markdown",
        })
    if _LLM_DIR.is_dir():
        for path in sorted(_LLM_DIR.rglob("*.md")):
            resources.append({
                "uri": _uri_for(path),
                "name": _name_for(path),
                "description": _description_for(path),
                "mimeType": "text/markdown",
            })
    return resources


def read_llm_resource(uri: str) -> str | None:
    """Return the markdown body for *uri*, or None if no match.

    Callers wrap the text in the MCP ``contents`` envelope; this helper
    just does the URI-to-bytes resolution.
    """
    path = _path_for(uri)
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None
