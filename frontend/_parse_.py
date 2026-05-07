# File: _parse_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from . import _rust  # type: ignore[attr-defined]


class CppRustDocStringParser:
    """Parser for logging-related docstrings in the C++/Rust sources.

    ``get_logging_docstrings(root)`` scans ``root`` for ``// Name:`` /
    ``logging.push("...")`` entries in ``.cu`` and ``.rs`` files and
    returns a sorted ``name -> entry`` dict (underscores replaced by
    hyphens in keys). The implementation is
    :func:`_ppf_cts_py.get_logging_docstrings` (defined in
    ``crates/ppf-cts-core/src/parsers.rs``).

    Example:
        Harvest the log-name table from the bundled C++/Rust sources
        so a session's recorded log keys can be annotated::

            import os
            from frontend import App, CppRustDocStringParser

            src_dir = os.path.join(App.get_proj_root(), "src")
            entries = CppRustDocStringParser.get_logging_docstrings(src_dir)
            print(sorted(entries)[:5])
    """

    get_logging_docstrings = staticmethod(_rust.get_logging_docstrings)
