#!/usr/bin/env python3
"""One-shot: reorganize docs/blender_addon/workflow/ into four subdirs.

Mapping (old -> new, both relative to docs/blender_addon/):

  workflow/object_groups.md        -> workflow/scene/object_groups.md
  workflow/static_objects.md       -> workflow/scene/static_objects.md
  workflow/material_params.md      -> workflow/params/material.md
  workflow/scene_params.md         -> workflow/params/scene.md
  workflow/dynamic_parameters.md   -> workflow/params/dynamic.md
  workflow/pins_and_operations.md  -> workflow/constraints/pins.md
  workflow/invisible_colliders.md  -> workflow/constraints/colliders.md
  workflow/snap_and_merge.md       -> workflow/constraints/snap_merge.md
  workflow/simulating.md           -> workflow/sim/simulating.md
  workflow/jupyterlab.md           -> workflow/sim/jupyterlab.md
  workflow/baking.md               -> workflow/sim/baking.md

Also:
  - Delete previous-attempt stubs (scene.md, parameters.md, constraints.md,
    simulation_and_output.md) under workflow/.
  - Create {scene,params,constraints,sim}/index.md with short intros + toctrees.
  - Rewrite workflow/index.md's toctree to reference the four subdir indexes.
  - Inside moved files: prepend one `../` to every path starting with `../`
    (image references, up-and-over references to other doc sections).
  - Inside moved files: rewrite same-directory links to siblings based on the
    mapping (e.g. `./material_params.md#foo` -> `../params/material.md#foo`
    or just `material.md#foo` if the target is in the same new subdir).
  - In every other .md/.rst file across the repo, rewrite
    `../workflow/<old>.md(#...)` (or same-dir variants from index.md-like
    places) to the new location.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # docs/
ADDON_DOCS = REPO / "blender_addon"
WORKFLOW = ADDON_DOCS / "workflow"

# old basename -> (subdir, new basename without extension)
MAPPING: dict[str, tuple[str, str]] = {
    "object_groups":       ("scene",       "object_groups"),
    "static_objects":      ("scene",       "static_objects"),
    "material_params":     ("params",      "material"),
    "scene_params":        ("params",      "scene"),
    "dynamic_parameters":  ("params",      "dynamic"),
    "pins_and_operations": ("constraints", "pins"),
    "invisible_colliders": ("constraints", "colliders"),
    "snap_and_merge":      ("constraints", "snap_merge"),
    "simulating":          ("sim",         "simulating"),
    "jupyterlab":          ("sim",         "jupyterlab"),
    "baking":              ("sim",         "baking"),
}

SUBDIR_ORDER = ["scene", "params", "constraints", "sim"]

SUBDIR_INDEX_CONTENT: dict[str, str] = {
    "scene": """# Scene

How the solver sees your Blender scene: as a collection of **object
groups** that carry type, material, and assigned meshes, plus a
**Static** variant for non-deforming colliders and props.

```{toctree}
:maxdepth: 1

object_groups
static_objects
```
""",
    "params": """# Parameters

Everything that tunes *how* the simulation behaves, from per-group
material properties to scene-wide solver settings and keyframed
dynamic overrides.

```{toctree}
:maxdepth: 1

material
scene
dynamic
```
""",
    "constraints": """# Constraints

Shape the motion by pinning vertices, stitching groups together, or
introducing parametric collision boundaries that never appear in the
scene.

```{toctree}
:maxdepth: 1

pins
colliders
snap_merge
```
""",
    "sim": """# Simulation and output

Running the solve and getting the result back onto your Blender
meshes, whether interactively from the sidebar, from JupyterLab, or
baked down to plain keyframes.

```{toctree}
:maxdepth: 1

simulating
jupyterlab
baking
```
""",
}

WORKFLOW_INDEX_TOCTREE = """```{toctree}
:maxdepth: 2

scene/index
params/index
constraints/index
sim/index
```"""


# --- Helpers -----------------------------------------------------------

def rewrite_content_in_moved_file(text: str, new_subdir: str) -> str:
    """Rewrite a workflow/*.md that is being moved to workflow/<new_subdir>/<new>.md.

    - Every `../x` -> `../../x` (images, up-and-over refs to other sections).
    - Every same-dir markdown link `./<old>.md[#...]` or `<old>.md[#...]`
      is rewritten:
        same new_subdir target -> `<new>.md[#...]` (or `./<new>.md`)
        different subdir       -> `../<tgt_subdir>/<new>.md[#...]`
    """
    # Phase 1: add extra ../ to any link starting with '..' segment. Match
    # inside markdown link targets and MyST figure directives / bare paths.
    # We only touch paths that begin with `..` (anywhere after `](`, `:alt:` is
    # not a path, `{figure}` and `{image}` directive args take bare paths).
    def prepend_dotdot(m: re.Match[str]) -> str:
        return m.group(1) + "../" + m.group(2)

    # markdown link or image: (](   |   {figure} / {image}   ) followed by '..'
    text = re.sub(
        r"(\]\()\.\.(/)",
        lambda m: m.group(1) + "../.." + m.group(2),
        text,
    )
    # MyST {figure} / {image} directive path on its own line (``` `{figure} ../...`` )
    text = re.sub(
        r"(\{figure\}\s*|\{image\}\s*)\.\.(/)",
        lambda m: m.group(1) + "../.." + m.group(2),
        text,
    )

    # Phase 2: same-dir sibling refs inside markdown links.
    # Matches: ](./<old>.md[#anchor]) or ](<old>.md[#anchor])
    def same_dir_link(m: re.Match[str]) -> str:
        opener = m.group(1)          # "](" or "](./"
        old = m.group(2)              # old basename
        tail = m.group(3) or ""       # optional "#..."
        if old not in MAPPING:
            # Not one we handle; leave untouched.
            return m.group(0)
        tgt_subdir, new_base = MAPPING[old]
        if tgt_subdir == new_subdir:
            new_path = f"{new_base}.md{tail}"
        else:
            new_path = f"../{tgt_subdir}/{new_base}.md{tail}"
        return f"{opener.rstrip('/')}{'/' if opener.endswith('/') else ''}{new_path}"

    # Rebuild cleanly: use explicit regex to capture `](./` or `](`.
    pattern = re.compile(r"(\]\((?:\./)?)([a-z_][a-z0-9_]*)\.md(#[^)]*)?\)")

    def repl(m: re.Match[str]) -> str:
        prefix = m.group(1)            # "](" or "](./"
        old = m.group(2)
        anchor = m.group(3) or ""
        if old not in MAPPING:
            return m.group(0)
        tgt_subdir, new_base = MAPPING[old]
        if tgt_subdir == new_subdir:
            # same subdir -> bare sibling link
            return f"](./{new_base}.md{anchor})" if prefix.endswith("/") else f"]({new_base}.md{anchor})"
        return f"](../{tgt_subdir}/{new_base}.md{anchor})"

    text = pattern.sub(repl, text)
    return text


def rewrite_external_refs(text: str) -> str:
    """In files OUTSIDE the workflow/ dir, rewrite `../workflow/<old>.md` links
    (and `workflow/<old>.md` on the top-level index.md) to the new paths."""
    # Pattern: any number of ../ then `workflow/<old>.md[#anchor]` inside a link.
    pattern = re.compile(r"(\]\(((?:\.\./)*|\./)?workflow/)([a-z_][a-z0-9_]*)\.md(#[^)]*)?\)")

    def repl(m: re.Match[str]) -> str:
        link_prefix = m.group(1)       # "](../workflow/" or "](workflow/"
        # (m.group(2) is the dot-slash prefix captured inside, already part of link_prefix)
        old = m.group(3)
        anchor = m.group(4) or ""
        if old not in MAPPING:
            return m.group(0)
        tgt_subdir, new_base = MAPPING[old]
        return f"{link_prefix}{tgt_subdir}/{new_base}.md{anchor})"

    return pattern.sub(repl, text)


# --- Main phases ---------------------------------------------------

def phase_create_subdirs() -> None:
    for sub in SUBDIR_ORDER:
        (WORKFLOW / sub).mkdir(parents=True, exist_ok=True)


def phase_move_and_rewrite_moved_files() -> None:
    for old_base, (sub, new_base) in MAPPING.items():
        old_path = WORKFLOW / f"{old_base}.md"
        new_path = WORKFLOW / sub / f"{new_base}.md"
        if not old_path.exists():
            print(f"WARN: missing source {old_path}")
            continue
        text = old_path.read_text()
        new_text = rewrite_content_in_moved_file(text, sub)
        new_path.write_text(new_text)
        old_path.unlink()
        print(f"moved  {old_path.relative_to(REPO)} -> {new_path.relative_to(REPO)}")


def phase_delete_prev_attempt_stubs() -> None:
    for stub in ("scene.md", "parameters.md", "constraints.md", "simulation_and_output.md"):
        p = WORKFLOW / stub
        if p.exists():
            p.unlink()
            print(f"deleted stub {p.relative_to(REPO)}")


def phase_write_subdir_indexes() -> None:
    for sub, content in SUBDIR_INDEX_CONTENT.items():
        (WORKFLOW / sub / "index.md").write_text(content)
        print(f"wrote  workflow/{sub}/index.md")


def phase_update_workflow_index() -> None:
    p = WORKFLOW / "index.md"
    text = p.read_text()
    # Replace any existing toctree block with the new one.
    new_text, n = re.subn(
        r"```\{toctree\}[\s\S]*?```",
        WORKFLOW_INDEX_TOCTREE,
        text,
        count=1,
    )
    if n != 1:
        # No toctree found; append.
        new_text = text.rstrip() + "\n\n" + WORKFLOW_INDEX_TOCTREE + "\n"
    p.write_text(new_text)
    print(f"updated {p.relative_to(REPO)}")


def phase_rewrite_external_refs() -> None:
    """Sweep every .md / .rst file OUTSIDE workflow/ for ../workflow/<old>.md
    references and rewrite them."""
    for p in ADDON_DOCS.rglob("*"):
        if not p.is_file() or p.suffix not in (".md", ".rst"):
            continue
        # Skip files inside workflow/ itself; their intra-workflow refs were
        # handled in phase_move_and_rewrite_moved_files. The new subdir
        # indexes don't need rewriting either.
        if WORKFLOW in p.parents or p == WORKFLOW:
            continue
        text = p.read_text()
        new_text = rewrite_external_refs(text)
        if new_text != text:
            p.write_text(new_text)
            print(f"rewrote external refs in  {p.relative_to(REPO)}")


def main() -> int:
    phase_create_subdirs()
    phase_delete_prev_attempt_stubs()
    phase_move_and_rewrite_moved_files()
    phase_write_subdir_indexes()
    phase_update_workflow_index()
    phase_rewrite_external_refs()
    print("\nDone. Clean build recommended: rm -rf docs/_build && docs/build.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
