#!/usr/bin/env python3
# File: shorten_webpack_chunks.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""
Shorten webpack chunk IDs to reduce path length.

Scans only jupyter-related directories for speed.
"""

import os
import re
import hashlib
import sys
from pathlib import Path


def short_id(chunk_id: str) -> str:
    """Generate 8-char hash from chunk ID."""
    return "c" + hashlib.md5(chunk_id.encode()).hexdigest()[:7]


def find_chunk_map(content: str) -> dict:
    """Extract chunk ID to hash mapping from remoteEntry.js content."""
    pattern = r'\{("[^"]+":"[a-f0-9]+",?\s*)+\}\[chunkId\]'
    match = re.search(pattern, content)
    if not match:
        return {}

    map_text = match.group(0).rsplit('[chunkId]', 1)[0]
    chunk_map = {}
    for m in re.finditer(r'"([^"]+)":"([a-f0-9]+)"', map_text):
        chunk_map[m.group(1)] = m.group(2)
    return chunk_map


def replace_in_file(file_path: Path, renames: dict) -> bool:
    """Replace old chunk IDs with new ones. Returns True if modified.

    Uses quoted replacement to avoid substring matches (e.g., replacing
    "lib_index_js" inside "packages_base_lib_index_js").
    """
    try:
        content = file_path.read_text(encoding='utf-8')
    except:
        return False

    modified = False
    # Sort by length descending to replace longer IDs first
    # This prevents shorter IDs from matching inside longer ones
    for old_id, new_id in sorted(renames.items(), key=lambda x: len(x[0]), reverse=True):
        # Replace quoted versions (in JS code)
        for quote in ['"', "'"]:
            old_quoted = f'{quote}{old_id}{quote}'
            new_quoted = f'{quote}{new_id}{quote}'
            if old_quoted in content:
                content = content.replace(old_quoted, new_quoted)
                modified = True

        # Replace in filenames (for RECORD files and sourceMappingURL)
        old_file = f'{old_id}.'
        new_file = f'{new_id}.'
        if old_file in content:
            content = content.replace(old_file, new_file)
            modified = True

    if modified:
        file_path.write_text(content, encoding='utf-8')
    return modified


def main():
    dist_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dist")
    python_dir = dist_dir / "python"
    labext_dir = python_dir / "share" / "jupyter" / "labextensions"

    if not labext_dir.exists():
        print("No labextensions directory found")
        return

    # Phase 1: Collect chunk IDs that have actual files
    print("Phase 1: Collecting chunk IDs with actual files...")
    all_renames = {}
    all_chunk_maps = {}

    for static_dir in list(labext_dir.glob("*/static")) + list(labext_dir.glob("@*/*/static")):
        for remote_entry in static_dir.glob("remoteEntry.*.js"):
            chunk_map = find_chunk_map(remote_entry.read_text(encoding='utf-8'))
            for chunk_id, hash_val in chunk_map.items():
                if len(chunk_id) > 10:
                    # Only include if actual file exists (not virtual webpack sharing chunks)
                    chunk_file = static_dir / f"{chunk_id}.{hash_val}.js"
                    if chunk_file.exists():
                        all_renames[chunk_id] = short_id(chunk_id)
                        all_chunk_maps[chunk_id] = hash_val

    if not all_renames:
        print("No long chunk IDs with files found")
        return

    print(f"  Found {len(all_renames)} long chunk IDs with files")

    # Phase 2: Replace in jupyter-related directories only
    print("\nPhase 2: Updating references...")
    files_updated = 0

    # 1. All files in labextensions
    for root, dirs, files in os.walk(labext_dir):
        for fname in files:
            if replace_in_file(Path(root) / fname, all_renames):
                print(f"    {fname}")
                files_updated += 1

    # 2. RECORD files in site-packages for jupyter/widget packages
    site_packages = python_dir / "Lib" / "site-packages"
    if site_packages.exists():
        for dist_info in site_packages.glob("*jupyter*.dist-info"):
            record_file = dist_info / "RECORD"
            if record_file.exists() and replace_in_file(record_file, all_renames):
                print(f"    {record_file.name} ({dist_info.name})")
                files_updated += 1

        for dist_info in site_packages.glob("*widget*.dist-info"):
            record_file = dist_info / "RECORD"
            if record_file.exists() and replace_in_file(record_file, all_renames):
                print(f"    {record_file.name} ({dist_info.name})")
                files_updated += 1

    print(f"  Updated {files_updated} files")

    # Phase 3: Rename chunk files
    print("\nPhase 3: Renaming files...")
    files_renamed = 0

    for static_dir in list(labext_dir.glob("*/static")) + list(labext_dir.glob("@*/*/static")):
        for old_id, new_id in all_renames.items():
            hash_val = all_chunk_maps.get(old_id)
            if not hash_val:
                continue

            for ext in ['.js', '.js.map']:
                old_file = static_dir / f"{old_id}.{hash_val}{ext}"
                new_file = static_dir / f"{new_id}.{hash_val}{ext}"
                if old_file.exists():
                    old_file.rename(new_file)
                    print(f"    {new_id}{ext}")
                    files_renamed += 1

    print(f"  Renamed {files_renamed} files")

    # Report longest path
    max_len = 0
    max_rel = None
    for root, _, files in os.walk(dist_dir):
        for fname in files:
            rel = os.path.relpath(os.path.join(root, fname), dist_dir)
            if len(rel) > max_len:
                max_len = len(rel)
                max_rel = rel

    print(f"\nLongest relative path: {max_len} chars")
    if max_rel:
        print(f"  {max_rel}")


if __name__ == "__main__":
    main()
