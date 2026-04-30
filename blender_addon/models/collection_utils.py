# File: collection_utils.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0


def sort_keyframes_by_frame(keyframes):
    """Sort a Blender keyframe collection in-place by frame number (insertion sort)."""
    n = len(keyframes)
    idx = n - 1
    while idx > 0 and keyframes[idx].frame < keyframes[idx - 1].frame:
        keyframes.move(idx, idx - 1)
        idx -= 1
    return idx


def validate_no_duplicate_frame(keyframes, frame):
    """Raise ValueError if a keyframe at the given frame already exists."""
    for kf in keyframes:
        if kf.frame == frame:
            raise ValueError(f"Keyframe at frame {frame} already exists")


def safe_update_index(current_index, new_length):
    """Return a valid index after collection size changes."""
    return min(current_index, max(0, new_length - 1))


def generate_unique_name(prefix, existing_names):
    """Generate 'Prefix N' name that doesn't collide with existing_names."""
    i = 1
    while f"{prefix} {i}" in existing_names:
        i += 1
    return f"{prefix} {i}"
