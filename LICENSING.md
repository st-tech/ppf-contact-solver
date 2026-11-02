# Licensing

This repository is licensed under the Apache License, Version 2.0
(`LICENSE`), with one exception: code that calls Blender's `bpy` module is
licensed under the GNU General Public License, version 3 or later
(SPDX: `GPL-3.0-or-later`). Blender treats a script that calls `bpy` as an
extension that works with Blender as a combined program, and requires it to
preserve the same freedoms as Blender.

| Path | License | Text |
| ---- | ------- | ---- |
| `blender_addon/` | GPL-3.0-or-later | `blender_addon/LICENSE` |
| `examples/blender/` | GPL-3.0-or-later | `examples/blender/LICENSE` |
| `addon_host_tests/` | GPL-3.0-or-later | `addon_host_tests/LICENSE` |
| everything else | Apache-2.0 | `LICENSE` |

Notes:

- The add-on ships without `frontend/` or any Rust code. Three files in
  `blender_addon/core/` (`noise.py`, `script_api.py`, `field_eval.py`) are
  copies, identical but for the license line, of `frontend/_noise_.py`, `frontend/_script_api_.py`
  and `frontend/_field_eval_.py`. The copyright holder offers each copy under
  the license of the directory it sits in.
- Do not paste code from `blender_addon/` or `examples/blender/` into the
  Apache-licensed tree, or the reverse for third-party GPL code.
- Each source file under the three GPL directories carries a
  `# License: GPL-3.0-or-later` header.
- `crates/ppf-cts-formats/tests/scripts/gen_fixtures.py` encodes with the
  Apache-licensed `frontend/_cbor_bridge_.py` and runs no add-on code. That
  bridge and `blender_addon/core/encoder/cbor_encode.py` are twins whose
  emitted bytes are compared by the rig scenario `bl_cbor_twin_agrees`.
- `addon_host_tests/` loads add-on modules to test them, so it is GPL too. No
  Apache-licensed file executes or imports add-on code.
