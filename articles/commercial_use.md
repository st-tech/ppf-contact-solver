# 💼 Commercial Use and Beyond

This project is released under the [Apache License 2.0](../LICENSE). In plain terms, you may use, modify, and redistribute the code in commercial products, including proprietary software, without paying royalties or open-sourcing your own code. You only need to preserve the license notice and the attribution required by the license.

The one exception is the code that calls Blender's Python API (`bpy`). Blender requires an add-on to preserve the same freedoms Blender grants, so the Blender add-on (`blender_addon/`), its example scripts (`examples/blender/`) and the tests that load it (`addon_host_tests/`) are licensed under the [GNU General Public License, version 3 or later](../blender_addon/LICENSE) (`GPL-3.0-or-later`). Everything else, including the solver and the Python frontend, remains Apache 2.0. [LICENSING.md](../LICENSING.md) maps each path to its license.

If you build something on top of this solver, we would love to hear about it, but you are not obligated to disclose anything.

## 🔧 Building Your Own Solver

If you like, we suggest taking the whole source code apart and building your own solver tailored to your specific needs. With the recent advances in coding agents, this is very likely possible even without any expertise in physics or coding.

Once you build your own, you are free to publish it as your own project or commercialize it, as long as you follow the license terms described above.

You don't need to use this project as is. Customize it as you like!
