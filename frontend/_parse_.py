# File: _parse_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import os


class CppRustDocStringParser:
    """Parser for logging-related docstrings in the C++/Rust sources.

    Example:
        Harvest the log-name table from the bundled C++/Rust sources
        so a session's recorded log keys can be annotated::

            import os
            from frontend import App, CppRustDocStringParser

            src_dir = os.path.join(App.get_proj_root(), "src")
            entries = CppRustDocStringParser.get_logging_docstrings(src_dir)
            print(sorted(entries)[:5])
    """

    @staticmethod
    def get_logging_docstrings(root: str) -> dict[str, dict[str, str]]:
        """Scan ``root`` for logging docstrings in ``.cu`` and ``.rs`` files.

        Walks ``root`` recursively (skipping ``args.rs``) and parses ``//``
        comment blocks describing ``SimpleLog``, ``logging.push``, and
        ``logging.mark`` call sites. Each block contributes a ``Label: value``
        dictionary plus a free-form ``Description`` and a derived ``filename``.

        Args:
            root (str): Directory to search.

        Returns:
            dict[str, dict[str, str]]: Mapping from entry name (with
            underscores replaced by hyphens) to the parsed docstring fields,
            sorted by key.

        Example:
            Look up the metadata associated with a specific log name
            under the bundled solver sources::

                import os
                from frontend import App, CppRustDocStringParser

                src_dir = os.path.join(App.get_proj_root(), "src")
                entries = CppRustDocStringParser.get_logging_docstrings(src_dir)
                if "time-per-frame" in entries:
                    print(entries["time-per-frame"].get("Description"))
        """
        result = {}
        doc = {}
        par_name = None
        desc = ""
        description_mode = False

        def clear():
            nonlocal doc
            nonlocal desc
            nonlocal description_mode
            doc = {}
            desc = ""
            description_mode = False

        def register(name):
            nonlocal par_name
            nonlocal doc
            nonlocal desc
            nonlocal description_mode

            if "Name" in doc:
                if desc:
                    doc["Description"] = desc
                if par_name:
                    doc["filename"] = f"{par_name}.{name}.out"
                else:
                    doc["filename"] = f"{name}.out"
                if "Map" in doc:
                    name = doc["Map"]
                    del doc["Map"]
                result[name.replace("_", "-")] = doc.copy()
            clear()

        def extract_name(line):
            start = line.find('"') + 1
            end = line.find('"', start)
            name = line[start:end].replace(" ", "_")
            return name

        def parse_line(line: str):
            nonlocal par_name
            nonlocal description_mode
            nonlocal desc
            nonlocal doc

            if line.strip() == "":
                clear()

            skip_lables = ["File", "Author", "License", "https"]
            if line.startswith("//"):
                content = line[2:].strip()  # Remove first 2 characters ("//")
                if description_mode:
                    if desc:
                        desc += " "
                    desc += content
                elif content.startswith("Description:"):
                    description_mode = True
                elif ":" in content:
                    fields = content.split(":")
                    label = fields[0].strip()
                    for skip_label in skip_lables:
                        if label == skip_label:
                            return
                    content = fields[1].strip()
                    doc[label] = content
            elif line.startswith("SimpleLog logging"):
                par_name = ""
                name = extract_name(line)
                register(name)
                par_name = name
            elif line.startswith("/*== push") or "logging.push(" in line or "logging.mark(" in line:
                register(extract_name(line))

        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                par_name = ""
                if filename != "args.rs" and filename.endswith((".cu", ".rs")):
                    path = os.path.join(dirpath, filename)
                    with open(path, encoding="utf-8") as f:
                        lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        if "#include" not in line:
                            parse_line(line)

        result = dict(sorted(result.items()))
        return result
