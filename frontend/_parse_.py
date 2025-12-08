# File: _parse_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import os


class CppRustDocStringParser:
    @staticmethod
    def get_logging_docstrings(root: str) -> dict[str, dict[str, str]]:
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
