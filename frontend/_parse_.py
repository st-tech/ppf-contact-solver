import re
import os
from typing import Any


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

            if "Name" in doc.keys():
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
                content = line.strip("//").strip()
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
            elif line.startswith("/*== push"):
                register(extract_name(line))
            elif "logging.push(" in line:
                register(extract_name(line))
            elif "logging.mark(" in line:
                register(extract_name(line))

        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                par_name = ""
                if filename != "args.rs":
                    if filename.endswith(".cu") or filename.endswith(".rs"):
                        path = os.path.join(dirpath, filename)
                        lines = open(path, "r").readlines()
                        for line in lines:
                            line = line.strip()
                            if "#include" not in line:
                                parse_line(line)

        result = dict(sorted(result.items()))
        return result


class ParamParser:
    @staticmethod
    def get_default_params(path: str) -> dict[str, dict[str, Any]]:
        """Get the default parameters.

        Args:
            path (str): The path to the args.rs file.

        Returns:
            dict[str, Any]: The default parameters.
        """
        att_pattern = re.compile(r"#\[(.*?)\]")
        field_pattern = re.compile(r"pub\s+(\w+):\s*([^,]+),?")
        struct_start_pattern = re.compile(r"^pub\s+struct\s+Args\s*\{")
        struct_end_pattern = re.compile(r"^\s*\}")
        curr_attributes = []
        inside_struct = False
        result = {}
        doc = {}
        var_type = None
        description_mode = False
        description = ""

        def parse_line(line):
            nonlocal doc
            nonlocal description_mode
            nonlocal description
            nonlocal var_type
            if line.strip().startswith("pub"):
                parts = line.strip().split()
                if len(parts) > 2:
                    var_type = parts[2].rstrip(",")
            if line.strip().startswith("//"):
                line = line.strip("// ").strip()
                if "Do not list" in line:
                    doc["list"] = False
                if line.startswith("Description:"):
                    description_mode = True
                else:
                    if description_mode:
                        if description:
                            description += " "
                        description += line
                    else:
                        try:
                            fields = line.split(":")
                            field = fields[0].strip()
                            text = fields[1].strip()
                            doc[field] = text
                        except Exception as _:
                            pass

        def clear_doc():
            nonlocal doc
            nonlocal description_mode
            nonlocal description
            nonlocal var_type
            doc = {"list": True}
            description_mode = False
            description = ""
            var_type = None

        with open(path, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if not inside_struct:
                    if struct_start_pattern.match(line.strip()):
                        inside_struct = True
                    continue
                parse_line(line)
                if struct_end_pattern.match(line.strip()):
                    break
                if not line.strip() or line.strip().startswith("//"):
                    continue
                attr_match = att_pattern.match(line.strip())
                if attr_match:
                    curr_attributes.append(attr_match.group(1).strip())
                else:
                    field_match = field_pattern.match(line.strip())
                    if field_match:
                        field_name = field_match.group(1).replace("_", "-")
                        default_value = None
                        for attr in curr_attributes:
                            clap_match = re.match(r"clap\((.*?)\)", attr)
                            if clap_match:
                                args = clap_match.group(1)
                                arg_list = re.findall(
                                    r'(?:[^,"]|"(?:\\.|[^"\\])*")+', args
                                )
                                for arg in arg_list:
                                    arg = arg.strip()
                                    if "=" in arg:
                                        key, value = map(str.strip, arg.split("=", 1))
                                        value = value.strip('"').strip("'")
                                        if "default_value" in key:
                                            default_value = value
                        if default_value is not None:
                            try:
                                float_value = float(default_value)
                                default_value = (
                                    int(float_value)
                                    if float_value.is_integer()
                                    and var_type != "f32"
                                    and var_type != "f64"
                                    else float_value
                                )
                            except ValueError:
                                pass
                        doc["Description"] = description
                        result[field_name] = {
                            "value": default_value,
                            "type": var_type,
                            "doc": doc,
                        }
                        clear_doc()
                        curr_attributes = []
                    else:
                        curr_attributes = []

        result = dict(sorted(result.items()))
        return result
