#!/usr/bin/env python3
# File: embed_source.py
# Code: GitHub Copilot
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import json
import pathlib
import sys

symbol, source_path, output_path = sys.argv[1:4]
source = pathlib.Path(source_path).read_text(encoding="utf-8")
output = (
    "// Generated from " + source_path + ". Do not edit.\n"
    "static const char " + symbol + "[] = " + json.dumps(source) + ";\n"
)
pathlib.Path(output_path).write_text(output, encoding="utf-8")
