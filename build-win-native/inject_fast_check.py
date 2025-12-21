#!/usr/bin/env python
# File: inject_fast_check.py
# Injects App.set_fast_check() after App.create/load calls

import re
import sys

if len(sys.argv) < 2:
    print("Usage: inject_fast_check.py <python_file>")
    sys.exit(1)

filepath = sys.argv[1]
with open(filepath, 'r') as f:
    content = f.read()

# Insert App.set_fast_check() after app = App.create(...) or App.load(...)
pattern = r'(app = App\.(create|load)\([^)]+\))'
replacement = r'\1; App.set_fast_check()'
content = re.sub(pattern, replacement, content)

with open(filepath, 'w') as f:
    f.write(content)
