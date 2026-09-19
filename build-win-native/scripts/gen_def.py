#!/usr/bin/env python3
"""Generate a Windows module-definition (.def) file exporting the backend's
C ABI from the single source of truth, the ABI header.

The ABI functions (``be_*``) are declared ``extern "C"`` with no export
attribute, because on Linux and macOS a shared library exports every symbol
with default visibility, so the reference build needs nothing. MSVC exports
nothing from a DLL unless it is told to, and has no ``--export-all-symbols``
equivalent, so the Windows device-link must be handed an explicit EXPORTS
list. Reading it from the header keeps that list from drifting: a new ABI
function is exported the moment it is declared, and a removed one drops out.
"""
import re
import sys

if len(sys.argv) != 2:
    sys.stderr.write("usage: gen_def.py <backend_abi.h>\n")
    sys.exit(2)

text = open(sys.argv[1], encoding="utf-8").read()
# Every ABI entry point is a `be_<name>(` at a call/declaration position. The
# header is pure declarations, so every match is a function to export.
names = sorted(set(re.findall(r"\b(be_[a-z0-9_]+)\s*\(", text)))
if not names:
    sys.stderr.write("gen_def.py: no be_* symbols found in %s\n" % sys.argv[1])
    sys.exit(1)

out = sys.stdout
out.write("EXPORTS\n")
for n in names:
    out.write("    %s\n" % n)
