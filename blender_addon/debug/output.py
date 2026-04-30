# File: output.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Response-formatting helpers shared by every CLI in this package.

Unified because MCP ``tools/call`` and the debug TCP ``exec`` port return
different envelopes, and we want the end-user experience of ``main.py exec``
vs ``perf.py report`` to be identical — code prints, errors go to stderr,
nothing else.
"""

import json
import sys


def print_json(obj):
    """Pretty-print a Python object as JSON to stdout."""
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def print_response(resp):
    """Print whatever ``resp`` carries as if the caller had executed the
    code locally. Handles both response shapes:

        MCP tools/call:  {"result": {"content": [{"text": "<json>"}]}}
        debug TCP exec:  {"status": "ok", "output": "..."}

    Writes script stdout to our stdout; errors go to stderr.
    """
    if isinstance(resp, dict):
        if "result" in resp and isinstance(resp["result"], dict):
            for item in resp["result"].get("content") or []:
                if not (isinstance(item, dict) and "text" in item):
                    continue
                text = item["text"]
                try:
                    inner = json.loads(text)
                except (ValueError, TypeError):
                    print(text)
                    continue
                if not isinstance(inner, dict):
                    print(text)
                    continue
                # Prefer "output" (run_python_script's stdout capture), fall
                # back to "message" (most other MCP tools' human-facing reply).
                out = inner.get("output") or inner.get("message") or ""
                if out:
                    sys.stdout.write(out)
                    if not out.endswith("\n"):
                        sys.stdout.write("\n")
                else:
                    # Structured reply with no message/output — show the
                    # payload so the caller can still see the result.
                    payload = {k: v for k, v in inner.items()
                               if k not in {"status", "success"}}
                    if payload:
                        json.dump(payload, sys.stdout, indent=2,
                                  ensure_ascii=False)
                        sys.stdout.write("\n")
                if inner.get("error"):
                    print(f"error: {inner['error']}", file=sys.stderr)
                elif inner.get("status") == "error":
                    print(f"error: {inner.get('message', 'unknown')}",
                          file=sys.stderr)
                elif inner.get("success") is False:
                    print(f"failure: {inner.get('message')}", file=sys.stderr)
            return
        if resp.get("status") == "error":
            if resp.get("output"):
                sys.stdout.write(resp["output"])
            print(f"error: {resp.get('error', 'unknown')}", file=sys.stderr)
            return
        if "output" in resp:
            sys.stdout.write(resp["output"])
            return
    json.dump(resp, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
