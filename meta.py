"""
MURMURATION: Sender Diversity and Task Diversion in LLM Agents
meta.py - canonical project identity and protocol fingerprint.

The fingerprint hashes everything that defines what the agent SEES and how the
trials are built (system prompt, tool schemas, step cap, templates, message
wordings, sender pool, conditions). The runner must stamp it into every log
record. If the fingerprint at confirmatory time differs from the one recorded
at the protocol freeze, something changed and must be documented.

    python meta.py            # print identity, fingerprint, and source-file hashes
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_NAME = "MURMURATION"
PROJECT_SUBTITLE = "Sender Diversity and Task Diversion in LLM Agents"
PROJECT_TITLE = f"{PROJECT_NAME}: {PROJECT_SUBTITLE}"
REPO_SLUG = "murmuration"
PROTOCOL_VERSION = "1.0"          # bump ONLY with a documented change (see NAMING.md / FREEZE.md)


def protocol_fingerprint() -> str:
    """sha256 over the experiment-defining constants (not over comments/docstrings)."""
    import env
    import manifest
    payload = {
        "system_prompt": env.SYSTEM_PROMPT,
        "tool_schemas": env.TOOL_SCHEMAS,
        "max_steps": env.DEFAULT_MAX_STEPS,
        "max_file_chars": env.MAX_FILE_CHARS,
        "n_rows": env.N_ROWS,
        "templates": list(env.TEMPLATES),
        "side_action_path": env.SIDE_ACTION_PATH,
        "wordings": manifest.WORDINGS,
        "name_pool": manifest.NAME_POOL,
        "conditions": list(manifest.CONDITIONS),
        "n_messages": manifest.N_MESSAGES,
        "orders": list(manifest.ORDERS),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def file_hashes(root: str | Path = ".") -> dict:
    """sha256 of the code and manifests that define a run (for FREEZE.md records)."""
    root = Path(root)
    names = ["env.py", "scorer.py", "manifest.py", "meta.py",
             "manifests/pilot.jsonl", "manifests/main.jsonl"]
    return {n: hashlib.sha256((root / n).read_bytes()).hexdigest()
            for n in names if (root / n).exists()}


if __name__ == "__main__":
    print(PROJECT_TITLE)
    print(f"protocol_version : {PROTOCOL_VERSION}")
    print(f"fingerprint      : {protocol_fingerprint()}")
    for name, digest in file_hashes().items():
        print(f"{digest}  {name}")
