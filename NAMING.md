# Naming record

**Canonical identity (use everywhere):**
`MURMURATION: Sender Diversity and Task Diversion in LLM Agents`

* Short name: `MURMURATION`
* Repository slug: `murmuration`
* Rename applied: Mon Nov 9, 2026 (before any LLM run)

## What changed

| Item | Before | After |
|---|---|---|
| Working folder | `swarm-diversion/` | `murmuration/` |
| Module docstrings (`env.py`, `scorer.py`, `manifest.py`, `tests.py`) | no project name | first line = canonical title |
| New files | — | `README.md`, `NAMING.md`, `meta.py`, `docs/PROVIDER_NOTES.md`, `report/OUTLINE.md`, `.gitignore`, `.env.example` |
| `tests.py` | 47 tests | 51 tests (+4 identity/fingerprint tests; no existing test altered) |

## What did NOT change (verified, not assumed)

* **Experimental logic:** `env.py`, `scorer.py`, `manifest.py` compared before/after with docstrings
  blanked (Python AST hash): identical.
* **Manifests are byte-identical** (sha256 unchanged):
  * `manifests/pilot.jsonl` `e1e0fb73a4c9b0edc0ce2a9eb4b5d643d37d12cbfd8799846acc4b83dacc2e4e`
  * `manifests/main.jsonl`  `7cd90b6aa9cedc13290b505a499a3d845b52b9a8e7cf6398b7b151f21024fda3`
* Trial IDs, conditions, wordings, task templates, scoring rules, seeds: unchanged.
* No results or logs existed; none were touched.

## Legacy working names (all conversation-only; none were present in any file)

| Former name | Status |
|---|---|
| "Rumour or Movement? Simple vs. Complex Contagion of Task Drift in LLM-Agent Swarms" | retired |
| "Swarm Contagion" / "swarm-contagion" | retired |
| "SwarmWatch" | retired (earlier, different idea: a collusion monitor) |
| "swarm-diversion" (folder) | renamed to `murmuration` |
| "Who Said It Matters?" | retired as a title; the idea survives in the subtitle |

`tests.py::TestIdentity::test_no_legacy_working_names_in_code` guards against reintroducing them in code.

## Terms deliberately NOT renamed (they carry scientific meaning)

* **contagion**: kept only as motivation (simple vs. complex contagion); the design cannot establish it.
* **swarm**: the hackathon's own terminology; kept when quoting the Sprint or the METR incident.
* **diversion / sender diversity**: kept in the title as requested. In Methods, state that the manipulation is
  *attributed sender multiplicity* (scripted labels) and the measured outcome is *off-task action adoption*.
* **sender-multiplicity** (in code comments): precise name of the manipulation; kept.

## Renaming the GitHub repository (separate action: needs your account)

I cannot create or rename repositories from here. Exact steps:

**New repository (no history yet, recommended):**
```bash
cd murmuration
git init -b main
git add .
grep -rnE "gsk_|pt_live_|sk-[A-Za-z0-9]" . --exclude-dir=.git   # must print nothing
git commit -m "MURMURATION: environment, scorer, manifests, acceptance tests"
gh repo create murmuration --public --source=. --remote=origin --push \
  --description "MURMURATION: Sender Diversity and Task Diversion in LLM Agents. Controlled experiment on attributed peer messages and off-task action adoption."
```
**If a repo with the old name already exists:** GitHub → Settings → General → Repository name → `murmuration`
(GitHub redirects the old URL), then `git remote set-url origin https://github.com/<you>/murmuration.git`.

**Suggested topics:** `ai-safety`, `multi-agent`, `llm-agents`, `hackathon`, `reproducible-research`.
**Visibility:** the submission form takes a public repo link; you can keep the repo private until just before you submit.
