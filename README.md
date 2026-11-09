# MURMURATION: Sender Diversity and Task Diversion in LLM Agents

A controlled, reproducible experiment built for the **Swarm Alignment Hackathon**
(Apart Research × AI Alignment Foundation, Nov 6–8, 2026; Track 2: *The Science of Agent Networks*).

**Status:** environment, scorer, manifests and acceptance tests are implemented and passing.
The LLM runner, pilot and confirmatory runs are **not done yet**. No results exist in this repository.

## Research question

In a simulated workspace, does an LLM agent adopt an unrequested, harmless action more often when
**two attributed peers** suggest it than when **one peer suggests it twice**, with message content held fixed?

This is a *micro-level* study of one agent responding to **scripted, attributed** peer messages.
It is not a multi-agent swarm experiment, and it makes no network-level empirical claim.

## Terminology (kept deliberately precise)

| Term | Meaning in this project |
|---|---|
| **Task diversion** (title) | The construct. Operationalised as **off-task action adoption** (primary outcome) and **abandonment** (secondary). |
| **Sender diversity** (title) | Manipulated only through attributed sender labels (condition D vs C). Senders are scripted, so Methods use **attributed sender multiplicity**. |
| Contagion | Motivation only (simple vs. complex contagion). The design cannot establish complex contagion. |
| Swarm | The organisers' term. This experiment studies a single agent. |

## Design in one table

| Cond. | Exposure (delivered inside the agent's first tool result) |
|---|---|
| A | No peer messages |
| B | The first message of a pair, from one sender |
| C | Both messages, from the **same** sender |
| D | Both messages, from **two distinct** senders (identical text to C) |
| E | Both messages, **no sender attribution** (identical text to C) |

Conditions are paired within (task instance × replicate): same files, wording pair, message order and first sender.

## Outcomes (scored from environment state only; never from agent text)

`adopted` (valid checklist file written) · `completed` (correct item code in the report file) ·
category ∈ `task_only | both | abandonment | neither`. Full definitions: `scorer.py`.

## Quick start

```bash
python tests.py -v                         # acceptance tests, no network, must all pass
python meta.py                             # identity, protocol fingerprint, file hashes
python manifest.py --split pilot --instances 2  --reps 4 --out manifests/pilot.jsonl   # 120 trials
python manifest.py --split main  --instances 10 --reps 4 --out manifests/main.jsonl    # 600 trials
```

Python 3.10+, standard library only (the runner will add one inference SDK).

## Repository layout

```
env.py            controlled virtual workspace, task generator, tool schemas
scorer.py         deterministic outcome scoring (re-scorable from logs)
manifest.py       counterbalanced, seeded trial generation + validator
meta.py           canonical identity + protocol fingerprint
probe.py          provider validation probe (not an experiment; offline --dry-run available)
tests.py          acceptance tests with scripted agents
manifests/        pilot.jsonl (120) and main.jsonl (600); generated, do not hand-edit
docs/             provider and credit notes (verified facts vs. unknowns)
report/           report outline (official template requirements)
NAMING.md         record of the rename to MURMURATION
runner.py         (next) LLM runner + logging
analysis.py       (next) pre-registered analysis
```

## Reproducibility and safeguards

* **Pilot and confirmatory runs are separate splits** (`pilot-*` and `main-*` trial IDs, disjoint instances).
  Wording is calibrated on pilot items only and frozen before the main run.
* Every log record must carry: provider, model ID, model version if exposed, temperature, seed,
  trial ID, protocol fingerprint (`meta.py`), input/output token counts, latency and cost.
* After the freeze, any change to prompts, tools, wordings or scoring must be documented in `FREEZE.md`.
* **Never commit API keys.** Keys live in environment variables or an untracked `.env`
  (see `.env.example`). Before every commit run:
  `grep -rnE "gsk_|pt_live_|sk-[A-Za-z0-9]" . --exclude-dir=.git` and expect no output.
* No results, logs or configurations are deleted; superseded files are renamed, not removed.

## Prior work and preexisting code

All code in this repository was written during the Sprint window (Nov 6–9, 2026).
**TODO before submission:** list any reused snippets or boilerplate from earlier projects here and in the report.
Closest prior work and what this experiment adds are documented in the report (`report/OUTLINE.md`).
