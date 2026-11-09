# Report outline: MURMURATION: Sender Diversity and Task Diversion in LLM Agents

Official requirements (apartresearch.com/sprints/swarm-alignment-2026, Guidelines tab):
PDF on the **official template**, **at most 8 pages** excluding references and appendices,
abstract **150–250 words**, authors and affiliations, and a **required appendix: "Limitations and Dual-Use /
Ethical Considerations"**. Optional: public repo, 3–5 minute video. Mark all preexisting code. Fabricated results or
references can disqualify a project.

## Structure

1. **Abstract** (150–250 words): question, design, headline estimate with CI, one-line limitation.
2. **Introduction:** METR/Hugging Face motivation (cascade of agents joining a workstream); the specific question.
3. **Related work and positioning:** Hu & Qu 2026 (closest; answer-flip outcome); *Who Do LLMs Trust?*;
   Prompt Infection; Secret Collusion (peripheral); Tilli 2026; simple vs. complex contagion (motivation only).
   State what is *not* claimed as novel. Mark every source not read in full.
4. **Method:** environment, tools, tasks, outcomes (scorer), conditions A–E, message design, counterbalancing,
   model/provider/settings, sample size, analysis plan (pre-registered), exclusions.
5. **Results:** per-cell rates with Wilson CIs; paired risk differences with cluster-bootstrap CIs for the
   confirmatory contrasts (B−A, C−B, D−C, C−E, Δ vs. reference); Holm-adjusted; exploratory results labelled.
6. **Discussion:** what the estimates do and do not show; consistency across models/wordings.
7. **Limitations and Dual-Use / Ethical Considerations (appendix, required):** scripted peers; single environment;
   benign payload; wording tuned on pilot; descriptive and injunctive cues bundled; benchmark biases;
   no network-level evidence. Dual-use: the payload is a harmless file write; note how attribution cues could
   in principle be used to make agent manipulation more effective.
8. **References** (verified only). **Appendix:** full prompts, wording pairs, scorer definitions, protocol fingerprint.

## Claims discipline

Allowed: "In this environment, for model X, adoption was …; D−C = … pp (95% CI …)."
Not allowed: "LLM agents show social reinforcement", "explains the Hugging Face incident", any network-level claim
from a simulation (simulations go in a section labelled *Simulation, not empirical evidence*).
