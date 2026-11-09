#!/usr/bin/env python3
"""
MURMURATION: Sender Diversity and Task Diversion in LLM Agents
probe.py - provider validation probe. This is NOT an experiment run: its output is
never used in analysis. It answers four questions before we spend time on a pilot:

  1. Does your Adaption key work, and which hosted models can it see?
  2. Can each candidate model complete our tool-calling loop (read -> write -> finish)?
  3. How many tokens does one trial use (so we can budget), and how fast is a call?
  4. What happens under light parallel load (rate limits)?

Cost: about 20 calls per model, a few tens of thousands of tokens in total. At the posted
prices that is well under 1 credit. It asks for confirmation before any paid call.

Safety: the key is read from the ADAPTION_API_KEY environment variable ONLY. It is never
printed, logged or written to disk.

Usage:
    python probe.py --dry-run            # offline, no key, no network: checks your local setup
    python probe.py                      # real probe (asks for confirmation)
    python probe.py --models qwen3.5-397b --yes
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import env
import manifest
import meta
import scorer

DEFAULT_MODELS = ["gemma-4-31b-it", "qwen3.5-397b", "deepseek-v4-flash"]

# Posted prices, credits per 1M tokens (input, output). Source: docs.adaptionlabs.ai/inference-stacks/models
# (read Nov 9, 2026). Cached-input discounts ignored (conservative). Unit conversion credit->USD is NOT known.
POSTED_PRICES = {
    "gemma-4-31b-it": (0.09, 0.34),
    "gemma-4-26b-a4b-it": (0.0765, 0.255),
    "qwen3.5-397b": (0.55, 3.50),
    "qwen3.8-27b": (0.039, 1.25),
    "glm-5.2": (1.40, 4.40),
    "glm-53-flash": (0.15, 0.50),
    "deepseek-v4-flash": (0.04444, 0.13332),
}


# --------------------------------------------------------------------------- #
# Normalised response + providers
# --------------------------------------------------------------------------- #
@dataclass
class NormResponse:
    model: str = ""
    response_id: str = ""
    status: str = ""
    items: list = field(default_factory=list)        # original output items (replayed into history)
    item_dicts: list = field(default_factory=list)   # JSON-able copies (for logs)
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    headers: dict = field(default_factory=dict)
    temperature_used: bool = True
    error: str = ""


def _get(item, key, default=None):
    return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)


def _to_dict(item) -> dict:
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump(exclude_none=True)
    return {"repr": repr(item)}


def flat_tools() -> list[dict]:
    """Our canonical (chat-style) schemas -> flat Responses-API function tools. Same names/descriptions/params."""
    out = []
    for t in env.TOOL_SCHEMAS:
        f = t["function"]
        out.append({"type": "function", "name": f["name"], "description": f["description"],
                    "parameters": f["parameters"]})
    return out


class AdaptionProvider:
    name = "adaption"

    def __init__(self, client=None):
        if client is None:
            from adaption import Adaption          # reads ADAPTION_API_KEY itself
            client = Adaption()
        self.client = client

    def list_hosted_models(self) -> list[str]:
        return sorted(m.id for m in self.client.models.list(type="hosted").data)

    def create(self, model, instructions, input_items, tools, temperature, max_output_tokens) -> NormResponse:
        kwargs = dict(model=model, instructions=instructions, input=input_items,
                      max_output_tokens=max_output_tokens, store=False)
        if tools:
            kwargs["tools"] = tools
        used_temp = temperature is not None
        t0 = time.perf_counter()
        try:
            try:
                raw = self.client.responses.with_raw_response.create(
                    **({"temperature": temperature} if used_temp else {}), **kwargs)
            except Exception as exc:                      # some models may reject temperature
                if used_temp and "temperature" in str(exc).lower():
                    used_temp = False
                    raw = self.client.responses.with_raw_response.create(**kwargs)
                else:
                    raise
            latency = time.perf_counter() - t0
            resp = raw.parse()
            hdrs = {k: v for k, v in dict(raw.headers).items()
                    if re.search(r"rate|limit|remaining|retry|credit|request-id", k, re.I)}
            usage = getattr(resp, "usage", None)
            items = list(resp.output or [])
            return NormResponse(
                model=getattr(resp, "model", ""), response_id=getattr(resp, "id", ""),
                status=getattr(resp, "status", ""), items=items,
                item_dicts=[_to_dict(i) for i in items], text=getattr(resp, "output_text", "") or "",
                input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
                output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
                latency_s=latency, headers=hdrs, temperature_used=used_temp,
                error=(getattr(getattr(resp, "error", None), "message", "") or ""))
        except Exception as exc:                          # never crash the probe on one bad call
            return NormResponse(latency_s=time.perf_counter() - t0, temperature_used=used_temp,
                                error=f"{type(exc).__name__}: {str(exc)[:300]}")


class FakeProvider:
    """Offline scripted 'model' used by --dry-run and tests: read -> write correct answer -> finish."""
    name = "fake"

    def list_hosted_models(self):
        return list(DEFAULT_MODELS)

    def create(self, model, instructions, input_items, tools, temperature, max_output_tokens) -> NormResponse:
        t0 = time.perf_counter()
        outs = [i for i in input_items if _get(i, "type") == "function_call_output"]
        n = len(outs)
        if not tools:
            items, text = [{"type": "message", "role": "assistant",
                            "content": [{"type": "output_text", "text": "OK"}]}], "OK"
        else:
            prompt = _get(input_items[0], "content", "")
            if n == 0:
                path = re.search(r"/data/\S+?\.csv", prompt).group(0)
                call = ("read_file", {"path": path})
            elif n == 1:
                rows = [ln.split(",") for ln in outs[0]["output"].strip().splitlines()[1:]]
                rows = [r for r in rows if r[0].startswith("ITEM-")]
                report = re.search(r"/reports/\S+?\.txt", prompt).group(0)
                if "highest amount" in prompt:
                    best = max(rows, key=lambda r: int(r[1]))
                elif "fewest units" in prompt:
                    best = min(rows, key=lambda r: int(r[1]))
                else:
                    best = max(rows, key=lambda r: r[1])
                call = ("write_file", {"path": report, "content": best[0]})
            else:
                call = ("finish", {})
            items = [{"type": "function_call", "name": call[0], "arguments": json.dumps(call[1]),
                      "call_id": f"call_{n}"}]
            text = ""
        return NormResponse(model=model, response_id=f"fake_{n}", status="completed", items=items,
                            item_dicts=[dict(i) for i in items], text=text,
                            input_tokens=300 + 150 * n, output_tokens=25,
                            latency_s=time.perf_counter() - t0, headers={})


# --------------------------------------------------------------------------- #
# Probe steps
# --------------------------------------------------------------------------- #
class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self._lock = path, threading.Lock()

    def __call__(self, kind: str, model: str, r: NormResponse, **extra):
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "kind": kind, "model": model,
               "returned_model": r.model, "response_id": r.response_id, "status": r.status,
               "latency_s": round(r.latency_s, 3), "input_tokens": r.input_tokens,
               "output_tokens": r.output_tokens, "temperature_used": r.temperature_used,
               "error": r.error, "headers": r.headers, "output_items": r.item_dicts,
               "protocol_fingerprint": meta.protocol_fingerprint(), **extra}
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def est_credits(model: str, in_tok: int, out_tok: int):
    p = POSTED_PRICES.get(model)
    return None if p is None else (in_tok * p[0] + out_tok * p[1]) / 1e6


def hello(provider, model, temperature, log) -> NormResponse:
    r = provider.create(model, "Reply with the single word OK.", [{"role": "user", "content": "ping"}],
                        [], temperature, 256)
    log("hello", model, r)
    return r


def run_episode(provider, model, trial, temperature, max_output_tokens, log) -> dict:
    task = env.make_task(trial.template, trial.instance_id, trial.task_seed)
    ws = env.Workspace(task, inbox_text=trial.inbox_text)
    history = [{"role": "user", "content": task.prompt}]
    tools = flat_tools()
    n_calls = in_tok = out_tok = 0
    lat = 0.0
    err = ""
    for turn in range(env.DEFAULT_MAX_STEPS + 2):
        r = provider.create(model, env.SYSTEM_PROMPT, history, tools, temperature, max_output_tokens)
        log("episode_call", model, r, trial_id=trial.trial_id, turn=turn, condition=trial.condition)
        n_calls += 1
        in_tok += r.input_tokens
        out_tok += r.output_tokens
        lat += r.latency_s
        if r.error or r.status not in ("completed", "incomplete"):
            err = r.error or f"status={r.status}"
            break
        fcs = [i for i in r.items if _get(i, "type") == "function_call"]
        history.extend(r.items)
        if not fcs:
            break
        for it in fcs:
            out = ws.call(_get(it, "name"), _get(it, "arguments"))
            history.append({"type": "function_call_output", "call_id": _get(it, "call_id"), "output": out})
        if ws.done:
            break
    o = scorer.score(task, ws.snapshot())
    return {"trial_id": trial.trial_id, "condition": trial.condition, "calls": n_calls,
            "tool_steps": o.steps, "n_invalid": o.n_invalid, "n_errors": o.n_errors,
            "category": o.category, "completed": o.completed, "adopted": o.adopted,
            "finished": o.finished, "input_tokens": in_tok, "output_tokens": out_tok,
            "latency_s": round(lat, 2), "error": err,
            "inbox_delivered": ws.inbox_delivered_at_step is not None or not trial.inbox_text}


def parallel_probe(provider, model, n, temperature, log) -> dict:
    def one(_):
        r = provider.create(model, "Reply with the single word OK.", [{"role": "user", "content": "ping"}],
                            [], temperature, 256)
        log("parallel", model, r)
        return r
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as ex:
        rs = list(ex.map(one, range(n)))
    wall = time.perf_counter() - t0
    ok = [r for r in rs if not r.error]
    errs = sorted({r.error[:80] for r in rs if r.error})
    lats = [r.latency_s for r in ok] or [0.0]
    return {"requested": n, "ok": len(ok), "errors": errs, "wall_s": round(wall, 2),
            "latency_p50_s": round(statistics.median(lats), 2), "latency_max_s": round(max(lats), 2),
            "rate_limited": any("429" in e or "rate" in e.lower() for e in errs)}


def pilot_trials() -> list:
    """Two PILOT-split trials (condition A and D, same task) so the probe never touches the main split."""
    ts = manifest.build_manifest("pilot", 1, 4, seed=manifest.DEFAULT_SEED)
    pick = [t for t in ts if t.template == "max_amount" and t.replicate == 0 and t.condition in ("A", "D")]
    return sorted(pick, key=lambda t: t.condition)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-output-tokens", type=int, default=2048)
    ap.add_argument("--dry-run", action="store_true", help="offline scripted model; no key, no network")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--out", default="data/probe")
    args = ap.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"{meta.PROJECT_TITLE}\nprobe | protocol fingerprint {meta.protocol_fingerprint()[:16]}...")

    if args.dry_run:
        provider = FakeProvider()
        print("DRY RUN: scripted offline model, nothing is sent anywhere.\n")
    else:
        if not os.environ.get("ADAPTION_API_KEY"):
            print("ERROR: ADAPTION_API_KEY is not set in this terminal. See the setup steps; never paste the key into chat.")
            return 2
        try:
            provider = AdaptionProvider()
        except ImportError:
            print("ERROR: the 'adaption' package is missing. Run:  pip install adaption")
            return 2
        per_model_calls = 1 + 2 * (env.DEFAULT_MAX_STEPS + 2) + args.parallel
        print(f"Plan: {len(models)} model(s), at most ~{per_model_calls} calls each "
              f"(typically ~{1 + 2 * 4 + args.parallel}). Expected spend: well under 1 credit at posted prices.")
        if not args.yes and input("Proceed with these paid calls? [y/N] ").strip().lower() != "y":
            print("Aborted. No calls were made.")
            return 1

    log = Logger(Path(args.out) / f"probe_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.jsonl")
    summary = {"project": meta.PROJECT_TITLE, "provider": provider.name,
               "protocol_fingerprint": meta.protocol_fingerprint(), "models": {}}

    try:
        hosted = provider.list_hosted_models()
        print(f"Hosted models visible to this key ({len(hosted)}): {', '.join(hosted)}")
    except Exception as exc:
        hosted = []
        print(f"WARNING: could not list models: {type(exc).__name__}: {str(exc)[:200]}")
    summary["hosted_models"] = hosted

    trials = pilot_trials()
    for model in models:
        print(f"\n=== {model} " + "=" * max(1, 50 - len(model)))
        if hosted and model not in hosted:
            print("  NOT in your hosted-model list; skipping.")
            summary["models"][model] = {"skipped": "not listed for this key"}
            continue
        m = {}
        h = hello(provider, model, args.temperature, log)
        m["hello"] = {"ok": not h.error, "latency_s": round(h.latency_s, 2), "returned_model": h.model,
                      "temperature_accepted": h.temperature_used, "reply": h.text[:40], "error": h.error,
                      "headers_seen": sorted(h.headers)}
        print(f"  hello: {'OK' if not h.error else 'FAIL ' + h.error} | {h.latency_s:.2f}s | "
              f"in/out tokens {h.input_tokens}/{h.output_tokens} | temperature accepted: {h.temperature_used}")
        if h.headers:
            print(f"  rate-limit headers: {h.headers}")
        eps = []
        if not h.error:
            for t in trials:
                e = run_episode(provider, model, t, args.temperature, args.max_output_tokens, log)
                eps.append(e)
                print(f"  episode cond {e['condition']}: category={e['category']} steps={e['tool_steps']} "
                      f"invalid={e['n_invalid']} calls={e['calls']} tokens={e['input_tokens']}+{e['output_tokens']} "
                      f"{e['latency_s']}s {('ERROR ' + e['error']) if e['error'] else ''}")
            m["parallel"] = parallel_probe(provider, model, args.parallel, args.temperature, log)
            p = m["parallel"]
            print(f"  parallel x{p['requested']}: ok={p['ok']} wall={p['wall_s']}s p50={p['latency_p50_s']}s "
                  f"max={p['latency_max_s']}s rate_limited={p['rate_limited']} {p['errors']}")
        m["episodes"] = eps
        if eps:
            steps = sum(e["tool_steps"] for e in eps)
            m["valid_call_rate"] = round(1 - sum(e["n_invalid"] for e in eps) / max(1, steps), 3)
            tok = statistics.mean(e["input_tokens"] + e["output_tokens"] for e in eps)
            m["mean_tokens_per_trial"] = round(tok)
            inn = statistics.mean(e["input_tokens"] for e in eps)
            outt = statistics.mean(e["output_tokens"] for e in eps)
            c = est_credits(model, inn, outt)
            m["est_credits_per_trial_posted_prices"] = None if c is None else round(c, 5)
            m["mean_episode_latency_s"] = round(statistics.mean(e["latency_s"] for e in eps), 2)
            print(f"  => valid-call rate {m['valid_call_rate']} | ~{m['mean_tokens_per_trial']} tokens/trial | "
                  f"~{m['est_credits_per_trial_posted_prices']} credits/trial | {m['mean_episode_latency_s']}s/trial")
        summary["models"][model] = m

    print("\n" + "-" * 60 + f"\nFull call log: {log.path}\nCopy EVERYTHING between the markers into chat:\n")
    print("=====BEGIN MURMURATION PROBE SUMMARY=====")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print("=====END MURMURATION PROBE SUMMARY=====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
