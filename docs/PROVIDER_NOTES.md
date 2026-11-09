# Provider and credit notes (MURMURATION)

Status as of Mon Nov 9, 2026. **Nothing here has been redeemed, purchased or called.** Items are tagged
`VERIFIED` (read from the cited page), `UNVERIFIED` (could not confirm), or `ASK USER` (only the user can supply).

## 1. The credits

| Question | Answer | Tag |
|---|---|---|
| Issuer | Adaption (adaptionlabs.ai), a hackathon partner | VERIFIED (sprint page) |
| Amount | $500 in Adaption Platform credits + 3 months Premium access, for all participants | VERIFIED (sprint page) |
| Winning teams | additional $1,000 in Adaption credits | VERIFIED (sprint page) |
| Redemption | sign up at adaptionlabs.ai/app with the **same email used to register for the Sprint**; an onboarding email was promised before the hackathon | VERIFIED (sprint page) |
| Is *this user's* credit really this one? | Plausible but not confirmed; the user must show the email/dashboard | ASK USER |
| Expiry | not stated | UNVERIFIED |
| Eligibility / usage restrictions beyond the above | not stated | UNVERIFIED |
| Value of one "credit" in USD | not stated in docs I could read. Do not assume $1 | UNVERIFIED |
| Do platform credits pay for the Responses (inference) API? | Docs price inference **in credits**, which suggests yes, but the sprint page describes credits for AutoScientist / Adaptive Data / Invent-a-Dataset | UNVERIFIED, test with 1 call |
| Rate limits / spend limits on the Responses API | not published in the pages read | UNVERIFIED, read headers on a real call |
| Combine with Groq? | They are separate accounts; no combining. A mixed-provider design must not silently change conditions | n/a |

## 2. What Adaption offers that matters for MURMURATION (docs.adaptionlabs.ai)

* A **Responses API** (`client.responses.create`, Python SDK `pip install adaption`, key in `ADAPTION_API_KEY`).
* **Function tools** are supported: the response ends with a `function_call` item; the caller runs it and returns
  a `function_call_output` item. This maps directly onto the `env.Workspace.call` loop.
* Responses are **not stored by default** (`store=True` opt-in); stateless history is passed explicitly.
* Hosted models (credits per 1M tokens; context):

| Model ID | Input (cached) | Output | Context |
|---|---:|---:|---:|
| `gemma-4-31b-it` | 0.09 (0.05) | 0.34 | 256K |
| `gemma-4-26b-a4b-it` | 0.0765 (0.0425) | 0.255 | 256K |
| `qwen3.5-397b` | 0.55 (0.225) | 3.50 | 131K |
| `qwen3.8-27b` | 0.039 (0.0039) | 1.25 | 262K |
| `glm-5.2` | 1.40 (0.26) | 4.40 | 786K |
| `glm-53-flash` | 0.15 (0.03) | 0.50 | 1M |
| `deepseek-v4-flash` | 0.04444 (0.001414) | 0.13332 | 1M |

  Docs note: availability can vary by API key; list with `client.models.list(type="hosted")`.
  Docs also note the hosted deployments' limits may differ from the upstream model cards.
* **Not documented:** per-model tool-calling reliability, determinism/seed support, `temperature` handling,
  version pinning, and rate limits. All must be measured in the pilot.

## 3. Groq (existing option)

* Free tier is far too small for this workload (published per-model daily token caps of roughly 100K–500K tokens).
* Paid (Developer) tier is pay-as-you-go; limits on your account must be read from the console and response headers.
* Reported prices (aggregators, 2026): Llama 3.3 70B about $0.59/$0.79 per 1M tokens in/out; gpt-oss-20b about $0.075/$0.30.
* Billing requires a payment method and **explicit user approval**. Not done.

## 4. Rules that protect validity

1. One provider and one model = one confirmatory arm. A provider swap = a new arm with its own pilot, never a silent substitute.
2. Every log record carries provider, model ID, version if exposed, temperature, seed, trial ID,
   protocol fingerprint (`meta.py`), token counts, latency, cost/credits consumed.
3. No spend beyond granted credits without explicit approval; redeem only after eligibility is confirmed.
4. Pilot data never enters confirmatory analysis.

## 5. Open items for the user

* Forward or paste the onboarding email / screenshot of the credits page (redact the API key).
* Confirm what the dashboard shows as the balance, its unit, and any expiry.
