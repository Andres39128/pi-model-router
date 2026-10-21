# pi-model-router

A hybrid automatic model-selection extension for [Pi](https://github.com/belgiumtech/pi) (the coding agent). It exposes one **virtual model** — `router/auto` — that picks a physical model and an effort level for every request: local heuristics + an optional LLM classifier + hard rules, with cascade fallback when a provider dies.

Runs in production against a 3-tier ladder: a local chat model, a cheap cloud dev model, and a frontier model — plus dedicated escape hatches for vision, a scarce strong solver, privacy, and compaction.

## How it decides (precedence, high to low)

| # | Rule | Trigger | Target |
|---|------|---------|--------|
| R6 | **Privacy** | Secret VALUES in the current turn: assignments (`password=…`), key literals (`sk-…`, PEM, `AKIA…`), credential URLs, secret-named env bulk | local model, fail-closed (visible error if the local model is down) |
| F28 | **Context fit** | Conversation bigger than a model's context window | Skip every model that cannot fit it; overflow retries go to the largest window |
| F15 | **Solver mention** | User names the solver in the turn (`/gpt-6.1-sol/`) | solver model, that turn only |
| F7 | **Explicit level** | `router/auto:xhigh` | solver model |
| R4 | **Vision** | Image blocks in the current turn | vision model |
| R3 | **Escalation** | Retry after connection/quota error | next ladder step, skipping dead duplicates |
| F29 | **Cache hysteresis** | Voluntary tier flip on a big conversation without a strong signal | keep the previous model (`why:"cache"`) |
| R1 | **Tier routing** | Heuristic score / classifier / role floors | ladder tier |

The current-turn text drives intent: compaction summaries, pasted warnings, and history artifacts never tier a request (only real key shapes in history keep a session local).

### Tier semantics (roles, not scores)

| Tier | Role | Who lands here |
|------|------|----------------|
| `local` | chat & general topics only | Turns with smalltalk markers (greetings, thanks, opinions); explicit `:low` |
| `flash` | development without high precision | The dev floor: mechanical git/exec chores, simple dev questions, formatting/renames |
| `frontier` | hard development | RISK vocabulary (deploy, auth, migrations, destructive ops), complex architecture work |

## Install

```bash
mkdir -p ~/.pi/agent/extensions/model-router
cp index.ts ~/.pi/agent/extensions/model-router/
```

Then set the virtual model as your default in `~/.pi/agent/settings.json`:

```json
{ "defaultModel": "router/auto" }
```

Pi resolves `router/auto` through this extension's `resolveModel`. Pick a specific effort with a virtual level: `router/auto:low | :medium | :high | :xhigh` (also settable per message via `/model`).

## Configuration — `~/.pi/router-poc/router.json`

Loaded from the directory the extension was developed in; every block is optional except `ladder`.

```json
{
  "ladder": [
    { "provider": "my-local",          "id": "chat-model",       "tier": "local"   },
    { "provider": "my-cloud",          "id": "dev-flash-model",  "tier": "flash"   },
    { "provider": "my-cloud",          "id": "frontier-model",   "tier": "frontier" }
  ],
  "classifier": { "mode": "auto", "url": "http://127.0.0.1:8090", "timeoutMs": 1500, "complexFloor": 0.4 },
  "vision":      { "provider": "my-cloud", "id": "vision-model" },
  "solver":      { "provider": "my-cloud", "id": "strong-model" },
  "privacy":     { "provider": "my-local", "id": "chat-model" },
  "compaction":  { "provider": "my-cloud", "id": "long-ctx-model" },
  "cacheCareTokens": 24576
}
```

- **ladder** — the three tiers. One model may serve two tiers. Escalation wraps around; entries whose model equals the failed one are skipped.
- **classifier** — `mode: "auto"` enables a tiny HTTP classifier (`simple|standard|complex` + probabilities). `complexFloor` promotes borderline-simple answers (0.4 = confident-complex bumps one tier up, both directions). `"off"` disables it; heuristics decide.
- **vision / solver / privacy / compaction** — special-route targets. All are skipped (fall through to the ladder) if missing from the catalog or if their context window cannot fit the request.
- **cacheCareTokens** — conversations this big make voluntary model switches need a strong signal (risk rule, explicit level, or classifier complex-prob ≥ 0.75); weak flips stay on the current model to preserve the prompt cache. Default 24576.

The physical models must exist in Pi's catalog (`~/.pi/agent/models.json` or a provider config) — the router only picks among models Pi can already serve.

## Usage

Just talk to Pi. The footer shows every decision:

```
glm-5.3-flash:high [laya:simple]     ← classifier answer
glm-5.3:max [risk]                   ← RISK hard rule
Qwen3.8-27B:high [privacy]           ← secrets in the turn
glm-5.3:max [cache]                  ← switch held by hysteresis
deepseek-v4.1-flash:high [chat]      ← smalltalk on the chat tier
```

Escape hatches you control directly:

- `/model <physical-model>` bypasses the router entirely.
- `/model router/auto:xhigh` forces the solver for hard tasks.
- Naming the solver in a message (`"continue with gpt-6.1-sol"`) routes that turn to it.

Every decision is appended to `decisions.jsonl` next to the extension source (model, tier, reason, why, privacy/latency fields) for calibration.

## Behavior notes

- **Effort levels**: tiers dispatch `local/flash → high`, `frontier → max`, solver → `xhigh`. Add a `thinkingLevelMap` identity entry to models that support more than the default ladder so Pi does not clamp levels.
- **Fail-closed privacy**: secrets + dead local model = visible routing failure, never a silent cloud fallback.
- **Overflow recovery**: `exceed_context_size_error` retries route to the largest fitting window; if none fits, the router returns null (the session must compact).
- **Compaction**: `direct` compaction requests route to the configured `compaction` model — unless the conversation carries secrets, then local.

## Tests & calibration

```bash
npm install        # @types/node only
npm test           # 163 tests, Node ≥ 22 (native TS)
```

`calibrate.ts` replays a `decisions.jsonl` against the current rules and reports tier-distribution shifts per `why` — run it after changing thresholds.

```bash
node calibrate.ts            # summary
node calibrate.ts --detail   # per-rule deltas
```

## License

MIT
