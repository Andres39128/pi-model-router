/**
 * Router — F4 (hybrid decision: heuristics + optional local classifier).
 *
 * Registers the virtual model `router/auto`. For every request it picks the
 * physical model from a whitelist ladder using three rules:
 *
 *   R1 Tier routing — the virtual level is an override and the text is a
 *      signal: `router/auto:low` forces the local tier, `:high` forces the
 *      frontier tier, and `:medium` (the default) lets the heuristics
 *      (risk hard-rule + complexity score) choose.
 *   R2 Sticky — continuations and non-connection retries keep the model that
 *      handled the turn (prompt cache + thinking signatures).
 *   R3 Escalation — a retry whose `failed.message.errorMessage` looks like a
 *      connection/provider failure walks one step up the ladder (wrapping),
 *      so a dead local server lands on the cloud flash model.
 *   R4 Modality (F6) — fresh user requests carrying image blocks route to
 *      the configured vision model (default minimax/MiniMax-M3): images
 *      outrank tier heuristics because the chat ladder takes no image input.
 *   R5 Solver (F7) — the explicit `router/auto:xhigh` level routes to the
 *      configured scarce strong model (default openai/gpt-6.1-sol), above
 *      the vision rule; fallback is the frontier tier. Never automatic:
 *      only the user's explicit level spends that budget.
 *   R6 Privacy (F11) — turns carrying secret VALUES (assignments like
 *      `api_key=...`, key/token literals, credential URLs, bulk env-file
 *      content) route to the configured local model (default
 *      llamacpp/Qwen3.8-27B). Highest precedence: above xhigh, vision, the
 *      risk rule and the classifier; retries never escalate a secrets turn
 *      to the cloud. Credential TOPICS without values are NOT matched.
 *   F8 — every route() call publishes the selected physical model to the
 *      footer/status bar via ctx.ui.setStatus (e.g. "glm-5.3:high [laya:complex]").
 *
 * F2 additions:
 *   - The ladder lives in ~/.pi/router-poc/router.json (validated; falls
 *     back to the built-in default when missing or invalid).
 *   - A cold/empty model catalog is retried a few times before erroring
 *     (the 11:16 UNRESOLVED incident: catalog empty at route() time).
 *   - Calibration hooks: the prediction is logged for EVERY user request,
 *     including when the virtual level overrides it (forcedMismatch).
 *   - Hybrid decision (F4): fresh user requests in the ambiguous heuristic
 *     band (score 1..4) ask a local decision model (Laya via llama.cpp
 *     /v1/systemone, HTTP) simple|standard|complex -> local|flash|frontier.
 *     Fast paths (risk hard-rule, level override, confident scores) and every
 *     classifier failure mode fall back to the pure heuristics.
 *
 * F24 additions (vision-intent):
 *   - When the user pastes a path or writes "imagen" / "screenshot" but pi's
 *     TUI fails to attach the image block, route to the vision model (R24)
 *     so the attachment does not fall on a text-only frontier that rejects it.
 *
 * F23 additions (signal hygiene):
 *   - Harness-injected messages (permission banners, gentle-jobs notices,
 *     pi-warden steers, gentle-agents wake calls) are NEVER user intent:
 *     they are dropped from userTextSignal and currentTurnText before any
 *     heuristic sees them. Evidence: a job notice quoting
 *     "-DCMAKE_BUILD_TYPE=Release" matched the RISK regex and locked a whole
 *     session to frontier (frontier creep 15%->71% in three days).
 *   - Privacy (R6) still scans the unfiltered authored+compaction signal —
 *     over-matching secrets is the fail-safe direction.
 *   - Classifier probabilities ride on the decision and the log, so the F9
 *     complexFloor becomes measurable; solver/direct paths fill their
 *     prediction fields (no more "?" predictedWhy rows).
 *
 * Master copy: ~/.pi/router-poc/index.ts
 * Installed at:  ~/.pi/agent/extensions/model-router/index.ts (global discovery)
 * Log:           ~/.pi/router-poc/decisions.jsonl
 * Tests:         node --test decide.test.ts
 */
import { appendFileSync, mkdirSync, readFileSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const POC_DIR = join(homedir(), ".pi", "router-poc");
const LOG_PATH = join(POC_DIR, "decisions.jsonl");
const CONFIG_PATH = join(POC_DIR, "router.json");

// ---------------------------------------------------------------------------
// Pure routing core (unit-tested in decide.test.ts)
// ---------------------------------------------------------------------------

/** Whitelist ladder, cheapest first. Also the fallback chain per tier.
 * F10: Qwen3.8-27B (local server) retired — cheap work goes to the cloud
 * flash model; the local and flash tiers share it. */
export const LADDER: ReadonlyArray<{ provider: string; id: string }> = [
	{ provider: "zhipuai-coding-plan", id: "glm-5.3-flash" },
	{ provider: "zhipuai-coding-plan", id: "glm-5.3-flash" },
	{ provider: "zhipuai-coding-plan", id: "glm-5.3" },
];

type Tier = "local" | "flash" | "frontier";

/** Ladder index each tier starts routing at. */
const TIER_START: Record<Tier, number> = { local: 0, flash: 1, frontier: 2 };

/** Resolved routing configuration: ladder order plus tier starting indexes. */
export interface ClassifierConfig {
	mode: "auto" | "off";
	/** Base URL of a decision-model server (llama.cpp /v1/systemone). */
	url?: string;
	/** Timeout for the classifier call in milliseconds. */
	timeoutMs?: number;
	/** F9: complex-probability floor. Argmax "simple" with complex-prob at or
	 * above this floor is promoted to the flash tier (0 disables, 0.4 default). */
	complexFloor?: number;
	/** F29: conversations with at least this many tokens make a prompt-cache
	 * refill at the destination a real cost — weak voluntary switches are
	 * held. Default 24_576. */
	cacheCareTokens?: number;
}

export interface RouterConfig {
	ladder: ReadonlyArray<{ provider: string; id: string }>;
	tierStart: Record<Tier, number>;
	/** F4: optional decision-model refinement of the ambiguous band. */
	classifier?: ClassifierConfig;
	/** F6/R4: preferred model when the turn carries images. */
	vision?: { provider: string; id: string };
	/** F7: scarce strong model for the explicit xhigh level. */
	solver?: { provider: string; id: string };
	/** F11/R6: model that keeps turns carrying secret values on this machine. */
	privacy?: { provider: string; id: string };
	/** F17: model that writes session compaction summaries (1M ctx helps). */
	compaction?: { provider: string; id: string };
}

/** Built-in ladder, used when router.json is missing or invalid. */
export const DEFAULT_CONFIG: RouterConfig = {
	ladder: LADDER,
	tierStart: TIER_START,
	classifier: { mode: "auto", url: "http://127.0.0.1:8090", timeoutMs: 1500, complexFloor: 0.4 },
	vision: { provider: "minimax", id: "MiniMax-M3" },
	solver: { provider: "openai", id: "gpt-6.1-sol" },
	privacy: { provider: "llamacpp", id: "Qwen3.8-27B" },
	compaction: { provider: "minimax", id: "MiniMax-M3" },
};

/** Parse and validate a router.json payload. Null means "keep the default". */
export function parseConfig(raw: unknown): RouterConfig | null {
	if (typeof raw !== "object" || raw === null) return null;
	const ladder = (raw as { ladder?: unknown }).ladder;
	if (!Array.isArray(ladder) || ladder.length === 0) return null;
	const tiers: readonly Tier[] = ["local", "flash", "frontier"];
	const seen = new Set<string>();
	const entries: Array<{ provider: string; id: string }> = [];
	const tierStart = {} as Record<Tier, number>;
	for (const item of ladder) {
		if (typeof item !== "object" || item === null) return null;
		const { provider, id, tier } = item as { provider?: unknown; id?: unknown; tier?: unknown };
		if (typeof provider !== "string" || provider === "") return null;
		if (typeof id !== "string" || id === "") return null;
		if (typeof tier !== "string" || !tiers.includes(tier as Tier)) return null;
		const key = `${provider}/${id}/${tier}`;
		if (seen.has(key)) return null;
		seen.add(key);
		const t = tier as Tier;
		if (tierStart[t] === undefined) tierStart[t] = entries.length;
		entries.push({ provider, id });
	}
	for (const t of tiers) if (tierStart[t] === undefined) return null;
	const rawClassifier = (raw as { classifier?: unknown }).classifier;
	let classifier: ClassifierConfig | undefined;
	let cacheCareTokens: number | undefined;
	if (rawClassifier !== undefined) {
		if (typeof rawClassifier !== "object" || rawClassifier === null) return null;
		const c = rawClassifier as { mode?: unknown; url?: unknown; timeoutMs?: unknown };
		if (c.mode !== "auto" && c.mode !== "off") return null;
		if (c.url !== undefined && typeof c.url !== "string") return null;
		if (c.timeoutMs !== undefined && (typeof c.timeoutMs !== "number" || c.timeoutMs <= 0)) return null;
		if (c.complexFloor !== undefined && (typeof c.complexFloor !== "number" || c.complexFloor < 0 || c.complexFloor > 1)) return null;
		classifier = { mode: c.mode, url: c.url, timeoutMs: c.timeoutMs, complexFloor: c.complexFloor };
	}
	if (raw.cacheCareTokens !== undefined) {
		if (typeof raw.cacheCareTokens !== "number" || raw.cacheCareTokens < 0) return null;
		cacheCareTokens = raw.cacheCareTokens;
	}
	const rawVision = (raw as { vision?: unknown }).vision;
	let vision: { provider: string; id: string } | undefined;
	if (rawVision !== undefined) {
		if (typeof rawVision !== "object" || rawVision === null) return null;
		const v = rawVision as { provider?: unknown; id?: unknown };
		if (typeof v.provider !== "string" || v.provider === "") return null;
		if (typeof v.id !== "string" || v.id === "") return null;
		vision = { provider: v.provider, id: v.id };
	}
	const rawSolver = (raw as { solver?: unknown }).solver;
	let solver: { provider: string; id: string } | undefined;
	if (rawSolver !== undefined) {
		if (typeof rawSolver !== "object" || rawSolver === null) return null;
		const s = rawSolver as { provider?: unknown; id?: unknown };
		if (typeof s.provider !== "string" || s.provider === "") return null;
		if (typeof s.id !== "string" || s.id === "") return null;
		solver = { provider: s.provider, id: s.id };
	}
	const rawPrivacy = (raw as { privacy?: unknown }).privacy;
	let privacy: { provider: string; id: string } | undefined;
	if (rawPrivacy !== undefined) {
		if (typeof rawPrivacy !== "object" || rawPrivacy === null) return null;
		const pv = rawPrivacy as { provider?: unknown; id?: unknown };
		if (typeof pv.provider !== "string" || pv.provider === "") return null;
		if (typeof pv.id !== "string" || pv.id === "") return null;
		privacy = { provider: pv.provider, id: pv.id };
	}
	const rawCompaction = (raw as { compaction?: unknown }).compaction;
	let compaction: { provider: string; id: string } | undefined;
	if (rawCompaction !== undefined) {
		if (typeof rawCompaction !== "object" || rawCompaction === null) return null;
		const cp = rawCompaction as { provider?: unknown; id?: unknown };
		if (typeof cp.provider !== "string" || cp.provider === "") return null;
		if (typeof cp.id !== "string" || cp.id === "") return null;
		compaction = { provider: cp.provider, id: cp.id };
	}
	return { ladder: entries, tierStart, classifier, vision, solver, privacy, compaction, cacheCareTokens };
}
/** F25 — inline level hint: the user types ":xhigh" / ":max" / ":high" /
 * ":low" in the message itself. The picker level (req.thinkingLevel) is the
 * real selector; this only fills in when the picker sits on the default
 * medium, so an explicit picker choice always wins. */
export function levelHint(turnText: string): "low" | "medium" | "high" | "xhigh" | "max" | null {
	const m = turnText.match(/(?:^|\s):(low|medium|high|xhigh|max)\b/i);
	return m ? (m[1].toLowerCase() as "low" | "medium" | "high" | "xhigh" | "max") : null;
}

/** F25 — effective level for one request: an explicit picker level wins, the
 * inline hint fills in when the picker is untouched (default medium). */
export function resolveLevel(input: { virtualLevel?: string; turnText?: string; text: string }): string {
	const picked = input.virtualLevel;
	if (picked && picked !== "medium") return picked;
	return levelHint(input.turnText ?? input.text) ?? picked ?? "medium";
}

/** Effort each tier dispatches with (Pi clamps to model support). F20:
 * the ladder runs high/max only — flash tokens are nearly free on the plan
 * and frontier gets glm-5.3's true maximum effort. */
const TIER_LEVEL: Record<Tier, "high" | "max"> = { local: "high", flash: "high", frontier: "max" };

const RISK =
	/\b(auth|password|contrase\w*|credential\w*|credencial\w*|secret\w*|migrat\w*|migraci\w*|(drop|delet\w*|eliminar|borrar|truncar|truncate)\w*\s+((el|la|los|las|the)\s+)?(table|tabla|data|datos|rows?|filas?)|public\s+api|breaking|release|deploy\w*|despleg\w*)\b/i;
const COMPLEX =
	/\b(refactor|architect\w*|arquitect\w*|redesign|redise\w+|redisen\w+|debug\w*|depura\w*|race\s+condition|condici\w*\s+de\s+carrera|deadlock|interbloqueo|optimi[sz]\w*|root\s+cause|causa\s+ra[ií]z|investigat\w*|investiga\w*)\b/i;
const TRIVIAL =
	/\b(format|formato|renam\w+|renombr\w+|typo|error\s+de\s+tipeo|boilerplate|lint|docstring|comentario|comment)\b/i;

/** F11/R6 — secret VALUES (not topics): an assignment of a secret-looking
 * keyword, a literal key/token shape, or bulk env-file content. F21: quoted
 * JSON properties count; bare type-name values (TS interfaces) do not. */
const SENSITIVE_ASSIGNMENT =
	/((?:api[_-]?key\w*|apikey\w*|secret\w*|password\w*|passwd|contrase\w*|credential\w*|credencial\w*|private[_-]?key\w*|token)["']?[ \t]*[:=][ \t]*["'`]?)([^\s"'`]{4,})/gi;
const TYPE_VALUE = /^(string|str|number|num|boolean|bool|any|unknown|never|void|object|function)(\s*;)?$/i;
const SENSITIVE_LITERAL =
	/(\b(sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,})|-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(postgres(ql)?|mysql|mongodb(\+srv)?|redis|amqp):\/\/[^\s"']*:[^\s"'@]*@)/;

/** True when the text carries secret values that must not leave the machine. */
export function hasSensitiveData(text: string): boolean {
	return hasSensitiveDataWhy(text) !== null;
}

/** F18 — the router's own test fixtures are fake by construction (constant
 * literals in decide.test.ts). Verifying this code must not trip privacy. */
const KNOWN_FAKES: ReadonlyArray<{ re: RegExp }> = [
	{ re: /sk-abcd1234efgh5678\w*/g },
	{ re: /sk-1234567890abcdefgh/g },
	{ re: /sk-test-1234567890abcd/g },
	{ re: /wJalrXUtnFEMIdeadbeef1234/g },
	{ re: /\bhunter2\b/g },
	{ re: /postgres:\/\/admin:s3creto@[^\s"']*/g },
];

/** Which privacy rule matched: "literal" | "assignment" | "env-bulk" | null.
 * Known fixture strings are stripped before matching and never count. */
export function hasSensitiveDataWhy(text: string): "literal" | "assignment" | "env-bulk" | null {
	let t = text;
	for (const { re } of KNOWN_FAKES) t = t.replace(re, "");
	if (SENSITIVE_LITERAL.test(t)) return "literal";
	for (const m of t.matchAll(SENSITIVE_ASSIGNMENT)) {
		if (!TYPE_VALUE.test(m[2] ?? "")) return "assignment";
	}
	// F19 — bulk counts only SECRET-NAMED variables: benign config listings
	// in summaries (PI_MODEL=..., flags) document the environment; a real
	// pasted .env carries PASSWORD/SECRET/TOKEN/KEY/ cred-named keys.
	// F21: the keyword must end at "_" or "=" — KEY inside KEYBOARD is a name,
	// not a secret marker.
	const envBulk = t.match(/^[ \t]*[A-Z0-9_]*(?:PASSWORD|SECRET|TOKEN|KEY|CRED\w*)(?=_|[ \t]*=)[A-Z0-9_]*[ \t]*=[ \t]*\S+$/gm) ?? [];
	return envBulk.length >= 2 ? "env-bulk" : null;
}

/** F15 — the user names the scarce solver: flexible separators. */
const SOLVER_MENTION = /\bgpt[.\-\s]*6[.,]1[.\-\s]*sol\b/i;

/** F16 — pi injects the compaction summary as its own user message; it is
 * system text, never user intent. Exported: calibrate.ts separates compaction
 * turns from user predictions with the same marker. */
export const COMPACT_MARK = "The conversation history before this point was compacted";

/** F23 — messages the harness injects on its own behalf. They ride the
 * conversation as user-role text, but no word in them is user intent, so no
 * heuristic (risk, complexity, solver mention, image reference) may see
 * them. Shapes observed in real sessions (2026-10-19/20 forensics):
 * pi-permission-suite-context, pi-warden-steer, gentle-jobs.notice,
 * gentle-agents.wake. Anchored at the head; matched per message. */
const HARNESS_NOISE: ReadonlyArray<RegExp> = [
	/^\[审批/, // permission banner (zh locale)
	/^\[Approval/i, // permission banner (en locale)
	/^Background job /, // gentle-jobs exit notice
	/^pi-warden:/i, // warden steer / injection notice
	/^Review the delivered subagent output/, // gentle-agents wake call
];

/** F24 — file extensions the user is likely to want a vision model to see.
 * PDF/SVG are not "imagen" in chat (they go to text/document flows). */
const IMAGE_EXT_RE = /\.(png|jpg|jpeg|gif|webp|heic|heif)(?=[^A-Za-z0-9]|$)/i;

/** F23 — true when a message's text is harness-injected, not user-authored. */
export function isHarnessNoise(text: string): boolean {
	const head = text.slice(0, 80);
	return HARNESS_NOISE.some((re) => re.test(head));
}

/** F23 — text carried by one message (string content or text blocks joined).
 * Shared by userTextSignal, currentTurnText and referencesSessionImage so the
 * extraction lives once. */
function messageText(m: { content?: unknown }): string {
	if (typeof m.content === "string") return m.content;
	const blocks = m.content as Array<{ type?: string; text?: string }> | undefined;
	if (!Array.isArray(blocks)) return "";
	return blocks.map((b) => (b.type === "text" ? b.text ?? "" : "")).join(" ");
}

/** Text of the CURRENT turn: the user messages after the last assistant
 * message, joined (approval banners belong to the turn). F16: messages that
 * are compaction summaries are DROPPED — a summary mentioning a model name
 * is not the user asking for it. */
export function currentTurnText(messages: readonly unknown[]): string {
	const rows = messages as Array<{ role?: string; content?: unknown }>;
	const parts: string[] = [];
	for (let i = rows.length - 1; i >= 0 && rows[i]?.role !== "assistant"; i--) {
		const m = rows[i];
		if (m?.role !== "user") continue;
		const text = messageText(m);
		// F23: compaction summaries and harness noise are not user intent.
		if (!text || text.startsWith(COMPACT_MARK) || isHarnessNoise(text)) continue;
		parts.unshift(text);
	}
	return parts.join("\n");
}

/** F28 — a context-overflow failure: the model is too small for the
 * conversation, not an outage of the provider. */
const CONTEXT_OVERFLOW =
	/exceed_context_size_error|exceeds? the available context|context (?:length|size) exceeded|maximum context|n_prompt_tokens/i;

/** Errors that mean "this provider/model is unavailable", not "this request
 * is bad". F12: quota exhaustion counts — rate limits and balance errors
 * are provider-level outages, so retries must escalate, not stick. */
const CONNECTION_ERROR =
	/econnrefused|econnreset|etimedout|eai_again|timed?\s*out|timeout|fetch\s*failed|network|connection|502|503|504|overloaded|rate[\s_-]?limit|rate_limit_error|quota|usage\s+limit|insufficient/i;

export interface RouteInput {
	reason: "user" | "continuation" | "retry" | "direct";
	/** Virtual level the user selected for `router/auto`. */
	virtualLevel?: string;
	/** Last user text, used by the heuristics when the level is medium. */
	text: string;
	/** Physical model of the latest successful response (continuation). */
	previous?: { provider: string; id: string; thinkingLevel?: string } | null;
	/** Failed attempt (retry): model plus the assistant errorMessage. */
	failed?: { provider: string; id: string; thinkingLevel?: string; errorMessage?: string } | null;
	/** F6/R4: the user turn carries image content blocks. */
	hasImages?: boolean;
	/** F28: estimated prompt size of the whole conversation, in tokens —
	 * models whose context window cannot fit it are skipped. */
	estimatedTokens?: number;
	/** F15: text of the CURRENT turn only (user messages after the last
	 * assistant message) — model-name mentions apply per request, not per
	 * session. Falls back to the full signal when the caller omits it. */
	turnText?: string;
}

/** Minimal catalog-lookup result the router relies on (truthy = available).
 * contextWindow feeds F28 context-aware routing when the registry exposes it. */
export interface FoundModel {
	provider: string;
	id: string;
	contextWindow?: number;
}

export interface RegistryLike {
	find(provider: string, id: string): FoundModel | undefined;
}

export interface RouteDecision {
	provider: string;
	id: string;
	thinkingLevel: "low" | "medium" | "high" | "xhigh" | "max";
	tier: Tier;
	sticky: boolean;
	escalated: boolean;
	predictedTier?: Tier;
	predictedScore?: number;
	predictedWhy?: string;
	/** True when the virtual level forced a tier the heuristics disagreed with. */
	forcedMismatch?: boolean;
	/** F4: the classifier label that refined an ambiguous heuristic band. */
	classifierAnswer?: "simple" | "standard" | "complex";
	/** F24: vision routed by text/path intent (no image block). */
	visionIntent?: "path-or-text";
	/** F24: the image-bearing path detected by visionPathname, if any. */
	visionPath?: string;
	/** F23/F9: the classifier's probability view, logged for calibration. */
	classifierProbabilities?: Record<string, number>;
	/** Where the classifier answer came from (for the log). */
	classifierSource?: string;
	/** F6/R4: true when the decision went to the configured vision model. */
	visionRouted?: boolean;
	/** F11/R6: true when secret values kept the turn on the local model. */
	privacyRouted?: boolean;
	/** F18: which privacy rule matched ("literal" | "assignment" | "env-bulk"). */
	privacyWhy?: "literal" | "assignment" | "env-bulk";
	/** F15: true when the user named the solver in the request text. */
	solverRouted?: boolean;
	/** F9: true when the complex floor promoted a "simple" argmax to flash. */
	classifierPromoted?: boolean;
	/** F22/F29: routing rationale ("mechanical", "cache") for logs and footer. */
	why?: string;
}

/** F22 — mechanical git/exec operations: terminal chores, not analysis. */
const MECHANICAL =
	/\b(commit\w*|pushe\w*|push|pull|merge\w*|branch\w*|tagge\w*|checkout|stash\w*|rebase|revert\w*|ejecut\w*|corr[eé]\w*)\b/i;

/** F30 — smalltalk markers. Pi is a coding agent: the base rate of a turn is
 * development, so the local tier (the chat model) only takes turns that look
 * like conversation — greetings, thanks, opinions, general topics. Everything
 * else in the low band rides the flash dev floor. */
const CHAT =
	/\b(hola|buenas|hey|chau|adiós|adios|bye|gracias|thank\w*|dale|perfecto|genial|excelente|jaja+|jeje+|opini\w*|recomend\w*|contame|cu[eé]ntame|clima|pel[ií]cula|m[uú]sica|f[uú]tbol|futbol|cine|comida|caf[eé]|chiste|broma|c[oó]mo and[aá]s|qu[eé] tal|que tal|buen d[ií]a|buenas noches|descans[aá])\b/i;

/** F0 heuristic, kept as the medium-level tiebreaker. F22: the caller passes
 * the CURRENT TURN's text (falling back to the full signal) — history
 * artifacts (compaction summaries, pasted warnings) no longer tier the
 * request; mechanical short turns without complex vocabulary stay cheap. */
function predict(text: string): { tier: Tier; score: number; why: string } {
	if (RISK.test(text)) return { tier: "frontier", score: 99, why: "hard-rule:risk" };
	if (text.length < 400 && MECHANICAL.test(text) && !COMPLEX.test(text)) {
		return { tier: "flash", score: 0, why: "mechanical" };
	}
	const len = text.length;
	const files = (text.match(/[\w./-]+\.[a-z]{1,4}/gi) ?? []).length;
	let score = 0;
	if (len > 1500) score += 2;
	if (len > 6000) score += 2;
	if (files >= 3) score += 1;
	if (files >= 8) score += 1;
	if (COMPLEX.test(text)) score += 3;
	if (TRIVIAL.test(text) && len < 400) score -= 3;
	if (score >= 5) return { tier: "frontier", score, why: "score" };
	if (score >= 2) return { tier: "flash", score, why: "score" };
	// F30 — the low band splits by INTENT, not by score: chat turns go to the
	// local chat model, development turns ride the flash dev floor.
	return CHAT.test(text)
		? { tier: "local", score, why: "chat" }
		: { tier: "flash", score, why: "dev-floor" };
}

/** F28 — a special-route target must exist AND fit the request's context. */
function fitsTarget(registry: RegistryLike, t: { provider: string; id: string }, need: number): boolean {
	try {
		const m = registry.find(t.provider, t.id);
		return !!m && (m.contextWindow === undefined || m.contextWindow >= need);
	} catch {
		return false;
	}
}

/** First ladder model from `start` (wrapping) that the registry knows.
 * `skip` (R3) walks past entries carrying the same dead model, so a ladder
 * with one model under two tiers does not retry the failed provider.
 * `need` (F28) skips models whose context window cannot fit the request. */
function pick(
	config: RouterConfig,
	start: number,
	registry: RegistryLike,
	skip?: { provider: string; id: string },
	need = 0,
): { provider: string; id: string; index: number } | null {
	for (let step = 0; step < config.ladder.length; step++) {
		const index = (start + step) % config.ladder.length;
		const entry = config.ladder[index];
		if (skip && entry.provider === skip.provider && entry.id === skip.id) continue;
		try {
			const m = registry.find(entry.provider, entry.id);
			if (m && (m.contextWindow === undefined || m.contextWindow >= need)) return { ...entry, index };
		} catch {
			/* registry refused this entry: try the next one */
		}
	}
	return null;
}

/** Ladder position of a (provider, id) pair, or -1 when it is not whitelisted. */
function ladderIndex(config: RouterConfig, provider: string, id: string): number {
	return config.ladder.findIndex((m) => m.provider === provider && m.id === id);
}

/** Tier whose routing starts at the given ladder index, or "local" when unknown. */
function tierForIndex(config: RouterConfig, index: number): Tier {
	return ((Object.keys(config.tierStart) as Tier[]).find((t) => config.tierStart[t] === index)) ?? "local";
}

/** The routing decision, pure over (input, registry, config). */
export function decide(
	input: RouteInput,
	registry: RegistryLike,
	config: RouterConfig = DEFAULT_CONFIG,
): RouteDecision | null {
	const failed = input.failed ?? null;
	const previous = input.previous ?? null;
	const level = resolveLevel(input);

	// R3 — connection-error retries escalate one ladder step (wrapping).
	if (input.reason === "retry" && failed && failed.errorMessage && CONNECTION_ERROR.test(failed.errorMessage)) {
		const from = ladderIndex(config, failed.provider, failed.id);
		const base = from >= 0 ? from : config.tierStart[predict(input.turnText ?? input.text).tier];
		const model = pick(config, base + 1, registry, { provider: failed.provider, id: failed.id }, input.estimatedTokens ?? 0);
		if (!model) return null;
		const tier = tierForIndex(config, model.index);
		return {
			provider: model.provider,
			id: model.id,
			thinkingLevel: TIER_LEVEL[tier],
			tier,
			sticky: false,
			escalated: true,
		};
	}

	// R2 — sticky: tool follow-ups and non-connection retries keep the model.
	const stickyModel = input.reason === "continuation" ? previous : input.reason === "retry" ? failed : null;
	if (stickyModel) {
		const stickyIndex = ladderIndex(config, stickyModel.provider, stickyModel.id);
		return {
			provider: stickyModel.provider,
			id: stickyModel.id,
			thinkingLevel: (stickyModel.thinkingLevel as "low" | "medium" | "high" | "xhigh" | "max" | undefined) ?? "medium",
			tier: stickyIndex >= 0 ? tierForIndex(config, stickyIndex) : "local",
			sticky: true,
			escalated: false,
		};
	}

	// R1 — tier routing: virtual level overrides, medium defers to heuristics.
	// The prediction runs for EVERY user request (F2): the log must calibrate
	// the bands even when the virtual level forces a different tier.
	const p = predict(input.turnText ?? input.text);
	const forced = level === "high" || level === "xhigh" || level === "max" ? "frontier" : level === "low" ? "local" : undefined;
	// F7 — xhigh: the scarce strong model, explicit opt-in only. Falls back
	// to the frontier tier when the solver is not in the catalog.
	if (level === "xhigh" && config.solver) {
		try {
			if (registry.find(config.solver.provider, config.solver.id)) {
				return {
					provider: config.solver.provider,
					id: config.solver.id,
					thinkingLevel: "xhigh",
					tier: "frontier",
					sticky: false,
					escalated: false,
					predictedTier: p.tier,
					predictedScore: p.score,
					predictedWhy: p.why,
					forcedMismatch: p.tier !== "frontier",
				};
			}
		} catch {
			/* solver unavailable: the frontier tier decides */
		}
	}
	let tier: Tier = forced ?? p.tier;
	// `direct` requests (compaction summaries) without a previous model are
	// medium work: send them to the flash tier.
	if (input.reason === "direct" && !previous) tier = "flash";

	const model = pick(config, config.tierStart[tier], registry, undefined, input.estimatedTokens ?? 0);
	if (!model) return null;
	return {
		provider: model.provider,
		id: model.id,
		thinkingLevel: TIER_LEVEL[tier],
		tier,
		sticky: false,
		escalated: false,
		predictedTier: p.tier,
		predictedScore: p.score,
		predictedWhy: p.why,
		forcedMismatch: forced ? forced !== p.tier : false,
	};
}

/** Resolve a decision against a possibly cold catalog: the catalog may still
 * be loading when route() first runs (the 11:16 UNRESOLVED incident), so
 * retry a bounded number of times before giving up. */
export async function resolveWithRetry(
	input: RouteInput,
	getRegistry: () => RegistryLike,
	config: RouterConfig = DEFAULT_CONFIG,
	attempts = 3,
	sleep: (ms: number) => Promise<void> | void = () => {},
	classifier?: ClassifierPort | null,
	signal?: AbortSignal,
): Promise<RouteDecision | null> {
	for (let attempt = 1; attempt <= attempts; attempt++) {
		const decision = await decideAsync(input, getRegistry(), config, classifier, signal);
		if (decision) return decision;
		if (attempt < attempts) await sleep(250);
	}
	return null;
}

/** A local decision model the router can consult for ambiguous prompts. */
export interface ClassifierPort {
	/** Human-readable origin for the log ("http:<url>", "registry", ...). */
	readonly source: string;
	/** Ask the complexity question; resolves a label, or null when unavailable. */
	classifyComplexity(text: string, signal?: AbortSignal): Promise<"simple" | "standard" | "complex" | null>;
	/** F9: optional probability view powering the complex floor policy. */
	classifyWithProbabilities?(
		text: string,
		signal?: AbortSignal,
	): Promise<{ label: "simple" | "standard" | "complex";
			probabilities?: Record<string, number> } | null>;
}

const CLASSIFIER_TIER: Record<"simple" | "standard" | "complex", Tier> = {
	simple: "local",
	standard: "flash",
	complex: "frontier",
};

/** Hybrid decision (F4): heuristics decide everything except the ambiguous
 * band (score 1..4 on a fresh user request), which the classifier refines.
 * Every fast path and every classifier failure falls back to decide(). */
export async function decideAsync(
	input: RouteInput,
	registry: RegistryLike,
	config: RouterConfig = DEFAULT_CONFIG,
	classifier?: ClassifierPort | null,
	signal?: AbortSignal,
): Promise<RouteDecision | null> {
	const heuristics = () => decide(input, registry, config);
	// F11/R6 — privacy: secret values in the turn (assignments, key literals,
	// credential URLs, bulk env content) never leave the machine. This rule
	// outranks everything else — xhigh, vision, the risk hard-rule, the
	// classifier — and a connection-error retry on a secrets turn returns to
	// the local model instead of escalating to the cloud ladder.
	if (config.privacy && (input.reason === "user" || input.reason === "retry")) {
		// F27 — the TURN decides for assignments and env-bulk (docs, tests and
		// examples living in summaries must not hijack ordinary work); only a
		// real key SHAPE in history keeps holding the session local, because a
		// pasted key stays in the context any cloud model would receive.
		const turnWhy = hasSensitiveDataWhy(input.turnText ?? input.text);
		const histLiteral = hasSensitiveDataWhy(input.text) === "literal";
		if (turnWhy !== null || histLiteral) {
		// F21 — fail CLOSED: detected secrets with an unavailable privacy model
		// never fall through to the cloud ladder, and a connection-failed
			// privacy model is not re-selected in a dead loop.
		let found = false;
		try {
			found = fitsTarget(registry, config.privacy, input.estimatedTokens ?? 0);
		} catch {
			found = false;
		}
		if (!found) return null;
		if (
			input.reason === "retry" &&
			input.failed?.errorMessage &&
			CONNECTION_ERROR.test(input.failed.errorMessage) &&
			input.failed.provider === config.privacy.provider &&
			input.failed.id === config.privacy.id
		) {
			return null;
		}
		const p = predict(input.turnText ?? input.text);
		return {
			provider: config.privacy.provider,
			id: config.privacy.id,
			thinkingLevel: "low",
			tier: "local",
			sticky: false,
			escalated: false,
			predictedTier: p.tier,
			predictedScore: p.score,
			predictedWhy: p.why,
			forcedMismatch: false,
			privacyRouted: true,
			privacyWhy: turnWhy ?? "literal",
		};
		}
	}
	// F17 — compaction ("direct"): the summary is structured work; it runs on
	// the configured compaction model (default minimax/MiniMax-M3) regardless
	// of topic — risk words inside the summarized content are not risk work.
	// Secrets in the conversation still keep it local (R6 wins): the summary
	// would carry them to the cloud otherwise.
	if (input.reason === "direct") {
		const secrets = hasSensitiveDataWhy(input.text) !== null;
		const target = secrets ? config.privacy : config.compaction;
		if (secrets) {
			// F21 — compaction rewriting a secrets conversation fails CLOSED
			// when the local model is unavailable: the summary must not leak.
			let found = false;
			try {
				found = !!(target && registry.find(target.provider, target.id));
			} catch {
				found = false;
			}
			if (!found) return null;
		}
		if (target && fitsTarget(registry, target, input.estimatedTokens ?? 0)) {
			try {
				if (registry.find(target.provider, target.id)) {
					const p = predict(input.turnText ?? input.text); // F23: log completeness
					return {
						provider: target.provider,
						id: target.id,
						thinkingLevel: "medium",
						tier: "local",
						sticky: false,
						escalated: false,
						privacyRouted: secrets || undefined,
						privacyWhy: secrets ? (hasSensitiveDataWhy(input.text) ?? undefined) : undefined,
						predictedTier: p.tier,
						predictedScore: p.score,
						predictedWhy: p.why,
					};
				}
			} catch {
				/* fall through to the direct flow */
			}
		}
	}
	if (input.reason === "retry" && input.failed?.errorMessage && CONTEXT_OVERFLOW.test(input.failed.errorMessage)) {
		// F28 — route to the ladder's largest context window that fits the
		// request; when nothing else fits, fail visible (the session must compact).
		const need = input.estimatedTokens ?? 0;
		let best: { provider: string; id: string; ctx: number } | null = null;
		for (const entry of config.ladder) {
			if (entry.provider === input.failed.provider && entry.id === input.failed.id) continue;
			try {
				const m = registry.find(entry.provider, entry.id);
				const ctx = m?.contextWindow ?? Number.POSITIVE_INFINITY;
				if (m && ctx >= need && (!best || ctx > best.ctx)) best = { ...entry, ctx };
			} catch {
				/* skip unavailable entries */
			}
		}
		if (!best) return null;
		const idx = ladderIndex(config, best.provider, best.id);
		return {
			provider: best.provider,
			id: best.id,
			thinkingLevel: TIER_LEVEL[tierForIndex(config, idx)],
			tier: tierForIndex(config, idx),
			sticky: false,
			escalated: true,
		};
	}
	// F15 — the user NAMES the solver in the current turn ("continuá con
	// gpt 6.1 sol"): explicit consent to spend the scarce budget, equivalent
	// to the xhigh level but per request. Secrets still outrank it (R6), and
	// the mention only buys its own turn.
	if (input.reason === "user" && config.solver && SOLVER_MENTION.test(input.turnText ?? input.text)) {
		const p = predict(input.turnText ?? input.text); // F23: log completeness
		try {
			if (fitsTarget(registry, config.solver, input.estimatedTokens ?? 0)) {
				return {
					provider: config.solver.provider,
					id: config.solver.id,
					thinkingLevel: "xhigh",
					tier: "frontier",
					sticky: false,
					escalated: false,
					solverRouted: true,
					predictedTier: p.tier,
					predictedScore: p.score,
					predictedWhy: p.why,
				};
			}
		} catch {
			/* solver unavailable: the normal flow applies */
		}
	}
	// F12 — a quota/connection failure of the vision model on an IMAGE turn
	// falls back to the image-capable local model: the ladder is text-only,
	// so the normal escalation walk would hand an image turn to a model that
	// cannot read it.
	if (
		input.reason === "retry" &&
		input.hasImages &&
		input.failed?.errorMessage &&
		CONNECTION_ERROR.test(input.failed.errorMessage) &&
		config.vision &&
		input.failed.provider === config.vision.provider &&
		input.failed.id === config.vision.id &&
		config.privacy
	) {
		try {
			if (registry.find(config.privacy.provider, config.privacy.id)) {
				return {
					provider: config.privacy.provider,
					id: config.privacy.id,
					thinkingLevel: "medium",
					tier: "local",
					sticky: false,
					escalated: true,
				};
			}
		} catch {
			/* fall through to the ladder walk */
		}
	}
	// F7 — xhigh: an explicit demand for the scarce strong model outranks the
	// vision rule too (the solver takes images); decide() resolves it.
	if (input.reason === "user" && resolveLevel(input) === "xhigh") return cacheAware(input, config, registry, heuristics());
	// R4 — modality: fresh user requests carrying images go to the vision
	// model. Images outrank tier heuristics and even the risk hard-rule: the
	// frontier chat model takes no image input, and the vision model is the
	// one the user elected for image analysis.
	if (input.reason === "user" && input.hasImages && config.vision) {
		try {
			if (fitsTarget(registry, config.vision, input.estimatedTokens ?? 0)) {
				const p = predict(input.turnText ?? input.text);
				return {
					provider: config.vision.provider,
					id: config.vision.id,
					thinkingLevel: "medium",
					tier: p.tier,
					sticky: false,
					escalated: false,
					predictedTier: p.tier,
					predictedScore: p.score,
					predictedWhy: p.why,
					forcedMismatch: false,
					visionRouted: true,
				};
			}
		} catch {
			/* vision model unavailable: fall through to the normal flow */
		}
	}
	// F24 — vision-intent: typed "imagen" / pasted path but no image block.
	// Route to the vision model so the attachment does not land on a text-only
	// frontier that rejects it. Privacy (R6) and solver (R5) above already
	// caught what outranks us; vision is below them.
	if (input.reason === "user" && config.vision && !input.hasImages && visionIntent(input.turnText ?? input.text)) {
		try {
			if (registry.find(config.vision.provider, config.vision.id)) {
				const p = predict(input.turnText ?? input.text);
				const path = visionPathname(input.turnText ?? input.text, "/");
				const tier: Tier = p.tier === "local" ? "flash" : p.tier;
				return {
					provider: config.vision.provider,
					id: config.vision.id,
					thinkingLevel: "medium",
					tier,
					sticky: false,
					escalated: false,
					predictedTier: p.tier,
					predictedScore: p.score,
					predictedWhy: p.why,
					forcedMismatch: p.tier !== "vision",
					visionRouted: true,
					visionIntent: "path-or-text",
					visionPath: path ?? undefined,
				};
			}
		} catch {
			/* vision model unavailable: fall through */
		}
	}
	if (!classifier || config.classifier?.mode === "off" || input.reason !== "user") return cacheAware(input, config, registry, heuristics());
	const p = predict(input.text);
	const forced = ["high", "low", "xhigh", "max"].includes(resolveLevel(input));
	// Fast paths with zero added latency: explicit level override, the risk
	// hard-rule, or a confident complex band never consult the classifier.
	// Everything else asks the model: keyword-free prompts (score 0) are
	// exactly what the heuristics cannot see (the "plan para un portal" case).
	if (forced || p.why === "hard-rule:risk" || p.score >= 5) return cacheAware(input, config, registry, heuristics());
	if (TRIVIAL.test(input.text) && input.text.length < 400) return cacheAware(input, config, registry, heuristics());
	try {
		// F9 — prefer the probability view: the argmax is unstable at the
		// simple/complex boundary, but the probabilities are not. A "simple"
		// argmax with complex-prob at or above the floor is promoted to flash.
		let answer: "simple" | "standard" | "complex" | null = null;
		let probabilities: Record<string, number> | undefined;
		let promoted = false;
		if (typeof classifier.classifyWithProbabilities === "function") {
			const detailed = await classifier.classifyWithProbabilities(input.text, signal);
			if (detailed) {
				answer = detailed.label;
				probabilities = detailed.probabilities;
			}
		} else {
			answer = await classifier.classifyComplexity(input.text, signal);
		}
		if (answer === "simple" && probabilities) {
			const floor = config.classifier?.complexFloor ?? 0.4;
			if (floor > 0 && (probabilities.complex ?? 0) >= floor) {
				answer = "standard";
				promoted = true;
			}
		} else if (answer === "standard" && probabilities) {
			// F26 — symmetric floor: the standard/complex boundary oscillates the
			// same way simple/complex did (flip #4); the safe side is frontier.
			const floor = config.classifier?.complexFloor ?? 0.4;
			if (floor > 0 && (probabilities.complex ?? 0) >= floor) {
				answer = "complex";
				promoted = true;
			}
		}
		if (answer === "simple" || answer === "standard" || answer === "complex") {
			let tier = CLASSIFIER_TIER[answer];
			// F30 — "simple" splits by intent like the heuristic band: chat turns
			// go local, simple DEVELOPMENT turns ride the flash dev floor.
			if (tier === "local" && !CHAT.test(input.turnText ?? input.text)) tier = "flash";
			const model = pick(config, config.tierStart[tier], registry, undefined, input.estimatedTokens ?? 0);
			if (model) {
			return cacheAware(input, config, registry, {
				provider: model.provider,
				id: model.id,
				thinkingLevel: TIER_LEVEL[tier],
				tier,
				sticky: false,
				escalated: false,
				predictedTier: p.tier,
				predictedScore: p.score,
				predictedWhy: p.why,
				forcedMismatch: false,
				classifierAnswer: answer,
				classifierSource: classifier.source,
				classifierPromoted: promoted || undefined,
				classifierProbabilities: probabilities, // F23/F9: calibration data
			});
			}
		}
	} catch {
		/* classifier unavailable: the heuristic band decides */
	}
	return cacheAware(input, config, registry, heuristics());
}

/** Conversations at least this big (tokens) make a prompt-cache refill at the
 * destination a real cost, so weak voluntary switches are held (F29). */
const CACHE_CARE_TOKENS = 24_576;
/** A classifier complex-probability at least this high is a strong signal:
 * it pays for the refill even on a big conversation (F29). */
const CACHE_STRONG_PROB = 0.75;

/** F29 — cache hysteresis. Prompt caches are per-model: switching models on a
 * big conversation re-prefills the whole context at the destination, and that
 * cost scales with conversation size. Voluntary tier flips therefore need a
 * strong signal (risk rule, confident classifier, explicit level); weak flips
 * ride the previous ladder model instead. Forced routes (privacy, overflow,
 * solver, vision, retries) never pass through here, and F28 wins over F29: a
 * previous model that cannot fit the conversation is never held. */
function cacheAware(
	input: RouteInput,
	config: RouterConfig,
	registry: RegistryLike,
	d: RouteDecision,
): RouteDecision {
	if (input.reason !== "user" || !d || d.sticky) return d;
	const est = input.estimatedTokens ?? 0;
	if (est < (config.cacheCareTokens ?? CACHE_CARE_TOKENS)) return d;
	const prev = input.previous;
	if (!prev || (prev.provider === d.provider && prev.id === d.id)) return d;
	const idx = ladderIndex(config, prev.provider, prev.id);
	if (idx < 0) return d; // off-ladder previous (vision/solver/privacy): nothing to hold
	if (!fitsTarget(registry, config.ladder[idx], est)) return d; // F28 outranks F29
	const prevTier = tierForIndex(config, idx);
	if (d.tier === prevTier) return d; // same tier: no refill at stake
	const strong =
		d.predictedWhy === "hard-rule:risk" ||
		resolveLevel(input) !== "medium" ||
		(d.classifierProbabilities?.complex ?? 0) >= CACHE_STRONG_PROB;
	if (strong) return d;
	return {
		provider: prev.provider,
		id: prev.id,
		thinkingLevel: (prev.thinkingLevel as RouteDecision["thinkingLevel"]) ?? TIER_LEVEL[prevTier],
		tier: prevTier,
		sticky: true,
		escalated: false,
		why: "cache",
		predictedTier: d.predictedTier,
		predictedScore: d.predictedScore,
		predictedWhy: d.predictedWhy,
		forcedMismatch: d.forcedMismatch,
	};
}

/** F8: compact footer status for the last routing decision. Undefined clears. */
export function statusText(
	decision: RouteDecision | null | undefined,
	selectedLevel?: string,
): string | undefined {
	if (!decision) return undefined;
	let tag: string;
	if (decision.privacyRouted) tag = "privacy";
	else if (decision.solverRouted) tag = "named";
	else if (decision.escalated) tag = "escalated";
	else if (decision.visionRouted) tag = "vision";
	else if (selectedLevel === "xhigh") tag = "xhigh";
	else if (decision.classifierAnswer) tag = `laya:${decision.classifierAnswer}`;
	else if (decision.why === "cache") tag = "cache";
	else if (decision.predictedWhy === "hard-rule:risk") tag = "risk";
	else if (decision.sticky) tag = "sticky";
	else tag = decision.tier;
	return `${decision.id}:${decision.thinkingLevel} [${tag}]`;
}

// ---------------------------------------------------------------------------
// Extension shell (logging + Pi adapter)
// ---------------------------------------------------------------------------

interface RouterState {
	count: number;
}

/** Concatenated user-text of the conversation. The harness appends banners
 * (e.g. approval notices) as user messages after the real prompt, so the
 * LAST user message alone is not a reliable signal — join them all.
 * F23: harness noise is dropped (intent signals must see user words only);
 * compaction summaries STAY (privacy R6 must keep seeing carried secrets). */
export function userTextSignal(messages: readonly unknown[]): string {
	const rows = messages as Array<{ role?: string; content?: unknown }>;
	return rows
		.filter((m) => m.role === "user")
		.flatMap((m) => {
			const t = messageText(m);
			return t !== "" && !isHarnessNoise(t) ? [t] : [];
		})
		.join("\n");
}

/** True when the CURRENT turn carries an image block: every user message
 * after the last assistant message (the harness appends approval banners as
 * extra user messages, so they must not hide a real image in the turn).
 * F13: images EARLIER in the session do not make later text turns vision
 * turns — code work after a screenshot routes by the normal tier rules. */
export function hasImageInput(messages: readonly unknown[]): boolean {
	const rows = messages as Array<{ role?: string; content?: unknown }>;
	for (let i = rows.length - 1; i >= 0 && rows[i]?.role !== "assistant"; i--) {
		const blocks = rows[i]?.content;
		if (Array.isArray(blocks) && blocks.some((b) => (b as { type?: string } | null)?.type === "image")) return true;
	}
	return false;
}

/** F14 — current-turn text that explicitly names an image, bilingual. */
const IMAGE_REF =
	/\b(imagen(es)?|capturas?|pantallazos?|screenshots?|fotos?|photos?|pictures?|images?|figures?|diagramas?|diagrams?)\b/i;

/** True when the current turn's TEXT names an image and the session actually
 * contains one (F14): "analiza esta imagen" after an earlier screenshot is a
 * vision turn even though the turn itself adds no image block. */
export function referencesSessionImage(messages: readonly unknown[]): boolean {
	const rows = messages as Array<{ role?: string; content?: unknown }>;
	const hasAnyImage = rows.some((m) => {
		const blocks = m?.content;
		return Array.isArray(blocks) && blocks.some((b) => (b as { type?: string } | null)?.type === "image");
	});
	if (!hasAnyImage) return false;
	let said = false;
	for (let i = rows.length - 1; i >= 0 && rows[i]?.role !== "assistant"; i--) {
		const m = rows[i];
		if (m?.role !== "user") continue;
		const text = messageText(m); // F23: shared extraction
		if (text && !isHarnessNoise(text) && IMAGE_REF.test(text)) said = true;
	}
	return said;
}

/** F24 — vision-intent: catch plain-text image references that R4 misses
 * (image R4 only fires when the message carries an image block; the pi TUI
 * pastes a tempfile path when that block is missing). Privacy (R6) above
 * already strips any vision mention out of secrets turns. */
export function visionIntent(text: string): boolean {
	if (!text) return false;
	const mentions = /\b(imagen(es)?|captura|screenshot|foto|photo|picture|figure|diagrama|diagram|images?|figures?)\b/i.test(text);
	return mentions || IMAGE_EXT_RE.test(text);
}

/** F24 — first image-bearing path in the text (absolute, cwd-relative, or
 * Windows). Returns null when no such path is present. */
export function visionPathname(text: string, cwd = "/"): string | null {
	if (!text) return null;
	const tokens = text.match(/["']?(?:[A-Za-z]:[\\/][^"'\s)\]<>]+|\.\.?\/[\w./-]+|\/[\w./-]+)["']?/g);
	if (!tokens) return null;
	for (const raw of tokens) {
		const token = raw.replace(/^["']|["']$/g, "");
		if (!IMAGE_EXT_RE.test(token)) continue;
		if (token.startsWith("./") || token.startsWith("../")) {
			return token.replace(/^\.\.?\//, cwd.endsWith("/") ? cwd : cwd + "/");
		}
		return token;
	}
	return null;
}

function log(entry: Record<string, unknown>): void {
	try {
		appendFileSync(LOG_PATH, `${JSON.stringify({ ts: new Date().toISOString(), ...entry })}\n`);
	} catch {
		/* logging must never break routing */
	}
}

let configCache: { mtimeMs: number; config: RouterConfig } | null = null;

/** Load router.json (mtime-cached). A missing file is the normal default
 * path; an invalid or unreadable file falls back to the default ladder and
 * surfaces the reason in the log. */
function loadConfig(): { config: RouterConfig; source: "file" | "default"; error?: string } {
	try {
		const st = statSync(CONFIG_PATH);
		if (configCache && configCache.mtimeMs === st.mtimeMs) return { config: configCache.config, source: "file" };
		const parsed = parseConfig(JSON.parse(readFileSync(CONFIG_PATH, "utf8")));
		if (!parsed) return { config: DEFAULT_CONFIG, source: "default", error: "invalid router.json: default ladder kept" };
		configCache = { mtimeMs: st.mtimeMs, config: parsed };
		return { config: parsed, source: "file" };
	} catch (err: unknown) {
		const missing = (err as { code?: unknown }).code === "ENOENT";
		return missing
			? { config: DEFAULT_CONFIG, source: "default" }
			: { config: DEFAULT_CONFIG, source: "default", error: String(err).slice(0, 120) };
	}
}

/** F16: classifier prompt cap — latency scales with prompt length. */
export const CLASSIFIER_MAX_CHARS = 4000;

function makeClassifier(config: RouterConfig): ClassifierPort | null {
	const c = config.classifier;
	if (!c || c.mode === "off" || !c.url) return null;
	const timeoutMs = c.timeoutMs ?? 1500;
	return {
		source: `laya:${c.url}`,
		async classifyComplexity(text: string, signal?: AbortSignal): Promise<"simple" | "standard" | "complex" | null> {
			const detailed = await classifyDetailed(text, signal);
			return detailed?.label ?? null;
		},
		async classifyWithProbabilities(text: string, signal?: AbortSignal) {
			return classifyDetailed(text, signal);
		},
	};

	async function classifyDetailed(
		text: string,
		signal?: AbortSignal,
	): Promise<{ label: "simple" | "standard" | "complex";
			probabilities?: Record<string, number> } | null> {
			const controller = new AbortController();
			const timer = setTimeout(() => controller.abort(), timeoutMs);
			// A request abort also aborts the classifier call.
			signal?.addEventListener("abort", () => controller.abort(), { once: true });
			try {
				const res = await fetch(`${c.url}/v1/systemone`, {
					method: "POST",
					headers: { "content-type": "application/json" },
					body: JSON.stringify({
						state: { prompt: text.slice(0, CLASSIFIER_MAX_CHARS) },
						questions: {
							complexity: {
								type: "choice",
								instructions: "How demanding is the software engineering work requested in `prompt`?",
								criteria: {
									simple: "Trivial questions, formatting, renames, or small lookups",
									standard: "Ordinary features, fixes, reviews, or questions",
									complex: "Subtle design, cross-cutting changes, or hard debugging",
								},
							},
						},
					}),
					signal: controller.signal,
				});
				if (!res.ok) return null;
				const data = (await res.json()) as { answers?: { complexity?: { probabilities?: Record<string, number> } } };
				const probs = data.answers?.complexity?.probabilities;
				if (!probs) return null;
				const best = (Object.entries(probs) as Array<[string, number]>).sort((a, b) => b[1] - a[1])[0]?.[0];
				if (best !== "simple" && best !== "standard" && best !== "complex") return null;
				return { label: best, probabilities: probs };
			} catch {
				return null;
			} finally {
				clearTimeout(timer);
			}
	}
}

export default function (pi: ExtensionAPI) {
	try {
		mkdirSync(POC_DIR, { recursive: true });
	} catch {
		/* already exists or no permission; appendFileSync will surface it */
	}

	pi.registerVirtualModel<RouterState>({
		provider: "router",
		id: "auto",
		name: "Auto (router F11)",
		thinkingLevels: ["low", "medium", "high", "xhigh", "max"],
		// Conservative shared floor of the whitelist. F10: the binding model is
		// the scarce solver gpt-6.1-sol (272K); every ladder model serves 1M.
		contextWindow: 262_144,
		maxTokens: 8_192,
		async route(request: unknown, ctx: { modelRegistry: RegistryLike; ui?: { setStatus?: (key: string, text: string | undefined) => void } }) {
			const t0 = Date.now();
			// Normalize the Pi route-request shapes into the pure core's input.
			const req = request as {
				reason: RouteInput["reason"];
				thinkingLevel?: string;
				messages?: readonly unknown[];
				state?: RouterState | null;
				signal?: AbortSignal;
				previous?: { model?: { provider?: string; id?: string }; thinkingLevel?: string } | null;
				failed?: { model?: { provider?: string; id?: string }; thinkingLevel?: string; message?: { errorMessage?: string } } | null;
			};
			const text = userTextSignal(req.messages ?? []);
			const turnText = currentTurnText(req.messages ?? []);
			const hasImages = hasImageInput(req.messages ?? []) || referencesSessionImage(req.messages ?? []);
			const { config, source: configSource, error: configError } = loadConfig();
			const classifier = makeClassifier(config);
			let decision: RouteDecision | null = null;
			try {
				decision = await resolveWithRetry(
					{
						reason: req.reason,
						virtualLevel: req.thinkingLevel,
						text,
						turnText,
						hasImages,
						estimatedTokens: Math.ceil(JSON.stringify(req.messages ?? []).length / 4),
						previous: req.previous?.model
							? {
									provider: req.previous.model.provider ?? "",
									id: req.previous.model.id ?? "",
									thinkingLevel: req.previous.thinkingLevel,
								}
							: null,
						failed: req.failed?.model
							? {
									provider: req.failed.model.provider ?? "",
									id: req.failed.model.id ?? "",
									thinkingLevel: req.failed.thinkingLevel,
									errorMessage: req.failed.message?.errorMessage,
								}
							: null,
					},
					() => ctx.modelRegistry,
					config,
					3,
					(ms) => new Promise<void>((resolve) => setTimeout(resolve, ms)),
					classifier,
					req.signal,
				);
			} catch {
				/* route() must never throw; decision stays null and the request errors */
			}

			const physical = decision ? ctx.modelRegistry.find(decision.provider, decision.id) : undefined;

			log({
				reason: req.reason,
				sticky: decision?.sticky ?? false,
				escalated: decision?.escalated ?? false,
				selectedLevel: req.thinkingLevel,
				configSource,
				configError,
				tier: decision?.tier,
				predictedTier: decision?.predictedTier,
				predictedScore: decision?.predictedScore,
				predictedWhy: decision?.predictedWhy,
				forcedMismatch: decision?.forcedMismatch,
				classifierAnswer: decision?.classifierAnswer,
				classifierSource: decision?.classifierSource,
				classifierPromoted: decision?.classifierPromoted ?? false,
				classifierProbabilities: decision?.classifierProbabilities
					? Object.fromEntries(
							Object.entries(decision.classifierProbabilities).map(([k, v]) => [k, Math.round(v * 1000) / 1000]),
						)
					: undefined,
				vision: decision?.visionRouted ?? false,
				visionIntent: decision?.visionIntent,
				visionPath: decision?.visionPath,
				solverNamed: decision?.solverRouted ?? false,
				privacy: decision?.privacyRouted ?? false,
				privacyWhy: decision?.privacyWhy ?? null,
				textLen: text.length,
				textHead: text.slice(0, 60),
				routedTo: decision ? `${decision.provider}/${decision.id}` : "UNRESOLVED",
				routedLevel: decision?.thinkingLevel,
				errSnippet: req.failed?.message?.errorMessage?.slice(0, 140),
				latencyMs: Date.now() - t0,
				stateCount: (req.state?.count ?? 0) + 1,
			});

			// F8 — publish the routed model to the footer/status bar per request.
			try {
				ctx.ui?.setStatus?.("router", statusText(decision, req.thinkingLevel));
			} catch {
				/* the status line must never break routing */
			}

			if (!decision || !physical) throw new Error("router/auto: no whitelisted model available");
			return {
				model: physical,
				thinkingLevel: decision.thinkingLevel,
				state: { count: (req.state?.count ?? 0) + 1 },
			};
		},
	});
}
