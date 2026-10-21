/**
 * F1 unit tests for the pure routing core `router.decide()` in index.ts.
 *
 * Run: node --test decide.test.ts   (Node >= 23.6 strips types natively)
 *
 * Covers the three F1 rules against a stubbed model registry:
 *   R1 tier routing: virtual level + text heuristics pick the model
 *   R2 sticky: continuation/retry (non-connection) keep the previous model
 *   R3 escalation: connection-error retries walk the fallback ladder
 */
import assert from "node:assert/strict";
import { test } from "node:test";
import * as router from "./index.ts";

/** Registry stub: catalog = the (provider, id) pairs that exist. */
function reg(...catalog: Array<[string, string, number?]>): router.RegistryLike {
	return {
		find(provider: string, id: string) {
			for (const [p, i, ctx] of catalog) {
				if (p === provider && i === id) return { provider, id, contextWindow: ctx };
			}
			return undefined;
		},
	};
}

const FULL = reg(
	["llamacpp", "Qwen3.8-27B"],
	["zhipuai-coding-plan", "glm-5.3-flash"],
	["zhipuai-coding-plan", "glm-5.3"],
	["minimax", "MiniMax-M3"],
);

const QWEN = { provider: "llamacpp", id: "Qwen3.8-27B" };
const FLASH = { provider: "zhipuai-coding-plan", id: "glm-5.3-flash" };
const GLM = { provider: "zhipuai-coding-plan", id: "glm-5.3" };

test("R1: trivial user text at medium routes to local tier (glm-5.3-flash, high)", () => {
	const d = router.decide({ reason: "user", virtualLevel: "medium", text: "hola, contame un chiste" }, FULL);
	assert.deepEqual({ provider: d?.provider, id: d?.id, tier: d?.tier, thinkingLevel: d?.thinkingLevel }, {
		provider: "zhipuai-coding-plan",
		id: "glm-5.3-flash",
		tier: "local",
		thinkingLevel: "high",
	});
	assert.equal(d?.sticky, false);
});

test("R1: risk text routes to frontier tier (glm-5.3, high)", () => {
	const d = router.decide(
		{ reason: "user", virtualLevel: "medium", text: "Revisá esta migración de auth y password antes del deploy" },
		FULL,
	);
	assert.deepEqual({ id: d?.id, tier: d?.tier, thinkingLevel: d?.thinkingLevel }, {
		id: "glm-5.3",
		tier: "frontier",
		thinkingLevel: "max",
	});
});

test("R1: virtual level high overrides heuristics and forces frontier", () => {
	const d = router.decide({ reason: "user", virtualLevel: "high", text: "hola" }, FULL);
	assert.equal(d?.tier, "frontier");
	assert.equal(d?.id, "glm-5.3");
});

test("R1: virtual level low overrides heuristics and forces local", () => {
	const d = router.decide(
		{ reason: "user", virtualLevel: "low", text: "Revisá esta migración de auth y password antes del deploy" },
		FULL,
	);
	assert.equal(d?.tier, "local");
	assert.equal(d?.id, "glm-5.3-flash");
});

test("R2: continuation stays sticky on the previous model and level", () => {
	const d = router.decide(
		{
			reason: "continuation",
			virtualLevel: "medium",
			text: "",
			previous: { provider: "zhipuai-coding-plan", id: "glm-5.3", thinkingLevel: "high" },
		},
		FULL,
	);
	assert.deepEqual({ id: d?.id, thinkingLevel: d?.thinkingLevel }, { id: "glm-5.3", thinkingLevel: "high" });
	assert.equal(d?.sticky, true);
	assert.equal(d?.tier, "frontier");
});

test("R3: connection-error retry escalates to the next ladder entry", () => {
	const d = router.decide(
		{
			reason: "retry",
			virtualLevel: "medium",
			text: "",
			failed: { ...QWEN, errorMessage: "fetch failed: ECONNREFUSED 127.0.0.1:8081" },
		},
		FULL,
	);
	// F30: an empty turn has no chat markers, so the fallback base is the
	// flash dev floor — the off-ladder failed model retries on frontier.
	assert.deepEqual({ id: d?.id, tier: d?.tier }, { id: "glm-5.3", tier: "frontier" });
	assert.equal(d?.escalated, true);
});

test("R3: second connection-error retry keeps walking the ladder", () => {
	const d = router.decide(
		{
			reason: "retry",
			virtualLevel: "medium",
			text: "",
			failed: { ...FLASH, errorMessage: "request timed out after 30000ms" },
		},
		FULL,
	);
	assert.equal(d?.id, "glm-5.3");
	assert.equal(d?.escalated, true);
});

test("R3: ladder wraps when the last entry fails with a connection error", () => {
	const d = router.decide(
		{
			reason: "retry",
			virtualLevel: "medium",
			text: "",
			failed: { ...GLM, errorMessage: "503 overloaded" },
		},
		FULL,
	);
	assert.equal(d?.id, "glm-5.3-flash");
	assert.equal(d?.escalated, true);
});

test("R2: non-connection retry (context overflow) stays sticky", () => {
	const d = router.decide(
		{
			reason: "retry",
			virtualLevel: "medium",
			text: "",
			failed: { ...QWEN, errorMessage: "context length exceeded: 262144 tokens" },
		},
		FULL,
	);
	assert.equal(d?.id, "Qwen3.8-27B");
	assert.equal(d?.sticky, true);
	assert.equal(d?.escalated, false);
});

test("R1 fallback: missing tier model skips to the next available ladder entry", () => {
	const noQwen = reg(["zhipuai-coding-plan", "glm-5.3-flash"], ["zhipuai-coding-plan", "glm-5.3"]);
	const d = router.decide({ reason: "user", virtualLevel: "low", text: "hola" }, noQwen);
	assert.equal(d?.id, "glm-5.3-flash");
});

test("direct compaction request without previous defaults to flash tier", () => {
	const d = router.decide({ reason: "direct", virtualLevel: "medium", text: "" }, FULL);
	assert.deepEqual({ id: d?.id, tier: d?.tier }, { id: "glm-5.3-flash", tier: "flash" });
});

test("empty registry resolves to null (caller logs UNRESOLVED)", () => {
	const d = router.decide({ reason: "user", virtualLevel: "medium", text: "hola" }, reg());
	assert.equal(d, null);
});

test("userTextSignal joins every user message, not just the last (F23: harness banners excluded)", () => {
	const text = router.userTextSignal([
		{ role: "system", content: "sys" },
		{ role: "user", content: "Analiza el riesgo de esta migracion de auth" },
		{ role: "user", content: [{ type: "text", text: "[审批: ⚡ ACT] 完全权限" }] },
	]);
	assert.ok(text.includes("migracion de auth"));
	// F23 contract change: banners are harness-injected, never user intent —
	// they must NOT reach any intent heuristic (the 'Release' notice incident).
	assert.ok(!text.includes("[审批"));
});

test("ladder order is stable: local -> flash -> frontier", () => {
	assert.deepEqual(
		router.LADDER.map((m) => `${m.provider}/${m.id}`),
		["zhipuai-coding-plan/glm-5.3-flash", "zhipuai-coding-plan/glm-5.3-flash", "zhipuai-coding-plan/glm-5.3"],
	);
});

// ---------------------------------------------------------------------------
// F2: configurable whitelist, cold-registry retry, calibration hooks
// ---------------------------------------------------------------------------

test("parseConfig: valid ladder maps tier starts", () => {
	const c = router.parseConfig({
		ladder: [
			{ provider: "llamacpp", id: "Qwen3.8-27B", tier: "local" },
			{ provider: "zhipuai-coding-plan", id: "glm-5.3-flash", tier: "flash" },
			{ provider: "zhipuai-coding-plan", id: "glm-5.3", tier: "frontier" },
		],
	});
	assert.deepEqual(c?.tierStart, { local: 0, flash: 1, frontier: 2 });
	assert.equal(c?.ladder.length, 3);
});

test("parseConfig: repeated tier takes the first index as its start", () => {
	const c = router.parseConfig({
		ladder: [
			{ provider: "a", id: "m1", tier: "local" },
			{ provider: "b", id: "m2", tier: "flash" },
			{ provider: "c", id: "m3", tier: "flash" },
			{ provider: "d", id: "m4", tier: "frontier" },
		],
	});
	assert.deepEqual(c?.tierStart, { local: 0, flash: 1, frontier: 3 });
});

test("parseConfig: same model under two tiers is valid; same model+tier duplicate is invalid", () => {
		const dupTiers = router.parseConfig({
			ladder: [
				{ provider: "zhipuai-coding-plan", id: "glm-5.3-flash", tier: "local" },
				{ provider: "zhipuai-coding-plan", id: "glm-5.3-flash", tier: "flash" },
				{ provider: "zhipuai-coding-plan", id: "glm-5.3", tier: "frontier" },
			],
		});
		assert.equal(dupTiers?.ladder.length, 3);
		assert.equal(dupTiers?.tierStart.local, 0);
		assert.equal(dupTiers?.tierStart.flash, 1);
		const sameTier = router.parseConfig({
			ladder: [
				{ provider: "zhipuai-coding-plan", id: "glm-5.3-flash", tier: "local" },
				{ provider: "zhipuai-coding-plan", id: "glm-5.3-flash", tier: "local" },
				{ provider: "zhipuai-coding-plan", id: "glm-5.3", tier: "frontier" },
			],
		});
		assert.equal(sameTier, null);
	});

test("parseConfig: garbage input is invalid, not a crash", () => {
	assert.equal(router.parseConfig(null), null);
	assert.equal(router.parseConfig({}), null);
	assert.equal(router.parseConfig({ ladder: "nope" }), null);
});

test("resolveWithRetry: catalog that settles on the 3rd read resolves", async () => {
	let reads = 0;
	const getRegistry = () => {
		reads++;
		return reads >= 3 ? FULL : reg();
	};
	const d = await router.resolveWithRetry({ reason: "user", virtualLevel: "medium", text: "hola" }, getRegistry, router.DEFAULT_CONFIG, 4, () => {});
	assert.equal(d?.id, "glm-5.3-flash");
	assert.equal(reads, 3);
});

test("resolveWithRetry: cold registry gives up after the attempt budget", async () => {
	let reads = 0;
	const d = await router.resolveWithRetry({ reason: "user", virtualLevel: "medium", text: "hola" }, () => (reads++, reg()), router.DEFAULT_CONFIG, 3, () => {});
	assert.equal(d, null);
	assert.equal(reads, 3);
});

test("decide: forced low on risky text logs the prediction it overrode", () => {
	const d = router.decide(
		{ reason: "user", virtualLevel: "low", text: "Revisá esta migracion de auth y password antes del deploy" },
		FULL,
	);
	assert.equal(d?.tier, "local");
	assert.equal(d?.predictedTier, "frontier");
	assert.equal(d?.predictedScore, 99);
	assert.equal(d?.forcedMismatch, true);
});

test("decide: medium risk text predicts frontier with no mismatch", () => {
	const d = router.decide({ reason: "user", virtualLevel: "medium", text: "migracion de auth" }, FULL);
	assert.equal(d?.tier, "frontier");
	assert.equal(d?.predictedTier, "frontier");
	assert.equal(d?.forcedMismatch, false);
});

test("R1: Spanish destructive risk (eliminar tabla en produccion) routes to frontier", () => {
	const d = router.decide(
		{ reason: "user", virtualLevel: "medium", text: "Evalúa el riesgo de eliminar la tabla de users en produccion, una linea" },
		FULL,
	);
	assert.equal(d?.tier, "frontier");
	assert.equal(d?.predictedWhy, "hard-rule:risk");
	assert.equal(d?.predictedScore, 99);
});

test("R1: Spanish credentials risk (contraseña/credenciales/despliegue) routes to frontier", () => {
	const d = router.decide(
		{ reason: "user", virtualLevel: "medium", text: "Revisa el cambio de contraseña y credenciales antes del despliegue" },
		FULL,
	);
	assert.equal(d?.tier, "frontier");
	assert.equal(d?.predictedWhy, "hard-rule:risk");
});

test("R1: Spanish complex text reaches the flash band", () => {
	const d = router.decide(
		{ reason: "user", virtualLevel: "medium", text: "Investiga la causa raiz del interbloqueo del worker" },
		FULL,
	);
	assert.equal(d?.tier, "flash");
	assert.ok((d?.predictedScore ?? 0) >= 2);
});

// ---------------------------------------------------------------------------
// F4: hybrid decision (heuristics + optional classifier)
// ---------------------------------------------------------------------------

/** Long text with no regex hits: heuristic score 2 (len>1500) = ambiguous band. */
const AMBIGUOUS = "palabra ".repeat(220);

function fakeClassifier(
	answer: "simple" | "standard" | "complex" | null,
	opts: { fail?: boolean; probs?: Record<string, number> } = {},
) {
	let calls = 0;
	return {
		source: "mock",
		calls: () => calls,
		async classifyComplexity(): Promise<"simple" | "standard" | "complex" | null> {
			calls++;
			if (opts.fail) throw new Error("classifier boom");
			return answer;
		},
		async classifyWithProbabilities(): Promise<{
			label: "simple" | "standard" | "complex";
			probabilities?: Record<string, number>;
		} | null> {
			calls++;
			if (opts.fail) throw new Error("classifier boom");
			return { label: answer!, probabilities: opts.probs };
		},
	};
}

test("F4: RISK fast path never consults the classifier", async () => {
	const c = fakeClassifier("simple");
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "Revisa esta migracion de auth con password antes del deploy" },
		FULL,
		router.DEFAULT_CONFIG,
		c,
	);
	assert.equal(d?.tier, "frontier");
	assert.equal(c.calls(), 0);
});

test("F4: ambiguous score consults the classifier; complex -> frontier", async () => {
	const c = fakeClassifier("complex");
	const d = await router.decideAsync({ reason: "user", virtualLevel: "medium", text: AMBIGUOUS }, FULL, router.DEFAULT_CONFIG, c);
	assert.equal(d?.tier, "frontier");
	assert.equal(d?.classifierAnswer, "complex");
	assert.equal(d?.classifierSource, "mock");
	assert.equal(c.calls(), 1);
});

test("F4: ambiguous score + classifier simple without chat markers -> flash", async () => {
	const c = fakeClassifier("simple");
	const d = await router.decideAsync({ reason: "user", virtualLevel: "medium", text: AMBIGUOUS }, FULL, router.DEFAULT_CONFIG, c);
	assert.equal(d?.tier, "flash");
	assert.equal(d?.id, "glm-5.3-flash");
});

test("F4: classifier failure falls back to the heuristic band", async () => {
	const c = fakeClassifier(null, { fail: true });
	const d = await router.decideAsync({ reason: "user", virtualLevel: "medium", text: AMBIGUOUS }, FULL, router.DEFAULT_CONFIG, c);
	assert.equal(d?.tier, "flash");
	assert.equal(d?.classifierAnswer, undefined);
});

test("F4: classifier off in config never calls it", async () => {
	const c = fakeClassifier("complex");
	const config = { ...router.DEFAULT_CONFIG, classifier: { mode: "off" as const } };
	const d = await router.decideAsync({ reason: "user", virtualLevel: "medium", text: AMBIGUOUS }, FULL, config, c);
	assert.equal(d?.tier, "flash");
	assert.equal(c.calls(), 0);
});

test("F4: confident score (>=5) skips the classifier", async () => {
	const c = fakeClassifier("simple");
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: `${AMBIGUOUS} refactor completo del modulo ` },
		FULL,
		router.DEFAULT_CONFIG,
		c,
	);
	assert.equal(d?.tier, "frontier");
	assert.equal(c.calls(), 0);
});

test("F4: short plain feature request (score 0, no TRIVIAL) consults the classifier", async () => {
	const c = fakeClassifier("complex");
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "Haceme un plan para crear un portal de turnos medicos con login y base de datos" },
		FULL,
		router.DEFAULT_CONFIG,
		c,
	);
	assert.equal(d?.tier, "frontier");
	assert.equal(d?.classifierAnswer, "complex");
	assert.equal(c.calls(), 1);
});

test("F4: explicitly trivial short text skips the classifier", async () => {
	const c = fakeClassifier("complex");
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "renombrar esta variable a usuarioId, typo fix" },
		FULL,
		router.DEFAULT_CONFIG,
		c,
	);
	assert.equal(d?.tier, "flash");
	assert.equal(c.calls(), 0);
});

test("parseConfig: optional classifier block accepted, bad mode rejected", () => {
	const ok = router.parseConfig({
		ladder: [
			{ provider: "llamacpp", id: "Qwen3.8-27B", tier: "local" },
			{ provider: "zhipuai-coding-plan", id: "glm-5.3-flash", tier: "flash" },
			{ provider: "zhipuai-coding-plan", id: "glm-5.3", tier: "frontier" },
		],
		classifier: { mode: "auto", timeoutMs: 900 },
	});
	assert.equal(ok?.classifier?.mode, "auto");
	assert.equal(ok?.classifier?.timeoutMs, 900);
	const bad = router.parseConfig({
		ladder: [
			{ provider: "llamacpp", id: "Qwen3.8-27B", tier: "local" },
			{ provider: "zhipuai-coding-plan", id: "glm-5.3-flash", tier: "flash" },
			{ provider: "zhipuai-coding-plan", id: "glm-5.3", tier: "frontier" },
		],
		classifier: { mode: "sometimes" },
	});
	assert.equal(bad, null);
});

// ---------------------------------------------------------------------------
// F6: R4 modality routing (images -> vision model)
// ---------------------------------------------------------------------------

const LADDER_JSON = [
	{ provider: "llamacpp", id: "Qwen3.8-27B", tier: "local" },
	{ provider: "zhipuai-coding-plan", id: "glm-5.3-flash", tier: "flash" },
	{ provider: "zhipuai-coding-plan", id: "glm-5.3", tier: "frontier" },
];

test("R4: image input routes to the vision model, skipping the classifier", async () => {
	const c = fakeClassifier("simple");
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "que ves en esta imagen", hasImages: true },
		FULL,
		router.DEFAULT_CONFIG,
		c,
	);
	assert.equal(d?.provider, "minimax");
	assert.equal(d?.id, "MiniMax-M3");
	assert.equal(c.calls(), 0);
	assert.equal(d?.visionRouted, true);
});

test("R4: images outrank the risk hard-rule (frontier glm has no image input)", async () => {
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "revisa el deploy y auth de esta captura", hasImages: true },
		FULL,
	);
	assert.equal(d?.id, "MiniMax-M3");
});

test("R4: vision model missing from the catalog falls back to the tier flow", async () => {
	const noMinimax = reg(
		["llamacpp", "Qwen3.8-27B"],
		["zhipuai-coding-plan", "glm-5.3-flash"],
		["zhipuai-coding-plan", "glm-5.3"],
	);
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "que ves en esta imagen", hasImages: true },
		noMinimax,
	);
	assert.equal(d?.id, "glm-5.3-flash");
});

test("R4: continuations stay sticky even with images in the turn", async () => {
	const d = await router.decideAsync(
		{
			reason: "continuation",
			virtualLevel: "medium",
			text: "",
			hasImages: true,
			previous: { provider: "minimax", id: "MiniMax-M3", thinkingLevel: "medium" },
		},
		FULL,
	);
	assert.equal(d?.sticky, true);
	assert.equal(d?.id, "MiniMax-M3");
});

test("parseConfig: optional vision block accepted, empty provider rejected", () => {
	const ok = router.parseConfig({ ladder: LADDER_JSON, vision: { provider: "minimax", id: "MiniMax-M3" } });
	assert.equal(ok?.vision?.id, "MiniMax-M3");
	const bad = router.parseConfig({ ladder: LADDER_JSON, vision: { provider: "", id: "X" } });
	assert.equal(bad, null);
});

// ---------------------------------------------------------------------------
// F7: xhigh level — the scarce strong model (gpt-6.1-sol), explicit opt-in
// ---------------------------------------------------------------------------

test("xhigh: routes to the configured solver model, skipping the classifier", async () => {
	const c = fakeClassifier("simple");
	const withSol = reg(
		["llamacpp", "Qwen3.8-27B"],
		["zhipuai-coding-plan", "glm-5.3-flash"],
		["zhipuai-coding-plan", "glm-5.3"],
		["openai", "gpt-6.1-sol"],
	);
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "xhigh", text: "hola" },
		withSol,
		router.DEFAULT_CONFIG,
		c,
	);
	assert.equal(d?.provider, "openai");
	assert.equal(d?.id, "gpt-6.1-sol");
	assert.equal(c.calls(), 0);
});

test("xhigh: solver missing from the catalog falls back to frontier", async () => {
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "xhigh", text: "hola" },
		FULL,
		router.DEFAULT_CONFIG,
	);
	assert.equal(d?.id, "glm-5.3");
	assert.equal(d?.tier, "frontier");
});

test("xhigh: outranks the vision rule (the solver takes images too)", async () => {
	const withSol = reg(
		["llamacpp", "Qwen3.8-27B"],
		["zhipuai-coding-plan", "glm-5.3-flash"],
		["zhipuai-coding-plan", "glm-5.3"],
		["minimax", "MiniMax-M3"],
		["openai", "gpt-6.1-sol"],
	);
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "xhigh", text: "analiza esta captura", hasImages: true },
		withSol,
	);
	assert.equal(d?.id, "gpt-6.1-sol");
});

test("xhigh: continuations stay sticky on the solver", async () => {
	const d = await router.decideAsync(
		{
			reason: "continuation",
			virtualLevel: "xhigh",
			text: "",
			previous: { provider: "openai", id: "gpt-6.1-sol", thinkingLevel: "high" },
		},
		FULL,
	);
	assert.equal(d?.sticky, true);
	assert.equal(d?.id, "gpt-6.1-sol");
});

test("parseConfig: optional solver block accepted, bad shape rejected", () => {
	const ok = router.parseConfig({ ladder: LADDER_JSON, solver: { provider: "openai", id: "gpt-6.1-sol" } });
	assert.equal(ok?.solver?.id, "gpt-6.1-sol");
	const bad = router.parseConfig({ ladder: LADDER_JSON, solver: { provider: "openai" } });
	assert.equal(bad, null);
});

// --- F8: footer status text -------------------------------------------------

test("F8: null decision clears the status", () => {
	assert.equal(router.statusText(null), undefined);
});

test("F8: classifier decision shows model, level and laya label", () => {
	// craft a classifier-shaped decision directly (same pattern as the [vision] test)
	const decision = {
		provider: "zhipuai-coding-plan",
		id: "glm-5.3",
		thinkingLevel: "high" as const,
		tier: "frontier" as const,
		sticky: false,
		escalated: false,
		classifierAnswer: "complex" as const,
	};
	assert.equal(router.statusText(decision), "glm-5.3:high [laya:complex]");
});

test("F8: risk fast path tags [risk]", () => {
	const d = router.decide(
		{ reason: "user", virtualLevel: "medium", text: "analiza el riesgo de esta migracion de auth" },
		FULL,
	);
	assert.equal(router.statusText(d), "glm-5.3:max [risk]");
});

test("F8: sticky continuation tags [sticky]", () => {
	const d = router.decide(
		{ reason: "continuation", text: "", previous: { provider: "llamacpp", id: "Qwen3.8-27B", thinkingLevel: "low" } },
		FULL,
	);
	assert.equal(router.statusText(d), "Qwen3.8-27B:low [sticky]");
});

test("F8: escalated retry tags [escalated]", () => {
	const d = router.decide(
		{
			reason: "retry",
			text: "hola",
			failed: { provider: "llamacpp", id: "Qwen3.8-27B", errorMessage: "fetch failed: ECONNREFUSED" },
		},
		FULL,
	);
	assert.equal(router.statusText(d), "glm-5.3-flash:high [escalated]");
});

test("F8: vision routing tags [vision]", () => {
	const decision = {
		provider: "minimax",
		id: "MiniMax-M3",
		thinkingLevel: "medium" as const,
		tier: "local" as const,
		sticky: false,
		escalated: false,
		visionRouted: true,
	};
	assert.equal(router.statusText(decision), "MiniMax-M3:medium [vision]");
});

test("F8: xhigh solver tags [xhigh] via selected level", () => {
	const d = router.decide(
		{ reason: "user", virtualLevel: "xhigh", text: "hola" },
		reg(
			["llamacpp", "Qwen3.8-27B"],
			["zhipuai-coding-plan", "glm-5.3-flash"],
			["zhipuai-coding-plan", "glm-5.3"],
			["openai", "gpt-6.1-sol"],
		),
	);
	assert.equal(router.statusText(d, "xhigh"), "gpt-6.1-sol:xhigh [xhigh]");
});

// --- F9: complex probability floor -------------------------------------------

const FLOOR_CONFIG = { ...router.DEFAULT_CONFIG, classifier: { mode: "auto" as const, complexFloor: 0.4 } };

test("F9: simple argmax with complex-prob >= floor routes to flash, not local", async () => {
	const c = fakeClassifier("simple", { probs: { simple: 0.5, standard: 0.05, complex: 0.45 } });
	const d = await router.decideAsync({ reason: "user", virtualLevel: "medium", text: AMBIGUOUS }, FULL, FLOOR_CONFIG, c);
	assert.equal(d?.tier, "flash");
	assert.equal(c.calls(), 1);
});

test("F9: simple argmax with low complex-prob rides the dev floor (flash)", async () => {
	const c = fakeClassifier("simple", { probs: { simple: 0.75, standard: 0.05, complex: 0.2 } });
	const d = await router.decideAsync({ reason: "user", virtualLevel: "medium", text: AMBIGUOUS }, FULL, FLOOR_CONFIG, c);
	assert.equal(d?.tier, "flash");
});

test("F9: complex argmax still routes to frontier through the detailed port", async () => {
	const c = fakeClassifier("complex", { probs: { simple: 0.2, standard: 0.1, complex: 0.7 } });
	const d = await router.decideAsync({ reason: "user", virtualLevel: "medium", text: AMBIGUOUS }, FULL, FLOOR_CONFIG, c);
	assert.equal(d?.tier, "frontier");
});

test("F9: floor disabled (complexFloor 0) keeps pure argmax", async () => {
	const off = { ...router.DEFAULT_CONFIG, classifier: { mode: "auto" as const, complexFloor: 0 } };
	const c = fakeClassifier("simple", { probs: { simple: 0.5, standard: 0.05, complex: 0.45 } });
	const d = await router.decideAsync({ reason: "user", virtualLevel: "medium", text: AMBIGUOUS }, FULL, off, c);
	assert.equal(d?.tier, "flash");
});

test("F9: port without the detailed method keeps working (back-compat)", async () => {
	const c = { source: "mock", classifyComplexity: async () => "simple" as const };
	const d = await router.decideAsync({ reason: "user", virtualLevel: "medium", text: AMBIGUOUS }, FULL, FLOOR_CONFIG, c);
	assert.equal(d?.tier, "flash");
});

test("F9: parseConfig accepts complexFloor in range and rejects out of range", () => {
	const ok = router.parseConfig({ ladder: LADDER_JSON, classifier: { mode: "auto", complexFloor: 0.4 } });
	assert.equal(ok?.classifier?.complexFloor, 0.4);
	assert.equal(router.parseConfig({ ladder: LADDER_JSON, classifier: { mode: "auto", complexFloor: 1.5 } }), null);
	assert.equal(router.parseConfig({ ladder: LADDER_JSON, classifier: { mode: "auto", complexFloor: -0.1 } }), null);
	assert.equal(router.parseConfig({ ladder: LADDER_JSON, classifier: { mode: "auto", complexFloor: "high" } }), null);
});

// --- F10: local tier swap (Qwen out, glm-5.3-flash in) -----------------------

test("F10: default local tier routes cheap work to glm-5.3-flash", () => {
	const d = router.decide({ reason: "user", virtualLevel: "medium", text: "hola" }, FULL);
	assert.equal(d?.id, "glm-5.3-flash");
	assert.equal(d?.tier, "local");
	assert.equal(d?.thinkingLevel, "high");
});

test("F10: R3 escalation skips a ladder entry with the same dead model", () => {
	const d = router.decide(
		{
			reason: "retry",
			text: "hola",
			failed: { provider: "zhipuai-coding-plan", id: "glm-5.3-flash", errorMessage: "fetch failed: 503 overloaded" },
		},
		FULL,
	);
	assert.equal(d?.id, "glm-5.3");
	assert.equal(d?.escalated, true);
});

// --- F11: privacy routing (secrets stay on the local model) -------------------

test("F11: pasted api-key assignment routes to the local model, classifier skipped", async () => {
	const c = fakeClassifier("complex");
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "revisa esto: OPENAI_API_KEY=sk-zz99real7key1not2fake3xx antes de seguir" },
		FULL,
		router.DEFAULT_CONFIG,
		c,
	);
	assert.equal(d?.provider, "llamacpp");
	assert.equal(d?.id, "Qwen3.8-27B");
	assert.equal(d?.privacyRouted, true);
	assert.equal(c.calls(), 0);
});

test("F11: private-key block and connection string with password route local", async () => {
	const key = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow...\n-----END RSA PRIVATE KEY-----";
	const d1 = await router.decideAsync({ reason: "user", virtualLevel: "medium", text: `firmá con ${key}` }, FULL);
	assert.equal(d1?.id, "Qwen3.8-27B");
	const d2 = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "conecta: postgres://admin:ClaveReal77@db.interno:5432/prod" },
		FULL,
	);
	assert.equal(d2?.id, "Qwen3.8-27B");
});

test("F11: bulk env-file content routes local", async () => {
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "mi .env:\nDB_PASSWORD=PassReal88\nSTRIPE_TOKEN=sk-zz77real7key1not2fake5zz" },
		FULL,
	);
	assert.equal(d?.privacyRouted, true);
});

test("F11: credential TOPICS without values keep normal routing", async () => {
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "Revisa el cambio de contraseña y credenciales antes del despliegue" },
		FULL,
	);
	assert.equal(d?.privacyRouted, undefined);
	assert.equal(d?.id, "glm-5.3");
});

test("F11: privacy outranks xhigh — secrets never reach the solver", async () => {
	const withSol = reg(
		["llamacpp", "Qwen3.8-27B"],
		["zhipuai-coding-plan", "glm-5.3-flash"],
		["zhipuai-coding-plan", "glm-5.3"],
		["openai", "gpt-6.1-sol"],
	);
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "xhigh", text: "usa esto AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIreal12345xx99" },
		withSol,
	);
	assert.equal(d?.id, "Qwen3.8-27B");
	assert.equal(d?.privacyRouted, true);
});

test("F11: privacy outranks the vision rule (local model takes images too)", async () => {
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "que ves en esta captura, api_key=sk-zz88real7key1not2fake4yy" , hasImages: true },
		FULL,
	);
	assert.equal(d?.id, "Qwen3.8-27B");
});

test("F11: connection-error retry on a secrets turn fails closed — no cloud escalation, no dead-model loop", async () => {
	const d = await router.decideAsync(
		{
			reason: "retry",
			text: "revisa: OPENAI_API_KEY=sk-zz99real7key1not2fake3xx",
			failed: { provider: "llamacpp", id: "Qwen3.8-27B", errorMessage: "fetch failed: ECONNREFUSED" },
		},
		FULL,
	);
	assert.equal(d, null);
});

test("F11: privacy model missing from the catalog fails closed", async () => {
	const noLocal = reg(["zhipuai-coding-plan", "glm-5.3-flash"], ["zhipuai-coding-plan", "glm-5.3"]);
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "revisa: api_key=sk-zz99real7key1not2fake3xx" },
		noLocal,
	);
	assert.equal(d, null);
});

test("F11: parseConfig accepts a privacy block and rejects an empty provider", () => {
	const ok = router.parseConfig({
		ladder: LADDER_JSON,
		privacy: { provider: "llamacpp", id: "Qwen3.8-27B" },
	});
	assert.equal(ok?.privacy?.id, "Qwen3.8-27B");
	assert.equal(router.parseConfig({ ladder: LADDER_JSON, privacy: { provider: "", id: "x" } }), null);
});

test("F11: statusText tags privacy routing", async () => {
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "api_key=sk-zz99real7key1not2fake3xx" },
		FULL,
	);
	assert.equal(router.statusText(d), "Qwen3.8-27B:low [privacy]");
});

// --- F12: quota-aware retries (rate-limit errors escalate, vision falls back) ---

test("F12: 429 rate_limit_error counts as a connection error and escalates", async () => {
	const d = await router.decideAsync(
		{
			reason: "retry",
			text: "revisa la infraestructura",
			failed: {
				provider: "zhipuai-coding-plan",
				id: "glm-5.3-flash",
				errorMessage: '429 {"type":"error","error":{"type":"rate_limit_error","message":"The Token Plan usage limit has been reached"}}',
			},
		},
		FULL,
	);
	assert.equal(d?.id, "glm-5.3");
	assert.equal(d?.escalated, true);
});

test("F12: vision-model quota failure on an image turn falls back to the image-capable local model", async () => {
	const d = await router.decideAsync(
		{
			reason: "retry",
			text: "ubica la funcion de plantillas",
			hasImages: true,
			failed: {
				provider: "minimax",
				id: "MiniMax-M3",
				errorMessage: '429 {"type":"error","error":{"type":"rate_limit_error","message":"The Token Plan usage limit has been reached"}}',
			},
		},
		FULL,
	);
	assert.equal(d?.provider, "llamacpp");
	assert.equal(d?.id, "Qwen3.8-27B");
	assert.equal(d?.escalated, true);
});

test("F12: vision failure on a TEXT turn (no images) escalates the normal ladder", async () => {
	const d = await router.decideAsync(
		{
			reason: "retry",
			text: "ubica la funcion de plantillas",
			failed: {
				provider: "minimax",
				id: "MiniMax-M3",
				errorMessage: "Error: 503 overloaded",
			},
		},
		FULL,
	);
	assert.equal(d?.provider, "zhipuai-coding-plan");
	assert.equal(d?.escalated, true);
});

// --- F13: vision routing keys on the CURRENT turn's images, not history -------

const imgMsg = { role: "user", content: [{ type: "image" }, { type: "text", text: "mirá esto" }] };
const txtMsg = { role: "user", content: [{ type: "text", text: "ahora seguí con el código" }] };
const asstMsg = { role: "assistant", content: [{ type: "text", text: "listo" }] };

test("F13: an image earlier in the session does not make later text turns vision turns", () => {
	const messages = [imgMsg, asstMsg, txtMsg];
	assert.equal(router.hasImageInput(messages), false);
});

test("F13: an image in the current turn routes to vision", () => {
	const messages = [imgMsg, asstMsg, { ...txtMsg }, imgMsg];
	assert.equal(router.hasImageInput(messages), true);
});

test("F13: an approval banner after an image prompt does not hide the image", () => {
	const banner = { role: "user", content: [{ type: "text", text: "[审批: ⚡ ACT] 完全权限" }] };
	const messages = [asstMsg, imgMsg, banner];
	assert.equal(router.hasImageInput(messages), true);
});

// --- F14: text references to a session image also route to vision -----------

const histImg = { role: "user", content: [{ type: "image" }, { type: "text", text: "mirá" }] };
const asst = { role: "assistant", content: [{ type: "text", text: "ok" }] };
const askRef = { role: "user", content: [{ type: "text", text: "analiza esta imagen con detalle" }] };
const askCode = { role: "user", content: [{ type: "text", text: "seguí con el código del endpoint" }] };

test("F14: 'analiza esta imagen' with a session image is a vision turn", () => {
	assert.equal(router.referencesSessionImage([histImg, asst, askRef]), true);
});

test("F14: image words without any image in the session are NOT a vision turn", () => {
	assert.equal(router.referencesSessionImage([asst, askRef]), false);
});

test("F14: plain code text with a session image is NOT a vision turn", () => {
	assert.equal(router.referencesSessionImage([histImg, asst, askCode]), false);
});

test("F14: english reference ('look at this screenshot') with session image", () => {
	const en = { role: "user", content: [{ type: "text", text: "look at this screenshot again" }] };
	assert.equal(router.referencesSessionImage([histImg, asst, en]), true);
});

// --- F15: naming the solver in the request routes the turn to it ------------

test("F15: 'continuá con gpt 6.1 sol' routes the turn to the solver", async () => {
	const c = fakeClassifier("simple");
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "hola", turnText: "continuá con gpt 6.1 sol en esto" },
		reg(
			["zhipuai-coding-plan", "glm-5.3-flash"],
			["zhipuai-coding-plan", "glm-5.3"],
			["openai", "gpt-6.1-sol"],
		),
		router.DEFAULT_CONFIG,
		c,
	);
	assert.equal(d?.id, "gpt-6.1-sol");
	assert.equal(d?.solverRouted, true);
	assert.equal(c.calls(), 0);
});

test("F15: exact hyphenated form 'gpt-6.1-sol' also matches", async () => {
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "x", turnText: "revisalo con gpt-6.1-sol por favor" },
		reg(["zhipuai-coding-plan", "glm-5.3-flash"], ["zhipuai-coding-plan", "glm-5.3"], ["openai", "gpt-6.1-sol"]),
	);
	assert.equal(d?.id, "gpt-6.1-sol");
});

test("F15: secrets outrank the named solver", async () => {
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "api_key=sk-zz99real7key1not2fake3xx", turnText: "analizalo con gpt 6.1 sol" },
		reg(["llamacpp", "Qwen3.8-27B"], ["zhipuai-coding-plan", "glm-5.3-flash"], ["openai", "gpt-6.1-sol"]),
	);
	assert.equal(d?.id, "Qwen3.8-27B");
	assert.equal(d?.privacyRouted, true);
});

test("F15: no mention keeps normal routing", async () => {
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "hola", turnText: "continuá con el endpoint" },
		reg(["zhipuai-coding-plan", "glm-5.3-flash"], ["zhipuai-coding-plan", "glm-5.3"], ["openai", "gpt-6.1-sol"]),
	);
	assert.equal(d?.solverRouted, undefined);
});

test("F15: a mention only affects its own turn; the next plain turn re-routes", async () => {
	const cat = reg(["zhipuai-coding-plan", "glm-5.3-flash"], ["zhipuai-coding-plan", "glm-5.3"], ["openai", "gpt-6.1-sol"]);
	const first = await router.decideAsync({ reason: "user", virtualLevel: "medium", text: "x", turnText: "usa gpt 6.1 sol" }, cat);
	assert.equal(first?.id, "gpt-6.1-sol");
	const next = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "hola", turnText: "listo, seguimos", previous: { provider: "openai", id: "gpt-6.1-sol", thinkingLevel: "high" } },
		cat,
	);
	assert.equal(next?.id, "glm-5.3-flash");
});

test("F15: statusText tags a named-solver turn", () => {
	assert.equal(
		router.statusText({ provider: "openai", id: "gpt-6.1-sol", thinkingLevel: "xhigh", tier: "frontier", sticky: false, escalated: false, solverRouted: true }),
		"gpt-6.1-sol:xhigh [named]",
	);
});

// --- F16: compaction summaries must not trigger the named-solver rule --------


const COMPACT_HEADER = "The conversation history before this point was compacted into the following summary:";

test("F16: a compaction summary mentioning the solver does NOT route to sol", async () => {
	// Contract: route() passes userIntentText, which drops compaction-summary
	// messages — so the router never sees the summary's own solver mentions.
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "hola", turnText: "" },
		reg(["zhipuai-coding-plan", "glm-5.3-flash"], ["zhipuai-coding-plan", "glm-5.3"], ["openai", "gpt-6.1-sol"]),
	);
	assert.equal(d?.solverRouted, undefined);
	assert.equal(d?.id, "glm-5.3-flash");
});

test("F16: currentTurnText drops compaction-summary messages, keeps user intent", () => {
	const summaryMsg = {
		role: "user",
		content: [{ type: "text", text: `${COMPACT_HEADER}\n\n## Goal\n- router F15 usa gpt-6.1-sol cuando el usuario lo menciona` }],
	};
	const userMsg = { role: "user", content: [{ type: "text", text: "continuá esto con gpt 6.1 sol" }] };
	const asst = { role: "assistant", content: [{ type: "text", text: "ok" }] };
	assert.equal(router.currentTurnText([asst, summaryMsg, userMsg]), "continuá esto con gpt 6.1 sol");
	assert.equal(router.currentTurnText([asst, summaryMsg]), "");
});

test("F16: a genuine user mention still routes to sol after compaction", async () => {
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "hola", turnText: "continuá esto con gpt 6.1 sol" },
		reg(["zhipuai-coding-plan", "glm-5.3-flash"], ["zhipuai-coding-plan", "glm-5.3"], ["openai", "gpt-6.1-sol"]),
	);
	assert.equal(d?.id, "gpt-6.1-sol");
});

// --- F17: compaction runs on the configured model (MiniMax-M3) --------------

test("F17: direct compaction requests route to MiniMax regardless of topic", async () => {
	const d = await router.decideAsync(
		{
			reason: "direct",
			text: "resume: cambiamos la migracion de auth con password y deploy",
			previous: { provider: "zhipuai-coding-plan", id: "glm-5.3", thinkingLevel: "high" },
		},
		FULL,
	);
	assert.equal(d?.provider, "minimax");
	assert.equal(d?.id, "MiniMax-M3");
	assert.equal(d?.thinkingLevel, "medium");
});

test("F17: compaction of a session with secrets stays local", async () => {
	const d = await router.decideAsync(
		{ reason: "direct", text: "resume: el usuario pego api_key=sk-zz99real7key1not2fake3xx y seguimos" },
		FULL,
	);
	assert.equal(d?.id, "Qwen3.8-27B");
	assert.equal(d?.privacyRouted, true);
});

test("F17: compaction model missing from catalog falls back to the direct flow", async () => {
	const noMini = reg(["zhipuai-coding-plan", "glm-5.3-flash"], ["zhipuai-coding-plan", "glm-5.3"], ["llamacpp", "Qwen3.8-27B"]);
	const d = await router.decideAsync({ reason: "direct", text: "resume esto" }, noMini);
	assert.equal(d?.id, "glm-5.3-flash");
});

test("F17: parseConfig accepts a compaction block, rejects empty provider", () => {
	const ok = router.parseConfig({ ladder: LADDER_JSON, compaction: { provider: "minimax", id: "MiniMax-M3" } });
	assert.equal(ok?.compaction?.id, "MiniMax-M3");
	assert.equal(router.parseConfig({ ladder: LADDER_JSON, compaction: { provider: "", id: "x" } }), null);
});

// --- F18: known test fixtures are not secrets; privacy explains itself -------

test("F18: the router's own test fixtures do not trigger privacy", async () => {
	const taskText =
		'revisa decide.test.ts: text "revisa esto: OPENAI_API_KEY=sk-abcd1234efgh5678" y el fixture .env con DB_PASSWORD=hunter2';
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "xhigh", text: taskText, turnText: taskText },
		reg(["llamacpp", "Qwen3.8-27B"], ["zhipuai-coding-plan", "glm-5.3-flash"], ["zhipuai-coding-plan", "glm-5.3"], ["openai", "gpt-6.1-sol"]),
	);
	assert.equal(d?.privacyRouted, undefined);
	assert.equal(d?.id, "gpt-6.1-sol");
});

test("F18: real-shaped keys next to the fixtures still trigger privacy", async () => {
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "mi clave: api_key=sk-zz99real7key1not2fake3xx" , turnText: "x" },
		reg(["llamacpp", "Qwen3.8-27B"], ["zhipuai-coding-plan", "glm-5.3-flash"]),
	);
	assert.equal(d?.privacyRouted, true);
});

test("F18: hasSensitiveDataWhy names the matched rule", () => {
	assert.equal(router.hasSensitiveDataWhy("postgres://admin:s3creto@db/prod"), null); // fixture URL
	assert.equal(router.hasSensitiveDataWhy("postgres://admin:RealPass99@db/prod"), "literal");
	assert.equal(router.hasSensitiveDataWhy("token=abcdefgh1234"), "assignment");
	assert.equal(router.hasSensitiveDataWhy("A_KEY=1\nB_KEY=2"), "env-bulk");
	assert.equal(router.hasSensitiveDataWhy("hola mundo"), null);
});

// --- F19: env-bulk counts only secret-named variables ------------------------

test("F19: a benign config listing (PI_MODEL, flags) is NOT env-bulk privacy", () => {
	const listing = "PI_MODEL=glm-5.3\nPI_PROVIDER=zhipuai-coding-plan\nPI_WARDEN_ENABLED=1\nPI_CACHE_RETENTION=long";
	assert.equal(router.hasSensitiveDataWhy(listing), null);
});

test("F19: a real .env paste with secret-named vars still triggers privacy", () => {
	const env = "DB_PASSWORD=realpass99\nSTRIPE_TOKEN=sk-zz88real7key1not2fake4yy";
	// values >= 4 chars also match the assignment rule; detection is what matters
	assert.ok(router.hasSensitiveDataWhy(env) !== null);
	// short values isolate the bulk rule itself
	assert.equal(router.hasSensitiveDataWhy("DB_PASSWORD=1\nX_TOKEN=2"), "env-bulk");
});

test("F19: a session summary documenting config routes normally (not Qwen)", async () => {
	const summary = "The conversation history before this point was compacted:\n\nAjustes: PI_MODEL=glm-5.3 y MINIMAX_API_KEY en ~/.zshrc; PI_WARDEN_ENABLED=1. Router OK.";
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: summary, turnText: "seguimos con el trabajo" },
		reg(["llamacpp", "Qwen3.8-27B"], ["zhipuai-coding-plan", "glm-5.3-flash"], ["zhipuai-coding-plan", "glm-5.3"]),
	);
	assert.equal(d?.privacyRouted, undefined);
	assert.equal(d?.id, "glm-5.3-flash");
});

// --- F21: independent-verification findings (privacy fail-closed, precision) --

test("F21: secrets fail CLOSED when the privacy model is unavailable (user)", async () => {
	const noLocal = reg(["zhipuai-coding-plan", "glm-5.3-flash"], ["zhipuai-coding-plan", "glm-5.3"]);
	const d = await router.decideAsync({ reason: "user", virtualLevel: "medium", text: "password=RivertonExample47!" }, noLocal);
	assert.equal(d, null);
});

test("F21: secrets fail CLOSED when the privacy model is unavailable (compaction)", async () => {
	const noLocal = reg(["zhipuai-coding-plan", "glm-5.3-flash"], ["zhipuai-coding-plan", "glm-5.3"]);
	const d = await router.decideAsync({ reason: "direct", text: "resume: password=RivertonExample47!" }, noLocal);
	assert.equal(d, null);
});

test("F21: quoted JSON credentials are detected", async () => {
	const d = await router.decideAsync({ reason: "user", virtualLevel: "medium", text: '{"password":"RivertonExample47!"}' }, FULL);
	assert.equal(d?.privacyRouted, true);
});

test("F21: a TS interface field is NOT a secret", async () => {
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "high", text: "Implement this interface: { token: string; }" },
		FULL,
	);
	assert.equal(d?.privacyRouted, undefined);
	assert.equal(d?.id, "glm-5.3");
});

test("F21: benign KEYBOARD_ env vars are not secret-named", () => {
	assert.equal(router.hasSensitiveDataWhy("KEYBOARD_LAYOUT=us\nKEYBOARD_VARIANT=intl"), null);
});

test("F21: a connection-failed privacy model is not re-selected on retry", async () => {
	const d = await router.decideAsync(
		{
			reason: "retry",
			text: "password=RivertonExample47!",
			failed: { provider: "llamacpp", id: "Qwen3.8-27B", errorMessage: "fetch failed: ECONNREFUSED" },
		},
		FULL,
	);
	assert.equal(d, null);
});

test("F21: omitted turnText falls back to the full text signal (documented contract)", async () => {
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "Use gpt-6.1-sol for this task" },
		reg(["zhipuai-coding-plan", "glm-5.3-flash"], ["zhipuai-coding-plan", "glm-5.3"], ["openai", "gpt-6.1-sol"]),
	);
	assert.equal(d?.id, "gpt-6.1-sol");
});

// --- F22: current-turn heuristics + mechanical git/exec operations -----------

test("F22: a short commit/push/merge turn routes to the dev-light tier", () => {
	const d = router.decide({ reason: "user", virtualLevel: "medium", text: "commiteá y pusheá estos cambios" }, FULL);
	assert.equal(d?.tier, "flash");
	assert.equal(d?.id, "glm-5.3-flash");
	assert.equal(d?.predictedWhy, "mechanical");
});

test("F22: history risk words do not tier a clean turn — the current turn decides", async () => {
	const d = await router.decideAsync(
		{
			reason: "user",
			virtualLevel: "medium",
			text: "The conversation history: migracion de auth con password y deploy en produccion",
			turnText: "commiteá y pusheá estos cambios",
		},
		FULL,
	);
	assert.equal(d?.tier, "flash");
	assert.equal(d?.predictedWhy, "mechanical");
});

test("F22: an explicit deploy request still wins over mechanical vocabulary", () => {
	const d = router.decide({ reason: "user", virtualLevel: "medium", text: "commiteá y desplegá el release a produccion" }, FULL);
	assert.equal(d?.tier, "frontier");
});

test("F22: complex vocabulary blocks the mechanical fast path", () => {
	const d = router.decide({ reason: "user", virtualLevel: "medium", text: "debuggeá el merge que rompe el build" }, FULL);
	assert.notEqual(d?.predictedWhy, "mechanical");
});

test("F22: long mechanical-looking text takes the scoring path", () => {
	const d = router.decide({ reason: "user", virtualLevel: "medium", text: "commiteá. " + "contexto del problema ".repeat(80) }, FULL);
	assert.notEqual(d?.predictedWhy, "mechanical");
});

test("F22: a risk-worded CURRENT turn still routes to frontier (turn text is the signal)", () => {
	const d = router.decide(
		{ reason: "user", virtualLevel: "medium", text: "x", turnText: "revisá la migracion de la tabla antes del deploy" },
		FULL,
	);
	assert.equal(d?.tier, "frontier");
	assert.equal(d?.predictedWhy, "hard-rule:risk");
});

// ---------------------------------------------------------------------------
// F23 — signal hygiene: harness-injected text is never user intent.
// Evidence: a gentle-jobs notice quoting "-DCMAKE_BUILD_TYPE=Release" matched
// the RISK regex (release\w*) and locked a whole session to frontier.
// ---------------------------------------------------------------------------

const SOLVER_REG = reg(
	["llamacpp", "Qwen3.8-27B"],
	["zhipuai-coding-plan", "glm-5.3-flash"],
	["zhipuai-coding-plan", "glm-5.3"],
	["minimax", "MiniMax-M3"],
	["openai", "gpt-6.1-sol"],
);

test("F23: isHarnessNoise matches the four observed harness-injected shapes", () => {
	assert.equal(router.isHarnessNoise("[审批: ⚡ ACT] 完全权限"), true);
	assert.equal(router.isHarnessNoise("Background job job-1 (\"Compilar\") exited with code 0 after 2m 13s."), true);
	assert.equal(router.isHarnessNoise("pi-warden: your recent replies read as unexplained technical terms"), true);
	assert.equal(router.isHarnessNoise("Review the delivered subagent output and continue."), true);
	assert.equal(router.isHarnessNoise("Hazlo"), false);
	assert.equal(router.isHarnessNoise("Revisá esta migración de auth antes del deploy"), false);
});

test("F23: userTextSignal keeps authored text, drops harness noise, keeps compaction (privacy)", () => {
	const msgs = [
		{ role: "user", content: "Analiza en mi sistema qué está comiendo recursos" },
		{ role: "assistant", content: "listo" },
		{ role: "user", content: "[审批: ⚡ ACT] 完全权限" },
		{ role: "user", content: "Background job job-1 exited. Command: cmake -DCMAKE_BUILD_TYPE=Release" },
		{ role: "user", content: "The conversation history before this point was compacted into this summary." },
		{ role: "user", content: "Hazlo" },
	];
	const signal = router.userTextSignal(msgs);
	assert.ok(signal.includes("Analiza en mi sistema"));
	assert.ok(signal.includes("Hazlo"));
	// compaction stays in the cumulative signal: privacy (R6) must still see it
	assert.ok(signal.includes("was compacted"));
	assert.ok(!signal.includes("Background job"));
	assert.ok(!signal.includes("Release"));
	assert.ok(!signal.includes("审批"));
});

test("F23: currentTurnText drops harness noise from the turn window", () => {
	const msgs = [
		{ role: "user", content: "primer mensaje" },
		{ role: "assistant", content: "respuesta" },
		{ role: "user", content: "pi-warden: Possible prompt injection notice" },
		{ role: "user", content: "Hazlo" },
	];
	assert.equal(router.currentTurnText(msgs), "Hazlo");
});

test("F23: a job notice quoting risk words never locks the tier (the 'Release' case)", () => {
	const msgs = [
		{ role: "user", content: "Analiza en mi sistema que es lo que está comiendo recursos de mi hadware" },
		{ role: "assistant", content: "respuesta previa" },
		{ role: "user", content: 'Background job job-1 ("Compilar") exited with code 0. Command: cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc' },
		{ role: "user", content: "Review the delivered subagent output and continue." },
	];
	const text = router.userTextSignal(msgs);
	const turnText = router.currentTurnText(msgs);
	const d = router.decide({ reason: "user", virtualLevel: "medium", text, turnText }, FULL);
	assert.notEqual(d?.predictedWhy, "hard-rule:risk");
	assert.notEqual(d?.tier, "frontier");
});

test("F23: authored risk still routes frontier (noise filter is not a risk amnesty)", () => {
	const d = router.decide(
		{ reason: "user", virtualLevel: "medium", text: "Revisá esta migración de auth y password antes del deploy" },
		FULL,
	);
	assert.equal(d?.predictedWhy, "hard-rule:risk");
	assert.equal(d?.tier, "frontier");
});

test("F23: contaminated cumulative signal no longer skips the classifier band", async () => {
	let calls = 0;
	const stub = {
		source: "stub",
		classifyComplexity: async () => { calls++; return "standard" as const; },
	};
	const msgs = [
		{ role: "user", content: "Baja qwen y vuelvelo a levantar" },
		{ role: "assistant", content: "ok" },
		{ role: "user", content: 'Background job job-1 exited. Command: cmake -DCMAKE_BUILD_TYPE=Release' },
		{ role: "user", content: "seguí con el ajuste del script" },
	];
	const text = router.userTextSignal(msgs);
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text, turnText: router.currentTurnText(msgs) },
		FULL,
		router.DEFAULT_CONFIG,
		stub,
	);
	assert.equal(calls, 1, "classifier must be consulted once the signal is clean");
	assert.equal(d?.classifierAnswer, "standard");
	assert.equal(d?.tier, "flash");
});

test("F23: classifier probabilities ride on the decision (F9 becomes measurable)", async () => {
	const stub = {
		source: "stub",
		classifyWithProbabilities: async () => ({
			label: "simple" as const,
			probabilities: { simple: 0.42, standard: 0.33, complex: 0.25 },
		}),
	};
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "cambiá el nombre de esta variable", turnText: "cambiá el nombre de esta variable" },
		FULL,
		router.DEFAULT_CONFIG,
		stub,
	);
	assert.equal(d?.classifierAnswer, "simple");
	assert.deepEqual(d?.classifierProbabilities, { simple: 0.42, standard: 0.33, complex: 0.25 });
});

test("F23: solver-named turn fills the prediction logging fields", async () => {
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "continuá con gpt 6.1 sol: respondé", turnText: "continuá con gpt 6.1 sol: respondé" },
		SOLVER_REG,
	);
	assert.equal(d?.solverRouted, true);
	assert.ok(typeof d?.predictedWhy === "string" && d.predictedWhy.length > 0, "predictedWhy must be logged");
	assert.ok(typeof d?.predictedTier === "string");
});

test("F23: xhigh solver turn fills the prediction logging fields", () => {
	const d = router.decide(
		{ reason: "user", virtualLevel: "xhigh", text: "Independent read-only verification", turnText: "Independent read-only verification" },
		SOLVER_REG,
	);
	assert.equal(d?.id, "gpt-6.1-sol");
	assert.ok(typeof d?.predictedWhy === "string" && d.predictedWhy.length > 0, "predictedWhy must be logged");
});

test("F23: direct (compaction) turn fills the prediction logging fields", async () => {
	const d = await router.decideAsync(
		{ reason: "direct", virtualLevel: "medium", text: "The conversation history before this point was compacted. Palabras varias.", turnText: "" },
		FULL,
	);
	assert.equal(d?.id, "MiniMax-M3");
	assert.ok(typeof d?.predictedTier === "string", "predictedTier must be logged");
});

// F24 — vision-intent: catch plain-text image references that R4 misses
// (text typing like "analizá esta imagen" or a /tmp path ending in an image
// extension). Privacy and R6 always win; vision is below them in precedence.
test("F24: visionIntent detects path with image extension in the current turn", () => {
	assert.equal(router.visionIntent(""), false);
	assert.equal(router.visionIntent("revisá el log"), false);
	assert.equal(router.visionIntent("/tmp/pi-clipboard-6efba440.png"), true);
	assert.equal(router.visionIntent("analizá esta imagen: /tmp/x.jpg"), true);
	assert.equal(router.visionIntent("/home/andres/Documentos/a.pdf"), false);
	assert.equal(router.visionIntent("C:\\Users\\me\\pic.PNG"), true);
});

test("F24: solve() without filename argument returns falsy", async () => {
	const p = router.visionPathname("revisá el log de error");
	assert.equal(p, null);
});

test("F24: vision rules below privacy — a secrets turn never routes to vision", async () => {
	const stub = {
		source: "stub",
		classifyComplexity: async () => "complex" as const,
	};
	const d = await router.decideAsync(
		{
			reason: "user",
			virtualLevel: "medium",
			text: 'api_key=AKIA1234567890ABCDEF and image /tmp/x.png',
			turnText: 'api_key=AKIA1234567890ABCDEF and image /tmp/x.png',
		},
		FULL,
		router.DEFAULT_CONFIG,
		stub,
	);
	assert.equal(d?.privacyRouted, true, "R6 must outrank F24 vision");
});

// F25 — inline level hint: ":xhigh" typed in the message text selects the
// level when the picker is still on the default medium.
test("F25: levelHint parses :low/:high/:xhigh/:max from the turn text", () => {
	assert.equal(router.levelHint(":xhigh ajusta el pipeline"), "xhigh");
	assert.equal(router.levelHint("arreglá esto :max"), "max");
	assert.equal(router.levelHint(":low resumí"), "low");
	assert.equal(router.levelHint("sin hint acá"), null);
	assert.equal(router.levelHint("puerto :8090 no es nivel"), null);
});

test("F25: inline :xhigh routes to the solver when the picker stays on medium", () => {
	const d = router.decide(
		{ reason: "user", virtualLevel: "medium", text: ":xhigh ajustá el pipeline", turnText: ":xhigh ajustá el pipeline" },
		SOLVER_REG,
	);
	assert.equal(d?.id, "gpt-6.1-sol");
	assert.equal(d?.thinkingLevel, "xhigh");
});

test("F25: an explicit picker level beats the inline hint", () => {
	const d = router.decide(
		{ reason: "user", virtualLevel: "low", text: ":xhigh ajustá el pipeline", turnText: ":xhigh ajustá el pipeline" },
		SOLVER_REG,
	);
	assert.equal(d?.tier, "local", "picker :low wins over the text hint");
});

// --- F26: symmetric floor on the standard/complex boundary -------------------

const F26_CONFIG = { ...router.DEFAULT_CONFIG, classifier: { mode: "auto" as const, complexFloor: 0.4 } };

test("F26: standard argmax with complex-prob >= floor promotes to frontier", async () => {
	const c = fakeClassifier("standard", { probs: { simple: 0.1, standard: 0.45, complex: 0.42 } });
	const d = await router.decideAsync({ reason: "user", virtualLevel: "medium", text: AMBIGUOUS }, FULL, F26_CONFIG, c);
	assert.equal(d?.tier, "frontier");
});

test("F26: standard argmax with complex-prob at the floor promotes", async () => {
	const c = fakeClassifier("standard", { probs: { simple: 0.2, standard: 0.4, complex: 0.4 } });
	const d = await router.decideAsync({ reason: "user", virtualLevel: "medium", text: AMBIGUOUS }, FULL, F26_CONFIG, c);
	assert.equal(d?.tier, "frontier");
});

test("F26: standard argmax with low complex-prob stays flash", async () => {
	const c = fakeClassifier("standard", { probs: { simple: 0.25, standard: 0.6, complex: 0.15 } });
	const d = await router.decideAsync({ reason: "user", virtualLevel: "medium", text: AMBIGUOUS }, FULL, F26_CONFIG, c);
	assert.equal(d?.tier, "flash");
});

test("F26: a simple promotion does NOT chain to frontier (F9 semantics preserved)", async () => {
	const c = fakeClassifier("simple", { probs: { simple: 0.5, standard: 0.05, complex: 0.45 } });
	const d = await router.decideAsync({ reason: "user", virtualLevel: "medium", text: AMBIGUOUS }, FULL, F26_CONFIG, c);
	assert.equal(d?.tier, "flash");
});

test("F26: floor 0 disables the standard promotion too", async () => {
	const off = { ...router.DEFAULT_CONFIG, classifier: { mode: "auto" as const, complexFloor: 0 } };
	const c = fakeClassifier("standard", { probs: { simple: 0.3, standard: 0.32, complex: 0.38 } });
	const d = await router.decideAsync({ reason: "user", virtualLevel: "medium", text: AMBIGUOUS }, FULL, off, c);
	assert.equal(d?.tier, "flash");
});

// --- F27: privacy keys on the turn; only real shapes hold the session -------

test("F27: an assignment-shaped string in the SUMMARY does not hijack a clean turn", async () => {
	const d = await router.decideAsync(
		{
			reason: "user",
			virtualLevel: "medium",
			text: "resumen: probamos con password=RivertonExample47! y funciono",
			turnText: "seguí con el endpoint de plantillas",
		},
		FULL,
	);
	assert.equal(d?.privacyRouted, undefined);
	assert.equal(d?.id, "glm-5.3-flash");
});

test("F27: a real key SHAPE in history still keeps the session local", async () => {
	const d = await router.decideAsync(
		{
			reason: "user",
			virtualLevel: "medium",
			text: "resumen: el usuario pego sk-zz99real7key1not2fake3xx ayer",
			turnText: "seguí con el endpoint de plantillas",
		},
		FULL,
	);
	assert.equal(d?.privacyRouted, true);
	assert.equal(d?.privacyWhy, "literal");
});

test("F27: a fresh paste in the turn hijacks as before", async () => {
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "x", turnText: "revisá: password=RivertonExample47!" },
		FULL,
	);
	assert.equal(d?.privacyRouted, true);
	assert.equal(d?.privacyWhy, "assignment");
});

test("F27: env-bulk in the summary alone does not hijack a clean turn", async () => {
	const d = await router.decideAsync(
		{
			reason: "user",
			virtualLevel: "medium",
			text: "resumen:\nDB_PASSWORD=PassReal88\nSTRIPE_TOKEN=tok_zz77NoRealValue99",
			turnText: "listo, seguimos con otra cosa",
		},
		FULL,
	);
	assert.equal(d?.privacyRouted, undefined);
});

// --- F28: context-aware routing — the model must fit the conversation -------

const CTX_LADDER = router.parseConfig({
	ladder: [
		{ provider: "zhipuai-coding-plan", id: "glm-5.3-flash", tier: "local" },
		{ provider: "zhipuai-coding-plan", id: "glm-5.3-flash", tier: "flash" },
		{ provider: "zhipuai-coding-plan", id: "glm-5.3", tier: "frontier" },
	],
	classifier: { mode: "off" },
	privacy: { provider: "llamacpp", id: "Qwen3.8-27B" },
});
function ctxReg(): router.RegistryLike {
	return reg(
		["zhipuai-coding-plan", "glm-5.3-flash", 1_000_000],
		["zhipuai-coding-plan", "glm-5.3", 1_000_000],
		["llamacpp", "Qwen3.8-27B", 131_072],
	);
}
const SMALL_LOCAL = () => router.parseConfig({
	ladder: [
		{ provider: "llamacpp", id: "Qwen3.8-27B", tier: "local" },
		{ provider: "zhipuai-coding-plan", id: "glm-5.3-flash", tier: "flash" },
		{ provider: "zhipuai-coding-plan", id: "glm-5.3", tier: "frontier" },
	],
	classifier: { mode: "off" },
})!;

test("F28: a huge conversation skips models whose context window cannot fit it", async () => {
	const cat = reg(["llamacpp", "Qwen3.8-27B", 131_072], ["zhipuai-coding-plan", "glm-5.3", 1_000_000]);
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "hola", turnText: "hola", estimatedTokens: 400_000 },
		cat,
		SMALL_LOCAL(),
	);
	assert.equal(d?.id, "glm-5.3");
});

test("F28: a small conversation keeps normal tiering", async () => {
	const cat = reg(["llamacpp", "Qwen3.8-27B", 131_072], ["zhipuai-coding-plan", "glm-5.3", 1_000_000]);
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "hola", turnText: "hola", estimatedTokens: 5_000 },
		cat,
		SMALL_LOCAL(),
	);
	assert.equal(d?.id, "Qwen3.8-27B");
});

test("F28: secrets in a conversation too big for the privacy model fail closed", async () => {
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "password=RivertonExample47!", turnText: "password=RivertonExample47!", estimatedTokens: 400_000 },
		ctxReg(),
		CTX_LADDER!,
	);
	assert.equal(d, null);
});

test("F28: a context-overflow retry escalates to the largest-context model", async () => {
	const cat = reg(["llamacpp", "Qwen3.8-27B", 131_072], ["zhipuai-coding-plan", "glm-5.3-flash", 200_000], ["zhipuai-coding-plan", "glm-5.3", 1_000_000]);
	const d = await router.decideAsync(
		{
			reason: "retry",
			text: "por que tan pocas hojas por minuto",
			turnText: "por que",
			failed: { provider: "llamacpp", id: "Qwen3.8-27B", errorMessage: '400: {"code":400,"message":"request (435911 tokens) exceeds the available context size (162304)","type":"exceed_context_size_error"}' },
		},
		cat,
		SMALL_LOCAL(),
	);
	assert.equal(d?.id, "glm-5.3");
});

test("F28: overflow retry when nothing else fits fails visible (null)", async () => {
	const cat = reg(["zhipuai-coding-plan", "glm-5.3-flash", 200_000], ["zhipuai-coding-plan", "glm-5.3", 1_000_000]);
	const d = await router.decideAsync(
		{
			reason: "retry",
			text: "seguimos",
			turnText: "seguimos",
			estimatedTokens: 500_000,
			failed: { provider: "zhipuai-coding-plan", id: "glm-5.3", errorMessage: "400 exceed_context_size_error: request exceeds the available context size" },
		},
		cat,
		CTX_LADDER!,
	);
	assert.equal(d, null);
});

// --- F29: cache hysteresis — voluntary switches must pay for themselves ------
// A prompt-cache refill at the destination scales with conversation size, so
// weak tier flips on big conversations ride the previous model; strong
// signals (risk rule, confident classifier, explicit level) still switch.

const CLF_ON = () => router.parseConfig({
	ladder: [
		{ provider: "llamacpp", id: "Qwen3.8-27B", tier: "local" },
		{ provider: "zhipuai-coding-plan", id: "glm-5.3-flash", tier: "flash" },
		{ provider: "zhipuai-coding-plan", id: "glm-5.3", tier: "frontier" },
	],
	classifier: { mode: "auto", url: "http://127.0.0.1:1", timeoutMs: 100 },
})!;
const PREV_GLM = { provider: "zhipuai-coding-plan", id: "glm-5.3", thinkingLevel: "max" };
const PREV_QWEN = { provider: "llamacpp", id: "Qwen3.8-27B", thinkingLevel: "high" };

test("F29: a weak flip on a big conversation holds the previous model", async () => {
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "hola", turnText: "hola", estimatedTokens: 100_000, previous: PREV_GLM },
		reg(["llamacpp", "Qwen3.8-27B", 131_072], ["zhipuai-coding-plan", "glm-5.3", 1_000_000]),
		SMALL_LOCAL(),
	);
	assert.equal(d?.id, "glm-5.3");
	assert.equal(d?.why, "cache");
	assert.equal(d?.sticky, true);
});

test("F29: a small conversation switches freely", async () => {
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "hola", turnText: "hola", estimatedTokens: 5_000, previous: PREV_GLM },
		reg(["llamacpp", "Qwen3.8-27B", 131_072], ["zhipuai-coding-plan", "glm-5.3", 1_000_000]),
		SMALL_LOCAL(),
	);
	assert.equal(d?.id, "Qwen3.8-27B");
	assert.notEqual(d?.why, "cache");
});

test("F29: a RISK turn switches even on a big conversation", async () => {
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "hay que hacer el deploy y release a produccion", turnText: "deploy y release a produccion", estimatedTokens: 100_000, previous: PREV_QWEN },
		reg(["llamacpp", "Qwen3.8-27B", 131_072], ["zhipuai-coding-plan", "glm-5.3", 1_000_000]),
		SMALL_LOCAL(),
	);
	assert.equal(d?.id, "glm-5.3");
	assert.notEqual(d?.why, "cache");
});

test("F29: mechanical turns on a big conversation stay on the sticky cloud model", async () => {
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "commiteá todo", turnText: "commiteá todo", estimatedTokens: 100_000, previous: PREV_GLM },
		reg(["llamacpp", "Qwen3.8-27B", 131_072], ["zhipuai-coding-plan", "glm-5.3-flash", 1_000_000], ["zhipuai-coding-plan", "glm-5.3", 1_000_000]),
		SMALL_LOCAL(),
	);
	assert.equal(d?.id, "glm-5.3");
	assert.equal(d?.why, "cache");
});

test("F29: a confident classifier still escalates on a big conversation", async () => {
	const classifier = fakeClassifier("standard", { probs: { simple: 0.05, standard: 0.05, complex: 0.9 } });
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: AMBIGUOUS, estimatedTokens: 100_000, previous: PREV_QWEN },
		reg(["llamacpp", "Qwen3.8-27B", 131_072], ["zhipuai-coding-plan", "glm-5.3", 1_000_000]),
		CLF_ON(),
		classifier,
	);
	assert.equal(d?.id, "glm-5.3");
	assert.notEqual(d?.why, "cache");
});

test("F29: a borderline classifier promotion is held on a big conversation", async () => {
	const classifier = fakeClassifier("standard", { probs: { simple: 0.45, standard: 0.05, complex: 0.5 } });
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: AMBIGUOUS, estimatedTokens: 100_000, previous: PREV_QWEN },
		reg(["llamacpp", "Qwen3.8-27B", 131_072], ["zhipuai-coding-plan", "glm-5.3", 1_000_000]),
		CLF_ON(),
		classifier,
	);
	assert.equal(d?.id, "Qwen3.8-27B");
	assert.equal(d?.why, "cache");
});

test("F29: an explicit virtual level wins over the cache hold", async () => {
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "high", text: "hola", turnText: "hola", estimatedTokens: 100_000, previous: PREV_QWEN },
		reg(["llamacpp", "Qwen3.8-27B", 131_072], ["zhipuai-coding-plan", "glm-5.3", 1_000_000]),
		SMALL_LOCAL(),
	);
	assert.equal(d?.id, "glm-5.3");
	assert.notEqual(d?.why, "cache");
});

test("F29: an off-ladder previous model cannot be held", async () => {
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "hola", turnText: "hola", estimatedTokens: 100_000, previous: { provider: "minimax", id: "MiniMax-M3" } },
		reg(["llamacpp", "Qwen3.8-27B", 131_072], ["zhipuai-coding-plan", "glm-5.3", 1_000_000], ["minimax", "MiniMax-M3", 1_000_000]),
		SMALL_LOCAL(),
	);
	assert.equal(d?.id, "Qwen3.8-27B");
	assert.notEqual(d?.why, "cache");
});

test("F29: a previous model that cannot fit the conversation is not held (F28 wins)", async () => {
	const d = await router.decideAsync(
		{ reason: "user", virtualLevel: "medium", text: "hola", turnText: "hola", estimatedTokens: 400_000, previous: PREV_QWEN },
		reg(["llamacpp", "Qwen3.8-27B", 131_072], ["zhipuai-coding-plan", "glm-5.3", 1_000_000]),
		SMALL_LOCAL(),
	);
	assert.equal(d?.id, "glm-5.3");
	assert.notEqual(d?.why, "cache");
});

// --- F30: tier semantics — local is CHAT-ONLY; dev-light rides flash -------

test("F30: a low-score DEV turn rides the flash floor, not local", () => {
	const d = router.decide({ reason: "user", virtualLevel: "medium", text: "Revisemos los jobs como van en aws" }, FULL);
	assert.equal(d?.tier, "flash");
});

test("F30: chat turns with smalltalk markers stay local", () => {
	const d = router.decide({ reason: "user", virtualLevel: "medium", text: "hola, cómo andás? todo bien por ahí?" }, FULL);
	assert.equal(d?.tier, "local");
	assert.equal(d?.predictedWhy, "chat");
});

test("F30: thanks/acks stay local", () => {
	const d = router.decide({ reason: "user", virtualLevel: "medium", text: "gracias, perfecto" }, FULL);
	assert.equal(d?.tier, "local");
});

test("F30: a general-opinion question stays local", () => {
	const d = router.decide({ reason: "user", virtualLevel: "medium", text: "qué opinás del clima hoy para salir a caminar?" }, FULL);
	assert.equal(d?.tier, "local");
});

test("F30: dev chores (typo/renames) ride flash", () => {
	const d = router.decide({ reason: "user", virtualLevel: "medium", text: "renombrar esta variable a usuarioId, typo fix" }, FULL);
	assert.equal(d?.tier, "flash");
});

test("F30: classifier simple with chat markers stays local", async () => {
	const c = fakeClassifier("simple");
	const d = await router.decideAsync({ reason: "user", virtualLevel: "medium", text: "hola, cómo andás? todo bien?" }, FULL, router.DEFAULT_CONFIG, c);
	assert.equal(d?.tier, "local");
});

test("F30: classifier simple on a DEV turn rides flash", async () => {
	const c = fakeClassifier("simple");
	const d = await router.decideAsync({ reason: "user", virtualLevel: "medium", text: "fijate el log del error del job en aws" }, FULL, router.DEFAULT_CONFIG, c);
	assert.equal(d?.tier, "flash");
});

test("F30: explicit virtual low still forces local on a dev turn", () => {
	const d = router.decide({ reason: "user", virtualLevel: "low", text: "fijate el log del error del job en aws" }, FULL);
	assert.equal(d?.tier, "local");
});
