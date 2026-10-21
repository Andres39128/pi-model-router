/**
 * F3 unit tests for the calibration report core `summarize()` in calibrate.ts.
 *
 * Run: node --test calibrate.test.ts
 */
import assert from "node:assert/strict";
import { test } from "node:test";
import * as cal from "./calibrate.ts";

const FIXTURE = [
	{ ts: "t1", reason: "user", predictedTier: "local", predictedScore: 0, forcedMismatch: false, escalated: false, sticky: false, routedTo: "llamacpp/Qwen3.8-27B", configSource: "file" },
	{ ts: "t2", reason: "continuation", sticky: true, escalated: false, routedTo: "llamacpp/Qwen3.8-27B", configSource: "file" },
	{ ts: "t3", reason: "user", predictedTier: "frontier", predictedScore: 99, forcedMismatch: true, escalated: false, sticky: false, routedTo: "llamacpp/Qwen3.8-27B", configSource: "default" },
	{ ts: "t4", reason: "retry", escalated: true, sticky: false, routedTo: "zhipuai-coding-plan/glm-5.3-flash", errSnippet: "Connection error.", configSource: "default" },
	{ ts: "t5", reason: "user", escalated: false, sticky: false, routedTo: "UNRESOLVED", configSource: "default" },
];

test("summarize counts totals, reasons, and config sources", () => {
	const s = cal.summarize(FIXTURE);
	assert.equal(s.total, 5);
	assert.equal(s.byReason.user, 3);
	assert.equal(s.byReason.continuation, 1);
	assert.equal(s.byReason.retry, 1);
	assert.equal(s.configFile, 2);
	assert.equal(s.configDefault, 3);
});

test("summarize counts prediction outcomes on user requests", () => {
	const s = cal.summarize(FIXTURE);
	assert.equal(s.userPredicted.local, 1);
	assert.equal(s.userPredicted.frontier, 1);
	assert.equal(s.userForcedMismatch, 1);
	assert.equal(s.escalated, 1);
	assert.equal(s.unresolved, 1);
});

test("summarize tolerates missing fields and empty input", () => {
	const s = cal.summarize([]);
	assert.equal(s.total, 0);
	assert.equal(s.userForcedMismatch, 0);
	const s2 = cal.summarize([{ reason: "user" }]);
	assert.equal(s2.total, 1);
	assert.equal(s2.userPredicted.local ?? 0, 0);
});

test("summarize counts classifier refinement per label", () => {
	const s = cal.summarize([
		{ reason: "user", classifierAnswer: "complex" },
		{ reason: "user", classifierAnswer: "simple" },
		{ reason: "user", classifierAnswer: "complex" },
		{ reason: "user" },
		{ reason: "continuation" },
	]);
	assert.equal(s.classifierConsulted, 3);
	assert.deepEqual(s.classifierAnswers, { complex: 2, simple: 1 });
});

test("F23: compaction summaries are separated from user predictions", () => {
	const s = cal.summarize([
		{ reason: "user", predictedTier: "local", textHead: "The conversation history before this point was compacted into this summary." },
		{ reason: "user", predictedTier: "frontier", textHead: "Analiza el sistema" },
		{ reason: "user", predictedTier: "local", forcedMismatch: true, textHead: "The conversation history before this point was compacted again." },
	]);
	assert.equal(s.userCompaction, 2, "both compaction turns counted separately");
	assert.equal(s.userPredicted.frontier, 1, "only the real user turn predicts");
	assert.equal(s.userPredicted.local ?? 0, 0, "compaction does not predict");
	assert.equal(s.userForcedMismatch, 0, "compaction never mismatches: it is not user intent");
});
