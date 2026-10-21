/**
 * F3 calibration report for the router decision log.
 *
 * Run: node calibrate.ts          (prints the report for decisions.jsonl)
 * Test: node --test calibrate.test.ts
 *
 * The bands themselves stay in index.ts; this tool only summarizes the
 * evidence so future calibration rounds have a repeatable starting point.
 */
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { COMPACT_MARK } from "./index.ts";

const LOG_PATH = join(homedir(), ".pi", "router-poc", "decisions.jsonl");

export interface Summary {
	total: number;
	byReason: Record<string, number>;
	/** predictedTier -> count, user requests only ("n/a" when absent). */
	userPredicted: Record<string, number>;
	/** F23: user-reason turns that were compaction summaries — system text,
	 * not user intent; excluded from userPredicted and forcedMismatch. */
	userCompaction: number;
	userForcedMismatch: number;
	escalated: number;
	unresolved: number;
	configFile: number;
	configDefault: number;
	/** F4/F5: how many decisions the classifier refined, per label. */
	classifierConsulted: number;
	classifierAnswers: Record<string, number>;
}

/** Aggregate a decisions.jsonl entry list. Pure; tolerates missing fields. */
export function summarize(entries: readonly unknown[]): Summary {
	const s: Summary = {
		total: 0,
		byReason: {},
		userPredicted: {},
		userCompaction: 0,
		userForcedMismatch: 0,
		escalated: 0,
		unresolved: 0,
		configFile: 0,
		configDefault: 0,
		classifierConsulted: 0,
		classifierAnswers: {},
	};
	for (const raw of entries) {
		const e = (raw ?? {}) as Record<string, unknown>;
		s.total++;
		const reason = typeof e.reason === "string" ? e.reason : "unknown";
		s.byReason[reason] = (s.byReason[reason] ?? 0) + 1;
		if (e.escalated === true) s.escalated++;
		if (e.routedTo === "UNRESOLVED") s.unresolved++;
		if (e.configSource === "file") s.configFile++;
		else if (e.configSource === "default") s.configDefault++;
		if (typeof e.classifierAnswer === "string") {
			s.classifierConsulted++;
			s.classifierAnswers[e.classifierAnswer] = (s.classifierAnswers[e.classifierAnswer] ?? 0) + 1;
		}
		if (reason === "user") {
			const head = typeof e.textHead === "string" ? e.textHead : "";
			if (head.startsWith(COMPACT_MARK)) {
				// F23: compaction summaries ride the user reason but are not user
				// intent — they calibrate nothing.
				s.userCompaction++;
				continue;
			}
			const pt = typeof e.predictedTier === "string" ? e.predictedTier : "n/a";
			s.userPredicted[pt] = (s.userPredicted[pt] ?? 0) + 1;
			if (e.forcedMismatch === true) s.userForcedMismatch++;
		}
	}
	return s;
}

function loadEntries(): unknown[] {
	try {
		return readFileSync(LOG_PATH, "utf8")
			.split("\n")
			.filter((line: string) => line.trim() !== "")
			.map((line: string) => {
				try {
					return JSON.parse(line);
				} catch {
					return { reason: "unparseable" };
				}
			});
	} catch {
		return [];
	}
}

function report(): void {
	const entries = loadEntries();
	const s = summarize(entries);
	console.log(`== Router calibration report (${LOG_PATH}) ==`);
	console.log(`total=${s.total} byReason=${JSON.stringify(s.byReason)}`);
	console.log(`config: file=${s.configFile} default=${s.configDefault}`);
	console.log(`user predictions: ${JSON.stringify(s.userPredicted)} forcedMismatch=${s.userForcedMismatch} compactionTurns=${s.userCompaction}`);
	console.log(`escalated=${s.escalated} unresolved=${s.unresolved}`);
	console.log(`classifier: consulted=${s.classifierConsulted} ${JSON.stringify(s.classifierAnswers)}`);
	console.log("\n-- user requests (last 12) --");
	const users = entries
		.map((raw) => (raw ?? {}) as Record<string, unknown>)
		.filter((e) => e.reason === "user")
		.slice(-12);
	for (const e of users) {
		const ts = typeof e.ts === "string" ? e.ts.slice(5, 16) : "?";
		const pred = `${e.predictedTier ?? "-"}/${e.predictedScore ?? "-"}/${e.predictedWhy ?? "-"}`;
		const mm = e.forcedMismatch === true ? " MISMATCH" : "";
		const len = typeof e.textLen === "number" ? e.textLen : "?";
		const head = typeof e.textHead === "string" ? e.textHead.slice(0, 40).replace(/\n/g, " | ") : "";
		console.log(`${ts} pred=${pred}${mm} len=${String(len).padStart(4)} :: ${head}`);
	}
}

const isMain = process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) report();
