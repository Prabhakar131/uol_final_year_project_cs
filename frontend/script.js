// CloudIR Trainer frontend.
// Shared state stays global so run_controls.js (saved runs) can read and restore it.
let appState = null;
let turnPayload = null;
let generatedEvidence = [];
let selectedAction = null;
let selectedEvidence = null;
let finalDebrief = null;
let mediaRecorder = null;
let recordedAudioChunks = [];
let recordedAudioBlob = null;
let learnerJustification = null;
let currentReasoningPrompt = null;
let testModeEnabled = false;
let skipNextTurnEnabled = false;
let isEvaluatingTurn = false;
let isContinuingTurn = false;
let isPreparingDataset = false;
let scenarioComplete = false;
let activeRunId = null;
let activeRunMetadata = null;
let activeScenarioId = "identity-management";

const MAX_JUSTIFICATION_RECORDING_MS = 45_000;
const TEST_MODE_STORAGE_KEY = "cloudirTestMode";
const SKIP_NEXT_TURN_STORAGE_KEY = "cloudirSkipNextTurn";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const wait = ms => new Promise(r => setTimeout(r, ms));
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[c]));
const escapeHtml = esc;
const clock = () => new Date().toLocaleTimeString();
const narrow = () => matchMedia("(max-width: 760px)").matches;

// Pages in the order run_controls.js knows them.
const pageEls = Object.fromEntries($$(".page").map(p => [p.dataset.page, p]));
const screens = {
  runs: pageEls.runs, home: pageEls.landing, datasetSelection: pageEls.case, datasetPreparation: pageEls.prep,
  action: pageEls.action, evidence: pageEls.evidence, justification: pageEls.reasoning, evaluation: pageEls.evaluation, end: pageEls.debrief,
};

async function callApi(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      "X-Test-Mode": testModeEnabled ? "1" : "0",
      ...(activeRunId ? { "X-Run-ID": activeRunId } : {}),
      ...(options.headers || {}),
    },
  });
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(data.error || `Request failed with status ${response.status}`);
    error.data = data;
    error.status = response.status;
    throw error;
  }
  return data;
}

function addClickListener(elementId, handler) {
  const element = document.getElementById(elementId);
  if (element) element.addEventListener("click", handler);
}

function parseServerSentEvent(rawEvent) {
  const dataLines = rawEvent.split("\n").filter(line => line.startsWith("data:")).map(line => line.slice(5).trim());
  if (!dataLines.length) return null;
  const dataText = dataLines.join("\n");
  try {
    return JSON.parse(dataText);
  } catch (error) {
    console.error("Failed to parse stream event:", dataText, error);
    return null;
  }
}

// Reads a server-sent event stream and hands each event to onEvent in order.
async function readEventStream(path, body, onEvent) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Accept": "text/event-stream", "X-Test-Mode": testModeEnabled ? "1" : "0", ...(activeRunId ? { "X-Run-ID": activeRunId } : {}) },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;
    try { message = (await response.json()).error || message; } catch { /* not JSON */ }
    throw new Error(message);
  }
  if (!response.body) throw new Error("Streaming response body is not available in this browser.");
  const reader = response.body.getReader(), decoder = new TextDecoder("utf-8");
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) {
      const event = parseServerSentEvent(part);
      if (event) await onEvent(event);
    }
  }
}

// ---------- adapters: server payloads to the shapes the interface draws ----------
const clean = s => String(s ?? "").replace(/\*\*/g, "").replace(/\s+/g, " ").trim();
const stateOf = s => s ? { c: Number(s.containment ?? 0), v: Number(s.visibility ?? 0), risk: s.risk || "Unknown", phase: s.phase || "" } : null;
const maxTurns = () => Number(appState?.maxTurns) || 5;
const turnNo = () => Number(appState?.currentTurn) || 1;
const getActionChoiceRole = a => String(a?.choiceRole || a?.choice_role || "").trim().toLowerCase();
const getEvidenceSupportRole = e => String(e?.supportRole || e?.support_role || "").trim().toLowerCase();
const evidenceTemplate = e => String(e?.template || e?.type || "").toLowerCase();

function adaptVlm(o = {}) {
  const rows = Array.isArray(o.event_rows) ? o.event_rows : null;
  const facts = rows
    ? [...rows.map((r, i) => `Row ${i + 1}: ${r.timestamp ?? "Unreadable time"} · ${r.message ?? "Unreadable message"}${r.truncated ? " [truncated in screenshot]" : ""}`),
       ...(o.log_group ? [`Log group: ${o.log_group}`] : []), ...(o.time_range ? [`Time range: ${o.time_range}`] : []),
       ...(o.matched_records != null ? [`Matched records: ${o.matched_records}`] : [])]
    : (o.visible_facts_extracted || o.visible_facts || o.visibleFacts || []);
  return { type: String(o.evidence_type || "unknown").toLowerCase(), summary: o.visible_evidence_summary || "", facts: facts.map(String),
    warnings: Array.isArray(o.extraction_warnings) ? o.extraction_warnings : [], query: rows ? (o.query_text || "") : "" };
}
function adaptSecurity(o = {}) {
  return { verdict: o.verdict || "Unknown", key: o.key_fact || "", whose: o.whose_evidence || "", reasoning: o.reasoning || "",
    claims: (o.justification_claims || []).map(c => ({ id: c.statement_id, a: c.assessment, q: c.transcript_quote })),
    assessment: o.justification_assessment || "", next: o.recommended_next_focus || "", risk: o.risk_of_wrong_interpretation || "",
    rows: Array.isArray(o.supporting_row_numbers) ? o.supporting_row_numbers : null };
}
const adaptCoach = (o = {}) => ({ feedback: clean(o.feedback), nudge: clean(o.next_turn_guidance) });

// The marking scheme arrives with each turn as expected outcomes.
function schemeOf(turn) {
  const eo = turn?.expectedOutcomes || turn?.expected_outcomes || {}, out = {};
  Object.entries(eo).forEach(([k, v]) => { const m = /^(strong|partial|weak)/i.exec(k); if (m && typeof v === "string") out[m[1].toLowerCase()] = clean(v); });
  return out;
}
const turnConfig = () => turnPayload?.turnConfig || {};
// Turn files keep snake_case keys inside turnConfig.
const knownContext = () => { const c = turnConfig(); return c.known_context || c.knownContext || []; };
const coachGuidance = () => { const c = turnConfig(); return c.coach_guidance || c.coachGuidance || ""; };

// ---------- constants ----------
const STAGES = ["Detection and initial triage", "Correlation and scope", "Scope and blast radius", "Containment and remediation", "Recovery and final debrief"];
const stageName = n => STAGES[n - 1] || `Turn ${n}`;
const PROMPTS = {
  cloudtrail: ["Explain how the CloudTrail event supports your action", "Explain how the CloudTrail event supports your selected response action. Connect the visible log fields to the decision you want to take.", ["event name", "user identity", "source IP", "event time", "MFA status", "region"]],
  guardduty: ["Explain how the finding supports your action", "Explain how the GuardDuty-style finding supports your selected response action. Connect the finding details to the incident response decision.", ["finding type", "severity", "affected identity or resource", "remote IP", "first seen", "last seen"]],
  cloudwatch: ["Explain how the monitoring evidence supports your action", "Explain how the CloudWatch-style evidence supports your selected response action. Connect the metric or anomaly to what you would investigate or contain.", ["metric name", "time range", "anomaly", "threshold", "affected resource"]],
  iam_activity: ["Explain how the IAM activity supports your action", "Explain how the IAM activity evidence supports your selected response action. Connect the identity activity to the response decision.", ["principal", "action", "timestamp", "source IP", "policy change", "authentication context"]],
  access_key: ["Explain how the access key evidence supports your action", "Explain how the access key evidence supports your selected response action. Connect key usage or rotation risk to the response decision.", ["key age", "status", "last used time", "user", "permissions", "rotation risk"]],
  billing: ["Explain how the billing evidence supports your action", "Explain how the billing evidence supports your selected response action. Connect the usage or spend signal to the response decision.", ["unusual spend", "affected service", "time period", "usage spike", "account or region"]],
};
const DEFAULT_PROMPT = ["Explain how this evidence supports your action", "Explain how the selected evidence supports your response action. Refer only to what is visible or described in the evidence.", ["visible field", "identity or resource", "time or source", "risk signal", "what is not proven"]];
// Field regions on each screenshot template, in image pixels: [field, x, y, width|null, height, char width]
const RMAP = {
  cloudtrail: [["event name", 453, 286, null, 28, 13], ["event time", 648, 291, null, 18, 6.6], ["region", 882, 291, null, 18, 6.4], ["user identity", 618, 347, null, 19, 6.6], ["source ip", 618, 377, null, 19, 6.6], ["event source", 618, 407, null, 19, 6.6], ["user agent", 618, 437, null, 19, 6.6], ["recipient account", 618, 467, null, 19, 6.6], ["error code", 614, 497, 24, 19], ["event id", 618, 527, null, 19, 6.6], ["request parameters", 466, 618, null, 20, 6.3]],
  guardduty: [["finding type", 100, 316, 640, 38], ["severity", 882, 300, 190, 48], ["service", 828, 478, null, 18, 7], ["status", 828, 504, null, 18, 7], ["active", 828, 504, 60, 18], ["principal", 214, 499, null, 19, 7], ["resource", 214, 533, null, 19, 7], ["remote ip", 214, 567, null, 19, 7], ["first seen", 214, 601, null, 19, 7], ["last seen", 214, 635, null, 19, 7], ["affected resource", 98, 436, 190, 28]],
  access_key: [["access key id", 86, 440, null, 19, 8], ["status", 356, 440, null, 19, 7.5], ["last used service", 476, 440, null, 19, 7], ["region", 716, 440, null, 19, 6.6], ["last used", 876, 440, null, 19, 6.6], ["owner", 228, 518, null, 19, 7.1], ["source ip", 228, 552, null, 19, 7.1], ["rotation status", 228, 586, null, 19, 7.1], ["console user", 228, 620, null, 19, 7.1]],
  iam_activity: [["principal", 176, 318, 146, 26], ["source ip", 184, 443, null, 19, 7], ["mfa", 184, 397, null, 19, 7], ["console password", 412, 384, 200, 20], ["access key status", 412, 414, 200, 20], ["security credentials", 410, 378, 420, 56], ["recent activit", 390, 465, 340, 22], ["actions", 792, 548, 150, 100], ["event times", 892, 184, 184, 30]],
  billing: [["estimated current spend", 100, 318, 100, 36], ["previous average", 406, 320, 70, 28], ["change", 680, 320, 56, 30], ["region", 906, 320, 140, 30], ["service cost driver", 770, 490, 70, 32], ["largest service", 770, 490, 70, 32]],
  cloudwatch: [["matched records", 66, 224, 170, 24], ["log group", 82, 364, 340, 24], ["time range", 82, 388, 310, 24], ["row 1", 72, 536, 1058, 90], ["row 2", 72, 628, 1058, 90], ["row 3", 72, 720, 1058, 90], ["row 4", 72, 812, 1058, 90]],
};
// The 100 case names in data/acse_eval.jsonl, for the build screen's case reel.
const ACSE_CASES = ["app-runner", "automated-security-response", "aws-ai-chat-bot", "aws-vdi-architecture", "big-data-processing-and-analytics-platform", "biological-compute-platform", "biometric-auth-system", "bird-game", "budgeting-application", "centralized-network-inspection", "cicd-microservices", "code-execution-platform", "connected-vehicle-system", "contact-center-platform", "containerized-microservices", "content-localization", "cost-management", "crm-system", "cryptocurrency", "data-viz-platform", "datazone-arch", "digital-twin-ecosystem", "distributed-cache-system", "distributed-graph-processing", "distributed-load-testing", "distributed-storage", "document-management-system", "efs-filesystem", "eks-outposts", "email-delivery-system", "enhanced-document-understanding", "ev-charging-system", "flight-information-management", "fraud-detection", "game-analytics-pipeline", "gaming-screenshot-processor", "healthcare-system", "hpc-cluster", "hpc-grid-system", "human-data-collection-system", "hybrid-integration", "identity-management", "image-video-recognition-system", "iot-device-management-system", "iot-predictive-maintenance", "kanban-board-system", "live-streaming-platform", "log-analysis-monitoring-system", "ml-platform", "ml-training-platform", "multi-region-active-active-db", "multi-region-backup-archival", "multi-region-consensus-protocol", "multi-region-dr-system", "multi-region-eks", "multi-region-hyperpod", "multiplayer-game-server", "music-streaming-service", "neuromorphic-computing-platform", "opensearch-migration-system", "orgs-assessment-system", "payroll-system", "photo-sharing-app", "pki-private-ca", "push-notification-service", "qa-chatbot", "quantum-computing-cluster", "real-time-collaborative-editing", "realtime-analytics", "realtime-personalization", "recommendation-engine", "renewable-energy-system", "rtb-platform", "sagemaker-platform", "sandboxed-game-emulators", "search-engine", "security-insights-on-aws", "serverless-anomaly-detection-system", "serverless-cicd", "serverless-data-pipeline", "serverless-etl-pipeline", "serverless-event-driven-architecture", "serverless-optimization", "serverless-payment-processing-system", "service-quotas-monitor", "simple-image-upload-download-website", "social-media-backend", "sovereign-cloud-platform", "sports-betting-system", "streaming-media-service", "support-ticketing-system", "temporary-access-system", "text-paste-service", "transit-gateway", "unreal-engine-streaming", "version-control", "virtual-waiting-room", "voice-processing-system", "weather-forecasting-system", "workflow-management"];
// Scenario cards: artwork and copy per known case; any other prepared case gets a plain card.
const CASES = {
  identity: { no: "01", code: "IAM", blurb: "Unusual AssumeRole and access-key activity around audit logging.", tags: ["CloudTrail", "IAM", "GuardDuty", "Access key"],
    art: '<i class="lan"></i><span class="card"><i class="hole"></i><i class="photo"></i><i class="ln a"></i><i class="ln b"></i><i class="ln c"></i><span class="nm">s.brennan</span><i class="bars"></i><i class="scan"></i><span class="bflag">AssumeRole</span></span>' },
  asr: { no: "02", code: "Auto-response", blurb: "A remediation workflow and its Lambda path behave unexpectedly.", tags: ["GuardDuty", "EventBridge", "Lambda"],
    art: '<svg viewBox="0 0 170 150"><path d="M27 29 C90 20 150 40 143 71 C136 110 90 120 57 127"/></svg><span class="node n1">GD</span><span class="node n2">EB</span><span class="node n3">λ</span><i class="pulse"></i>' },
  cost: { no: "03", code: "Billing", blurb: "Spend drifts above the daily average while logs tell part of the story.", tags: ["Billing", "CloudWatch", "CloudTrail"],
    art: [.42, .47, .4, .46, .44, .45, .92].map((h, i) => `<span class="cbar${i === 6 ? " spike" : ""}" style="--h:${h}"></span>`).join("") + '<i class="avg"></i><span class="tag">+13%</span>' },
};
const ART = { identity: "art-badge", asr: "art-flow", cost: "art-chart" };
const caseKey = id => /identity/.test(id) ? "identity" : /security-response|asr/.test(id) ? "asr" : /cost/.test(id) ? "cost" : "identity";
const titleCase = id => { const s = String(id || "").replace(/[-_]+/g, " ").trim(); return s ? s[0].toUpperCase() + s.slice(1) : "Case"; };
const archUrl = id => `/dataset-assets/${encodeURIComponent(id)}/architecture.png`;
const vkey = v => /strong/i.test(v) ? "strong" : /partial/i.test(v) ? "partial" : "weak";
const VWORD = { strong: "Strong", partial: "Partial", weak: "Weak" };
const CLAIM = { supported: "Supported", overclaim: "Overclaim", needs_detail: "Needs detail" };
const TYPE_LABEL = { cloudtrail: "CloudTrail event", guardduty: "GuardDuty finding", cloudwatch: "CloudWatch logs", iam_activity: "IAM activity", access_key: "Access key", billing: "Billing", unknown: "Type not classified" };

// ---------- avatars ----------
const AV = {
  coach: '<div class="head"><div class="face"><i class="eye l"></i><i class="eye r"></i><i class="mouth"></i></div><i class="set"></i><i class="mic"></i></div><div class="body"><i class="led"></i></div><div class="dots"><i></i><i></i><i></i></div>',
  lens: '<i class="hump"></i><div class="body"><i class="led"></i></div><div class="head"><i class="ring"><i class="iris"></i><i class="pupil"></i><i class="glint"></i></i></div><i class="br"></i>',
  warden: '<div class="scales"><i class="post"></i><i class="beam"><i class="pan l"></i><i class="pan r"></i></i></div><div class="body"><i class="led"></i></div><div class="head"><i class="rim"></i><i class="shield"></i><i class="visor"><i class="glow"></i></i><i class="key"></i></div>',
  echo: '<i class="ring"></i><i class="ring r2"></i><div class="body"><i class="led"></i></div><div class="head"><div class="face"><i class="eye l"></i><i class="eye r"></i></div><i class="wave"><i></i><i></i><i></i><i></i><i></i></i></div><i class="band"></i><i class="cup l"></i><i class="cup r"></i>',
};
function initAvatars(root = document) {
  $$("[data-av]:not(.av)", root).forEach(el => {
    const t = el.dataset.av;
    const keep = [...el.classList].filter(c => c !== "io-work").join(" ");
    el.dataset.base = "av " + t + (el.classList.contains("io-work") ? " io-work" : "");
    el.className = el.dataset.base + (keep ? " " + keep : "");
    el.innerHTML = AV[t];
    el.setAttribute("aria-hidden", "true");
    el.style.setProperty("--d", (-Math.random() * 2.8).toFixed(2) + "s");
    el.addEventListener("animationend", e => { if (e.target === el && e.animationName === "hop") el.classList.remove("hop"); });
  });
}
const setAv = (el, st) => { if (el) el.className = el.dataset.base + (st ? " " + st : ""); };
const hop = el => { if (!el) return; el.classList.remove("hop"); void el.offsetWidth; el.classList.add("hop"); };
initAvatars();

// ---------- ornaments ----------
function drawRosettes(root = document) {
  $$("svg[data-rosette]", root).forEach(svg => {
    if (svg.childElementCount) return;
    const [R1, R2, k] = svg.dataset.rosette.split(",").map(Number);
    let out = "";
    for (let j = 0; j < 18; j++) {
      const ph = j * Math.PI / 9, pts = [];
      for (let t = 0; t <= Math.PI * 2 + .01; t += .02) {
        const r = 60 + 30 * Math.sin(R1 * t + ph) * Math.cos(R2 * t * k);
        pts.push((r * Math.cos(t)).toFixed(1) + "," + (r * Math.sin(t)).toFixed(1));
      }
      out += `<polyline fill="none" stroke="currentColor" stroke-width=".25" points="${pts.join(" ")}"/>`;
    }
    svg.innerHTML = out + '<circle r="98" fill="none" stroke="currentColor" stroke-width=".3"/>';
  });
}
drawRosettes();
$("#mq").innerHTML = (Array(2).fill(["AssumeRole", "CreateAccessKey", "GetBucketLocation", "DescribeVolumes", "PutObject", "DeleteAccessKey", "UnauthorizedAccess:IAMUser/InstanceCredentialExfiltration", "Persistence:IAMUser/AnomalousBehavior", "UpdateFunctionCode", "198.51.100.1", "ap-southeast-1"].join("<i>/</i>")).join("<i>/</i>")) + "<i>/</i>";
$("#micro").textContent = "CLOUDIR·TRAINER·ACSE-EVAL·EVIDENCE·FIRST·".repeat(40);

// ---------- motion ----------
const motionMQ = matchMedia("(prefers-reduced-motion: reduce)");
let reduce = motionMQ.matches;
let skipAnim = false;
// Decorative pacing only: skipped when motion is reduced or the learner skips animations.
const pace = ms => wait(reduce || skipAnim ? 0 : ms);
const CALM_KEY = "cloudir-reduce-motion";
const savedPref = () => { try { return localStorage.getItem(CALM_KEY); } catch (e) { return null; } };
let toastT = 0;
function motionNote(on) {
  const t = $("#mtoast"); clearTimeout(toastT);
  t.textContent = on ? "Reduced motion on. Animations are off and results appear straight away." : "Reduced motion off. Animations are back on.";
  t.classList.add("on");
  toastT = setTimeout(() => { t.classList.remove("on"); toastT = setTimeout(() => t.textContent = "", 400); }, 4000);
}
function setCalm(on, save) {
  reduce = on;
  document.body.classList.toggle("calm", on); document.body.classList.toggle("motion-ok", !on);
  $$("[data-calm]").forEach(b => b.setAttribute("aria-pressed", on));
  if (on) {
    $$(".av").forEach(b => { b.style.removeProperty("--lx"); b.style.removeProperty("--ly"); });
    document.getAnimations().forEach(a => { if (!(a instanceof CSSAnimation) && !(a instanceof CSSTransition)) try { a.finish(); } catch (e) { /* already done */ } });
  }
  if (save) { try { localStorage.setItem(CALM_KEY, on ? "1" : "0"); } catch (e) { /* storage blocked */ } motionNote(on); }
}
$$("[data-calm]").forEach(b => b.onclick = () => setCalm(!reduce, true));
{ const pref = savedPref(); setCalm(pref === null ? motionMQ.matches : pref === "1"); }
motionMQ.addEventListener("change", e => { if (savedPref() === null) setCalm(e.matches); });
let raf = 0, px = 0, py = 0;
addEventListener("pointermove", e => { px = e.clientX; py = e.clientY; if (!raf) raf = requestAnimationFrame(look); }, { passive: true });
function look() {
  raf = 0; if (reduce) return;
  $$(".av").forEach(b => {
    if (!b.offsetParent) return;
    const r = b.getBoundingClientRect(); if (r.bottom < 0 || r.top > innerHeight) return;
    const dx = Math.max(-1, Math.min(1, (px - r.left - r.width / 2) / 300)), dy = Math.max(-1, Math.min(1, (py - r.top - r.height / 3) / 300));
    b.style.setProperty("--lx", dx.toFixed(2)); b.style.setProperty("--ly", dy.toFixed(2));
  });
}
let idleT = 0, lastWake = 0;
function sleepNow() { if (isEvaluatingTurn || isPreparingDataset) { idleT = setTimeout(sleepNow, 15000); return; } document.body.classList.add("idle"); }
function wake() {
  const now = performance.now(), was = document.body.classList.contains("idle");
  if (!was && now - lastWake < 800) return;
  lastWake = now; clearTimeout(idleT); idleT = setTimeout(sleepNow, 45000);
  if (was) { document.body.classList.remove("idle"); $$(".av").forEach(a => { if (a.offsetParent && !a.classList.contains("work")) hop(a); }); }
}
["pointermove", "pointerdown", "keydown", "wheel", "touchstart", "scroll"].forEach(t => addEventListener(t, wake, { passive: true }));

// ---------- router ----------
const TURN_PAGES = ["action", "evidence", "reasoning"];
let current = null, wiping = false, wipeTok = 0, booted = false;
const label = id => ({ landing: "CloudIR", case: "Choose a case", prep: "Preparing", action: `Turn ${turnNo()} · Action`, evidence: `Turn ${turnNo()} · Evidence`, reasoning: `Turn ${turnNo()} · Reasoning`, evaluation: "AI evaluation", handoff: `Turn ${turnNo()}`, debrief: "Debrief", runs: "Saved runs", tutorial: "Tutorial" }[id]);
function setPills(st) { if (!st) return; pillSet("#pC", st.c + "%"); pillSet("#pV", st.v + "%"); pillSet("#pR", st.risk); }
function pillSet(sel, txt) { const el = $(sel); if (el.textContent === txt) return; const was = el.textContent; el.textContent = txt; if (was !== "--" && !reduce) { const p = el.parentElement; p.classList.remove("bump"); void p.offsetWidth; p.classList.add("bump"); } }
const scenarioTitle = () => titleCase(activeScenarioId);
function barFor(id) {
  const n = turnNo(), max = maxTurns();
  const map = {
    case: ["ACSE-Eval dataset", false], prep: ["ACSE-Eval dataset · building", false], runs: ["Saved runs", false], tutorial: ["How to use CloudIR Trainer", false],
    handoff: [`${scenarioTitle()} · Turn ${n} of ${max} · ${stageName(n)}`, true],
    debrief: [`${report ? report.name : scenarioTitle()} · Complete`, true],
  };
  const [txt, pills] = map[id] || (appState ? [`${scenarioTitle()} · Turn ${n} of ${max} · ${stageName(n)}`, true] : ["CloudIR Trainer", false]);
  $("#barTurn").textContent = txt; $("#barPills").hidden = !pills;
}
function show(id) {
  if (current && LEAVE[current]) LEAVE[current]();
  Object.entries(pageEls).forEach(([k, el]) => { el.hidden = k !== id; el.classList.toggle("active", k === id); });
  current = id;
  document.body.classList.toggle("on-landing", id === "landing");
  if (id !== "landing") barFor(id);
  if (RENDER[id]) RENDER[id]();
  updateRunControls(); setStick();
  try { history.replaceState(null, "", "#" + id); } catch (e) { /* file URL */ }
  scrollTo(0, 0);
  document.title = id === "landing" ? "CloudIR Trainer" : `${label(id)} · CloudIR Trainer`;
  if (ENTER[id]) ENTER[id]();
  if (booted) focusView();
  booted = true;
}
function focusView() {
  const h = pageEls[current] && pageEls[current].querySelector("h1");
  if (!h) return;
  h.setAttribute("tabindex", "-1"); h.setAttribute("data-view-heading", "");
  h.focus({ preventScroll: true });
}
function go(id) {
  if (!pageEls[id] || wiping) return;
  if ($("#lb").open) $("#lb").close();
  if (id === current && !TURN_PAGES.includes(id)) { show(id); return; }
  if (reduce || !current) { show(id); return; }
  if (TURN_PAGES.includes(id) && TURN_PAGES.includes(current)) {
    show(id);
    pageEls[id].animate([{ opacity: .2, transform: "translateX(28px)" }, { opacity: 1, transform: "none" }], { duration: 380, easing: "cubic-bezier(.2,.8,.2,1)" });
    return;
  }
  wiping = true;
  const w = $("#wipe");
  $("#wipeLabel").innerHTML = `<small>CloudIR Trainer</small>${esc(label(id))}`;
  w.className = "wipe in";
  const tok = ++wipeTok;
  // The new page is live as soon as it is shown; the exit sweep never blocks clicks.
  setTimeout(() => { show(id); wiping = false; w.className = "wipe out"; setTimeout(() => { if (tok === wipeTok) w.className = "wipe"; }, 520); }, 440);
}
// Guards for pages that need a loaded turn.
const NEEDS_TURN = ["action", "evidence", "reasoning", "evaluation", "handoff"];
document.addEventListener("click", e => {
  const g = e.target.closest("[data-go]");
  if (g && !g.disabled) {
    e.preventDefault();
    const to = g.dataset.go;
    if (to === "evidence" && !selectedAction) return;
    if (to === "reasoning" && !(selectedAction && selectedEvidence)) return;
    if (NEEDS_TURN.includes(to) && !turnPayload) { go("case"); return; }
    if (to === "handoff") { continueToNextTurn(); return; }
    if (to === "debrief") { openFinalReport(); return; }
    go(to); return;
  }
  const sc = e.target.closest("[data-scroll]");
  if (sc) { const t = document.getElementById(sc.dataset.scroll); if (t) t.scrollIntoView({ behavior: reduce ? "auto" : "smooth" }); return; }
  const pc = e.target.closest("[data-pick-case]");
  if (pc) { activeScenarioId = pc.dataset.pickCase; go("case"); return; }
  const z = e.target.closest("[data-zoom]");
  if (z) openLightbox(z.dataset.zoom, z.dataset.cap || "");
});
function openLightbox(src, cap) { $("#lbImg").src = src; $("#lbImg").alt = cap; $("#lbCap").textContent = cap; $("#lb").showModal(); $("#lbClose").focus(); }
function closeEvidenceModal() { if ($("#lb").open) $("#lb").close(); }
$(".skip-link").onclick = e => { e.preventDefault(); focusView(); };
$("#lbClose").onclick = () => $("#lb").close();
$("#lb").onclick = e => { if (e.target.id === "lb") $("#lb").close(); };
const setStick = () => document.documentElement.style.setProperty("--stick", ($("#bar").offsetHeight + 16) + "px");

// ---------- Test Mode ----------
function applyTestModeState() {
  document.body.classList.toggle("testmode", testModeEnabled);
  $("#testModeToggleBtn").setAttribute("aria-pressed", testModeEnabled ? "true" : "false");
  $("#skipNextTurnBtn").setAttribute("aria-pressed", skipNextTurnEnabled ? "true" : "false");
  updateRunControls(); setStick();
}
function initialiseTestMode() {
  try {
    testModeEnabled = localStorage.getItem(TEST_MODE_STORAGE_KEY) === "1";
    skipNextTurnEnabled = localStorage.getItem(SKIP_NEXT_TURN_STORAGE_KEY) === "1";
  } catch (e) { /* storage blocked */ }
  applyTestModeState();
}
const shouldSkipNextTurn = () => testModeEnabled && skipNextTurnEnabled;
addClickListener("testModeToggleBtn", () => {
  testModeEnabled = !testModeEnabled;
  try { localStorage.setItem(TEST_MODE_STORAGE_KEY, testModeEnabled ? "1" : "0"); } catch (e) { /* storage blocked */ }
  applyTestModeState();
  if (RENDER[current] && current !== "evaluation") RENDER[current]();
  if (current === "evaluation") fillPayload();
});
addClickListener("skipNextTurnBtn", () => {
  skipNextTurnEnabled = !skipNextTurnEnabled;
  try { localStorage.setItem(SKIP_NEXT_TURN_STORAGE_KEY, skipNextTurnEnabled ? "1" : "0"); } catch (e) { /* storage blocked */ }
  applyTestModeState();
});
const jsonOut = o => esc(JSON.stringify(o, null, 2));
function updateRunControlsSafe() { if (typeof updateRunControls === "function") updateRunControls(); }
if (typeof updateRunControls !== "function") window.updateRunControls = () => {};

// ---------- landing ----------
const lines = ["Name the exact visible fields that support each response action.", "Explain what the evidence does not prove before deciding containment.", "Connect detection, correlation, and response decisions across the full investigation path."];
let li = 0;
setInterval(() => { if (current !== "landing" || reduce) return; li = (li + 1) % lines.length; $("#heroLine").textContent = lines[li]; }, 5200);
const heroBot = $("#heroBot");
$$(".hero .btn").forEach(b => { b.addEventListener("mouseenter", () => setAv(heroBot, "happy")); b.addEventListener("mouseleave", () => setAv(heroBot, "")); });
$$(".roster > div, .mcard").forEach(d => { const a = d.querySelector(".av"); d.addEventListener("mouseenter", () => setAv(a, "work")); d.addEventListener("mouseleave", () => setAv(a, "")); });
const io = new IntersectionObserver(es => es.forEach(e => {
  e.target.classList.toggle("inview", e.isIntersecting);
  $$(".io-work", e.target).forEach(a => setAv(a, e.isIntersecting ? "work" : ""));
  if (e.isIntersecting && e.target.id === "life") drawChart();
}), { threshold: .3 });
$$("[data-io]").forEach(s => io.observe(s));
let chartDrawn = false;
function drawChart() {
  const svg = $("#lifeChart"); if (chartDrawn && svg.innerHTML) return; chartDrawn = true;
  const C = [20, 30, 40, 50, 60, 70], V = [35, 50, 65, 80, 95, 100];
  const X = i => 50 + i * 104, Y = v => 280 - v * 2.4;
  const step = arr => arr.map((v, i) => `${X(i)},${Y(v)}`).join(" ");
  let g = "";
  [0, 25, 50, 75, 100].forEach(v => { g += `<line x1="50" x2="570" y1="${Y(v)}" y2="${Y(v)}" stroke="currentColor" stroke-opacity=".18"/><text x="40" y="${Y(v) + 4}" text-anchor="end" font-size="11" fill="currentColor">${v}%</text>`; });
  ["Start", "T1", "T2", "T3", "T4", "T5"].forEach((t, i) => { g += `<text x="${X(i)}" y="304" text-anchor="middle" font-size="11" fill="currentColor">${t}</text>`; });
  g += `<polyline points="${step(V)}" fill="none" stroke="#D9F99D" stroke-width="3" class="ln"/><polyline points="${step(C)}" fill="none" stroke="#C9A24E" stroke-width="3" class="ln"/>`;
  V.forEach((v, i) => g += `<circle cx="${X(i)}" cy="${Y(v)}" r="4" fill="#D9F99D"/>`);
  C.forEach((v, i) => g += `<circle cx="${X(i)}" cy="${Y(v)}" r="4" fill="#C9A24E"/>`);
  g += `<text x="${X(5)}" y="${Y(100) - 12}" text-anchor="end" font-size="12" font-weight="700" fill="#D9F99D">100%</text><text x="${X(5)}" y="${Y(70) + 20}" text-anchor="end" font-size="12" font-weight="700" fill="#C9A24E">70%</text>`;
  svg.innerHTML = g;
  if (!reduce) $$(".ln", svg).forEach(p => { const L = p.getTotalLength(); p.style.strokeDasharray = L; p.animate([{ strokeDashoffset: L }, { strokeDashoffset: 0 }], { duration: 1400, easing: "cubic-bezier(.3,.7,.2,1)" }); });
}
let resumable = null;
async function refreshResumeCard() {
  try {
    const data = await callApi("/api/runs");
    resumable = data.activeRun && data.activeRun.status !== "completed" ? data.activeRun : null;
  } catch (e) { resumable = null; }
  renderLanding();
}
function renderLanding() {
  const r = activeRunId && appState && !scenarioComplete
    ? { label: activeRunMetadata?.label || scenarioTitle(), turn: turnNo(), done: (appState.actionHistory || []).length }
    : resumable ? { label: resumable.label, turn: resumable.turn || 1, done: resumable.completedTurns || 0 } : null;
  $("#resumeCard").hidden = !r; $("#resumeBtn2").hidden = !r;
  if (r) {
    $("#resumeTitle").textContent = `${r.label} · Turn ${r.turn} of ${maxTurns()}`;
    $("#resumeSub").textContent = `${stageName(r.turn)} · ${r.done} turn${r.done === 1 ? "" : "s"} evaluated`;
  }
}
async function resumeActive() {
  if (activeRunId && turnPayload && !scenarioComplete) { go(selectedAction ? (selectedEvidence ? "reasoning" : "evidence") : "action"); return; }
  if (resumable && typeof resumeSavedRun === "function") { try { await resumeSavedRun(resumable.id); } catch (e) { alert(`Could not resume the run: ${e.message}`); } }
}
$("#resumeBtn").onclick = resumeActive; $("#resumeBtn2").onclick = resumeActive;

// ---------- case selection ----------
let scenarios = [], statuses = {};
const caseCard = (sc, attrs, px) => {
  const k = caseKey(sc.id), c = CASES[k], id = px + "-" + sc.id, st = statuses[sc.id];
  const prepared = st?.prepared === true || st?.savedCopy === true;
  return `<button class="case case-${k}" ${attrs} aria-labelledby="${id}-t" aria-describedby="${id}-s ${id}-d"><span class="case-top"><span class="case-no">Case ${c.no} · ${c.code}</span><span class="state${st && !prepared ? " need" : ""}" id="${id}-s"><i aria-hidden="true">●</i> ${st ? (prepared ? "Prepared" : "Needs build") : "Checking"}</span></span><span class="case-art ${ART[k]}" aria-hidden="true">${c.art}</span><span class="case-copy"><span class="ch" id="${id}-t">${esc(titleCase(sc.id))}</span><span class="cp" id="${id}-d">${esc(c.blurb)}</span></span><span class="tags" aria-hidden="true">${c.tags.map(t => `<span>${t}</span>`).join("")}</span></button>`;
};
async function loadScenarios() {
  const first = await callApi(`/api/dataset/status?scenario=${encodeURIComponent(activeScenarioId)}`);
  scenarios = (first.enabledScenarios || []).filter(s => s.sourceReady);
  statuses[activeScenarioId] = first;
  const order = ["identity", "asr", "cost"];
  scenarios.sort((a, b) => order.indexOf(caseKey(a.id)) - order.indexOf(caseKey(b.id)));
  await Promise.all(scenarios.filter(s => !statuses[s.id] || s.id !== activeScenarioId).map(async s => {
    try { statuses[s.id] = await callApi(`/api/dataset/status?scenario=${encodeURIComponent(s.id)}`); } catch (e) { statuses[s.id] = { error: e.message }; }
  }));
  if (!scenarios.some(s => s.id === activeScenarioId) && scenarios.length) activeScenarioId = scenarios[0].id;
}
function renderLandCases() {
  const list = scenarios.length ? scenarios : [{ id: "identity-management" }, { id: "automated-security-response" }, { id: "cost-management" }];
  $("#landCases").innerHTML = list.map(s => caseCard(s, `data-pick-case="${esc(s.id)}"`, "lc")).join("");
}
const isPrepared = id => statuses[id]?.prepared === true || statuses[id]?.savedCopy === true;
function renderCase() {
  const k = activeScenarioId, name = titleCase(k), st = statuses[k] || {};
  $("#caseCards").innerHTML = scenarios.map(s => caseCard(s, `data-case="${esc(s.id)}" aria-pressed="${s.id === k}"`, "cc")).join("") || '<p class="muted">Loading cases…</p>';
  $$("#caseCards .case").forEach(c => c.onclick = () => { activeScenarioId = c.dataset.case; renderCase(); });
  $(".case-detail").dataset.case = caseKey(k);
  const prepared = isPrepared(k);
  $("#cdEyebrow").textContent = `${name} · ${st.error ? "Status unavailable" : prepared ? "Prepared" : "Needs build"}`;
  $("#cdTitle").textContent = name + " architecture";
  const img = st.architectureAssetUrl || archUrl(k);
  $("#cdImg").src = img; $("#cdImg").alt = name + " architecture diagram";
  $("#cdZoom").dataset.zoom = img; $("#cdZoom").dataset.cap = name + " architecture. Context only, not selectable evidence.";
  $("#startCaseBtn").textContent = prepared ? "Start scenario" : "Build case";
  $("#startCaseBtn").disabled = !st.sourceReady;
  $("#rebuildBtn").hidden = !prepared; $("#rebuildBtn").disabled = !st.sourceReady;
  const files = [["data/acse_eval.jsonl", st.requiredSourceFiles?.data_jsonl], [st.sourcePaths?.architecture_png || "architecture.png", st.requiredSourceFiles?.architecture_png], [st.sourcePaths?.threat_model_json || "threat-model.json", st.requiredSourceFiles?.threat_model_json]];
  $("#files").innerHTML = files.map(([f, ok]) => `<li><span class="${ok ? "ok" : "missing"}">${ok ? "Found" : "Missing"}</span><code>${esc(f)}</code></li>`).join("");
  $("#cleanConfirm").hidden = true;
  renderLandCases();
}
async function openPrototypeWorkspace() {
  go("case");
  try { await loadScenarios(); } catch (e) { $("#caseStatus").textContent = `Could not check the dataset: ${e.message}`; }
  if (current === "case") renderCase();
}
async function refreshCaseStatus(msg) {
  try { statuses = {}; await loadScenarios(); $("#caseStatus").textContent = msg || `Checked ${clock()}. ${scenarios.filter(s => isPrepared(s.id)).length} of ${scenarios.length} cases prepared.`; }
  catch (e) { $("#caseStatus").textContent = `Status check failed: ${e.message}`; }
  if (current === "case") renderCase();
}
$("#startCaseBtn").onclick = () => { if (isPrepared(activeScenarioId)) startRuntimeScenario(); else startDatasetPreparation(); };
$("#rebuildBtn").onclick = () => startDatasetPreparation();
$("#refreshBtn").onclick = () => refreshCaseStatus();
$("#cleanBtn").onclick = () => {
  const all = $("#cleanScope").value === "all";
  $("#cleanMsg").textContent = all
    ? "This removes processed files and prepared copies for every case, the active runtime, evaluation traces and all generated screenshots. Source files and saved runs are kept."
    : `This removes the processed files and prepared copy of ${titleCase(activeScenarioId)}. If it is the active case, the runtime, evaluation traces and screenshots are cleared too. Source files are kept.`;
  $("#cleanConfirm").hidden = false; $("#cleanGo").focus();
};
$("#cleanCancel").onclick = () => { $("#cleanConfirm").hidden = true; $("#cleanBtn").focus(); };
$("#cleanGo").onclick = async () => {
  const btn = $("#cleanGo"); btn.disabled = true; btn.textContent = "Cleaning…";
  try {
    await callApi("/api/dataset/flush-generated", { method: "POST", body: JSON.stringify({ confirm: "flush-generated", scope: $("#cleanScope").value === "all" ? "all" : "scenario", scenarioId: activeScenarioId }) });
    await refreshCaseStatus(`Cleaned ${clock()}. Build the case to play it again.`);
  } catch (e) { $("#caseStatus").textContent = `Clean restart failed: ${e.message}`; }
  finally { btn.disabled = false; btn.textContent = "Clean restart"; $("#cleanConfirm").hidden = true; }
};

// ---------- case preparation (live build stream) ----------
const PREP = [
  { stage: "select_case", label: "Select ACSE case", msg: id => `Load ${id} from acse_eval.jsonl.`, who: "code" },
  { stage: "parse_architecture", label: "VLM reads architecture", msg: () => "architecture.png → architecture_facts.json.", who: "lens", model: "Vision · Qwen2.5-VL-3B" },
  { stage: "normalise_threat", label: "Normalise threat model", msg: () => "Merge threat-model.json with architecture facts.", who: "warden", model: "Security · Foundation-Sec-8B" },
  { stage: "build_runtime", label: "Create runtime files", msg: () => "Write seed, hidden truth, and config files.", who: "code" },
  { stage: "incident_timeline", label: "Write incident timeline", msg: () => "Security AI writes the timeline every turn's evidence comes from.", who: "warden", model: "Security · Foundation-Sec-8B" },
  { stage: "coach_turn_1", label: "Generate Turn 1", msg: () => "Coach AI creates briefing, actions, and choices.", who: "coach", model: "Coach · Qwen2.5-1.5B" },
  { stage: "render_evidence", label: "Render evidence", msg: () => "Create AWS-style evidence screenshots.", who: "code" },
  { stage: "complete", label: "Scenario ready", msg: () => "CloudIR runtime can start.", who: "coach", model: "Coach" },
];
let prepTick = 0, buildId = "", buildInfo = {};
const logLine = s => { const l = $("#log"); l.textContent += s + "\n"; l.scrollTop = l.scrollHeight; };
const setProgress = f => { $("#prepFill").style.transform = `scaleX(${f})`; $("#prepPct").textContent = Math.round(f * 100) + "%"; };
function buildVis(i, info = {}) {
  const box = $("#bstage"); if (!box) return;
  const name = titleCase(buildId);
  const it = (html, s, cls = "") => `<div class="it${cls ? " " + cls : ""}" style="--d:${s.toFixed(2)}s">${html}</div>`;
  const chips = (xs, s0, step) => `<div class="bchips">${xs.map((x, j) => `<span class="it bchip" style="--d:${(s0 + j * step).toFixed(2)}s">${esc(x)}</span>`).join("")}</div>`;
  const writing = n => `<div class="bv-writing">${Array.from({ length: n }, (_, j) => `<i class="it" style="--d:${(.2 + j * .25).toFixed(2)}s;--w:${60 + ((j * 37) % 35)}%"></i>`).join("")}</div>`;
  let h = "", cls = "";
  if (i === 0) {
    const names = ACSE_CASES.includes(buildId) ? ACSE_CASES : [...ACSE_CASES, buildId].sort();
    const at = Math.max(0, names.indexOf(buildId)), from = Math.max(0, at - 10), list = names.slice(from, at + 5);
    cls = "bv0";
    h = `<p class="bv-h">data/acse_eval.jsonl · ACSE-Eval cases</p><div class="bv-reel"><ol style="--sel:${at - from}">${list.map((n, j) => `<li class="${j === at - from ? "sel" : ""}">${esc(n)}</li>`).join("")}</ol></div>
      <div class="bv-card it" style="--d:.95s"><b>${esc(name)}</b>${["architecture.png", "threat-model.json", "diagram.py", "cdk/"].map((f, j) => `<span class="it" style="--d:${(1.1 + j * .13).toFixed(2)}s"><i>✓</i>${f}</span>`).join("")}</div>`;
  } else if (i === 1) {
    cls = "bv1";
    h = `<div class="bv-img"><img src="${archUrl(buildId)}" alt=""><i class="bscan"></i><i class="focus"></i></div>
      <div><p class="bv-h">architecture_facts.json</p>${it('<p class="bv-k">Reading the diagram</p>', .3)}${chips(["Services", "Trust boundaries", "Data flows", "Identity flows", "Logging"], .5, .3)}${it(`<p class="bv-flow">The vision model lists what it can see. Nothing is inferred from outside the diagram.</p>`, 2.1)}</div>`;
  } else if (i === 2) {
    cls = "bv2";
    const threats = info.threats || [];
    h = `<div class="bv-src">${it("<b>threat-model.json</b>", .1)}${it("<i>+</i>", .3)}${it("<b>architecture facts</b>", .45)}</div>
      <div class="bcards">${threats.length ? it(`<p class="bv-k">Threats in the model</p><ul>${threats.slice(0, 5).map(x => `<li>${esc(x)}</li>`).join("")}</ul>`, .7, "bcard") : it(`<p class="bv-k">Primary risk</p>${writing(3)}`, .7, "bcard")}${it(`<p class="bv-k">Normalising</p><p>Primary risk, affected assets, attack path and security signals.</p>`, 1.2, "bcard")}</div>`;
  } else if (i === 3) {
    cls = "bv3";
    const F = [["scenario_seed.json", "Case facts and the starting state", ""], ["hidden_truth.json", "What really happened. Hidden from you.", "lock"], ["scenario_config.json", "Turns, phases and scoring", ""]];
    h = `<p class="bv-h">runtime/</p><div class="bv-files">${F.map(([f, t, c], j) => `<div class="it bfile ${c}" style="--d:${(.2 + j * .35).toFixed(2)}s"><b>${f.replace(/_/g, "_<wbr>")}</b><span>${t}</span><em>✓</em></div>`).join("")}</div><p class="bv-note it" style="--d:1.4s">Code step · no model needed. These files keep every turn consistent with the case.</p>`;
  } else if (i === 4) {
    h = `<p class="bv-h">incident_timeline.json${info.events ? ` · ${info.events} events` : " · writing"}</p>${info.retry ? `<div class="bv-warn it" style="--d:.1s"><i>!</i><span>${esc(info.retry)}</span></div>` : ""}<ol class="bv-tl">${Array.from({ length: 7 }, (_, j) => `<li class="it${j === 3 || j === 4 ? " sus" : ""}" style="--d:${(.15 + j * .3).toFixed(2)}s"><span>${String(9 + j).padStart(2, "0")}:${String((j * 7) % 60).padStart(2, "0")}</span><b><i class="skl" style="--w:${55 + (j * 13) % 40}%"></i></b><em></em></li>`).join("")}</ol>`;
  } else if (i === 5) {
    h = `<p class="bv-h">turns/turn_1 · written by the coach</p>${info.retry ? `<div class="bv-warn it" style="--d:.1s"><i>!</i><span>${esc(info.retry)}</span></div>` : ""}${writing(3)}<div class="bv-acts">${["A", "B", "C"].map((l, j) => `<div class="it bv-act" style="--d:${(1.1 + j * .3).toFixed(2)}s"><b>${l}</b><span><i class="skl" style="--w:${70 + j * 9}%"></i></span></div>`).join("")}</div>`;
  } else if (i === 6) {
    const evs = info.evidence || [];
    h = `<p class="bv-h">evidence/turn_1 · rendered screenshots</p><div class="bv-evs">${(evs.length ? evs : [{}, {}, {}]).slice(0, 3).map((e, j) => `<figure class="bv-ev"><div class="slot">${e.imageUrl ? `<img src="${esc(e.imageUrl)}" alt="" style="--d:${(.2 + j * .35).toFixed(2)}s">` : ""}</div><figcaption class="it" style="--d:${(1 + j * .35).toFixed(2)}s">Exhibit 1-${"ABC"[j]}${e.title ? ` · ${esc(e.title)}` : ""}</figcaption></figure>`).join("")}</div>`;
  } else {
    cls = "bv7";
    const sv = info.services || [];
    h = `<b>${esc(name)}</b><span>Five turns, three actions and three exhibits per turn.</span>${sv.length ? chips(sv.slice(0, 6), .3, .12) : ""}<div class="bv-stamp">Ready</div>`;
  }
  box.className = "bstage " + cls; box.innerHTML = h;
  requestAnimationFrame(() => requestAnimationFrame(() => box.classList.add("go")));
}
const listOf = v => Array.isArray(v) ? v : typeof v === "string" ? (() => { try { const p = JSON.parse(v.replace(/'/g, '"')); return Array.isArray(p) ? p : []; } catch { return []; } })() : [];
function renderPrepShell() {
  $("#prepEyebrow").textContent = "Building " + titleCase(buildId).toLowerCase();
  $("#stages").innerHTML = PREP.map((s, i) => `<li data-stage="${s.stage}"><span class="si">${i + 1}</span><div><strong>${s.label}</strong><span>${esc(s.msg(buildId))}</span></div><em class="kind ${s.model && i < 7 ? "ai" : ""}">${s.who === "code" ? "code" : i === 7 ? "output" : "AI"}</em></li>`).join("");
  $("#log").textContent = ""; $("#nowDone").hidden = true; $("#nowLoad").hidden = false; setProgress(0);
  $("#nowError").hidden = true;
}
function prepFocus(i, info) {
  const s = PREP[i], lis = $$("#stages li");
  lis.forEach((x, j) => { if (!x.classList.contains("done") && !x.classList.contains("err")) x.className = j === i ? "run" : ""; });
  $$(".performers [data-who]").forEach(el => el.hidden = el.dataset.who !== s.who);
  const p = $(`.performers [data-who="${s.who}"]`); if (p.dataset.base) setAv(p, i === 7 ? "happy" : "work");
  $("#nowWho").textContent = s.model || "Code step"; $("#nowLabel").textContent = s.label; $("#nowMsg").textContent = s.msg(buildId);
  buildVis(i, info);
  const t0 = performance.now(); clearInterval(prepTick); $("#nowTime").textContent = "0.0 s";
  prepTick = setInterval(() => $("#nowTime").textContent = ((performance.now() - t0) / 1000).toFixed(1) + " s", 250);
}
async function startDatasetPreparation() {
  if (isPreparingDataset) return;
  buildId = activeScenarioId; buildInfo = {};
  isPreparingDataset = true; updateRunControlsSafe();
  go("prep");
  await wait(reduce ? 0 : 480);
  renderPrepShell();
  let complete = false, lastStage = -1;
  try {
    await readEventStream("/api/dataset/prepare-stream", { scenarioId: buildId }, async ev => {
      const i = PREP.findIndex(s => s.stage === ev.stage);
      if (ev.status === "error" || ev.stage === "error") throw Object.assign(new Error(ev.message || "Dataset preparation failed."), { stageIndex: lastStage });
      if (i < 0) return;
      if (ev.status === "running") {
        const retry = i === lastStage && /Regenerating|rejected/i.test(ev.message || "") ? ev.message : "";
        lastStage = i;
        if (i === 2 && !buildInfo.threats) {
          try { const tm = await (await fetch(`/dataset-assets/${encodeURIComponent(buildId)}/threat-model.json`)).json(); buildInfo.threats = (Array.isArray(tm) ? tm : tm.threats || []).map(t => t.threatName).filter(Boolean); } catch (e) { buildInfo.threats = []; }
        }
        prepFocus(i, { ...buildInfo, retry });
        logLine(`▸ ${ev.label}: ${ev.message}`);
      } else if (ev.status === "success") {
        const li = $$("#stages li")[i]; if (li) li.className = "done";
        logLine(`  ✓ ${ev.message}${ev.output ? `\n    Output: ${ev.output}` : ""} (${$("#nowTime").textContent})`);
        if (i === 4) { const m = /with (\d+) events/.exec(ev.message || ""); buildInfo.events = m ? +m[1] : 0; buildVis(4, buildInfo); }
        if (i === 6) { buildInfo.evidence = Array.isArray(ev.data?.result) ? ev.data.result : []; buildVis(6, buildInfo); await pace(1600); }
        setProgress($$("#stages li.done").length / 7);
      } else if (ev.status === "complete") {
        complete = true;
        buildInfo.services = listOf(ev.result?.architecture_facts?.visible_services);
        $$("#stages li").forEach(x => x.className = "done"); setProgress(1);
        prepFocus(7, buildInfo); clearInterval(prepTick);
        hop($('.performers [data-who="coach"]')); $("#nowLoad").hidden = true; $("#nowDone").hidden = false;
        $("#barTurn").textContent = "ACSE-Eval dataset · case ready";
        logLine(`✓ ${ev.message}`);
      }
    });
    if (!complete) throw new Error("Dataset preparation stream ended before completion.");
    statuses = {}; loadScenarios().catch(() => {});
  } catch (error) {
    console.error(error);
    clearInterval(prepTick);
    const li = $$("#stages li")[Math.max(0, error.stageIndex ?? lastStage)]; if (li) li.className = "err";
    $$(".performers [data-who]").forEach(el => { if (!el.hidden && el.dataset.base) setAv(el, "err"); });
    $("#nowLoad").hidden = true; $("#nowError").hidden = false; $("#nowErrorMsg").textContent = error.message;
    $("#barTurn").textContent = "ACSE-Eval dataset · build stopped";
    logLine(`✗ ${error.message}`);
  } finally {
    isPreparingDataset = false; activeRunId = null; activeRunMetadata = null; updateRunControlsSafe();
  }
}
$("#prepStart").onclick = () => { activeScenarioId = buildId; startRuntimeScenario(); };
$("#prepRetry").onclick = () => { activeScenarioId = buildId; startDatasetPreparation(); };
$("#logBtn").onclick = () => { const l = $("#log"); l.hidden = !l.hidden; $("#logBtn").textContent = l.hidden ? "Show build log" : "Hide build log"; $("#logBtn").setAttribute("aria-expanded", !l.hidden); };

// ---------- starting and loading turns ----------
// Incident state at the start of each turn, for the gauges' "since last turn" deltas.
let turnStart = {}, railAnimKey = "";
function loadTurnData(data) {
  appState = data.state;
  turnPayload = data.turn;
  generatedEvidence = data.generatedEvidence || [];
  selectedAction = null; selectedEvidence = null; learnerJustification = null; recordedAudioBlob = null;
  scenarioComplete = !!appState?.completed || (turnNo() >= maxTurns() && (appState?.actionHistory || []).length >= maxTurns());
  if (appState && !turnStart[turnNo()]) turnStart[turnNo()] = stateOf(appState);
  transcripts = {};
}
async function startRuntimeScenario() {
  const btn = $("#startCaseBtn"); btn.disabled = true;
  try {
    const data = await callApi("/api/runs/start", { method: "POST", body: JSON.stringify({ scenarioId: activeScenarioId, testMode: testModeEnabled }) });
    activeRunId = data.run.id; activeRunMetadata = data.run; $("#runNameInput").value = data.run.label;
    activeScenarioId = data.run.scenarioId || activeScenarioId;
    turnStart = {}; finalDebrief = null; lastEval = null; report = null;
    loadTurnData(data);
    isEvaluatingTurn = false; isContinuingTurn = false;
    go("action");
  } catch (error) {
    if (error.data?.needs_dataset_preparation) { $("#caseStatus").textContent = "This case has not been prepared yet. Build it first."; await refreshCaseStatus(); }
    else { console.error(error); $("#caseStatus").textContent = `Could not start the scenario: ${error.message}`; }
    if (current !== "case") go("case");
  } finally { btn.disabled = false; }
}

// ---------- turn workspace ----------
const STEPS = [["action", "Action"], ["evidence", "Evidence"], ["reasoning", "Reasoning"], ["evaluation", "Evaluation"]];
const words = t => { t = (t || "").trim(); return t ? t.split(/\s+/).length : 0; };
let transcripts = {};
const txKey = () => `${activeRunId}:${turnNo()}`;
function renderStepper(el) {
  const i = +el.dataset.stepper, a = selectedAction, e = selectedEvidence;
  const subs = [a ? a.title : "Not chosen", e ? e.title : "Not chosen", words($("#learnerJustificationInput").value) + " words", i === 3 ? "Submitted" : "Not submitted"];
  el.innerHTML = `<p class="turnlabel"><span class="eyebrow">Turn ${turnNo()} of ${maxTurns()}</span>${esc(stageName(turnNo()))}</p><ol class="stepper">${STEPS.map(([p, l], j) => {
    const locked = i === 3 || j >= i;
    return `<li class="${j < i ? "done" : j === i ? "now" : ""}"><button type="button" ${locked ? "disabled" : `data-go="${p}"`} ${j === i ? 'aria-current="step"' : ""}><span class="n">${j < i ? "✓" : j + 1}</span><span class="l">${l}<small>${esc(subs[j])}</small></span></button></li>`;
  }).join("")}</ol>`;
}
const RISK = { reduced: 1, low: 1, medium: 2, elevated: 3, high: 3, critical: 4 };
const riskLvl = r => RISK[String(r || "").toLowerCase()] || 2;
function statePanel(st, o = {}) {
  const f = o.from, anim = !!f && !reduce, base = o.base, ti = o.ti ?? turnNo() - 1;
  const gauge = (key, lab, cls) => {
    const b = st[key], a = anim ? f[key] : b, d = base ? b - base[key] : 0;
    return `<div class="gauge ${cls}"><svg viewBox="0 0 100 100" aria-hidden="true"><circle class="gt" cx="50" cy="50" r="42"/><circle class="gf" cx="50" cy="50" r="42" pathLength="100" style="stroke-dashoffset:${100 - a}" data-to="${100 - b}"/></svg><div class="gv"><b class="cnt" data-to="${b}">${a}</b><small>%</small></div><span class="gl">${lab}</span>${d ? `<em class="gdelta${d < 0 ? " neg" : ""}">${d > 0 ? "+" : ""}${d}</em>` : ""}</div>`;
  };
  const lv = riskLvl(st.risk), lb = base ? riskLvl(base.risk) : lv;
  return `<div class="isp${anim ? " intro" : ""}" role="group" aria-label="Containment ${st.c}%, visibility ${st.v}%, risk ${esc(st.risk)}, phase ${esc(st.phase)}">
    <div class="gauges">${gauge("c", "Containment", "g-c")}${gauge("v", "Visibility", "g-v")}</div>
    <div class="riskm"><div class="rrow"><span class="gl">Risk</span><b class="rname r${lv}">${esc(st.risk)}</b>${lv !== lb ? `<em class="rchg ${lv < lb ? "down" : "up"}">${lv < lb ? "↓" : "↑"} from ${esc(base.risk)}</em>` : ""}</div>
      <div class="rsegs" data-lv="${anim ? 0 : lv}" data-to="${lv}"><i></i><i></i><i></i><i></i></div>
      <div class="rlabels"><span>Low</span><span>Medium</span><span>High</span><span>Critical</span></div></div>
    <div class="phasem"><div class="rrow"><span class="gl">Phase</span><b>${esc(st.phase)}</b></div><div class="pdots">${STAGES.map((s, i) => `<i class="${i < ti ? "done" : i === ti ? "now" : ""}" title="${esc(s)}"></i>`).join("")}</div></div>
  </div>`;
}
function animateState(root) {
  const box = root && $(".isp.intro", root); if (!box) return;
  setTimeout(() => requestAnimationFrame(() => requestAnimationFrame(() => {
    $$(".gf[data-to]", box).forEach(c => c.style.strokeDashoffset = c.dataset.to);
    const rs = $(".rsegs", box); if (rs) rs.dataset.lv = rs.dataset.to;
    box.classList.add("go"); countUp(box);
  })), wiping ? 560 : 0);
}
function countUp(root = document) {
  $$(".cnt", root).forEach(el => {
    const from = +el.textContent, to = +el.dataset.to;
    if (reduce || from === to) { el.textContent = to; return; }
    const t0 = performance.now(), D = 900;
    const f = t => { const p = Math.min(1, (t - t0) / D), e = 1 - Math.pow(1 - p, 3); el.textContent = Math.round(from + (to - from) * e); if (p < 1) requestAnimationFrame(f); };
    requestAnimationFrame(f);
  });
}
function railIntro() {
  if (railAnimKey === txKey()) return {};
  railAnimKey = txKey();
  const prev = turnStart[turnNo() - 1];
  return { from: { c: 0, v: 0 }, base: prev || null };
}
const history_ = () => uniqueTurnEntries(appState?.actionHistory || []);
function uniqueTurnEntries(entries) {
  const byTurn = new Map();
  (Array.isArray(entries) ? entries : []).forEach(entry => { const t = Number(entry?.turn); if (Number.isFinite(t) && !byTurn.has(t)) byTurn.set(t, entry); });
  return [...byTurn.values()].sort((a, b) => Number(a.turn) - Number(b.turn));
}
function renderRail(el) {
  const k = el.dataset.rail, a = selectedAction, e = selectedEvidence, cfg = turnConfig(), st = stateOf(appState), intro = railIntro(), hist = history_();
  el.innerHTML = `<details class="rail" ${narrow() ? "" : "open"}><summary>Briefing and state</summary><div class="rail-in">
    <div><p class="eyebrow">Incident briefing</p><p class="rb">${esc(cfg.briefing)}</p><ul class="rk">${knownContext().map(x => `<li>${esc(x)}</li>`).join("")}</ul></div>
    ${k !== "action" ? `<div class="picked"><p class="eyebrow">Your action</p><strong>${a ? esc(a.title) : "Not chosen"}</strong><button class="linkbtn" data-go="action">Change</button></div>` : ""}
    ${k === "reasoning" && e ? `<div class="picked"><p class="eyebrow">Your evidence</p><img src="${esc(e.imageUrl)}" alt=""><strong>${esc(e.title)}</strong><button class="linkbtn" data-go="evidence">Change</button></div>` : ""}
    <div><p class="eyebrow">Incident state</p>${st ? statePanel(st, intro) : ""}</div>
    <div><p class="eyebrow">Action history</p>${hist.length ? `<div class="stamps">${hist.map(h => `<span class="stamp-s ${vkey(h.verdict)}${intro.from && Number(h.turn) === turnNo() - 1 ? " fresh" : ""}">T${h.turn} · ${VWORD[vkey(h.verdict)]}</span>`).join("")}</div>` : `<p class="fine" style="margin-top:6px">No turns evaluated yet.</p>`}</div>
    <button class="linkbtn" data-zoom="${archUrl(activeScenarioId)}" data-cap="${esc(scenarioTitle())} architecture. Context only, not selectable evidence.">Open case architecture</button>
  </div></details>`;
  animateState(el);
}
function paintWorkspace(page) {
  $$("[data-stepper]", pageEls[page]).forEach(renderStepper);
  $$("[data-rail]", pageEls[page]).forEach(renderRail);
}
function coachSay(id, labelTxt, msg, st) {
  const bot = $("#coach" + id); $("#coach" + id + "Label").textContent = labelTxt; $("#coach" + id + "Msg").textContent = msg;
  setAv(bot, st); hop(bot);
  if (st === "talk") setTimeout(() => { if (bot.classList.contains("talk")) setAv(bot, ""); }, 1600);
}
// Role hints are Test Mode only; in a normal session the coach asks questions instead.
const ROLE_A = {
  best: ["Good Instinct", t => `${t} looks like a strong path. Now choose evidence that proves it directly.`, "happy"],
  partial: ["Check The Gap", t => `${t} may help, but check what it leaves uncertain before you commit.`, "caution"],
  weak: ["Careful", t => `${t} might be useful context, but it may not justify the next response on its own.`, "caution"],
};
const ROLE_E = {
  strong: ["Strong Support", t => `${t} gives you a strong support path. In your justification, name the exact visible fields.`, "happy"],
  partial: ["Partial Support", t => `${t} supports part of the story. Say what it proves and what still needs correlation.`, "caution"],
  weak: ["Weak Evidence", t => `${t} is weaker support. Be careful not to overclaim beyond what the screenshot shows.`, "caution"],
};
const roleTag = (r, good) => r ? `<span class="tmeta">${good} <span class="trole ${r === "best" || r === "strong" ? "" : esc(r)}">${esc(r)}</span></span>` : "";
function renderAction() {
  const actions = turnPayload?.actions || [];
  $("#actions").innerHTML = actions.map((a, i) => `<button class="choice" data-k="${"ABC"[i] || i + 1}" data-id="${esc(a.id)}" aria-pressed="${selectedAction?.id === a.id}"><strong>${esc(a.title)}</strong><span>${esc(a.description)}</span>${roleTag(getActionChoiceRole(a), "Correct test choice")}</button>`).join("");
  $$("#actions .choice").forEach(b => b.onclick = () => pickAction(b.dataset.id));
  if (selectedAction) actionCoach(selectedAction, false);
  else coachSay("A", "Decision", coachGuidance() || "Read the briefing, then choose the action you can defend with evidence.", "");
  $("#toEvidence").disabled = !selectedAction;
  $("#tpBest").disabled = !actions.some(a => getActionChoiceRole(a) === "best");
  paintWorkspace("action");
}
function actionCoach(a, animate = true) {
  const r = ROLE_A[getActionChoiceRole(a)];
  if (testModeEnabled && r) coachSay("A", r[0], r[1](a.title), animate ? r[2] : "");
  else coachSay("A", "Next Step", `You picked “${a.title}”. Before you move on: what would it show you, and what would it leave out?`, animate ? "talk" : "");
}
function pickAction(id) {
  const a = (turnPayload?.actions || []).find(x => x.id === id); if (!a) return;
  if (selectedAction?.id !== id) selectedEvidence = null;
  selectedAction = a;
  $$("#actions .choice").forEach(x => x.setAttribute("aria-pressed", x.dataset.id === id));
  actionCoach(a); $("#toEvidence").disabled = false; paintWorkspace("action");
}
$("#tpBest").onclick = () => { const a = (turnPayload?.actions || []).find(x => getActionChoiceRole(x) === "best"); if (a) pickAction(a.id); };
function renderEvidence() {
  const n = turnNo();
  $("#exhibits").innerHTML = generatedEvidence.map((e, i) => `<article class="exhibit ${selectedEvidence?.id === e.id ? "sel" : ""}" data-id="${esc(e.id)}"><button class="pick" aria-pressed="${selectedEvidence?.id === e.id}"><img src="${esc(e.imageUrl)}" alt="" loading="lazy"><strong>${esc(e.title)}</strong><span>${esc(e.summary)}</span>${roleTag(getEvidenceSupportRole(e), "Support role")}</button><span class="tag">EXHIBIT ${n}-${"ABC"[i] || i + 1}</span><button class="zoom" data-zoom="${esc(e.imageUrl)}" data-cap="Exhibit ${n}-${"ABC"[i] || i + 1} · ${esc(e.title)}">Zoom</button></article>`).join("");
  $$("#exhibits .exhibit").forEach(x => $(".pick", x).onclick = () => pickEvidence(x.dataset.id));
  if (selectedEvidence) evidenceCoach(selectedEvidence, false);
  else coachSay("E", "Evidence", "Look for the screenshot that directly proves your chosen action, not just something related.", "");
  $("#toReasoning").disabled = !selectedEvidence;
  $("#tpStrong").disabled = !generatedEvidence.some(e => getEvidenceSupportRole(e) === "strong");
  paintWorkspace("evidence");
}
function evidenceCoach(e, animate = true) {
  const r = ROLE_E[getEvidenceSupportRole(e)];
  if (testModeEnabled && r) coachSay("E", r[0], r[1](e.title), animate ? r[2] : "");
  else coachSay("E", "Evidence Chosen", `You picked ${e.title}. Which field on it names who acted, and which shows when it happened?`, animate ? "talk" : "");
}
function pickEvidence(id) {
  const e = generatedEvidence.find(x => x.id === id); if (!e) return;
  selectedEvidence = e;
  $$("#exhibits .exhibit").forEach(x => { x.classList.toggle("sel", x.dataset.id === id); $(".pick", x).setAttribute("aria-pressed", x.dataset.id === id); });
  evidenceCoach(e); $("#toReasoning").disabled = false; paintWorkspace("evidence");
}
$("#tpStrong").onclick = () => { const e = generatedEvidence.find(x => getEvidenceSupportRole(x) === "strong"); if (e) pickEvidence(e.id); };

// ---------- reasoning: prompt, recording, transcription ----------
function buildReasoningPrompt(action, evidence) {
  const type = evidenceTemplate(evidence), [title, text, cues] = PROMPTS[type] || DEFAULT_PROMPT;
  return { title, text, cues, actionTitle: action?.title, evidenceTitle: evidence?.title, evidenceType: type || "unknown" };
}
const tx = $("#learnerJustificationInput"), echo = $("#echo"), trBtn = $("#trBtn"), trRow = $("#trRow"), recBtn = $("#recBtn");
$("#wave").innerHTML = Array.from({ length: 36 }, () => `<i style="--h:${(.2 + Math.random() * .8).toFixed(2)};animation-delay:${(Math.random() * -.7).toFixed(2)}s"></i>`).join("");
let recordingTimer = null, recordingEndsAt = 0;
function setRecUi(on) {
  $("#voice").classList.toggle("recording", on);
  recBtn.textContent = on ? "STOP" : "REC";
  recBtn.setAttribute("aria-pressed", on ? "true" : "false");
  recBtn.setAttribute("aria-label", on ? "STOP: stop recording (45 seconds maximum)" : "REC: record your answer");
}
async function startJustificationRecording() {
  stopPromptSpeech();
  if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
    $("#trMsg").textContent = "Audio recording is not supported in this browser. Type your answer instead."; return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedAudioChunks = []; recordedAudioBlob = null;
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.addEventListener("dataavailable", e => { if (e.data && e.data.size > 0) recordedAudioChunks.push(e.data); });
    mediaRecorder.addEventListener("stop", () => {
      recordedAudioBlob = new Blob(recordedAudioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
      stream.getTracks().forEach(t => t.stop());
      clearInterval(recordingTimer); recordingTimer = null; setRecUi(false);
      trBtn.disabled = false;
      $("#trMsg").textContent = "Recording stopped. Transcribe it to add it to your answer.";
      $("#timer").textContent = "45";
    });
    mediaRecorder.start();
    setRecUi(true); trBtn.disabled = true;
    $("#trMsg").textContent = "Recording. Up to 45 seconds.";
    recordingEndsAt = Date.now() + MAX_JUSTIFICATION_RECORDING_MS;
    $("#timer").textContent = "45";
    recordingTimer = setInterval(() => {
      const left = Math.max(0, Math.ceil((recordingEndsAt - Date.now()) / 1000));
      $("#timer").textContent = String(left);
      if (left <= 0) stopJustificationRecording();
    }, 250);
  } catch (error) {
    console.error(error);
    $("#trMsg").textContent = `Could not start the microphone: ${error.message}. Type your answer instead.`;
  }
}
function stopJustificationRecording() { if (mediaRecorder && mediaRecorder.state === "recording") mediaRecorder.stop(); }
recBtn.onclick = () => { if (mediaRecorder && mediaRecorder.state === "recording") stopJustificationRecording(); else startJustificationRecording(); };
trBtn.onclick = async () => {
  if (!recordedAudioBlob) { $("#trMsg").textContent = "Record your answer before transcribing."; return; }
  trBtn.disabled = true; setAv(echo, "work"); trRow.classList.add("loading");
  $("#trMsg").textContent = "Transcribing with Whisper on this Mac";
  const tEl = $("#trTime"), t0 = performance.now();
  const iv = setInterval(() => tEl.textContent = ((performance.now() - t0) / 1000).toFixed(1) + " s", 250);
  try {
    const form = new FormData(); form.append("audio", recordedAudioBlob, "learner_justification.webm");
    const response = await fetch("/api/transcribe-justification", { method: "POST", body: form, headers: activeRunId ? { "X-Run-ID": activeRunId } : {} });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `Transcription failed with status ${response.status}`);
    // Anything already typed stays; the transcript is added after it.
    const typed = tx.value.trim();
    learnerJustification = typed ? { ...data, source: "edited_transcript" } : data;
    const full = data.transcript || "", w = full.split(" ");
    clearInterval(iv); trRow.classList.remove("loading");
    tx.value = typed ? typed + " " : "";
    if (reduce) tx.value += full; else for (let i = 0; i < w.length; i++) { tx.value += (i ? " " : "") + w[i]; await wait(45); }
    transcripts[txKey()] = tx.value; recordedAudioBlob = null;
    setAv(echo, "happy"); hop(echo);
    $("#trMsg").textContent = `Transcript ${typed ? "added after your typed text" : "ready"} from ${data.model || "the speech model"}. Edit it before you submit.`;
  } catch (error) {
    console.error(error);
    setAv(echo, "caution");
    $("#trMsg").textContent = `Transcription failed: ${error.message}. Type your answer instead.`;
    trBtn.disabled = false;
  } finally { clearInterval(iv); trRow.classList.remove("loading"); updateChecks(); paintWorkspace("reasoning"); }
};
function speakReasoningPrompt() {
  const p = currentReasoningPrompt; if (!p) return;
  if (!("speechSynthesis" in window)) { $("#trMsg").textContent = "Reading aloud is not supported in this browser."; return; }
  if (speechSynthesis.speaking) { stopPromptSpeech(); return; }
  const u = new SpeechSynthesisUtterance([p.title, p.text, p.cues.length ? `Try to refer to: ${p.cues.join(", ")}.` : ""].join(". "));
  u.rate = .92;
  u.onstart = () => { $("#speakBtn").textContent = "Stop reading"; };
  u.onend = u.onerror = () => { $("#speakBtn").textContent = "Read the prompt aloud"; };
  speechSynthesis.speak(u);
}
function stopPromptSpeech() { if ("speechSynthesis" in window && speechSynthesis.speaking) speechSynthesis.cancel(); }
$("#speakBtn").onclick = speakReasoningPrompt;
function getLearnerJustification() {
  const transcript = tx.value.trim();
  return {
    mode: learnerJustification?.source === "spoken_justification" ? "speech" : "typed",
    prompt: currentReasoningPrompt, transcript,
    model: learnerJustification?.model || null,
    source: learnerJustification?.source || "typed_justification",
  };
}
function renderReasoning() {
  currentReasoningPrompt = buildReasoningPrompt(selectedAction, selectedEvidence);
  const p = currentReasoningPrompt;
  $("#promptTitle").textContent = p.title; $("#promptText").textContent = p.text;
  $("#cues").innerHTML = p.cues.map(c => `<span data-cue="${esc(c.toLowerCase())}">${esc(c)}</span>`).join("");
  tx.value = transcripts[txKey()] ?? "";
  trBtn.disabled = !recordedAudioBlob;
  $("#tpRPayload").innerHTML = jsonOut({ selectedAction, selectedEvidence, supportRole: getEvidenceSupportRole(selectedEvidence) || "not set" });
  updateChecks(); paintWorkspace("reasoning");
}
function updateChecks() {
  const text = tx.value.toLowerCase(), n = words(tx.value), cues = (currentReasoningPrompt?.cues || []).map(c => c.toLowerCase());
  const hitCue = c => text.includes(c) || c.split(" ").some(w => w.length > 4 && text.includes(w));
  const cue = cues.some(hitCue);
  $$("#cues span").forEach(sp => { const hit = hitCue(sp.dataset.cue || ""); if (hit && !sp.classList.contains("hit")) { sp.classList.add("hit", "pop"); setTimeout(() => sp.classList.remove("pop"), 400); } else if (!hit) sp.classList.remove("hit"); });
  const items = [[!!selectedAction, "Action selected", "Choose an action"], [!!selectedEvidence, "Evidence selected", "Choose evidence"], [n >= 12, "Reasoning has enough detail", `Add more reasoning detail (${n} of 12 words)`], [cue, "References an evidence cue", "Mention a visible evidence cue"]];
  $("#checks").innerHTML = items.map(([ok, y, no]) => `<li class="${ok ? "ok" : "todo"}"><i aria-hidden="true">${ok ? "✓" : "!"}</i>${ok ? y : no}</li>`).join("");
  const open = items.filter(i => !i[0]).length;
  $("#submitHint").textContent = open ? `${open} check${open > 1 ? "s" : ""} open. You can still submit.` : "Ready to submit.";
  $("#submitBtn").disabled = !selectedAction || !selectedEvidence || isEvaluatingTurn;
}
tx.addEventListener("input", () => { transcripts[txKey()] = tx.value; if (learnerJustification && tx.value.trim() !== (learnerJustification.transcript || "").trim()) learnerJustification = { ...learnerJustification, source: "edited_transcript" }; updateChecks(); $$("[data-stepper]", pageEls.reasoning).forEach(renderStepper); });
$("#tpFill").onclick = () => {
  if (!selectedAction || !selectedEvidence) return;
  tx.value = [
    `I selected ${selectedAction.title} to investigate this incident.`,
    "I chose this screenshot because it might help review the recent activity.",
    "I will check the visible identity, source IP, event time, service, status, and risk signal to see what the evidence actually establishes.",
    "If those details do not prove the action, I would compare this with another source before drawing a conclusion.",
  ].join(" ");
  transcripts[txKey()] = tx.value; learnerJustification = null; updateChecks(); paintWorkspace("reasoning");
};
$("#tpClear").onclick = () => { tx.value = ""; transcripts[txKey()] = ""; learnerJustification = null; updateChecks(); paintWorkspace("reasoning"); };
$("#submitBtn").onclick = () => {
  if (!selectedAction || !selectedEvidence || isEvaluatingTurn) return;
  if (mediaRecorder && mediaRecorder.state === "recording") { $("#trMsg").textContent = "Stop recording and transcribe it before you submit."; return; }
  lastEval = { evidence: selectedEvidence, action: selectedAction, before: stateOf(appState), after: null, vlm: null, sec: null, coach: null, scheme: schemeOf(turnPayload), turn: turnNo(), live: true };
  go("evaluation");
};

// ---------- evaluation ----------
const frame = $("#frame"), stamp = $("#bigstamp"), nextStrip = $("#nextStrip");
const MX = [0, 1, 2].map(i => $("#M" + i)), MB = MX.map(m => $(".mbody", m)), ML = MX.map(m => $(".mload", m)), MS = MX.map(m => $(".mstat", m)), MT = MX.map(m => $(".mtoggle", m)), AVS = MX.map(m => $(".mhead .av", m));
const bodies = [$("#B1"), $("#B2"), $("#B3")];
let lastEval = null, run = 0, tick = 0, markEls = [], evSize = { w: 1200, h: 760 };
// Resolves once the evaluation stream has closed and the server has released its workspace lock.
let evalSettled = Promise.resolve();
function regionFor(template, fact) {
  const map = RMAP[template]; if (!map) return null;
  const text = fact.trim(); let key, val;
  const ci = text.indexOf(":");
  if (template === "guardduty" && /^[A-Z][A-Za-z]+:[A-Za-z]+\//.test(text)) { key = "finding type"; val = text; }
  else if (ci > 0 && ci < 40) { key = text.slice(0, ci).toLowerCase().trim(); val = text.slice(ci + 1).trim(); }
  else { key = text.toLowerCase(); val = text; }
  let best = null;
  for (const r of map) if (key.startsWith(r[0]) && (!best || r[0].length > best[0].length)) best = r;
  if (!best) return null;
  const [, x, y, w, h, cw] = best;
  return { x, y, w: w ?? Math.min(720, Math.round(val.length * cw + 12)), h };
}
function keywords(sec) {
  const src = `${sec.key} ${sec.whose}`.replace(/\b[A-Za-z][A-Za-z ]{1,24}:\s/g, " ");
  return new Set(src.split(/[\s,;]+/).map(t => t.replace(/^[("'“]+|[)"'”.,;]+$/g, "").toLowerCase()).filter(t => t.length >= 5 && !["shown", "event", "source", "address", "identity", "change"].includes(t)));
}
const isKw = (w, kw) => { const c = w.replace(/^[("'“]+|[)"'”.,;:]+$/g, "").toLowerCase(); if (c.length < 4) return false; if (kw.has(c)) return true; return c.length >= 7 && [...kw].some(k => k.includes(c) || c.includes(k)); };
function parseClaims(s) {
  const blocks = (s.assessment || "").split(/\n\s*\n/).map(b => b.trim()).filter(Boolean);
  const notes = blocks.map(b => b.split("\n").slice(1).join(" ").trim());
  if (s.claims.length) return s.claims.map((c, i) => ({ q: c.q, a: c.a, note: notes[i] || "" }));
  return blocks.length ? [{ q: blocks[0].split("\n")[0].replace(/^Learner statement:\s*/, "").replace(/[“”]/g, ""), a: "needs_detail", note: notes[0] || "" }] : [];
}
// Splits only where a full stop is followed by a space, so names like s.brennan stay whole.
const sentences = t => { const x = String(t || "").trim(); return x ? x.split(/(?<=[.!?])\s+/) : []; };
function coachHTML(text, done) {
  let cavDone = false;
  return sentences(text || "").map(sn => {
    const ws = sn.trim().split(/\s+/).map(w => `<span class="w${done ? " on" : ""}">${esc(w)}</span>`).join(" ");
    if (!cavDone && /\b(however|but|lacks|unclear|does not|cannot|not confirm|we need)\b/i.test(sn)) { cavDone = true; return `<span class="cav${done ? " on" : ""}">${ws}</span>`; }
    return ws;
  }).join(" ");
}
function wordsHTML(text, kw) { return (text || "").split(" ").map(w => `<span class="w on done${isKw(w, kw) ? " kw" : ""}">${esc(w)}</span>`).join(" "); }
const K = { weak: 0, partial: 1, strong: 2 };
const factHTML = f => { const ci = f.indexOf(": "); return ci > 0 && ci < 32 ? `<em>${esc(f.slice(0, ci))}</em>${esc(f.slice(ci + 2))}` : esc(f); };
function buildVision(ev, anim) {
  const v = ev.vlm;
  return `<div class="v-top"><span class="etype">${esc(TYPE_LABEL[v.type] || v.type.replace(/_/g, " "))}</span><span class="v-count"><b class="fc">${anim ? 0 : v.facts.length}</b>visible facts</span></div>
    <p class="v-sum">${esc(v.summary || "No summary returned.")}</p>
    ${v.query ? `<p class="v-q"><span class="eyebrow">Query editor text (not an event)</span><code>${esc(v.query)}</code></p>` : ""}
    <ol class="facts">${v.facts.map((f, i) => `<li data-i="${i}"><b>${i + 1}</b><span>${factHTML(f)}</span></li>`).join("")}</ol>
    <p class="v-warn">${v.warnings.length ? `<i class="warn">!</i>${esc(v.warnings.join(" · "))}` : `<i>✓</i>Extraction warnings: none reported`}</p>`;
}
function keyRows(key) {
  const parts = (key || "").split(/,\s(?=[A-Z][A-Za-z ]{1,30}:\s)/);
  const rows = parts.map(p => { const i = p.indexOf(": "); return i > 0 && i < 32 ? [p.slice(0, i), p.slice(i + 2)] : null; });
  return rows.every(Boolean) ? rows : null;
}
function buildSecurity(ev, anim) {
  const s = ev.sec, vk = vkey(s.verdict), kr = keyRows(s.key), sc = ev.scheme || {};
  return `<div class="meter3" style="--k:${anim ? 0 : K[vk]}"><i class="knob ${vk}"></i><span>Weak</span><span>Partial</span><span>Strong</span></div>
    <p class="vline"><span class="vt ${vk}">${esc(s.verdict)}</span><span class="whose">Whose evidence: ${esc(s.whose || "none shown")}</span></p>
    <div class="keyf"><span class="eyebrow">Key fact</span>${kr ? `<dl class="kf">${kr.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("")}</dl>` : `<p class="kf-s">${esc(s.key || "No key fact returned.")}</p>`}</div>
    <p class="reason">${anim ? "" : wordsHTML(s.reasoning, keywords(s))}</p>
    ${s.rows && s.rows.length ? `<p class="rowsref">Supporting event rows: ${s.rows.join(", ")}</p>` : ""}
    <div class="claims">${parseClaims(s).map(c => `<div class="claim"><p class="eyebrow">Your statement</p><p class="q">“${esc(c.q)}”</p><span class="cstamp ${esc(c.a)}">${CLAIM[c.a] || esc(c.a)}</span>${c.note ? `<p class="cnote">${esc(c.note)}</p>` : ""}</div>`).join("")}</div>
    ${s.risk ? `<div class="risk"><div class="tape"><span>NOT PROVEN</span></div><p><b>Risk of wrong interpretation.</b> ${esc(s.risk)}</p></div>` : ""}
    ${s.next ? `<p class="nextfocus"><span class="eyebrow">Recommended next focus</span><br>${esc(s.next)}</p>` : ""}
    ${sc.strong ? `<details class="scheme"><summary>Marking scheme for Turn ${ev.turn}</summary><ol>${["strong", "partial", "weak"].map(l => `<li class="${l}${l === vk ? " hit" : ""}"><b>${VWORD[l]}</b><span>${esc(sc[l] || "")}</span></li>`).join("")}</ol></details>` : ""}`;
}
function buildCoach(ev, anim) {
  return `<div class="c-bubble"><p class="c-text">${anim ? "" : coachHTML(ev.coach.feedback, true)}</p></div>
    ${ev.coach.nudge ? `<div class="nudge"><i class="pin"></i><b>Coach nudge</b>${esc(ev.coach.nudge)}</div>` : ""}`;
}
const BUILD = [buildVision, buildSecurity, buildCoach];
function buildMarks(ev) {
  $$(".mark", frame).forEach(m => m.remove());
  const t = evidenceTemplate(ev.evidence), e = evSize;
  markEls = ev.vlm.facts.map((f, i) => {
    const r = regionFor(t, f); if (!r) return null;
    const m = document.createElement("div"); m.className = "mark";
    m.style.cssText = `left:${(r.x / e.w * 100).toFixed(2)}%;top:${(r.y / e.h * 100).toFixed(2)}%;width:${Math.min(r.w / e.w * 100, 100 - r.x / e.w * 100).toFixed(2)}%;height:${(r.h / e.h * 100).toFixed(2)}%`;
    m.style.setProperty("--i", i); m.innerHTML = `<b>${i + 1}</b>`; frame.insertBefore(m, $("#pen")); return m;
  });
}
function hoverLinks() {
  $$(".facts li", bodies[0]).forEach(li => { const m = markEls[+li.dataset.i]; if (!m) return; li.onmouseenter = () => m.classList.add("hl"); li.onmouseleave = () => m.classList.remove("hl"); });
}
const short = f => f.length > 48 ? f.slice(0, 46) + "…" : f;
function say(who, msg) { $("#tkWho").textContent = who; $("#tkMsg").textContent = msg; const t = $("#ticker"); t.classList.remove("pop"); void t.offsetWidth; t.classList.add("pop"); }
function openCard(i, animate = true) {
  MX.forEach((m, j) => { const on = j === i; m.classList.toggle("open", on); MT[j].setAttribute("aria-expanded", on); });
  if (animate && i >= 0 && !reduce) MB[i].animate([{ opacity: 0, transform: "translateY(10px)" }, { opacity: 1, transform: "none" }], { duration: 320, easing: "cubic-bezier(.2,.8,.2,1)" });
}
const activate = i => MX.forEach((m, j) => m.classList.toggle("active", j === i));
function setStatus(i, html) { MS[i].innerHTML = html; MS[i].classList.remove("pop"); void MS[i].offsetWidth; MS[i].classList.add("pop"); }
function stageStart(i, msg) {
  MX[i].classList.remove("wait"); activate(i); openCard(i);
  ML[i].hidden = false; bodies[i].hidden = true; MB[i].scrollTop = 0;
  $(".ld-msg", ML[i]).textContent = msg;
  setAv(AVS[i], "work");
  const t0 = performance.now(), el = $(".ld-time", ML[i]); el.textContent = "0.0 s";
  MS[i].innerHTML = `<span class="work">Working · <span class="wt">0.0 s</span></span>`; const wt = $(".wt", MS[i]);
  clearInterval(tick);
  tick = setInterval(() => { const sec = ((performance.now() - t0) / 1000).toFixed(1) + " s"; el.textContent = sec; wt.textContent = sec; }, 250);
}
function stageOutput(i) { clearInterval(tick); ML[i].hidden = true; bodies[i].hidden = false; MB[i].scrollTop = 0; }
const scrollable = c => c.scrollHeight > c.clientHeight + 4 && getComputedStyle(c).overflowY !== "visible";
function follow(i, el, instant) {
  if (!el) return;
  const c = MB[i];
  if (scrollable(c)) {
    const top = el.offsetTop, bottom = top + el.offsetHeight, vt = c.scrollTop, vb = vt + c.clientHeight;
    let target = null;
    if (bottom > vb - 8) target = Math.min(top - 8, bottom - c.clientHeight + 16);
    else if (top < vt) target = top - 8;
    if (target !== null) c.scrollTo({ top: Math.max(0, target), behavior: instant || reduce ? "auto" : "smooth" });
  } else if (!reduce && !skipAnim) el.scrollIntoView({ block: "nearest", behavior: instant ? "auto" : "smooth" });
}
async function flyFacts(ev, me) {
  const lis = $$(".facts li", bodies[0]), fc = $(".fc", bodies[0]), flights = [];
  let count = 0;
  const land = i => { lis[i].classList.add("on"); fc.textContent = ++count; const f2 = $(".fc2", MS[0]); if (f2) f2.textContent = count; };
  for (let i = 0; i < lis.length; i++) {
    if (me !== run) return;
    const m = markEls[i]; if (m) m.classList.add("on");
    follow(0, lis[i], true);
    if (reduce || skipAnim) { land(i); continue; }
    const fr = frame.getBoundingClientRect();
    const src = m ? m.getBoundingClientRect() : { left: fr.left + fr.width / 2 - 60, top: fr.top + fr.height / 2, height: 20 };
    const dst = lis[i].getBoundingClientRect();
    const chip = document.createElement("div"); chip.className = "chip"; chip.setAttribute("aria-hidden", "true"); chip.innerHTML = `<b>${i + 1}</b>${esc(short(ev.vlm.facts[i]))}`;
    document.body.appendChild(chip);
    const x0 = src.left, y0 = src.top + (src.height || 20) / 2 - 12, x1 = dst.left, y1 = dst.top, xm = (x0 + x1) / 2, ym = Math.min(y0, y1) - 70;
    const an = chip.animate([
      { transform: `translate(${x0}px,${y0}px) scale(.5)`, opacity: 0 },
      { transform: `translate(${x0}px,${y0 - 20}px) scale(1.08)`, opacity: 1, offset: .18 },
      { transform: `translate(${xm}px,${ym}px) scale(1)`, opacity: 1, offset: .55 },
      { transform: `translate(${x1}px,${y1}px) scale(.96)`, opacity: .9 },
    ], { duration: 780, easing: "cubic-bezier(.45,0,.2,1)" });
    flights.push(an.finished.then(() => { chip.remove(); land(i); }).catch(() => chip.remove()));
    await pace(270);
  }
  await Promise.all(flights);
}
async function popWords(el, text, me) {
  el.innerHTML = coachHTML(text, false);
  const spans = $$(".w", el);
  for (let i = 0; i < spans.length; i++) { if (me !== run) return; spans[i].classList.add("on"); await pace(34); }
  $$(".cav", el).forEach(c => c.classList.add("on"));
  await pace(450);
}
async function unredact(el, text, kw, me) {
  el.innerHTML = (text || "").split(" ").map(w => `<span class="w${isKw(w, kw) ? " kw" : ""}">${esc(w)}</span>`).join(" ");
  const spans = $$(".w", el);
  for (let i = 0; i < spans.length; i++) { if (me !== run) return; spans[i].classList.add("on"); if (i % 2) await pace(26); }
  await pace(320);
}
function verdictHit(vk, verdict) {
  stamp.className = "bigstamp " + vk; stamp.textContent = verdict; void stamp.offsetWidth; stamp.classList.add("on");
  setTimeout(() => {
    if (reduce || skipAnim) return;
    const sh = $(".shock", frame); sh.className = "shock " + vk; void sh.offsetWidth; sh.classList.add("go");
    const fl = $(".flash", frame); fl.classList.remove("go"); void fl.offsetWidth; fl.classList.add("go");
    frame.classList.remove("shake"); void frame.offsetWidth; frame.classList.add("shake");
    for (let i = 0; i < 12; i++) {
      const d = document.createElement("i"); d.className = "splat " + vk; frame.appendChild(d);
      const ang = i / 12 * Math.PI * 2 + Math.random() * .4, dist = 80 + Math.random() * 100;
      d.animate([{ transform: "translate(-50%,-50%) scale(1)", opacity: 1 }, { transform: `translate(calc(-50% + ${Math.cos(ang) * dist}px), calc(-50% + ${Math.sin(ang) * dist}px)) scale(.3)`, opacity: 0 }], { duration: 650 + Math.random() * 250, easing: "cubic-bezier(.2,.8,.3,1)" }).finished.then(() => d.remove(), () => d.remove());
    }
  }, 230);
  setTimeout(() => { if (!stamp.classList.contains("on")) return; stamp.className = "bigstamp static " + vk; void stamp.offsetWidth; stamp.classList.add("settle"); }, 1700);
}
const DESK = "(min-width: 961px) and (min-height: 620px)";
function fitEval() {
  if (current !== "evaluation" || !lastEval) return;
  const g = $(".evgrid"), img = $("#frameImg"), e = evSize, left = $(".evleft");
  if (matchMedia(DESK).matches) {
    const H = Math.max(420, Math.floor(innerHeight - (g.getBoundingClientRect().top + scrollY) - 14));
    g.style.height = H + "px";
    const availH = H - nextStrip.offsetHeight - $(".capline").offsetHeight - 24;
    const sc = Math.min(left.clientWidth / e.w, availH / e.h, 1);
    img.style.width = Math.floor(e.w * sc) + "px";
  } else { g.style.height = ""; img.style.width = ""; }
}
addEventListener("resize", () => fitEval());
function renderNext(mode, message) {
  const st = lastEval.before, af = lastEval.after || lastEval.before, last = lastEval.turn >= maxTurns(), n = lastEval.turn + 1;
  nextStrip.className = "next" + (mode === "wait" ? " wait" : mode === "loading" ? " loading" : "");
  if (mode === "loading") nextStrip.innerHTML = `<div><p class="eyebrow">Coach · ${last ? "final debrief" : "next turn"}</p><div class="ld-top" style="margin-top:10px"><span class="ld-msg">${last ? "Preparing the final coach debrief" : `Generating Turn ${n} and checking its evidence`}</span><span class="ld-time" id="nxTime">0.0 s</span></div><div class="ld-bar"><i></i></div></div><button class="btn" disabled>${last ? "Preparing Final Report…" : `Continue to Turn ${n}`}</button>`;
  else if (mode === "wait") nextStrip.innerHTML = `<div><p class="eyebrow">${last ? "Final coach debrief" : `Turn ${n}`}</p><p style="margin-top:6px">Available when the three models finish. Your progress updates after the verdict.</p></div><button class="btn" disabled>${last ? "View final debrief" : `Continue to Turn ${n}`}</button>`;
  else if (mode === "fault") nextStrip.innerHTML = `<div><p class="eyebrow">Evaluation stopped</p><p style="margin-top:6px">No verdict or progress was awarded. Retry the evaluation or change your answer.</p></div><button class="btn" disabled>Continue</button>`;
  else if (mode === "skipped") nextStrip.innerHTML = `<div><p class="eyebrow">Next Turn Skipped (Test Mode)</p><p style="margin-top:6px">${esc(message || "Review the verdict and feedback above. This run cannot continue after a skipped turn.")}</p></div><button class="btn" disabled>Next turn skipped</button>`;
  else nextStrip.innerHTML = `<div><p class="eyebrow">${last ? "Final Coach Debrief Ready" : "Next Incident Turn Ready"}</p><div class="deltas"><span>Containment ${st.c} → <em>${af.c}%</em></span><span>Visibility ${st.v} → <em>${af.v}%</em></span><span>Risk <em>${esc(af.risk)}</em></span></div></div><button class="btn" data-go="${last ? "debrief" : "handoff"}" id="continueBtn">${last ? "View final debrief" : `Continue to Turn ${n}`}</button>`;
  fitEval();
  if (mode === "loading") { const t0 = performance.now(); clearInterval(tick); tick = setInterval(() => { const el = $("#nxTime"); if (el) el.textContent = ((performance.now() - t0) / 1000).toFixed(1) + " s"; }, 250); }
}
function fillPayload() { if (lastEval) $("#tpVPayload").innerHTML = jsonOut({ selectedAction: lastEval.action, selectedEvidence: lastEval.evidence, learnerJustification: lastEval.justification || getLearnerJustification() }); }
function renderEvaluation() {
  if (!lastEval) return;
  const e = lastEval.evidence;
  $("#evEyebrow").textContent = `AI evaluation · Turn ${lastEval.turn} of ${maxTurns()}`;
  $("#frameImg").src = e.imageUrl; $("#frameImg").alt = e.title;
  $("#frameCap").textContent = `Evaluated evidence: ${e.title}`;
  $$("[data-stepper]", pageEls.evaluation).forEach(renderStepper);
  fillPayload(); renderTrack();
  requestAnimationFrame(fitEval);
}
function keyIdx(ev) {
  const key = (ev.sec.key || "").toLowerCase(), rows = ev.sec.rows || [], out = [];
  ev.vlm.facts.forEach((f, i) => {
    const ci = f.indexOf(":"), has = ci > 0 && ci < 40;
    const lab = has ? f.slice(0, ci).toLowerCase().trim() : "", val = (has ? f.slice(ci + 1) : f).toLowerCase().trim();
    const rowHit = /^row \d+/.test(lab) && rows.includes(+lab.slice(4));
    const toks = val.split(/[\s,/:]+/).filter(t => t.length >= 6 && !/^\d+$/.test(t));
    if (rowHit || (lab.length > 3 && key.includes(lab)) || (val.length >= 4 && key.includes(val)) || toks.some(t => key.includes(t))) out.push(i);
  });
  return out.slice(0, 3);
}
function applyKeyMarks(keys) {
  markEls.forEach((m, i) => { if (!m) return; m.classList.toggle("key", keys.includes(i)); m.classList.toggle("k1", i === keys[0]); m.classList.toggle("dim", keys.length > 0 && !keys.includes(i)); });
}
const VROW = { strong: 0, partial: 1, weak: 2 };
function dossierHTML(ev) {
  const sc = ev.scheme || {};
  if (!sc.strong) return "";
  return `<p class="dz-h"><i class="dz-shield"></i>Marking scheme · Turn ${ev.turn}</p><ol class="dz-rows">${["strong", "partial", "weak"].map(l => `<li class="${l}"><b>${VWORD[l]}</b><span>${esc(sc[l] || "")}</span></li>`).join("")}<i class="dz-hl"></i></ol>`;
}
function scribble(cx, cy, rx, ry) {
  const pts = [], n = 40, a0 = -2.3 + Math.random() * .4;
  for (let k = 0; k <= n; k++) {
    const t = a0 + (k / n) * Math.PI * 2.2, j = 1 + Math.sin(k * 1.7) * .03 + (Math.random() - .5) * .03;
    pts.push((cx + Math.cos(t) * rx * j).toFixed(1) + " " + (cy + Math.sin(t) * ry * j).toFixed(1));
  }
  return "M" + pts.join(" L");
}
function penHTML(ev, keys) {
  const e = evSize, vk = vkey(ev.sec.verdict), t = evidenceTemplate(ev.evidence);
  const rs = keys.map(i => regionFor(t, ev.vlm.facts[i])).filter(Boolean);
  if (!rs.length) return "";
  const note = { strong: "this proves it", partial: "only part of it", weak: "doesn't show it" }[vk];
  const r0 = rs[0], right = r0.x + r0.w + 280 < e.w;
  const lx = Math.round(right ? r0.x + r0.w + 36 : Math.max(20, r0.x)), ly = Math.round(right ? r0.y + r0.h / 2 + 12 : Math.min(e.h - 24, r0.y + r0.h + 60));
  return rs.map(r => `<path d="${scribble(r.x + r.w / 2, r.y + r.h / 2, r.w / 2 + 24, r.h / 2 + 13)}" pathLength="1"/>`).join("") + `<text class="pnote" x="${lx}" y="${ly}" transform="rotate(-4 ${lx} ${ly})">${note}</text>`;
}
function setPen(ev, keys, drawn, stat) {
  const pen = $("#pen"), e = evSize, t = evidenceTemplate(ev.evidence);
  pen.setAttribute("viewBox", `0 0 ${e.w} ${e.h}`); pen.innerHTML = penHTML(ev, keys);
  pen.setAttribute("class", "pen" + (drawn ? " drawn" : "") + (stat ? " static" : ""));
  const nib = $(".nib", frame), r = keys.length && regionFor(t, ev.vlm.facts[keys[0]]);
  nib.style.left = (r ? (r.x + r.w / 2) / e.w * 100 : 50).toFixed(1) + "%"; nib.style.top = (r ? (r.y - 6) / e.h * 100 : 45).toFixed(1) + "%";
}

// ---------- next-turn build board ----------
const flip = $("#flip"), board = $("#board"), flipBtn = $("#flipBtn");
function boardHTML(ready, content) {
  const acts = content?.actions || [], evs = content?.evidence || [];
  const n = lastEval.turn + 1;
  if (lastEval.turn >= maxTurns()) {
    const secs = ["Executive summary", "Response timeline", "Indicators and findings", "Decision scorecard", "Recommendations", "Sign-off"];
    return `<div class="bd fin"><div class="bd-head"><p class="eyebrow">Coach · compiling</p><h2>Final <em>report</em></h2></div>
      <div class="bd-doc"><div class="mini"><i class="band"></i><b class="mt">Incident response report</b><span class="mid">${esc(scenarioTitle())} · five turns</span><ol>${secs.map((s, i) => `<li><span>${i + 1}</span>${s}<i class="tick">✓</i></li>`).join("")}</ol><i class="mseal">Filed</i></div></div>
      <ol class="bd-steps"><li><i>1</i>Review all five turns</li><li><i>2</i>Score the decisions</li><li><i>3</i>Compile the report</li></ol>
      <div class="bd-ready">Report ready</div></div>`;
  }
  // Outline while the coach writes; filled with the real turn once it has been generated.
  return `<div class="bd"><div class="bd-head"><p class="eyebrow">Coach · building the next turn</p><h2>Turn ${n}<em>${esc(stageName(n))}</em></h2><p class="bd-brief">${ready ? esc(ready) : '<i class="skl" style="--w:92%"></i><i class="skl" style="--w:74%"></i>'}</p></div>
    <div class="bd-cols"><div class="bd-acts"><p class="eyebrow">Response actions</p>${["A", "B", "C"].map((l, i) => `<div class="bd-act"><b>${l}</b><span>${acts[i] ? esc(acts[i]) : `<i class="skl" style="--w:${78 - i * 9}%"></i>`}</span></div>`).join("")}</div>
    <div class="bd-evs"><p class="eyebrow">Evidence screenshots</p>${["A", "B", "C"].map((l, i) => `<figure class="bd-ev"><div class="slot">${evs[i]?.imageUrl ? `<img src="${esc(evs[i].imageUrl)}" alt="">` : ""}<i class="lscan"></i><i class="chk">✓</i></div><figcaption>Exhibit ${n}-${l}</figcaption></figure>`).join("")}</div></div>
    <ol class="bd-steps"><li><i>1</i>Coach writes Turn ${n}</li><li><i>2</i>Render screenshots</li><li><i>3</i>Vision checks them</li></ol>
    <div class="bd-ready">Turn ${n} ready</div></div>`;
}
function flipLabel() { flipBtn.textContent = flip.classList.contains("flipped") ? "Show evaluated evidence" : lastEval.turn >= maxTurns() ? "Preview the final report" : `Preview Turn ${lastEval.turn + 1}`; }
flipBtn.onclick = () => { flip.classList.toggle("flipped"); flipLabel(); };
// Step 1 stays active while the coach writes; the rest plays once the turn really exists.
let boardRun = 0;
function startBoard() {
  boardRun++;
  board.innerHTML = boardHTML();
  const bd = $(".bd", board), steps = $$(".bd-steps li", bd);
  steps.forEach((s, j) => s.className = j === 0 ? "now" : "");
  flip.classList.add("flipped"); flipBtn.hidden = true;
}
async function boardReady(message, content) {
  const me = ++boardRun;
  board.innerHTML = boardHTML(message, content);
  const bd = $(".bd", board), steps = $$(".bd-steps li", bd);
  const step = i => steps.forEach((s, j) => s.className = j < i ? "done" : j === i ? "now" : "");
  if (reduce || !content || !flip.classList.contains("flipped")) {
    bd.classList.add("s1", "s2", "s3", "isready", "static"); steps.forEach(s => s.className = "done");
  } else {
    step(1); bd.classList.add("s1"); await pace(1300); if (me !== boardRun) return;
    step(2); bd.classList.add("s2"); await pace(1300); if (me !== boardRun) return;
    step(3); bd.classList.add("s3"); await pace(900); if (me !== boardRun) return;
    bd.classList.add("isready", "go");
  }
  setAv(AVS[2], "happy"); hop(AVS[2]);
}

// ---------- verdict track, faults ----------
function renderTrack() {
  const vt = $("#vtrack"); if (!vt) return;
  const hist = history_(), cur = lastEval?.turn || turnNo();
  vt.innerHTML = Array.from({ length: maxTurns() }, (_, i) => i + 1).map(n => {
    const h = n !== cur && hist.find(x => Number(x.turn) === n), k = h && vkey(h.verdict);
    return `<li class="${k || (n === cur ? "cur" : "")}" data-t="${n}"><span>T${n}</span>${k ? `<b>${VWORD[k]}</b>` : n === cur ? "<b>…</b>" : ""}</li>`;
  }).join("") + `<li class="sli"><em class="streak" hidden></em></li>`;
}
function streakOf(vk) {
  let n = 0; const hist = history_(), cur = lastEval.turn;
  for (let t = cur; t >= 1; t--) { const h = t === cur ? { verdict: vk } : hist.find(x => Number(x.turn) === t); if (h && vkey(h.verdict) === "strong") n++; else break; }
  return n;
}
function landStamp(vk, animate) {
  const li = $(`#vtrack li[data-t="${lastEval.turn}"]`); if (!li || li.classList.contains(vk)) return;
  const put = pop => {
    li.className = vk + (pop ? " land" : ""); li.innerHTML = `<span>T${lastEval.turn}</span><b>${VWORD[vk]}</b>`;
    const em = $("#vtrack .streak"), n = streakOf(vk);
    if (n >= 2) { em.hidden = false; em.textContent = `${n} strong in a row`; if (pop) { em.classList.remove("pop"); void em.offsetWidth; em.classList.add("pop"); } } else em.hidden = true;
  };
  if (!animate || reduce || skipAnim) { put(false); return; }
  const s = stamp.getBoundingClientRect(), d = li.getBoundingClientRect();
  const chip = document.createElement("div"); chip.className = "flystamp " + vk; chip.setAttribute("aria-hidden", "true"); chip.textContent = VWORD[vk]; document.body.appendChild(chip);
  const cw = chip.offsetWidth, ch = chip.offsetHeight;
  const x0 = s.left + s.width / 2 - cw / 2, y0 = s.top + s.height / 2 - ch / 2, x1 = d.left + d.width / 2 - cw / 2, y1 = d.top + d.height / 2 - ch / 2;
  chip.animate([
    { transform: `translate(${x0}px,${y0}px) rotate(-9deg) scale(1.7)`, opacity: 0 },
    { transform: `translate(${x0}px,${y0 - 24}px) rotate(-9deg) scale(1.4)`, opacity: 1, offset: .18 },
    { transform: `translate(${(x0 + x1) / 2}px,${Math.min(y0, y1) - 90}px) rotate(10deg) scale(1)`, opacity: 1, offset: .6 },
    { transform: `translate(${x1}px,${y1}px) rotate(-4deg) scale(.62)`, opacity: 1 },
  ], { duration: 950, easing: "cubic-bezier(.45,0,.2,1)" }).finished.then(() => { chip.remove(); put(true); }, () => chip.remove());
}
function clearFault() { $$(".merr", pageEls.evaluation).forEach(x => x.remove()); MX.forEach(m => m.classList.remove("failed")); frame.classList.remove("fault"); }
const WHO = ["Vision", "Security", "Coach"];
function fault(i, message, type) {
  run++; clearInterval(tick); boardRun++; endRun();
  frame.classList.remove("sweep", "judging", "coaching", "reread"); frame.classList.add("fault");
  if (i > 2) { flip.classList.remove("flipped"); i = 2; }
  ML[i].hidden = true; bodies[i].hidden = !lastEval[["vlm", "sec", "coach"][i]]; activate(i); openCard(i); MX[i].classList.add("failed");
  MB[i].insertAdjacentHTML("afterbegin", `<div class="merr" role="alert"><div class="merr-h"><i>!</i><b>${WHO[i]} model stopped</b></div><p>${esc(message)}</p>${type ? `<code>${esc(type)}</code>` : ""}<div class="cta-row"><button class="btn small" id="retryEval">Retry evaluation</button><button class="btn ghost small" id="backAnswer">Back to your answer</button></div></div>`);
  setStatus(i, `<span class="failed">Failed</span>`);
  MX.forEach((m, j) => { if (j > i) { MS[j].textContent = "Not run"; setAv(AVS[j], ""); } });
  setAv(AVS[i], "err");
  stamp.className = "bigstamp fault"; stamp.textContent = "No verdict"; void stamp.offsetWidth; stamp.classList.add("on");
  const li = $(`#vtrack li[data-t="${lastEval.turn}"]`); if (li) { li.className = "fail land"; li.innerHTML = `<span>T${lastEval.turn}</span><b>No verdict</b>`; }
  say(WHO[i], "The evaluation stopped. " + message);
  renderNext("fault");
  $("#retryEval").onclick = async () => { await evalSettled; clearFault(); startLiveEvaluation(); };
  $("#backAnswer").onclick = () => { clearFault(); go("reasoning"); };
}
function endRun() {
  isEvaluatingTurn = false; skipAnim = false;
  $("#showAll").hidden = true; MT.forEach(b => b.removeAttribute("aria-disabled"));
  document.body.classList.remove("models-running");
  updateRunControlsSafe();
}

// ---------- the live evaluation ----------
// Stream events queue their animations so each model's output plays in order,
// even when the next model finishes before the previous animation ends.
function resetEvalView() {
  clearFault(); renderTrack();
  bodies.forEach(b => { b.className = "lbody anim"; b.innerHTML = ""; b.hidden = true; });
  ML.forEach(l => l.hidden = true);
  MX.forEach((m, i) => { m.classList.add("wait"); m.classList.remove("active"); MS[i].textContent = "Waiting"; setAv(AVS[i], ""); });
  openCard(-1, false);
  $$(".mark", frame).forEach(m => m.remove()); markEls = [];
  stamp.className = "bigstamp"; stamp.textContent = "";
  frame.classList.remove("sweep", "judging", "coaching", "reread", "fault"); $("#pen").innerHTML = ""; $("#pen").setAttribute("class", "pen");
  $("#dossier").innerHTML = "";
  flip.classList.remove("flipped"); board.innerHTML = ""; flipBtn.hidden = true;
}
async function measureEvidence() {
  const img = $("#frameImg");
  try { await img.decode(); } catch (e) { /* use the default size */ }
  evSize = { w: img.naturalWidth || 1200, h: img.naturalHeight || 760 };
}
async function startLiveEvaluation() {
  const me = ++run, alive = () => me === run, ev = lastEval;
  let settle; evalSettled = new Promise(r => { settle = r; });
  ev.vlm = ev.sec = ev.coach = ev.done = ev.preview = null; ev.after = null; ev.justification = getLearnerJustification();
  isEvaluatingTurn = true; skipAnim = false; updateRunControlsSafe(); updateChecks();
  resetEvalView();
  $("#showAll").hidden = false; MT.forEach(b => b.setAttribute("aria-disabled", "true"));
  document.body.classList.add("models-running");
  renderNext("wait"); setPills(ev.before);
  await measureEvidence();
  let stage = 0, queue = Promise.resolve();
  const later = fn => { queue = queue.then(() => alive() ? fn() : null); return queue; };
  stageStart(0, "Reading the screenshot"); say("Vision", "The VLM is reading the selected evidence screenshot.");
  frame.classList.add("sweep");
  try {
    await readEventStream("/api/evaluate-stream", { selectedAction: ev.action, selectedEvidence: ev.evidence, learnerJustification: ev.justification, skipNextTurn: shouldSkipNextTurn() }, async event => {
      if (!alive()) return;
      const out = event.output || {};
      if (event.stage === "extraction_retry") later(() => {
        setAv(AVS[0], "work squint"); frame.classList.remove("sweep"); frame.classList.add("reread");
        $(".ld-msg", ML[0]).textContent = "Re-reading the screenshot (attempt 2 of 2)";
        say("Vision", out.message || "The evidence extraction was incomplete. Re-reading the screenshot once.");
      });
      else if (event.stage === "vlm") { ev.vlm = adaptVlm(out); later(() => playVision(ev, me)).then(() => { if (alive()) { stage = 1; startSecurity(ev); } }); }
      else if (event.stage === "security") { ev.sec = adaptSecurity(out); later(() => playSecurity(ev, me)).then(() => { if (alive()) { stage = 2; startCoach(ev); } }); }
      else if (event.stage === "coach") { ev.coach = adaptCoach(out); later(() => playCoach(ev, me)); }
      else if (event.stage === "next_turn_generation" || event.stage === "scenario_complete_generation") later(() => {
        stage = 3; activate(-1); setAv(AVS[2], "work");
        say("Coach", event.stage === "next_turn_generation" ? "I am preparing the next incident turn based on your previous action, evidence, and verdict." : "I am reviewing the full trace so the final debrief reflects all of your decisions.");
        renderNext("loading"); startBoard();
      });
      else if (event.stage === "next_turn_preview") ev.preview = out;
      else if (event.stage === "done") { ev.done = out; await later(() => finishLive(ev, me)); }
      else if (event.stage === "error") {
        const msg = out.error || "The evaluation failed.", type = out.error_type || "";
        await later(() => fault(Math.min(stage, 3), msg, type));
      }
    });
    await queue;
    // The run stays busy until the stream closes, so Continue never meets the server's lock.
    if (ev.done) endRun();
    if (alive() && !ev.done) fault(Math.min(stage, 3), "The evaluation stream ended before the final result arrived. Check the Flask terminal, then retry.", "");
  } catch (error) {
    console.error(error);
    await queue;
    if (alive()) fault(Math.min(stage, 3), error.message, "");
  } finally {
    settle();
    if (typeof finishQueuedRunLeave === "function") await finishQueuedRunLeave();
  }
}
async function playVision(ev, me) {
  const alive = () => me === run, B0 = bodies[0], nf = ev.vlm.facts.length;
  frame.classList.remove("sweep", "reread"); setAv(AVS[0], "happy");
  buildMarks(ev);
  B0.className = "lbody anim"; B0.innerHTML = buildVision(ev, true); stageOutput(0);
  setStatus(0, `<b class="fc2">0</b> facts`);
  say("Vision", `Extracted ${nf} visible fact${nf === 1 ? "" : "s"}${ev.vlm.warnings.length ? " with extraction warnings" : " with no extraction warnings"}.`);
  $(".etype", B0).classList.add("on"); await pace(250);
  $(".v-sum", B0).classList.add("on"); const vq = $(".v-q", B0); if (vq) vq.classList.add("on");
  await pace(300); if (!alive()) return;
  await flyFacts(ev, me); if (!alive()) return;
  const vw = $(".v-warn", B0); follow(0, vw); vw.classList.add("on"); await pace(600); if (!alive()) return;
  B0.classList.remove("anim"); hoverLinks(); setAv(AVS[0], "");
  const nw = ev.vlm.warnings.length;
  setStatus(0, `<b>${nf}</b> facts · ${nw ? nw + " warning" + (nw > 1 ? "s" : "") : "no warnings"}`);
  await pace(400);
}
function startSecurity(ev) {
  $("#dossier").innerHTML = dossierHTML(ev);
  stageStart(1, ev.scheme?.strong ? "Weighing the facts against the marking scheme" : "Weighing the facts against your action");
  say("Security", "The vision model has listed the visible facts. Now the security model is weighing them against this turn's marking scheme.");
  frame.classList.add("judging");
}
async function playSecurity(ev, me) {
  const alive = () => me === run, vk = vkey(ev.sec.verdict), keys = keyIdx(ev), B = bodies[1];
  B.className = "lbody anim"; B.innerHTML = buildSecurity(ev, true); stageOutput(1);
  const m3 = $(".meter3", B), hl = $(".dz-hl", frame);
  if (hl) { hl.classList.add("lock"); hl.style.setProperty("--row", VROW[vk]); }
  void m3.offsetWidth; m3.style.setProperty("--k", 2); await pace(420); m3.style.setProperty("--k", 0); await pace(420); m3.style.setProperty("--k", K[vk]);
  const row = $$(".dz-rows li", frame)[VROW[vk]]; if (row) row.classList.add("match");
  if (ev.scheme?.strong) say("Security", `Closest match in the marking scheme: ${VWORD[vk]}.`);
  await pace(800); if (!alive()) return;
  applyKeyMarks(keys);
  $(".vline", B).classList.add("on"); await pace(250);
  const kf = $(".keyf", B); kf.classList.add("on"); for (const r of $$(".kf > div", kf)) { r.classList.add("on"); await pace(170); }
  await pace(250); if (!alive()) return;
  frame.classList.remove("judging");
  const rEl = $(".reason", B), dec = unredact(rEl, ev.sec.reasoning, keywords(ev.sec), me); follow(1, rEl); await dec; if (!alive()) return;
  const rr = $(".rowsref", B); if (rr) rr.classList.add("on");
  verdictHit(vk, ev.sec.verdict); setAv(AVS[1], vk === "strong" ? "happy" : "caution"); hop(AVS[1]);
  setTimeout(() => { if (alive()) landStamp(vk, true); }, 800);
  setStatus(1, `<span class="vchip ${vk}">${esc(ev.sec.verdict)}</span>`);
  say("Security", `Verdict: ${ev.sec.verdict}.${ev.sec.key ? ` Key fact: ${ev.sec.key}` : ""}`);
  await pace(900); if (!alive()) return;
  for (const c of $$(".claim", B)) { follow(1, c); c.classList.add("on"); await pace(700); if (!alive()) return; }
  const rk = $(".risk", B); if (rk) { follow(1, rk); rk.classList.add("on"); await pace(900); if (!alive()) return; }
  const nx = $(".nextfocus", B); if (nx) { follow(1, nx); nx.classList.add("on"); await pace(800); if (!alive()) return; }
  B.classList.remove("anim");
  setPen(ev, keys, false, false);
}
function startCoach(ev) {
  stageStart(2, "Marking up the evidence and writing feedback");
  say("Coach", "The security verdict is ready. I am marking up the evidence and turning the verdict into feedback.");
  frame.classList.add("coaching");
}
async function playCoach(ev, me) {
  const alive = () => me === run, C = bodies[2];
  frame.classList.remove("coaching"); $("#pen").classList.add("drawn");
  C.className = "lbody anim"; C.innerHTML = buildCoach(ev, true); stageOutput(2); setAv(AVS[2], "talk");
  const pw = popWords($(".c-text", C), ev.coach.feedback, me); follow(2, $(".c-bubble", C)); await pw; if (!alive()) return;
  setAv(AVS[2], "happy");
  const nd = $(".nudge", C); if (nd) { follow(2, nd); nd.classList.add("on"); await pace(1100); if (!alive()) return; }
  setStatus(2, `Feedback${ev.coach.nudge ? " + nudge" : ""}`);
  say("Coach", "Feedback is ready. Read it as a coaching note: what you proved, what you assumed, and what to improve.");
  C.classList.remove("anim");
  await pace(700);
}
// Shows a finished evaluation in its final state (after the stream, or for a restored run).
function renderStatic(ev) {
  bodies.forEach((b, i) => { const has = [ev.vlm, ev.sec, ev.coach][i]; b.className = "lbody"; b.innerHTML = has ? BUILD[i](ev, false) : ""; b.hidden = !has; });
  ML.forEach(l => l.hidden = true);
  const vk = vkey(ev.sec.verdict);
  buildMarks(ev); markEls.forEach(m => m && m.classList.add("on"));
  const keys = keyIdx(ev); applyKeyMarks(keys); setPen(ev, keys, true, true);
  stamp.className = "bigstamp static settle " + vk; stamp.textContent = ev.sec.verdict;
  const nw = ev.vlm.warnings.length;
  setStatus(0, `<b>${ev.vlm.facts.length}</b> facts · ${nw ? nw + " warning" + (nw > 1 ? "s" : "") : "no warnings"}`);
  setStatus(1, `<span class="vchip ${vk}">${esc(ev.sec.verdict)}</span>`);
  setStatus(2, `Feedback${ev.coach.nudge ? " + nudge" : ""}`);
  MX.forEach(m => m.classList.remove("wait", "active", "failed"));
  openCard(2, false); hoverLinks();
  frame.classList.remove("sweep", "judging", "coaching", "reread", "fault");
  setAv(AVS[0], ""); setAv(AVS[1], vk === "strong" ? "happy" : "caution"); setAv(AVS[2], "happy");
}
function finishLive(ev, me) {
  if (me !== run) return;
  const out = ev.done || {};
  if (out.state) appState = out.state;
  if (out.finalDebrief && Object.keys(out.finalDebrief).length) finalDebrief = out.finalDebrief;
  if (Array.isArray(out.evaluationTraces)) evaluationTraces = out.evaluationTraces;
  ev.after = stateOf(out.turnTrace?.state_after_turn || out.state) || ev.before;
  const preview = out.nextTurnPreview || ev.preview || {};
  const vk = vkey(ev.sec.verdict);
  run++; clearInterval(tick); $$(".chip").forEach(c => c.remove());
  renderStatic(ev);
  if (preview.skipped) { board.innerHTML = ""; flipBtn.hidden = true; flip.classList.remove("flipped"); }
  else { const playing = flip.classList.contains("flipped"); boardReady(preview.nextBriefing || preview.message, preview.content); flipBtn.hidden = false; if (!playing) flip.classList.remove("flipped"); flipLabel(); }
  scenarioComplete = preview.isComplete === true || !!appState?.completed;
  say("Coach", preview.skipped ? "Test Mode skipped the next turn. Review the verdict and feedback above."
    : scenarioComplete ? "Final debrief is ready. Look for repeated habits in how you chose and justified evidence."
    : "The next turn is ready. Carry forward what the last evidence proved and what it did not.");
  renderNext(preview.skipped ? "skipped" : "ready", preview.message);
  landStamp(vk, false);
  setPills(ev.after);
}
MT.forEach((b, i) => b.onclick = () => { if (isEvaluatingTurn) return; openCard(MX[i].classList.contains("open") ? -1 : i); });
$("#showAll").onclick = () => {
  skipAnim = true; say("Coach", "Animations skipped. Each model's output appears as soon as it arrives.");
  document.getAnimations().forEach(a => { if (!(a instanceof CSSAnimation) && !(a instanceof CSSTransition)) try { a.finish(); } catch (e) { /* done */ } });
  $("#showAll").hidden = true;
};

// ---------- between turns ----------
let handoffFrom = null, evaluationTraces = [];
// Retries briefly while the server still reports the previous operation as running.
async function whenFree(request, tries = 6) {
  for (let i = 0; ; i++) {
    try { return await request(); } catch (error) { if (!error.data?.busy || i >= tries) throw error; await wait(400); }
  }
}
async function continueToNextTurn() {
  if (isContinuingTurn) return;
  if (isEvaluatingTurn) { await evalSettled; if (isEvaluatingTurn) return; }
  if (scenarioComplete) { openFinalReport(); return; }
  isContinuingTurn = true; updateRunControlsSafe();
  const btn = $("#continueBtn"); if (btn) { btn.disabled = true; btn.textContent = "Loading the next turn…"; }
  say("Coach", "Loading the next prepared turn and refreshing the incident state.");
  try {
    const data = await whenFree(() => callApi("/api/continue", { method: "POST" }));
    if (data.isComplete) { appState = data.state; scenarioComplete = true; openFinalReport(); return; }
    handoffFrom = lastEval;
    loadTurnData(data);
    lastEval = null;
    go("handoff");
  } catch (error) {
    console.error(error);
    say("Coach", `Could not load the next turn: ${error.message}`);
    if (btn) { btn.disabled = false; btn.textContent = `Continue to Turn ${turnNo() + 1}`; }
  } finally { isContinuingTurn = false; updateRunControlsSafe(); }
}
function renderHandoff() {
  const ev = handoffFrom, n = turnNo(), prev = n - 1, hist = history_();
  const vk = ev?.sec ? vkey(ev.sec.verdict) : hist.length ? vkey(hist[hist.length - 1].verdict) : "";
  const a = stateOf(appState), b = ev?.before || turnStart[prev] || a;
  $("#handoffBody").innerHTML = `<p class="eyebrow">Turn ${prev} complete${ev?.sec ? ` · ${esc(ev.sec.verdict)}` : ""}</p>
    <h1 class="huge">Turn ${n}</h1><p class="huge-sub">${esc(stageName(n))}</p>
    <ol class="phases">${STAGES.map((s, i) => `<li class="${i < prev ? "done" : i === prev ? "now" : ""}"><span>${i + 1}</span>${s}${i === prev - 1 && vk ? `<em class="phstamp ${vk}">${VWORD[vk]}</em>` : ""}</li>`).join("")}</ol>
    <div class="ho-grid">
      <div class="ho-card"><p class="eyebrow">Incident state</p>${statePanel(a, { from: b, base: b, ti: prev })}</div>
      <div class="ho-card"><p class="eyebrow">Carry forward</p><div class="coachcard"><div data-av="coach" id="coachH"></div><p>${esc(ev?.coach?.nudge || ev?.sec?.next || coachGuidance() || "Start from what the last evidence proved, and what it left open.")}</p></div></div>
      <div class="ho-card"><p class="eyebrow">Ready for you</p><div class="ready"><b>${(turnPayload?.actions || []).length}</b><span>response actions</span><b>${generatedEvidence.length}</b><span>evidence screenshots, generated for Turn ${n}</span></div></div>
    </div>
    <div class="cta-row" style="margin-top:32px"><button class="btn" id="startNext">Start Turn ${n}</button><span class="fine" style="color:var(--plate-ink2)">Verdicts so far: ${hist.map(h => VWORD[vkey(h.verdict)]).join(" · ") || "none"}</span></div>`;
  initAvatars($("#handoffBody"));
  $("#startNext").onclick = () => go("action");
}

// ---------- final report ----------
let report = null, reportLoading = false;
function normaliseFinalDebrief(d) {
  d = d && typeof d === "object" ? d : {};
  const pick = (...ks) => { for (const k of ks) if (d[k] !== undefined && d[k] !== null && d[k] !== "") return d[k]; return undefined; };
  const list = v => Array.isArray(v) ? v.filter(Boolean).map(String) : [];
  const sum = pick("incidentSummary", "incident_summary") || {};
  return {
    overallAssessment: pick("overallAssessment", "overall_assessment") || "",
    performanceLevel: pick("performanceLevel", "performance_level") || "Pending",
    evidenceQualityScore: pick("evidenceQualityScore", "evidence_quality_score") || "",
    strengths: list(pick("strengths")), missedDetails: list(pick("missedDetails", "missed_details")), riskyInterpretations: list(pick("riskyInterpretations", "risky_interpretations")),
    recommendedResponse: pick("recommendedResponse", "recommended_response") || "",
    responseChecklist: list(pick("responseChecklist", "response_checklist")),
    incidentSummary: { whatHappened: sum.whatHappened || sum.what_happened || "", whatWasProven: sum.whatWasProven || sum.what_was_proven || "", whatRemainsUncertain: sum.whatRemainsUncertain || sum.what_remains_uncertain || "" },
    decisionScorecard: (pick("decisionScorecard", "decision_scorecard") || []).map(s => ({ label: s.label || s.name || "Decision criterion", score: Math.max(0, Math.min(100, Number(s.score ?? s.value ?? 0) || 0)), feedback: s.feedback || s.note || "" })),
    reflectionPrompts: list(pick("reflectionPrompts", "reflection_prompts")), learningTargets: list(pick("learningTargets", "learning_targets")),
    turnDebriefs: uniqueTurnEntries(pick("turnDebriefs", "turn_debriefs") || []),
    generatedBy: pick("generatedBy", "generated_by") || "",
  };
}
function normaliseTimeline(tl) {
  if (!tl) return {};
  const keep = (x, ks) => Object.fromEntries(ks.map(k => [k, x?.[k]]));
  return { account: tl.account ?? tl.account_id, region: tl.region, cost: tl.cost,
    actors: (tl.actors || []).map(a => keep(a, ["id", "role", "principal", "source_ip", "user_agent", "mfa"])),
    events: (tl.events || []).map(e => keep(e, ["time", "actor", "event_name", "event_source", "result", "source_ip", "suspicious", "baseline", "turn"])),
    findings: (tl.findings || []).map(f => keep(f, ["finding_type", "severity", "resource", "description", "turn"])) };
}
function buildReport({ run, traces, debrief, timeline }) {
  const ts = uniqueTurnEntries((traces || []).map(t => ({ ...t, turn: Number(t.turn) }))).map(t => ({
    turn: t.turn, action: t.selected_action?.title || "", evidence: t.selected_evidence?.title || "", verdict: t.security_output?.verdict || "Unknown",
    before: stateOf(t.state_before_turn), after: stateOf(t.state_after_turn),
  }));
  return { name: titleCase(run?.scenarioId || activeScenarioId), slug: run?.scenarioId || activeScenarioId, runId: run?.id || activeRunId || "", runLabel: run?.label || "",
    created: run?.createdAt || "", updated: run?.updatedAt || new Date().toISOString(), turns: ts, debrief: normaliseFinalDebrief(debrief), timeline: normaliseTimeline(timeline) };
}
// Opens the report for the active run, or for a saved run when runId is given.
async function openFinalReport(runId) {
  if (reportLoading) return;
  if (!runId && isEvaluatingTurn) await evalSettled;
  reportLoading = true; report = null;
  go("debrief");
  try {
    if (runId) {
      const d = await callApi(`/api/runs/${encodeURIComponent(runId)}`);
      report = buildReport({ run: d.run, traces: d.evaluations, debrief: d.finalDebrief, timeline: d.timeline });
    } else {
      const [deb, tl, det] = await Promise.all([
        finalDebrief ? Promise.resolve({ debrief: finalDebrief }) : callApi("/api/final-debrief").catch(() => ({ debrief: null })),
        callApi("/api/incident-timeline").catch(() => ({ timeline: null })),
        evaluationTraces.length >= maxTurns() || !activeRunId ? Promise.resolve(null) : callApi(`/api/runs/${encodeURIComponent(activeRunId)}`).catch(() => null),
      ]);
      finalDebrief = deb.debrief || finalDebrief;
      report = buildReport({ run: det?.run || activeRunMetadata, traces: evaluationTraces.length ? evaluationTraces : det?.evaluations, debrief: finalDebrief, timeline: tl.timeline });
    }
  } catch (error) {
    console.error(error);
    $("#debriefBody").innerHTML = `<div class="phead"><p class="eyebrow">Final report</p><h1>Report unavailable</h1><p>${esc(error.message)}</p><div class="cta-row"><button class="btn line" data-go="case">Back to cases</button></div></div>`;
  } finally { reportLoading = false; }
  if (report && current === "debrief") { renderDebrief(); enterDebrief(); barFor("debrief"); }
}
const fmtDay = iso => { const d = new Date(iso); return isNaN(d) ? "" : d.toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric", timeZone: "UTC" }); };
const fmtTime = iso => { const d = new Date(iso); return isNaN(d) ? "" : d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", timeZone: "UTC" }); };
const reportNo = r => `IR-${(r.updated || "").slice(0, 10).replace(/-/g, "")}-${String(r.runId).replace(/^run-/, "").slice(0, 6).toUpperCase() || "LOCAL"}`;
const SIGN = [["lens", "Vision model", "Qwen2.5-VL-3B-Instruct", "Evidence extraction", "Qwen V. L."], ["warden", "Security model", "Foundation-Sec-8B-Instruct", "Evidence verdicts", "F. Sec-8B"], ["coach", "Coach model", "Qwen2.5-1.5B-Instruct", "Feedback and debrief", "Q. Coach"]];
function reportChart(states) {
  const n = states.length - 1, X = i => 60 + i * (500 / Math.max(1, n)), Y = v => 200 - v * 1.7;
  let g = "";
  [0, 25, 50, 75, 100].forEach(v => g += `<line class="gl" x1="60" x2="560" y1="${Y(v)}" y2="${Y(v)}"/><text x="50" y="${Y(v) + 4}" text-anchor="end">${v}%</text>`);
  states.forEach((_, i) => g += `<text x="${X(i)}" y="228" text-anchor="middle">${i ? "T" + i : "Start"}</text>`);
  const line = (k, cls) => `<polyline class="ln ${cls}" pathLength="1" points="${states.map((s, i) => `${X(i)},${Y(s[k])}`).join(" ")}"/>` + states.map((s, i) => `<circle class="pt ${cls}" cx="${X(i)}" cy="${Y(s[k])}" r="4" style="--d:${(.18 * i).toFixed(2)}s"/>`).join("");
  const a = states[0], z = states[n];
  return `<svg viewBox="0 0 580 240" role="img" aria-label="Containment moves from ${a.c}% to ${z.c}% and visibility from ${a.v}% to ${z.v}% over ${n} turn${n === 1 ? "" : "s"}">${g}${line("v", "vis")}${line("c", "con")}</svg>`;
}
function renderDebrief() {
  if (!report) {
    $("#debriefBody").innerHTML = `<div class="phead"><p class="eyebrow">Final report</p><h1>Compiling your report</h1><p>Collecting the coach's debrief, your five turns and the incident timeline.</p><div class="ld-bar" style="max-width:360px;margin-top:18px"><i></i></div></div>`;
    return;
  }
  const r = report, deb = r.debrief, TT = r.turns, tl = r.timeline || {};
  const states = TT.length ? [TT[0].before || TT[0].after, ...TT.map(t => t.after || t.before)].filter(Boolean) : [];
  const start = states[0] || stateOf(appState) || { c: 0, v: 0, risk: "Unknown" }, fin = states[states.length - 1] || start;
  const m = /(\d+)\s*\/\s*(\d+)\s*(\w+)?/.exec(deb.evidenceQualityScore || "") || [];
  const no = reportNo(r), day = fmtDay(r.updated), win = r.created ? `${fmtDay(r.created)}, ${fmtTime(r.created)}–${fmtTime(r.updated)} UTC` : day;
  const actor = id => (tl.actors || []).find(a => a.id === id) || {};
  const shortArn = a => String(a || "").replace(/^arn:aws:iam::\d+:/, "");
  const ROLE = { attacker: "Suspected attacker", responder: "Incident responder", background: "Background service" };
  const iocs = [];
  (tl.actors || []).forEach(a => {
    const ctx = ROLE[a.role] || a.role;
    if (a.principal) iocs.push([a.principal, "IAM principal", ctx, a.role]);
    if (a.source_ip) iocs.push([a.source_ip, "Source IP", ctx, a.role]);
    if (a.role === "attacker" && a.user_agent) iocs.push([a.user_agent, "User agent", ctx, a.role]);
  });
  (tl.findings || []).forEach(f => iocs.push([f.finding_type, "GuardDuty finding", `${f.severity} severity · ${f.resource}`, "finding"]));
  if (tl.cost && tl.cost.baseline_daily_usd != null) iocs.push([`$${tl.cost.baseline_daily_usd} → $${tl.cost.incident_daily_usd} per day`, "Cost anomaly", `${tl.cost.service || "Service"} spend${tl.cost.normal_service ? ` (usually led by ${tl.cost.normal_service})` : ""}`, "cost"]);
  const evRows = (tl.events || []).map(e => {
    const a = actor(e.actor), cls = e.suspicious ? "sus" : e.baseline ? "base" : a.role === "responder" ? "resp" : "bg";
    const lab = { sus: "Suspicious", base: "Baseline", resp: "Response", bg: "Background" }[cls];
    return `<tr class="${cls}"><td class="mono">${esc((e.time || "").replace("T", " ").replace("Z", ""))}</td><td>${esc(shortArn(a.principal) || e.actor)}</td><td class="mono">${esc(e.event_name)}</td><td class="mono">${esc((e.event_source || "").replace(".amazonaws.com", ""))}</td><td class="mono">${esc(e.source_ip || a.source_ip || "")}</td><td><span class="flag ${cls}">${lab}</span>${e.turn ? ` <span class="tref">T${e.turn}</span>` : ""}</td></tr>`;
  }).join("");
  const kpi = (v, lab, sub) => `<div><b>${v}</b><span>${lab}</span>${sub ? `<em>${sub}</em>` : ""}</div>`;
  const strong = TT.filter(t => vkey(t.verdict) === "strong").length;
  const hasTl = (tl.events || []).length > 0, sum = deb.incidentSummary;
  const secs = [["r1", "Executive summary"], ["r2", "Response timeline"], ...(iocs.length ? [["r3", "Indicators and findings"]] : []), ["r4", "Decision scorecard"], ["r5", "Recommendations"], ["r6", "Lessons learned"], ["r7", "Review and sign-off"], ...(hasTl ? [["ra", "Event log"]] : [])];
  const num = id => id === "ra" ? "A" : secs.findIndex(s => s[0] === id) + 1;
  const ul = xs => xs.length ? `<ul>${xs.map(x => `<li>${esc(x)}</li>`).join("")}</ul>` : `<p class="r-note">Nothing recorded for this section.</p>`;
  const tlp = `<div class="tlp"><b>TLP:GREEN</b>Training exercise · synthetic data · not a real incident</div>`;
  const turnDeb = t => deb.turnDebriefs.find(x => Number(x.turn) === t.turn) || {};
  $("#debriefBody").innerHTML = `<div class="desk">
    <aside class="rtoc" aria-label="Report sections"><p class="eyebrow">Final report</p><p class="rtoc-no mono">${no}</p>
      <ol>${secs.map(([id, t]) => `<li><button data-rs="${id}"><span>${num(id)}</span>${t}</button></li>`).join("")}</ol>
      <div class="rtoc-cta"><button class="btn ink small" id="printBtn">Print or save as PDF</button><button class="btn line small" data-go="case">Start another case</button><button class="linkbtn" data-go="landing">Back to home</button></div></aside>
    <div class="paperwrap"><article class="paper${reduce ? "" : " live"}" id="report">
      ${tlp}
      <header class="lh"><div class="lh-brand"><svg data-rosette="7,13,0.62" viewBox="-100 -100 200 200" aria-hidden="true"></svg><div><b>CloudIR Trainer</b><span>Incident response report</span></div></div>
        <dl><dt>Report no.</dt><dd>${no}</dd><dt>Issued</dt><dd>${esc(day)}</dd><dt>Status</dt><dd>${TT.length >= maxTurns() ? "Closed" : "In progress"}</dd><dt>Classification</dt><dd>TLP:GREEN</dd></dl></header>
      <h1 class="r-title">${esc(r.name)} incident</h1>
      <p class="r-sub">Post-incident review of a ${TT.length}-turn response exercise built from the ACSE-Eval case <span class="mono">${esc(r.slug)}</span>.</p>
      <div class="tablewrap"><table class="r-meta"><tbody>
        <tr><th>AWS account</th><td class="mono">${esc([tl.account, tl.region].filter(Boolean).join(" · ") || "Not recorded")}</td><th>Exercise window</th><td>${esc(win)}</td></tr>
        <tr><th>Analyst</th><td>${esc(r.runLabel || "Learner (you)")}</td><th>Run ID</th><td class="mono">${esc(r.runId || "Not saved")}</td></tr>
        <tr><th>Reviewed by</th><td>Vision, Security and Coach models, running locally</td><th>Turns completed</th><td>${TT.length} of ${maxTurns()}</td></tr>
      </tbody></table></div>
      ${deb.generatedBy && deb.generatedBy !== "coach_ai" ? `<p class="r-note">The coach model's debrief was not available, so parts of this report were assembled from your evaluation records.</p>` : ""}

      <section class="rsec" id="r1"><h2><span>${num("r1")}</span>Executive summary</h2>
        <div class="r-sum"><div class="r-grade"><span class="eyebrow">Performance level</span><strong>${esc(deb.performanceLevel)}</strong></div><p>${esc(deb.overallAssessment || `You completed ${TT.length} evaluated turns.`)}</p>
          <div class="r-seal" role="img" aria-label="${m[1] ? `${m[1]} of ${m[2]} ${esc(m[3] || "")} verdicts` : `${strong} strong verdicts`}"><svg viewBox="0 0 120 120" aria-hidden="true"><defs><path id="sealp" d="M60 60 m-45 0 a45 45 0 1 1 90 0 a45 45 0 1 1 -90 0"/></defs><circle cx="60" cy="60" r="57"/><circle class="in" cx="60" cy="60" r="36"/><text><textPath href="#sealp">EVIDENCE FIRST · CLOUDIR TRAINER · VERIFIED ·</textPath></text></svg><b>${m[1] ? `${m[1]}/${m[2]}` : `${strong}/${TT.length}`}</b><small>${esc((m[3] || "strong").toUpperCase())}</small></div></div>
        <div class="r-kpis">${kpi(fin.c + "%", "Containment", `from ${start.c}%`)}${kpi(fin.v + "%", "Visibility", `from ${start.v}%`)}${kpi(esc(fin.risk), "Residual risk", `from ${esc(start.risk)}`)}${kpi(`${strong}/${TT.length}`, "Strong verdicts", "")}</div>
        ${sum.whatHappened || sum.whatWasProven || sum.whatRemainsUncertain ? `<div class="r-two r-summary"><div><p class="eyebrow r-h3">What happened</p><p>${esc(sum.whatHappened)}</p></div><div><p class="eyebrow r-h3">What you proved</p><p>${esc(sum.whatWasProven)}</p></div><div><p class="eyebrow r-h3">What remains uncertain</p><p>${esc(sum.whatRemainsUncertain)}</p></div></div>` : ""}
        ${states.length > 1 ? `<figure class="r-fig">${reportChart(states)}<figcaption><b>Figure 1.</b> Containment <i class="k con"></i> and visibility <i class="k vis"></i> at the start of the exercise and after each turn.</figcaption></figure>` : ""}</section>

      <section class="rsec" id="r2"><h2><span>${num("r2")}</span>Response timeline</h2>
        <div class="tablewrap fit"><table class="r-table tl"><thead><tr><th>Turn</th><th>Phase</th><th>Action taken</th><th>Evidence cited</th><th>Verdict</th></tr></thead><tbody>${TT.map((t, i) => `<tr><td class="mono">T${t.turn}</td><td>${esc(stageName(t.turn))}</td><td>${esc(t.action)}</td><td>${esc(t.evidence)}</td><td><span class="vtag ${vkey(t.verdict)}" style="--sd:${(.25 + i * .16).toFixed(2)}s">${VWORD[vkey(t.verdict)]}</span></td></tr>`).join("")}</tbody></table></div>
        ${TT.some(t => turnDeb(t).what_went_well || turnDeb(t).whatWentWell) ? `<div class="r-turns">${TT.map(t => { const d = turnDeb(t), facts = d.evidenceFactsUsed || d.evidence_facts_used || []; return `<details class="r-turn"><summary><span class="mono">T${t.turn}</span>${esc(t.action)}<span class="vtag ${vkey(t.verdict)}">${VWORD[vkey(t.verdict)]}</span></summary><dl>
          <dt>What went well</dt><dd>${esc(d.whatWentWell || d.what_went_well || "")}</dd><dt>What was missing</dt><dd>${esc(d.whatWasMissing || d.what_was_missing || "")}</dd>
          <dt>Risk warning</dt><dd>${esc(d.riskWarning || d.risk_warning || "")}</dd><dt>Next time</dt><dd>${esc(d.nextTimeImprove || d.next_time_improve || "")}</dd>
          ${facts.length ? `<dt>Evidence facts used</dt><dd><ul>${facts.map(f => `<li>${esc(f)}</li>`).join("")}</ul></dd>` : ""}</dl></details>`; }).join("")}</div>` : ""}</section>

      ${iocs.length ? `<section class="rsec" id="r3"><h2><span>${num("r3")}</span>Indicators and findings</h2>
        <p class="r-note">Taken from the incident timeline this case was generated from. All addresses and identities are synthetic.</p>
        <div class="tablewrap"><table class="r-table ioc"><thead><tr><th>Indicator</th><th>Type</th><th>Context</th></tr></thead><tbody>${iocs.map(([v, t, c, rl]) => `<tr class="${esc(rl)}"><td class="mono">${esc(v)}</td><td>${esc(t)}</td><td>${esc(c)}</td></tr>`).join("")}</tbody></table></div></section>` : ""}

      <section class="rsec" id="r4"><h2><span>${num("r4")}</span>Decision scorecard</h2>
        ${deb.decisionScorecard.length ? `<div class="scores">${deb.decisionScorecard.map((s, i) => `<div class="score"><div class="row"><span>${esc(s.label)}</span><b>${s.score}<small>/100</small></b></div><div class="track"><i style="--f:${s.score / 100};animation-delay:${(.2 + i * .12).toFixed(2)}s"></i></div><p>${esc(s.feedback)}</p></div>`).join("")}</div>
        <p class="r-cap"><b>Figure 2.</b> Decision scores awarded by the coach model.</p>` : `<p class="r-note">No scorecard was generated for this run.</p>`}</section>

      <section class="rsec" id="r5"><h2><span>${num("r5")}</span>Recommendations</h2>
        <p class="r-lead">${esc(deb.recommendedResponse)}</p>
        ${deb.responseChecklist.length ? `<p class="eyebrow r-h3">Follow-up checklist</p><ol class="r-check">${deb.responseChecklist.map(x => `<li><label><input type="checkbox"><span>${esc(x)}</span></label></li>`).join("")}</ol>` : ""}</section>

      <section class="rsec" id="r6"><h2><span>${num("r6")}</span>Lessons learned</h2><div class="r-two">
        <div><p class="eyebrow r-h3">Strengths</p>${ul(deb.strengths)}</div>
        <div><p class="eyebrow r-h3">Missed details</p>${ul(deb.missedDetails)}</div>
        <div><p class="eyebrow r-h3">Risky interpretations</p>${ul(deb.riskyInterpretations)}</div>
        <div><p class="eyebrow r-h3">Learning targets</p>${ul(deb.learningTargets)}</div>
        <div><p class="eyebrow r-h3">Reflection questions</p>${deb.reflectionPrompts.length ? `<ol>${deb.reflectionPrompts.map(x => `<li>${esc(x)}</li>`).join("")}</ol>` : ul([])}</div></div></section>

      <section class="rsec signoff" id="r7"><h2><span>${num("r7")}</span>Review and sign-off</h2>
        <div class="sigs">${SIGN.map(([av, role, model, duty, sig], i) => `<div class="sig"><div class="sigline"><span class="sigtext" style="--sd:${(.3 + i * .75).toFixed(2)}s">${sig}</span></div><div class="sigwho"><div data-av="${av}"></div><div><b>${role}</b><span>${model}</span><small>${duty} · ${esc(day)}</small></div></div></div>`).join("")}</div>
        <div class="r-stamp">Reviewed<small>${esc(day)}</small></div></section>

      ${hasTl ? `<section class="rsec" id="ra"><h2><span>A</span>Appendix · Reconstructed event log</h2>
        <div class="tablewrap"><table class="r-table ev"><thead><tr><th>Time (UTC)</th><th>Principal</th><th>Event</th><th>Service</th><th>Source IP</th><th>Classification</th></tr></thead><tbody>${evRows}</tbody></table></div></section>` : ""}

      <footer class="r-foot"><span>CloudIR Trainer · ${no}</span><span>TLP:GREEN</span><span>End of report</span></footer>
      ${tlp}
    </article></div>
  </div>
  <div class="testpanel"><p class="eyebrow">Test Mode</p><h2>Final Debrief Payload</h2><pre tabindex="0" role="region" aria-label="Final debrief payload">${jsonOut({ state: appState, finalDebrief: deb })}</pre></div>`;
  initAvatars($("#debriefBody"));
  drawRosettes($("#debriefBody"));
  $("#printBtn").onclick = () => { try { window.print(); } catch (e) { /* print blocked */ } };
  $$("#debriefBody [data-rs]").forEach(b => b.onclick = () => { const t = document.getElementById(b.dataset.rs); if (t) t.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" }); });
  if (r.turns.length) setPills(fin);
}
function enterDebrief() {
  const paper = $("#report"); if (!paper) return;
  const secs = $$(".rsec", paper), links = $$("#debriefBody [data-rs]");
  if (reduce) secs.forEach(x => x.classList.add("seen")); else paper.classList.add("drop");
  const seen = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add("seen"); seen.unobserve(e.target); } }), { threshold: .15 });
  const spy = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) links.forEach(l => l.classList.toggle("on", l.dataset.rs === e.target.id)); }), { rootMargin: "-35% 0px -60% 0px" });
  secs.forEach(x => { seen.observe(x); spy.observe(x); });
  LEAVE.debrief = () => { seen.disconnect(); spy.disconnect(); };
}

// ---------- restoring saved runs (called from run_controls.js) ----------
function evalFromTrace(trace) {
  const evidence = generatedEvidence.find(e => e.id === trace.selected_evidence?.id) || trace.selected_evidence || {};
  return { evidence, action: trace.selected_action || {}, before: stateOf(trace.state_before_turn), after: stateOf(trace.state_after_turn),
    vlm: adaptVlm(trace.vlm_output), sec: adaptSecurity(trace.security_output), coach: adaptCoach(trace.coach_output),
    scheme: schemeOf(turnPayload), turn: Number(trace.turn) || turnNo(), justification: trace.learner_justification, restored: true };
}
async function enterRunUI(data) {
  closeEvidenceModal(); stopPromptSpeech();
  activeRunMetadata = data.run; activeRunId = data.run.id; $("#runNameInput").value = data.run.label;
  activeScenarioId = data.run.scenarioId || activeScenarioId;
  turnStart = {}; railAnimKey = ""; report = null; finalDebrief = null; evaluationTraces = []; lastEval = null; handoffFrom = null;
  loadTurnData(data);
  isEvaluatingTurn = false; isContinuingTurn = false;
  const draft = data.draft || {}, trace = data.currentEvaluation;
  if (scenarioComplete) { await openFinalReport(); return; }
  if (trace && appState.pendingTurn) {
    selectedAction = trace.selected_action; lastEval = evalFromTrace(trace); selectedEvidence = lastEval.evidence;
    go("evaluation"); return;
  }
  if (draft.selectedAction) {
    selectedAction = (turnPayload?.actions || []).find(a => a.id === draft.selectedAction.id) || null;
    if (selectedAction && ["evidence", "justification", "evaluation"].includes(draft.screen)) selectedEvidence = generatedEvidence.find(e => e.id === draft.selectedEvidence?.id) || null;
    if (selectedEvidence && ["justification", "evaluation"].includes(draft.screen)) {
      transcripts[txKey()] = draft.justification?.transcript || draft.justification?.typed_text || "";
      learnerJustification = draft.justification?.source ? draft.justification : null;
      go("reasoning"); return;
    }
    go(selectedEvidence ? "reasoning" : selectedAction && draft.screen !== "action" ? "evidence" : "action"); return;
  }
  go("action");
}
function showScreen(name) {
  const id = { runs: "runs", home: "landing", datasetSelection: "case", datasetPreparation: "prep", action: "action", evidence: "evidence", justification: "reasoning", evaluation: "evaluation", end: "debrief" }[name] || name;
  if (current !== id) go(id);
}

// ---------- tutorial ----------
const TUT = [
  { id: "case", title: "Choose a case", lead: "Everything starts on the case page. Each case is a different ACSE-Eval incident with its own architecture.", go: "case", goLabel: "Open the case page",
    shot: { w: 1280, h: 1200, pins: [[4.1, 18.1, 29.38, 30], [5.9, 51.85, 38.57, 33.5], [46.65, 75.18, 11.72, 4.4], [4.18, 89.02, 90.47, 4.47]] },
    steps: [["Scenario cards", "Pick one of the three incidents. The selected card is outlined in gold."], ["Architecture diagram", "Context for the case. Select it to enlarge. You can't submit it as evidence."], ["Start scenario", "Starts Turn 1 straight away. A case that isn't prepared shows Build case instead, and the models build it first."], ["Maintenance", "Refresh the status, clean restart a case, and check its source files."]] },
  { id: "action", title: "Pick an action", lead: "Each turn has three steps before the models evaluate it. The first is choosing what to do next.",
    shot: { w: 1280, h: 900, pins: [[4.1, 12.63, 90.63, 5.47], [5.43, 65.64, 20, 25.48], [28.95, 46.73, 65.78, 11.8], [28.95, 70.81, 65.78, 12.41], [79.79, 85.88, 14.94, 5.87]] },
    steps: [["Turn steps", "Shows where you are in the turn. Select a finished step to go back and change it."], ["Incident state", "Containment, visibility, risk and phase. The gauges update after every evaluated turn."], ["Response actions", "Three options. Pick the one you can prove with a screenshot."], ["Coach", "Asks about your choice, but never tells you which answer is right."], ["Choose evidence", "Moves on once an action is selected."]] },
  { id: "evidence", title: "Choose your evidence", lead: "Pick the one screenshot that proves your action. The coach will ask what it shows.",
    shot: { w: 1280, h: 900, pins: [[73.63, 36.22, 21.09, 33.31], [30.51, 38.45, 3.97, 2.78], [5.43, 63.06, 20, 9.4], [76.55, 86.51, 18.18, 5.87]] },
    steps: [["Exhibits", "Three AWS-style screenshots generated for this turn. Select one to use it."], ["Zoom", "Opens the screenshot full size so you can read every field. Press Esc to close it."], ["Your action", "The briefing panel keeps your chosen action in view. Change takes you back a step."], ["Explain your reasoning", "Moves on to the last step before evaluation."]] },
  { id: "reasoning", title: "Explain your reasoning", lead: "Say why the screenshot backs your action. Name the fields you can see, and what they don't prove.",
    shot: { w: 1280, h: 1180, pins: [[30.74, 38.5, 62.19, 2.12], [30.74, 45.87, 5, 5.42], [30.74, 61.98, 62.19, 9.83], [30.74, 77.06, 62.19, 5.08], [74.06, 84.61, 18.87, 4.47]] },
    steps: [["Evidence cues", "Fields worth mentioning for this type of screenshot. Each one lights up when your answer mentions it."], ["Record", "Speak for up to 45 seconds, then select Transcribe. Whisper turns it into text on your Mac."], ["Your answer", "Type here, or edit the transcript before you submit."], ["Checklist", "Four quick checks. You can still submit with some of them open."], ["Submit for AI evaluation", "Sends your action, evidence and reasoning to the three models."]] },
  { id: "evaluation", title: "Read the evaluation", lead: "The three models work one after another, and each one marks the screenshot in its own way.",
    shot: { w: 1280, h: 860, pins: [[4.1, 25.57, 49.48, 46.85], [55.22, 25.69, 39.43, 7.35], [55.22, 34.2, 39.43, 8.46], [55.14, 43.71, 39.58, 54.65], [46.6, 73.7, 6.98, 1.86], [37.81, 88.63, 14.28, 6.14]] },
    steps: [["Your screenshot", "Numbered boxes are the fields Vision read. Security marks the key fact in gold, the coach circles it in red pen, and the verdict stamp lands in the corner."], ["Vision", "The facts it read. Hover a fact to find it on the screenshot."], ["Security", "The verdict, the key fact, the marking scheme it used, and a check of each statement you made."], ["Coach", "Feedback on this turn and a nudge for the next. Select any card's header to reopen it."], ["Preview", "Flips the screenshot to show the coach building the next turn."], ["Continue", "Moves on to the next turn, or to your final report after Turn 5."]] },
  { id: "handoff", title: "Between turns", lead: "After each evaluation you see how the incident has moved on before the next turn starts.",
    shot: { w: 1280, h: 900, pins: [[4.1, 43.07, 90.63, 3.68], [5.74, 55.22, 26.09, 25.48], [34.73, 50.31, 29.38, 37.77], [4.1, 91.64, 10.34, 5.87]] },
    steps: [["Phases", "The five response phases. Your verdict is stamped on the phase you just finished."], ["Incident state", "How containment, visibility and risk changed this turn."], ["Carry forward", "The coach's nudge for the next turn."], ["Start the next turn", "Opens the action step for the new turn."]] },
  { id: "debrief", title: "Your final report", lead: "After Turn 5 you get an incident response report you can print or save as a PDF. Finished runs keep theirs under Saved runs.",
    shot: { w: 1280, h: 900, pins: [[4.1, 15.09, 17.97, 35.06], [4.1, 52.15, 17.97, 4.63], [29.84, 13.11, 60.13, 15.58], [29.84, 65.95, 60.13, 34.05]] },
    steps: [["Contents", "Jump to any section. The section you're reading is highlighted."], ["Print or save as PDF", "Prints the report on its own, without the rest of the app."], ["Report header", "Report number, date, status and classification."], ["Executive summary", "Your performance level, the final incident state and a chart of every turn."]] },
  { id: "runs", title: "Test Mode and saved runs", lead: "Test Mode is for trying out paths and saving runs. Leave it off for a normal session.", go: "runs", goLabel: "Open saved runs",
    shot: { w: 1280, h: 900, pins: [[75.6, 1.33, 6.52, 2.78], [41.07, 6.9, 53.65, 4.24], [4.1, 33.92, 48.41, 10.12], [56.18, 66.5, 36.75, 4.86]] },
    steps: [["Test Mode", "Turns the testing tools on or off. Skip next turn appears beside it."], ["Run tools", "Name the run, save it, save and leave, retry this turn, or open saved runs."], ["Saved runs", "Every run with its verdicts. Select one to see its details."], ["Run actions", "Resume, review the report, start a new attempt from a checkpoint, or export a ZIP."]] },
];
let tutBuilt = false, tutSeen = null, tutSpy = null;
function renderTutorial() {
  if (tutBuilt) return; tutBuilt = true;
  const ROUTE = ["Case", "Action", "Evidence", "Reasoning", "Evaluation", "Next turn", "Report", "Test Mode"];
  $("#route").innerHTML = TUT.map((c, i) => `<li><button data-ch="t-${c.id}" title="${esc(c.title)}"><b>${i + 1}</b>${ROUTE[i]}</button></li>`).join("");
  $("#chapters").innerHTML = TUT.map((c, i) => `<article class="chap" id="t-${c.id}">
    <div class="chap-head"><span class="chap-no">${String(i + 1).padStart(2, "0")}</span><div><h2>${esc(c.title)}</h2><p>${esc(c.lead)}</p></div></div>
    <div class="chap-body"><figure class="tshot"><img src="assets/tutorial/${c.id}.jpg" alt="${esc(c.title)} screen" loading="lazy" width="${c.shot.w}" height="${c.shot.h}">${c.shot.pins.map((p, j) => p ? `<button class="tpin" data-n="${j}" aria-label="Step ${j + 1}: ${esc(c.steps[j][0])}" style="left:${p[0]}%;top:${p[1]}%;width:${p[2]}%;height:${p[3]}%"><b>${j + 1}</b></button>` : "").join("")}</figure>
      <div><ol class="callouts">${c.steps.map(([t, d], j) => `<li data-n="${j}"><b>${j + 1}</b><div><strong>${esc(t)}</strong><span>${esc(d)}</span></div></li>`).join("")}</ol>${c.go ? `<div class="cta-row"><button class="btn line small" data-go="${c.go}">${esc(c.goLabel)}</button></div>` : ""}</div></div></article>`).join("");
  $$("#chapters .chap").forEach(ch => {
    const pins = $$(".tpin", ch), lis = $$(".callouts li", ch);
    const hl = (n, on) => { pins.forEach(p => p.classList.toggle("hl", on && p.dataset.n === n)); lis.forEach(l => l.classList.toggle("hl", on && l.dataset.n === n)); };
    lis.forEach(l => { l.onmouseenter = () => hl(l.dataset.n, true); l.onmouseleave = () => hl(l.dataset.n, false); });
    pins.forEach(p => { p.onmouseenter = p.onfocus = () => hl(p.dataset.n, true); p.onmouseleave = p.onblur = () => hl(p.dataset.n, false); p.onclick = () => { hl(p.dataset.n, true); const li = lis[+p.dataset.n]; if (li && narrow()) li.scrollIntoView({ block: "center", behavior: reduce ? "auto" : "smooth" }); }; });
  });
  $$("#route [data-ch]").forEach(b => b.onclick = () => { const t = document.getElementById(b.dataset.ch); if (t) t.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" }); });
  initAvatars(pageEls.tutorial);
}
function enterTutorial() {
  const chaps = $$("#chapters .chap"), btns = $$("#route [data-ch]");
  if (reduce) chaps.forEach(c => c.classList.add("seen"));
  tutSeen = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add("seen"); tutSeen.unobserve(e.target); } }), { threshold: .25 });
  tutSpy = new IntersectionObserver(es => es.forEach(e => { if (!e.isIntersecting) return; const i = chaps.indexOf(e.target); btns.forEach((b, j) => { b.classList.toggle("on", j === i); b.classList.toggle("past", j < i); }); const on = btns[i], nav = $(".route"); if (on && nav.scrollWidth > nav.clientWidth) { const li = on.parentElement; nav.scrollTo({ left: li.offsetLeft - (nav.clientWidth - li.offsetWidth) / 2, behavior: reduce ? "auto" : "smooth" }); } }), { rootMargin: "-40% 0px -55% 0px" });
  chaps.forEach(c => { tutSeen.observe(c); tutSpy.observe(c); });
  const bot = $("#tutBot"); if (bot) { setAv(bot, "talk"); setTimeout(() => setAv(bot, "happy"), 1800); }
}

// ---------- page hooks ----------
const RENDER = { landing: renderLanding, case: renderCase, action: renderAction, evidence: renderEvidence, reasoning: renderReasoning, evaluation: renderEvaluation, handoff: renderHandoff, debrief: renderDebrief, tutorial: renderTutorial };
const ENTER = {
  tutorial: enterTutorial, landing: refreshResumeCard,
  action: () => setPills(stateOf(appState)), evidence: () => setPills(stateOf(appState)), reasoning: () => setPills(stateOf(appState)),
  evaluation: () => {
    if (!lastEval) return;
    if (lastEval.live && !lastEval.started) { lastEval.started = true; startLiveEvaluation(); return; }
    if (lastEval.restored && !isEvaluatingTurn) {
      resetEvalView(); measureEvidence().then(() => {
        renderStatic(lastEval); renderNext("ready"); boardReady("The next turn was prepared before this run was saved.");
        flipBtn.hidden = false; flipLabel(); landStamp(vkey(lastEval.sec.verdict), false); setPills(lastEval.after);
        say("Coach", "Saved evaluation restored. Continue without rerunning it.");
      });
    }
  },
  handoff: () => {
    setPills(stateOf(appState));
    setTimeout(() => animateState($("#handoffBody")), 350);
    const h = $("#coachH"); setAv(h, "talk"); setTimeout(() => setAv(h, "happy"), 2000);
  },
  debrief: () => { if (report) enterDebrief(); },
};
const LEAVE = {
  tutorial: () => { if (tutSeen) tutSeen.disconnect(); if (tutSpy) tutSpy.disconnect(); },
  // Leaving mid-run cannot stop the models, so their animations are skipped instead.
  evaluation: () => { if (isEvaluatingTurn) skipAnim = true; },
  reasoning: () => stopPromptSpeech(),
};
addEventListener("keydown", e => { if (e.key === "Escape" && speechSynthesis?.speaking) stopPromptSpeech(); });

// ---------- start ----------
addEventListener("DOMContentLoaded", () => {
  initialiseTestMode(); wake();
  const h0 = (location.hash || "").slice(1);
  if (h0 === "tutorial") show("tutorial");
  else if (h0 === "case") { show("landing"); openPrototypeWorkspace(); }
  else if (h0 === "runs" && typeof showSavedRuns === "function") { show("landing"); withRunControl(showSavedRuns); }
  else show("landing");
  loadScenarios().then(renderLandCases, renderLandCases);
});
