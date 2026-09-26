// Saved runs and Test Mode run controls. They share the main interface and the real evaluation pipeline.
let leaveRunQueued = false;
let runControlBusy = false;
let lastSavedRunDraft = {};

function runMessage(message) {
  document.getElementById("runSaveStatus").textContent = message;
}

function updateRunControls() {
  document.getElementById("runToolbar").classList.toggle("hidden", !testModeEnabled);
  document.getElementById("activeRunLabel").textContent = activeRunMetadata
    ? `${activeRunMetadata.label} · Turn ${appState?.currentTurn || activeRunMetadata.turn}`
    : "No active run";
  const busy = runControlBusy || isPreparingDataset || isContinuingTurn;
  document.getElementById("saveRunBtn").disabled = !activeRunId || busy || isEvaluatingTurn;
  document.getElementById("leaveRunBtn").disabled = !activeRunId || busy || leaveRunQueued;
  document.getElementById("leaveRunBtn").textContent = leaveRunQueued
    ? "Will save & leave when ready" : isEvaluatingTurn ? "Save & Leave When Ready" : "Save & Leave";
  document.getElementById("repeatTurnBtn").disabled = !activeRunId || busy || isEvaluatingTurn;
  document.getElementById("savedRunsBtn").disabled = busy || isEvaluatingTurn;
  document.getElementById("testModeToggleBtn").disabled = leaveRunQueued;
}

function runDraft() {
  const screen = Object.entries(screens).find(([, element]) => element?.classList.contains("active"))?.[0];
  if (!["action", "evidence", "justification", "evaluation", "end"].includes(screen)) return lastSavedRunDraft;
  return {
    screen, selectedAction, selectedEvidence,
    justification: getLearnerJustification(),
  };
}

async function withRunControl(action) {
  if (runControlBusy) return;
  if (mediaRecorder?.state === "recording") {
    runMessage("Stop recording and transcribe your audio before saving or leaving.");
    return;
  }
  if (recordedAudioBlob && !document.getElementById("learnerJustificationInput")?.value.trim()) {
    runMessage("Transcribe your recording first. Saved runs preserve the transcript, not an untranscribed recording.");
    return;
  }
  runControlBusy = true;
  updateRunControls();
  try { await action(); }
  catch (error) { runMessage(error.message); }
  finally { runControlBusy = false; updateRunControls(); }
}

async function saveCurrentRun(leaving = false) {
  const draft = runDraft();
  const result = await callApi(leaving ? "/api/runs/leave" : "/api/runs/save", {
    method: "POST", body: JSON.stringify({ draft, label: document.getElementById("runNameInput").value || undefined }),
  });
  lastSavedRunDraft = draft;
  activeRunMetadata = result.run;
  runMessage(`Saved ${new Date(result.checkpoint.createdAt).toLocaleTimeString()}.`);
  if (leaving) {
    closeEvidenceModal();
    if (window.speechSynthesis) window.speechSynthesis.cancel();
    activeRunId = null;
    activeRunMetadata = null;
    selectedAction = null;
    selectedEvidence = null;
    recordedAudioBlob = null;
    learnerJustification = null;
    await openPrototypeWorkspace();
    runMessage("Run saved. Resume it from Saved Runs whenever you are ready.");
  }
}

async function requestRunLeave() {
  if (isEvaluatingTurn) {
    leaveRunQueued = true;
    runMessage("Your run will be saved after the current evaluation and turn preparation finish.");
    updateRunControls();
    return;
  }
  await withRunControl(() => saveCurrentRun(true));
}

async function finishQueuedRunLeave() {
  if (leaveRunQueued) {
    leaveRunQueued = false;
    await withRunControl(() => saveCurrentRun(true));
  } else {
    updateRunControls();
  }
}

const RUN_STATUS = { active: "Active", paused: "Paused", completed: "Completed", legacy: "Legacy" };
let selectedRunId = null;

async function showSavedRuns() {
  if (activeRunId) await saveCurrentRun();
  const data = await callApi("/api/runs");
  const list = document.getElementById("savedRunsList");
  if (!data.runs.some(run => run.id === selectedRunId)) selectedRunId = data.runs[0]?.id || null;
  list.innerHTML = data.runs.length ? "" : '<p class="fine">No saved runs yet. Start a prepared case to create one.</p>';
  for (const run of data.runs) {
    const status = RUN_STATUS[run.status] || run.status;
    const row = document.createElement("button");
    row.className = "runrow";
    row.dataset.runView = run.id;
    row.setAttribute("aria-pressed", run.id === selectedRunId ? "true" : "false");
    row.innerHTML = `<span><span class="eyebrow">${escapeHtml(run.scenarioId)}${run.testMode ? " · Test Mode" : ""}</span><br><strong>${escapeHtml(run.label)}</strong></span>
      <span class="status ${escapeHtml(status)}">${escapeHtml(status)}</span>
      <span class="meta"><span>${run.completedTurns} of 5 turns evaluated</span><span>${escapeHtml(new Date(run.updatedAt).toLocaleString())}</span></span>`;
    list.appendChild(row);
  }
  document.getElementById("savedRunDetails").innerHTML = selectedRunId ? "" : '<p class="fine">Select a run to see its results.</p>';
  showScreen("runs");
  if (selectedRunId) await viewSavedRun(selectedRunId);
}

async function viewSavedRun(runId) {
  selectedRunId = runId;
  document.querySelectorAll("#savedRunsList .runrow").forEach(row => row.setAttribute("aria-pressed", row.dataset.runView === runId ? "true" : "false"));
  const data = await callApi(`/api/runs/${runId}`);
  const panel = document.getElementById("savedRunDetails");
  const run = data.run, status = RUN_STATUS[run.status] || run.status;
  const choices = data.checkpoints.filter(cp => ["initial", "before_turn"].includes(cp.kind));
  const verdicts = data.evaluations.map(t => String(t.security_output?.verdict || ""));
  const vk = v => /strong/i.test(v) ? "strong" : /partial/i.test(v) ? "partial" : "weak";
  const word = { strong: "Strong", partial: "Partial", weak: "Weak" };
  panel.innerHTML = `<p class="eyebrow">${escapeHtml(run.scenarioId)} · ${escapeHtml(status)}</p><h2>${escapeHtml(run.label)}</h2>
    <dl><dt>Run ID</dt><dd class="mono">${escapeHtml(run.id)}</dd><dt>Turns</dt><dd>${data.evaluations.length} of 5 evaluated</dd><dt>Updated</dt><dd>${escapeHtml(new Date(run.updatedAt).toLocaleString())}</dd>${run.parent ? `<dt>Repeat of</dt><dd class="mono">${escapeHtml(run.parent.runId)}</dd>` : ""}</dl>
    <div class="stamps" style="margin-top:16px">${verdicts.map((v, i) => `<span class="stamp-s ${vk(v)}">T${i + 1} · ${word[vk(v)]}</span>`).join("") || '<p class="fine">No turns evaluated yet.</p>'}</div>
    <div class="cta-row">
      ${run.resumable ? `<button class="btn ink small" data-run-resume="${run.id}">${run.status === "completed" ? "Open run" : "Resume run"}</button>` : ""}
      ${data.state?.completed ? `<button class="btn ink small" data-run-report="${run.id}">Review report</button>` : ""}
      <a class="btn line small" href="/api/runs/${run.id}/export" download>Export ZIP</a>
    </div>
    ${run.resumable && choices.length ? `<div class="repeat"><label for="repeatCheckpoint" class="eyebrow">New attempt from a checkpoint</label>
      <div class="cta-row" style="margin-top:8px"><select id="repeatCheckpoint">${choices.map(cp => `<option value="${cp.id}">${cp.kind === "initial" ? "Run starting point" : "Before evaluation"} · Turn ${cp.state.current_turn} · ${escapeHtml(new Date(cp.createdAt).toLocaleTimeString())}</option>`).join("")}</select>
      <button class="btn line small" data-run-repeat="${run.id}">Start new attempt</button></div>
      <p class="fine" style="margin-top:6px">Each attempt is a separate run linked to this one. The original results are kept.</p></div>`
      : run.resumable ? "" : '<p class="fine" style="margin-top:12px">This is unverified legacy data. You can review and export it, but it cannot be resumed.</p>'}
    <details class="rawrun"><summary>Recorded evaluations</summary>
      <div class="tablewrap"><table class="r-table"><thead><tr><th>Turn</th><th>Action</th><th>Evidence</th><th>Verdict</th></tr></thead><tbody>
      ${data.evaluations.map(t => `<tr><td>${t.turn}</td><td>${escapeHtml(t.selected_action?.title)}</td><td>${escapeHtml(t.selected_evidence?.title)}</td><td>${escapeHtml(t.security_output?.verdict)}</td></tr>`).join("") || '<tr><td colspan="4">No evaluated turns saved yet.</td></tr>'}
      </tbody></table></div></details>
    <details class="rawrun"><summary>Full saved results, checkpoints and timings</summary><pre class="run-json" tabindex="0" role="region" aria-label="Saved run data">${escapeHtml(JSON.stringify(data, null, 2))}</pre></details>`;
}

async function restoreRunUI(data) {
  lastSavedRunDraft = data.draft || {};
  await enterRunUI(data);
  runMessage("Run restored. Previous results and evidence are preserved.");
  updateRunControls();
}

async function resumeSavedRun(runId) {
  if (activeRunId) await saveCurrentRun();
  await restoreRunUI(await callApi(`/api/runs/${runId}/resume`, { method: "POST" }));
}

async function repeatSavedCheckpoint(runId, checkpointId) {
  if (activeRunId) await saveCurrentRun();
  await restoreRunUI(await callApi(`/api/runs/${runId}/repeat`, {
    method: "POST", body: JSON.stringify({ checkpointId, label: `Repeat of Turn ${appState?.currentTurn || 1}` }),
  }));
}

async function retryCurrentTurn() {
  const data = await callApi(`/api/runs/${activeRunId}`);
  const checkpoint = [...data.checkpoints].reverse().find(cp => cp.state.current_turn === appState.currentTurn && ["before_turn", "initial"].includes(cp.kind));
  if (!checkpoint) throw new Error("No starting checkpoint for this turn yet. Save or evaluate the turn first.");
  await repeatSavedCheckpoint(activeRunId, checkpoint.id);
}

addClickListener("saveRunBtn", () => withRunControl(() => saveCurrentRun()));
addClickListener("leaveRunBtn", requestRunLeave);
addClickListener("repeatTurnBtn", () => withRunControl(retryCurrentTurn));
addClickListener("savedRunsBtn", () => withRunControl(showSavedRuns));
addClickListener("refreshRunsBtn", () => withRunControl(showSavedRuns));
addClickListener("runsToScenariosBtn", () => withRunControl(async () => {
  if (activeRunId) await saveCurrentRun(true); else await openPrototypeWorkspace();
}));
document.addEventListener("click", event => {
  const button = event.target.closest("button");
  if (button?.dataset.runView) withRunControl(() => viewSavedRun(button.dataset.runView));
  if (button?.dataset.runReport) withRunControl(() => openFinalReport(button.dataset.runReport));
  if (button?.dataset.runResume) withRunControl(() => resumeSavedRun(button.dataset.runResume));
  if (button?.dataset.runRepeat) withRunControl(() => repeatSavedCheckpoint(button.dataset.runRepeat, document.getElementById("repeatCheckpoint").value));
});
updateRunControls();
