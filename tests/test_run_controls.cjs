// Unit tests for run-control behaviour; no browser or AI models are launched.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function harness() {
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) {
      const classes = new Set();
      elements.set(id, { value: '', textContent: '', disabled: false, dataset: {},
        classList: { add: c => classes.add(c), remove: c => classes.delete(c),
          contains: c => classes.has(c), toggle(c, enabled) { if (enabled) classes.add(c); else classes.delete(c); } },
        scrollIntoView() {}, appendChild() {},
      });
    }
    return elements.get(id);
  }
  const calls = [];
  const context = vm.createContext({
    console, Date, JSON, Object, Array, Error,
    document: { getElementById: element, querySelectorAll: () => [], addEventListener() {}, createElement: () => element(Math.random()) },
    window: {}, screens: { action: element('action'), justification: element('justification'), runs: element('runs'), evaluation: element('evaluation') },
    testModeEnabled: true, activeRunId: 'run-test', activeRunMetadata: { id: 'run-test', label: 'Test run', turn: 1 },
    appState: { currentTurn: 1 }, selectedAction: { id: 'action1' }, selectedEvidence: { id: 'evidence1' },
    isPreparingDataset: false, isContinuingTurn: false, isEvaluatingTurn: false,
    mediaRecorder: null, recordedAudioBlob: null, learnerJustification: null,
    getLearnerJustification: () => ({ transcript: 'My evidence justification' }),
    addClickListener() {}, closeEvidenceModal() {},
    async openPrototypeWorkspace() { calls.push({ path: 'open-scenarios' }); },
    async callApi(url, options = {}) {
      calls.push({ path: url, body: options.body ? JSON.parse(options.body) : null });
      return { run: { id: 'run-test', label: 'Test run', turn: 1 }, checkpoint: { createdAt: new Date().toISOString() } };
    },
  });
  element('justification').classList.add('active');
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../frontend/run_controls.js'), 'utf8'), context);
  return { context, element, calls, run: code => vm.runInContext(code, context) };
}

test('save preserves choices and draft even after opening Saved Runs', async () => {
  const h = harness();
  h.element('runNameInput').value = 'Weak evidence test';
  await h.run('saveCurrentRun()');
  h.element('justification').classList.remove('active');
  h.element('runs').classList.add('active');
  await h.run('saveCurrentRun()');
  assert.equal(h.calls[0].body.label, 'Weak evidence test');
  assert.equal(h.calls[1].body.draft.screen, 'justification');
  assert.equal(h.calls[1].body.draft.selectedEvidence.id, 'evidence1');
  assert.equal(h.calls[1].body.draft.justification.transcript, 'My evidence justification');
});

test('leave during inference queues one save until evaluation finishes', async () => {
  const h = harness();
  h.context.isEvaluatingTurn = true;
  await h.run('requestRunLeave()');
  assert.equal(h.calls.length, 0);
  assert.equal(h.element('leaveRunBtn').disabled, true);
  h.context.isEvaluatingTurn = false;
  await h.run('finishQueuedRunLeave()');
  assert.equal(h.calls.filter(c => c.path === '/api/runs/leave').length, 1);
  assert.equal(h.calls.at(-1).path, 'open-scenarios');
  assert.equal(h.context.activeRunId, null);
});

test('failed save retains the active run and shows the error', async () => {
  const h = harness();
  h.context.callApi = async () => { throw new Error('Disk is full'); };
  await h.run('withRunControl(() => saveCurrentRun(true))');
  assert.equal(h.context.activeRunId, 'run-test');
  assert.equal(h.element('runSaveStatus').textContent, 'Disk is full');
});

test('untranscribed recording prevents an exit that would lose audio', async () => {
  const h = harness();
  h.context.recordedAudioBlob = {};
  await h.run('withRunControl(() => saveCurrentRun(true))');
  assert.equal(h.calls.length, 0);
  assert.match(h.element('runSaveStatus').textContent, /Transcribe/);
});
