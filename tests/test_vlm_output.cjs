const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');

const source = fs.readFileSync(path.join(__dirname, '../frontend/script.js'), 'utf8');
const start = source.indexOf('function formatVlmOutput(');
const end = source.indexOf('function formatCoachOutput(', start);
const context = {};
vm.runInNewContext(source.slice(start, end), context);

test('structured extraction displays rows and labels query text separately', () => {
  const output = context.formatVlmOutput({
    evidence_type: 'cloudwatch', query_text: 'search text', time_range: '10:10-10:30 UTC',
    matched_records: 5, event_rows: [
      {timestamp: '10:12', log_stream: 'audit', message: 'event...', truncated: true},
    ], visible_facts_extracted: ['compatibility duplicate'], extraction_warnings: [],
  });
  assert.match(output, /Query editor text \(not an event\):\nsearch text/);
  assert.match(output, /Matched records badge: 5; extracted rows: 1/);
  assert.match(output, /event\.\.\. \[truncated in screenshot\]/);
  assert.doesNotMatch(output, /compatibility duplicate|Security relevance/);
});

test('older saved extractions still display their facts', () => {
  const output = context.formatVlmOutput({visible_facts_extracted: ['Status: Inactive']});
  assert.match(output, /Status: Inactive/);
  assert.doesNotMatch(output, /Query editor text/);
});

test('security display preserves complete server-grounded learner quote', () => {
  const assessment = 'Learner statement: “I did not acknowledge truncation.”\nThis statement needs a more specific connection between a visible fact and the selected action.';
  const output = context.formatSecurityOutput({
    verdict: 'Strong Support', reasoning: 'The event supports review.',
    justification_assessment: assessment,
  });
  assert.ok(output.includes(assessment));
  assert.match(output, /Verdict: Strong Support/);
});

test('strong evidence with no transcript still displays no reasoning credit', () => {
  const output = context.formatSecurityOutput({
    verdict: 'Strong Support', reasoning: 'The event supports review.',
    justification_assessment: 'No learner justification was provided. The evidence verdict does not imply demonstrated learner reasoning.',
  });
  assert.match(output, /No learner justification was provided/);
  assert.doesNotMatch(output, /Great work|correctly acknowledged/);
});
