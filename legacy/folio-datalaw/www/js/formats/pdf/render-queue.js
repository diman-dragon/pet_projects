// Render scheduler: one pdf.js render at a time, most urgent first; paused while the reader pinches / flings the page.
const jobs = new Map();
let running = null, held = 0, until = 0, timer = 0;
export function cancel(key) {
  jobs.delete(key);
  if (running?.key === key) { running.ctl.dead = true; running.ctl.task?.cancel(); }
}
export function clearQueue() {
  jobs.clear();
  if (running) { running.ctl.dead = true; running.ctl.task?.cancel(); }
}
// Short pause (fast scrolling): rendering resumes ms after the last call.
export const pauseFor = (ms) => { until = Math.max(until, performance.now() + ms); };
// Hard hold (pinch): the running render is stopped and restarted after release.
export function hold(on) {
  held = Math.max(0, held + (on ? 1 : -1));
  if (on && running) { running.again = true; running.ctl.dead = true; running.ctl.task?.cancel(); }
  if (!on) { pauseFor(160); pump(); }
}
async function pump() {
  clearTimeout(timer);
  if (running || !jobs.size) return;
  const wait = held ? 120 : until - performance.now();
  if (held || wait > 0) { timer = setTimeout(pump, Math.max(40, wait)); return; }
  let best = null;
  jobs.forEach((j) => { if (!best || j.prio < best.prio) best = j; });
  jobs.delete(best.key);
  running = { key: best.key, job: best, ctl: { dead: false, task: null }, again: false };
  try { await best.run(running.ctl); } catch { /* cancelled or failed job: its owner retries on the next sync */ }
  if (running.again && !jobs.has(best.key)) jobs.set(best.key, best);
  running = null;
  timer = setTimeout(pump, 0);
}
// run(ctl) does the whole job; it must store its pdf.js task in ctl.task and re-check ctl.dead after every await.
export function enqueue(key, prio, run) {
  cancel(key);
  jobs.set(key, { key, prio, run });
  pump();
}
