// Voting UI: prefetched pair queue, clip-is-the-button, feedback layers (spec §6).
import { loadSkeleton, ClipView, MultiView } from './renderer.js';

const $ = (id) => document.getElementById(id);

const sessionId =
  localStorage.getItem('sid') ||
  (() => {
    const s = crypto.randomUUID();
    localStorage.setItem('sid', s);
    return s;
  })();

let pairQueue = [];
let current = null;          // {pair_token, clips:[{id, trajectory}, ...]}
let shownAt = 0;
let votesThisSession = Number(sessionStorage.getItem('nv') || 0);
let voting = false;

const skel = await loadSkeleton();
const multi = new MultiView($('gl'));
const viewL = new ClipView(skel);
const viewR = new ClipView(skel);
multi.views = [
  { view: viewL, el: $('paneL') },
  { view: viewR, el: $('paneR') },
];

function loop(now) {
  multi.render(now);
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

async function refill() {
  if (pairQueue.length >= 3) return;
  try {
    const res = await fetch('/pairs?n=5');
    const data = await res.json();
    pairQueue.push(...data.pairs);
  } catch (e) {
    /* server unreachable; retry on next need */
  }
}

function showNext() {
  current = pairQueue.shift() || null;
  refill();
  if (!current) {
    $('feedback').textContent = 'no pairs available — the worker may be mid-cycle; retrying…';
    setTimeout(showNext, 1500);
    return;
  }
  viewL.setClip(current.clips[0].trajectory);   // index 0 renders left (server randomized)
  viewR.setClip(current.clips[1].trajectory);
  shownAt = performance.now();
  document.querySelectorAll('.pane').forEach((p) => p.classList.remove('chosen'));
}

function setTicks(n, target) {
  const ticks = $('ticks');
  ticks.innerHTML = '';
  for (let i = 0; i < target; i++) {
    const s = document.createElement('span');
    if (i < n) s.classList.add('on');
    ticks.appendChild(s);
  }
}

function overlay(kind, title, body) {
  const ov = $('overlay');
  ov.className = 'overlay show ' + kind;
  $('ovTitle').textContent = title;
  $('ovBody').textContent = body;
}
$('ovBtn').onclick = () => ($('overlay').className = 'overlay');

async function vote(sideIdx) {
  if (!current || voting) return;
  voting = true;
  const winner = current.clips[sideIdx].id;
  const dt = Math.round(performance.now() - shownAt);
  document.querySelectorAll('.pane')[sideIdx].classList.add('chosen');
  const body = {
    pair_token: current.pair_token,
    winner_clip: winner,
    dt_ms: dt,
    session_id: sessionId,
  };
  let fb = null;
  try {
    const res = await fetch('/vote', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (res.ok) fb = await res.json();
  } catch (e) { /* dropped vote; move on */ }

  votesThisSession += 1;
  sessionStorage.setItem('nv', votesThisSession);

  if (fb) {
    // Layer 1: every vote — agreement + contest ticks.
    $('feedback').textContent =
      fb.agreement_pct == null
        ? 'first vote on this pair'
        : `${fb.agreement_pct}% of voters agreed with you`;
    if (fb.contest_progress) setTicks(fb.contest_progress.n, fb.contest_progress.target);

    // Layer 2: resolution moments. Rejections MUST be shown (spec §6).
    if (fb.resolution) {
      if (fb.resolution.outcome === 'challenger_won') {
        overlay('promoted', 'New generation.',
          'Your vote resolved the contest — the challenger takes over the lineage. ' +
          'Everything you see from here descends from it.');
      } else {
        overlay('rejected', 'The internet rejected this one.',
          'The challenger lost its contest. The incumbent holds. ' +
          'A new challenger enters the pool.');
      }
      refreshHeader();
    }
  } else {
    $('feedback').textContent = 'vote not recorded (offline?) — next pair';
  }

  // Layer 3: session card every ~20 votes.
  if (votesThisSession > 0 && votesThisSession % 20 === 0) {
    overlay('session', `${votesThisSession} votes this session`,
      'You are one of the hands steering this thing. Share the link — ' +
      'the creature only moves while people are voting.');
  }

  voting = false;
  showNext();
}

$('paneL').onclick = () => vote(0);
$('paneR').onclick = () => vote(1);
addEventListener('keydown', (e) => {
  if (e.key === 'ArrowLeft') vote(0);
  if (e.key === 'ArrowRight') vote(1);
});

async function refreshHeader() {
  try {
    const s = await (await fetch('/state')).json();
    $('gen').textContent = s.generation;
    $('totalVotes').textContent = s.total_votes;
  } catch (e) { /* leave stale */ }
}

await refreshHeader();
setInterval(refreshHeader, 30_000);
await refill();
showNext();
