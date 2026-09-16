#!/usr/bin/env python
"""
prompt_studio.py — local web UI for the Bismuth prompt generator.

Run:  python prompt_studio.py        (opens http://localhost:7801)

ARCHITECTURE v2 front end: every generation goes through llm_bridge.generate
(engine structure + LLM bridges + mechanical verifiers). The controls
collapse to what the author's decided the UI is for -- dropboxes for the resolved
levers (engine, genre, spice, quality, period, appearance, detail) and CHECKBOXES THAT GOVERN
RANDOM GENERATION ONLY: a typed concept is always honoured whatever the
checkbox says; the checkbox only decides whether the dice may add that
section when the prompt is silent.

Pure stdlib. Runs its own llama-server from LLM/llama.cpp (revival:
`lms server start` then `lms load qwen3-8b-heretic --gpu max -y`).
"""

import hashlib
import json
import math
import os
import random
from collections import Counter
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from promptstudio.engine import enhancer as pe
from promptstudio.engine import bridge as lb
from promptstudio import paths as _paths

PORT = 7801
HERE = os.path.dirname(os.path.abspath(__file__))
# the browser remembers inputs per origin, and every install on a machine
# shares localhost:7801 -- so the storage key carries an install id and
# a fresh install never shows another install's inputs (2026-09-16)
INSTALL_ID = hashlib.sha1(os.path.abspath(os.path.join(HERE, "..", "..")).encode("utf-8")).hexdigest()[:10]
def llm_status():
    """the engine's state (llm_server): 'own' = our llama-server is
    serving; 'ready-to-start' = it will auto-start on the first generate
    (~20s); '' = no engine possible. There is no third-party fallback."""
    try:
        from promptstudio.llm import server as llm_server
        return llm_server.status()
    except Exception:
        return ""


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Bismuth Prompt Studio</title>
<style>
:root { --bg:#14151a; --panel:#1e2028; --acc:#c88df0; --acc2:#7fd3c9; --tx:#e8e6ef; --dim:#9a97a8; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--tx); font:15px/1.5 "Segoe UI",system-ui,sans-serif; }
.wrap { max-width:980px; margin:0 auto; padding:24px 16px 60px; }
h1 { font-size:22px; margin:0 0 4px; } h1 b { color:var(--acc); }
.sub { color:var(--dim); font-size:13px; margin-bottom:18px; }
.sub .ok { color:var(--acc2); } .sub .bad { color:#e0876a; }
.panel { background:var(--panel); border-radius:12px; padding:16px; margin-bottom:14px; }
textarea { width:100%; background:#12131a; color:var(--tx); border:1px solid #333646; border-radius:8px; padding:10px; font:15px monospace; resize:vertical; min-height:56px; }
.row { display:flex; flex-wrap:wrap; gap:12px; margin-top:12px; align-items:flex-end; }
.field { display:flex; flex-direction:column; gap:4px; font-size:12px; color:var(--dim); }
select,input[type=number] { background:#12131a; color:var(--tx); border:1px solid #333646; border-radius:7px; padding:7px 9px; font-size:14px; }
input:not([type]),input[type=text] { background:#12131a; color:var(--tx); border:1px solid #333646; border-radius:7px; padding:7px 9px; font:14px monospace; width:100%; box-sizing:border-box; }
.field textarea { min-height:44px; }
.chk { display:flex; gap:6px; align-items:center; color:var(--tx); font-size:14px; padding:7px 0; }
.chklbl { color:var(--dim); font-size:12px; align-self:center; padding-right:2px; }
button.go { background:var(--acc); color:#181322; border:0; border-radius:8px; padding:10px 26px; font-size:15px; font-weight:600; cursor:pointer; }
button.go[disabled] { opacity:.65; cursor:progress; }
.spin { display:inline-block; width:12px; height:12px; margin-right:8px; vertical-align:-1px;
        border:2px solid #18132255; border-top-color:#181322; border-radius:50%;
        animation:sp .7s linear infinite; }
@keyframes sp { to { transform:rotate(360deg); } }
.working { color:var(--acc2); font-size:13px; margin-top:10px; }
button.go:hover { filter:brightness(1.1); }
.out { background:#12131a; border:1px solid #333646; border-radius:8px; padding:12px; margin-top:10px; font:13.5px/1.55 monospace; white-space:pre-wrap; word-break:break-word; }
.out .nl { color:#cfd6b8; border-top:1px dashed #333646; margin-top:8px; padding-top:8px; }
.out .neg { color:var(--dim); border-top:1px dashed #333646; margin-top:8px; padding-top:8px; }
.out .rep { color:var(--dim); font:12px/1.6 "Segoe UI",system-ui,sans-serif; border-top:1px dashed #333646; margin-top:8px; padding-top:8px; }
.out .rep .warn { color:#d8a657; }
.out .rep .why { color:var(--acc2); }
.bar { display:flex; gap:8px; align-items:center; margin-bottom:6px; }
.copy { background:#2a2d3a; color:var(--acc2); border:0; border-radius:6px; padding:4px 10px; font-size:12px; cursor:pointer; }
.copy:hover { background:#343849; }
.tagline { color:var(--acc2); font-size:12px; flex:1; }
.note { color:#e0b66a; font-size:12px; margin-top:6px; }
.err { color:#e0876a; font-size:13px; margin-top:10px; white-space:pre-wrap; }
.acwrap { position:relative; }
.ac { position:absolute; left:0; right:0; top:100%; z-index:40; background:#12131a;
      border:1px solid #45496040; border-top:0; border-radius:0 0 8px 8px; max-height:260px;
      overflow-y:auto; box-shadow:0 10px 24px #0009; display:none; }
.ac div { padding:6px 11px; font:13.5px monospace; cursor:pointer; display:flex; gap:8px; }
.ac div .rank { color:var(--dim); font-size:11px; margin-left:auto; }
.ac div .kind { color:#8fb3a0; font-size:10.5px; font-style:italic; }
.ac div.look > span:first-child { color:#b9d6c6; }
.ac div.sel, .ac div:hover { background:#2c2f3d; color:var(--acc); }
.hint { color:var(--dim); font-size:11.5px; margin-top:5px; }
.tabs { display:flex; gap:6px; margin:8px 0 10px; }
.tab { background:#1b1d27; color:var(--dim); border:1px solid #2c2f3c;
       border-radius:6px; padding:4px 14px; font-size:12px; cursor:pointer; }
.tab:hover { color:#cfd3e2; }
.tab.on { background:#2a2f45; color:#fff; border-color:#4a5170; }
</style></head><body><div class="wrap">
<h1><b>Bismuth</b> Prompt Studio</h1>
<div class="sub">a few words in &rarr; a full two-part prompt out &middot; <span id="stats"></span> &middot; LLM: <span id="llmstat"></span> &middot; autocomplete: <span id="vocabsrc"></span></div>
<div class="panel">
  <div class="bar" style="margin-bottom:8px">
    <button class="copy" id="inventBtn" onclick="invent()" title="ask the local language model for a scene idea, guided by the controls below. It never reads what is in this box.">\u2726 let the LLM invent it</button>
    <span class="tagline" id="inventNote"></span>
  </div>
  <div class="acwrap">
    <textarea id="base" autocomplete="off" spellcheck="false" placeholder="1girl dancing on a rooftop at night  —  or tags, or nothing at all (empty = full creative mode)"></textarea>
    <div class="ac" id="ac"></div>
  </div>
  <div class="row" style="margin-top:8px; flex-wrap:wrap; gap:10px">
    <div class="field" style="flex:1 1 100%" title="what must NOT appear: goes to the negative line of both tabs, prunes the plan before the prose and gates the dice; a word you typed in the brief still wins"><span>what should not appear?</span><textarea id="exclude" rows="2" spellcheck="false" style="width:100%; box-sizing:border-box; resize:vertical" placeholder="glasses, hat, text on clothes, extra people..."></textarea></div>
    <div class="field acwrap" style="flex:1 1 48%" title="a treatment or look, kept apart from the sentence (appended to the brief): a style name the pool knows, a medium, a palette, a technique. Type two letters for the list of styles, mediums, palettes and techniques the studio knows."><span>style hint</span><textarea id="styleHint" rows="2" autocomplete="off" spellcheck="false" style="width:100%; box-sizing:border-box; resize:vertical" placeholder="ink and watercolor, film noir, soft pastel palette..."></textarea><div class="ac" id="acStyle"></div></div>
    <div class="field acwrap" style="flex:1 1 48%" title="artist references, comma separated, spelled as on danbooru (underscores or spaces). The @ is OPTIONAL for an artist the pool knows and REQUIRED for a name it does not know, or a name that is also an ordinary tag; the tabs write it their own way (anima adds @, illustrious never does)."><span>artist hint (the @ is optional for known artists; required for unknown names)</span><textarea id="artistHint" rows="2" autocomplete="off" spellcheck="false" style="width:100%; box-sizing:border-box; resize:vertical" placeholder="shirow masamune, @some_artist_the_pool_does_not_know, (wlop:0.8)"></textarea><div class="ac" id="acArtist"></div></div>
    <label class="chk" title="keep these inputs in this browser and restore them next time"><input type="checkbox" id="remember"> remember inputs</label>
  </div>
  <div class="hint">type to autocomplete (booru tags here, styles and artists in their own fields) · ↑↓ choose · Tab/Enter insert · Esc dismiss · Ctrl+Enter generate · typed concepts are ALWAYS honoured — the checkboxes only allow the dice to add a section you did not mention</div>
  <div class="row">
    <div class="field" title="fast = no LLM. The engine resolves the whole scene from the measured pools and the tag network, then verifies it exactly as the LLM path does. About a second per prompt instead of 15-40s; the prose is templated rather than written.">engine<select id="engine"><option value="llm" selected>llm (richer prose)</option><option value="fast">fast (no llm)</option></select></div>
    <div class="field" title="random = the dice choose a genre for the prompt (as before). A selected genre drives the prompt: its places, jobs, races, poses, clothes. A genre word typed in the prompt still wins.">genre<select id="genre"><option value="random" selected>random</option></select></div>
    <div class="field">spice<select id="spice"><option selected>auto</option><option>safe</option><option>sensitive</option><option>nsfw</option><option>explicit</option></select></div>
    <div class="field">quality<select id="quality"><option selected>standard</option><option>off</option><option>maximum</option></select></div>
    <div class="field" id="periodField" title="the year/era band (anima's 1c channel). The illustrious tab does not use it.">period<select id="period"></select></div>
    <div class="field">variants<input type="number" id="variants" value="1" min="1" max="10"></div>
    <div class="field">seed (blank = random)<input type="number" id="seed" placeholder="random"></div>
  </div>
  <div class="row">
    <span class="chklbl">generate when not typed:</span>
    <label class="chk" title="name unnamed subjects as EXISTING characters, gender-matched and series-scoped when the prompt names one"><input type="checkbox" id="genCharacters"> generate random characters</label>
    <label class="chk" title="draw 1-3 artists (50/30/20) matched to the resolved style by measured fingerprint, and to the spice level: a safe image draws only from artists whose own measured work is safe. Typed artists always pass regardless."><input type="checkbox" id="genArtists"> artists</label>
    <button class="go" style="margin-left:auto" onclick="gen()">Generate</button>
    <button class="copy" id="cancelBtn" title="stop after the current model call; finished variants stay">cancel</button>
  </div>
  <div class="row">
    <div class="field" title="automatic = the prompt first, then canon when the subject is a known character, then the ordinary dependencies (occupation, gender, spice, actions). random = the prompt first, then rolled — canon does not dress the character, but identity (hair and eye colour) still locks, so a named character stays recognisable.">character appearance<select id="appearance"><option selected>automatic</option><option>random</option></select></div>
    <div class="field" title="how much per-subject description depth is generated (scales all subject detail)">character detail<select id="detail"><option>minimal</option><option selected>standard</option><option>detailed</option></select></div>
  </div>
  <div class="row" style="border-top:1px dashed #333646; padding-top:10px; margin-top:14px">
    <button class="copy" id="releaseBtn" onclick="releaseLLM()" title="unload the LLM engine and free all its VRAM for image generation. The next Generate reloads it automatically (~20s).">&#9209; release LLM</button>
  </div>
</div>
<div id="working" class="working"></div>
<div id="results"></div>
<script>
function fillPeriod() {
  const p = document.getElementById('period');
  ['newest','recent','mid','early','old'].forEach(t => p.add(new Option(t)));
  for (let y = 2025; y >= 2000; y--) p.add(new Option(y));
}
function fillGenres(genres) {
  // idempotent (2026-09-14: boot() runs twice and every group was appended
  // twice); a bucket whose only leaf is itself is one option, not a group
  const g = document.getElementById('genre');
  if (!g || !genres) return;
  const keep = g.value;
  Array.from(g.querySelectorAll('optgroup, option')).forEach(n => { if (n.value !== 'random') n.remove(); });
  const byBucket = {};
  genres.forEach(e => { (byBucket[e.bucket] = byBucket[e.bucket] || []).push(e.leaf); });
  Object.keys(byBucket).forEach(b => {
    const leaves = byBucket[b];
    if (leaves.length === 1 && leaves[0] === b) { g.appendChild(new Option(b, b)); return; }
    const grp = document.createElement('optgroup'); grp.label = b;
    leaves.forEach(l => grp.appendChild(new Option(l, l)));
    g.appendChild(grp);
  });
  if (keep && Array.from(g.options).some(o => o.value === keep)) g.value = keep;
}
let INPUT_KEY = null;   // 'ps-inputs:' + this install's id, set by boot
// THE IDEA BUTTON. It replaces the brief, so what was typed is kept and
// can be put back; the ideas already seen ride along so the model is asked
// for something else each press.
let IDEAS = [], BEFORE_INVENT = null;
async function invent() {
  const btn = document.getElementById('inventBtn'), note = document.getElementById('inventNote');
  const box = document.getElementById('base');
  if (btn.disabled) return;
  btn.disabled = true; const lbl = btn.innerHTML;
  btn.innerHTML = '<span class="spin"></span>inventing\u2026';
  note.textContent = '';
  try {
    const r = await fetch('/api/invent', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({spice: document.getElementById('spice').value,
                            genre: document.getElementById('genre').value,
                            period: document.getElementById('period').value,
                            exclude: document.getElementById('exclude').value,
                            avoid: IDEAS.slice(-5)})});
    const j = await r.json();
    if (j.error) { note.innerHTML = '<span style="color:#e0876a">' + j.error + '</span>'; }
    else if (!j.idea) { note.textContent = j.note || 'the model returned nothing \u2014 press again'; }
    else {
      if (BEFORE_INVENT === null) BEFORE_INVENT = box.value;
      box.value = j.idea; IDEAS.push(j.idea);
      let msg = j.genre ? ('idea for ' + j.genre + (j.place ? ' \u00b7 ' + j.place : '') + ' at ' + j.level) : '';
      if (j.note) msg += (msg ? ' \u00b7 ' : '') + j.note;
      note.textContent = msg;
      if (BEFORE_INVENT) {
        const a = document.createElement('a');
        a.href = '#'; a.style.cssText = 'color:var(--acc2); margin-left:8px';
        a.textContent = 'put back what I typed';
        a.onclick = e => { e.preventDefault(); box.value = BEFORE_INVENT; BEFORE_INVENT = null; note.textContent = ''; };
        note.appendChild(a);
      }
    }
  } catch (e) { note.innerHTML = '<span style="color:#e0876a">' + e + '</span>'; }
  btn.disabled = false; btn.innerHTML = lbl;
  if (LLM_STATE === '') { btn.disabled = true; }
}

let LLM_STATE = '';
async function boot() {
  const j = await (await fetch('/api/info')).json();
  fillGenres(j.genres);
  INPUT_KEY = 'ps-inputs:' + (j.install || 'default'); restoreInputs();
  const el = document.getElementById('llmstat');
  if (j.llm === 'own') { el.textContent = 'built-in engine running'; el.className = 'ok'; }
  else if (j.llm === 'ready-to-start') { el.textContent = 'built-in engine — auto-starts on first generate (\\u224820s)'; el.className = 'ok'; }
  else { el.textContent = '\\u26a0 no engine: backend exe or GGUF missing — see llm_server.py'; el.className = 'bad'; }
  // the idea button needs the model: greyed out, and it says why
  LLM_STATE = j.llm || '';
  const ib = document.getElementById('inventBtn');
  if (!LLM_STATE) {
    ib.disabled = true; ib.style.opacity = '.45'; ib.style.cursor = 'not-allowed';
    ib.title = 'needs the local language model \u2014 see README, "Setting up the language model"';
  }
}

function repLine(r) {
  const bits = [];
  if (r.engine_note) bits.push('<span class="warn">engine: ' + r.engine_note + '</span>');
  const cast = Object.entries(r.cast || {}).filter(([k,v]) => v > 0)
    .map(([k,v]) => v + ' ' + k).join(', ') || 'no subjects';
  bits.push('cast: ' + cast);
  let sp = 'spice: ' + r.level;
  if (r.switched) sp += ` <span class="warn">\\u26a0 your prompt implies ${r.switched.to} — generated at ${r.switched.to}, dropdown said ${r.switched.from}</span>`;
  bits.push(sp);
  const du = r.dropped_unknown || {};
  Object.keys(du).forEach(m => { if ((du[m] || []).length) bits.push('not on ' + m + ' (its booru has no such tag; the prose keeps it): ' + du[m].join(', ')); });
  if ((r.refused || []).length) bits.push('<span class="warn">refused: ' + r.refused.join(', ') + ' (sexualised minors are never generated; youth-coded words are refused above safe)</span>');
  if (r.why && Object.keys(r.why).length) {
    const w = r.why, parts = [];
    if (w.genre) parts.push('genre ' + w.genre + (w.bucket && w.bucket !== w.genre ? ' (' + w.bucket + ')' : '') + (w.genre_src === 'typed' ? ' [typed]' : ''));
    if (w.place) parts.push('place ' + w.place + (w.place_src === 'typed' ? ' [typed]' : ''));
    if ((w.occupations || []).length) parts.push('occupation ' + w.occupations.join(' / '));
    if ((w.races || []).length) parts.push('race ' + w.races.join(' / '));
    if (w.event) parts.push('occasion ' + w.event);
    if (w.season) parts.push('season ' + w.season);
    if (w.weather) parts.push('weather ' + w.weather);
    if (w.act) parts.push('act ' + w.act + ' (frames the picture)');
    if (w.style) parts.push('style ' + w.style + (w.style_src === 'typed' ? ' [typed]' : ''));
    if (w.engine) parts.push('engine ' + w.engine);
    bits.push('<span class="why">why: ' + parts.join(' &middot; ') + '</span>');
  }
  if ((r.llm_calls || []).length) bits.push('llm: ' + r.llm_calls.map(c => c.stage + ' ' + (c.seconds == null ? '?' : c.seconds.toFixed(1)) + 's' + (c.completion_tokens ? ' ' + c.completion_tokens + ' tok' : '') + (c.truncated ? ' [cut, retried]' : '') + (c.status !== 'ok' ? ' [' + c.status + ']' : '')).join(', '));
  if ((r.injected_for_level || []).length) bits.push('added for level: ' + r.injected_for_level.join(', '));
  const onLine = new Set(((r.prompts||{}).anima || r.prompt || '').split('\\n')[0].split(',').map(t => t.trim().toLowerCase()));
  const nlOnly = (r.unmapped || []).filter(u => !onLine.has(String(u).toLowerCase()));
  if (nlOnly.length) bits.push('NL-only (no booru tag): ' + nlOnly.join(', '));
  if ((r.stripped_by_verifier || []).length) bits.push('<span class="warn">verifier stripped: ' + r.stripped_by_verifier.map(x => Array.isArray(x) ? x.join('\\u2192') : x).join(', ') + '</span>');
  if ((r.coverage || []).length) bits.push('coverage: ' + r.coverage.map(c => c.concept + ' \u2192 ' + c.fate).join(' \u00b7 '));
  return bits.join(' &middot; ');
}
let LAST_BODY = null, LAST_RESULTS = [];
function downloadBundle() {
  const blob = new Blob([JSON.stringify({request: LAST_BODY, results: LAST_RESULTS}, null, 1)], {type: 'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'prompt-studio-bundle.json'; a.click();
}
function restoreInputs() {
  try {
    if (!INPUT_KEY) return;
    const b = JSON.parse(localStorage.getItem(INPUT_KEY) || 'null'); if (!b) return;
    document.getElementById('remember').checked = true;
    const set = (id, v) => { const el = document.getElementById(id); if (el && v != null) el.value = v; };
    set('base', b.base); set('exclude', b.exclude); set('styleHint', b.style_hint); set('artistHint', b.artist_hint);
    set('spice', b.spice); set('quality', b.quality); set('appearance', b.appearance); set('detail', b.detail); set('variants', b.variants);
    document.getElementById('engine').value = b.fast ? 'fast' : 'llm';
    document.getElementById('genCharacters').checked = !!b.gen_characters; document.getElementById('genArtists').checked = !!b.gen_artists;
  } catch (e) {}
}

function card(r, i) {
  const d = document.createElement('div');
  d.className = 'out';
  const bar = document.createElement('div'); bar.className = 'bar';
  bar.innerHTML = `<span class="tagline">variant ${i+1} · seed ${r.seed}</span>`;
  const cp = document.createElement('button'); cp.className = 'copy'; cp.textContent = 'copy prompt';
  // the copy buttons copy what is ON THE CARD, edits included (the tag
  // line and the prose are editable in place)
  const current = () => tagsEl.textContent.trim() + (nlEl.textContent.trim() ? '\\n' + nlEl.textContent.trim() : '');
  cp.onclick = () => { navigator.clipboard.writeText(current()); cp.textContent='copied!'; setTimeout(()=>cp.textContent='copy prompt',1200); };
  const cs = document.createElement('button'); cs.className = 'copy'; cs.textContent = 'copy story';
  cs.onclick = () => { navigator.clipboard.writeText(nlEl.textContent.trim()); cs.textContent='copied!'; setTimeout(()=>cs.textContent='copy story',1200); };
  const note = document.createElement('div'); note.className = 'note';
  const us = document.createElement('button'); us.className = 'copy'; us.textContent = 'use story as start';
  us.title = 'put this story (edited or not) into the brief for a second pass';
  us.onclick = () => { document.getElementById('base').value = nlEl.textContent.trim() || tagsEl.textContent.trim(); window.scrollTo(0, 0); };
  const ru = document.createElement('button'); ru.className = 'copy'; ru.textContent = 'reuse inputs';
  ru.title = 'restore the inputs that made this card, seed included';
  ru.onclick = () => { if (LAST_BODY) { const b = Object.assign({}, LAST_BODY, {seed: r.seed}); localStorage.setItem(INPUT_KEY, JSON.stringify(b)); restoreInputs(); document.getElementById('seed').value = r.seed; window.scrollTo(0, 0); } };
  const dl = document.createElement('button'); dl.className = 'copy'; dl.textContent = 'download bundle'; dl.title = 'every result of this run as JSON';
  dl.onclick = downloadBundle;
  bar.append(cp, cs, us, ru, dl);

  // ONE PROMPT, TWO TABS (the author's). The scene is decided once; each tab
  // is that same scene packaged for its model -- anima gets its score /
  // year / safety tags, the '@' on artist names and the prose paragraph;
  // illustrious gets its own quality tags, bare artist names and comma
  // phrases. Nothing is generated twice.
  const prompts = r.prompts || {anima: r.prompt};
  const negatives = r.negatives || {anima: r.negative};
  const tabs = document.createElement('div'); tabs.className = 'tabs';
  const tagsEl = document.createElement('div'); tagsEl.contentEditable = 'true'; tagsEl.spellcheck = false; tagsEl.title = 'editable: click to edit, copy takes the edited text';
  const nlEl = document.createElement('div'); nlEl.className = 'nl'; nlEl.contentEditable = 'true'; nlEl.spellcheck = false; nlEl.title = 'editable';
  const edits = {};      // per-tab edits survive a tab switch
  tagsEl.addEventListener('input', () => { edits[d.dataset.tab] = current(); });
  nlEl.addEventListener('input', () => { edits[d.dataset.tab] = current(); });
  const ng = document.createElement('div'); ng.className = 'neg';
  const nb = document.createElement('button'); nb.className = 'copy';
  nb.style.cssFloat = 'right'; nb.textContent = 'copy negative';

  function show(which) {
    const text = edits[which] || prompts[which] || '';
    d.dataset.prompt = text; d.dataset.tab = which;
    const nlSplit = text.indexOf('\\n');
    if (nlSplit > -1) {
      tagsEl.textContent = text.slice(0, nlSplit);
      nlEl.textContent = text.slice(nlSplit + 1);
      nlEl.style.display = '';
    } else {
      tagsEl.textContent = text;
      nlEl.textContent = ''; nlEl.style.display = 'none';
    }
    const neg = negatives[which] || '';
    nb.onclick = () => { navigator.clipboard.writeText(neg); nb.textContent='copied!'; setTimeout(()=>nb.textContent='copy negative',1200); };
    ng.textContent = 'Negative: ' + neg;
    ng.prepend(nb);
    ng.style.display = neg ? '' : 'none';
    [...tabs.children].forEach(t2 => t2.classList.toggle('on', t2.dataset.m === which));
    cp.textContent = 'copy prompt';
  }
  ['anima', 'illustrious'].forEach(m2 => {
    if (!prompts[m2]) return;
    const b2 = document.createElement('button');
    b2.className = 'tab'; b2.dataset.m = m2; b2.textContent = m2;
    b2.onclick = () => show(m2);
    tabs.append(b2);
  });
  d.append(bar, tabs, tagsEl, nlEl, note, ng);
  show(prompts.anima ? 'anima' : Object.keys(prompts)[0]);
  const rep = document.createElement('div'); rep.className = 'rep';
  rep.innerHTML = repLine(r);
  d.append(rep);
  return d;
}
async function gen() {
  const body = {
    base: document.getElementById('base').value,
    fast: document.getElementById('engine').value === 'fast',
    spice: document.getElementById('spice').value,
    genre: document.getElementById('genre').value,
    quality: document.getElementById('quality').value,
    period: document.getElementById('period').value,
    appearance: document.getElementById('appearance').value,
    detail: document.getElementById('detail').value,
    variants: +document.getElementById('variants').value,
    seed: document.getElementById('seed').value,
    gen_characters: document.getElementById('genCharacters').checked,
    gen_artists: document.getElementById('genArtists').checked,
    exclude: document.getElementById('exclude').value,
    style_hint: document.getElementById('styleHint').value,
    artist_hint: document.getElementById('artistHint').value,
  };
  try { if (document.getElementById('remember').checked) localStorage.setItem(INPUT_KEY, JSON.stringify(body)); else localStorage.removeItem(INPUT_KEY); } catch (e) {}
  LAST_BODY = body;
  const btn = document.querySelector('button.go');
  const work = document.getElementById('working');
  const t0 = Date.now();
  btn.disabled = true;
  const label = btn.textContent;
  btn.innerHTML = '<span class="spin"></span>Generating…';
  let stage = '';
  const poll = setInterval(async () => {
    try { const st = await (await fetch('/api/status')).json(); stage = [st.stage, st.detail].filter(Boolean).join(' · ') + (st.calls ? ` · ${st.calls} model call${st.calls>1?'s':''}` : ''); } catch (e) {}
  }, 700);
  const tick = setInterval(() => {
    work.textContent = `building ${body.variants} prompt${body.variants>1?'s':''}… ` + (stage ? stage + ' · ' : '') + ((Date.now()-t0)/1000).toFixed(1) + 's';
  }, 100);
  let j;
  try {
    document.getElementById('cancelBtn').onclick = () => { fetch('/api/cancel', {method:'POST'}); work.textContent = 'cancelling after the current call\u2026'; };
  j = await (await fetch('/api/generate', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)})).json();
  } catch (e) {
    work.textContent = 'generation failed: ' + e;
    clearInterval(tick); clearInterval(poll); btn.disabled = false; btn.textContent = label;
    return;
  }
  clearInterval(tick); clearInterval(poll);
  LAST_RESULTS = j.results || [];
  btn.disabled = false; btn.textContent = label;
  work.textContent = `done in ${((Date.now()-t0)/1000).toFixed(1)}s`;
  const res = document.getElementById('results');
  res.innerHTML = '';
  (j.results || []).forEach((r, i) => {
    if (r.error) {
      const e = document.createElement('div'); e.className = 'err';
      e.textContent = `variant ${i+1} failed: ${r.error}`;
      res.appendChild(e);
    } else res.appendChild(card(r, i));
  });
}
async function releaseLLM() {
  const btn = document.getElementById('releaseBtn');
  btn.disabled = true; btn.textContent = 'releasing\\u2026';
  try {
    const j = await (await fetch('/api/llm_release', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'})).json();
    btn.textContent = j.released ? '\\u23f9 released \\u2713' : '\\u23f9 release LLM';
    boot();
  } catch (e) { btn.textContent = '\\u23f9 release LLM'; }
  setTimeout(() => { btn.textContent = '\\u23f9 release LLM'; btn.disabled = false; }, 2000);
  btn.disabled = false;
}
// ---------------- autocomplete ----------------
// ONE IMPLEMENTATION, THREE FIELDS (2026-09-16). The brief suggests the
// booru vocabulary; the style hint suggests the style axis's OWN
// vocabulary (the styles the pool measures, the mediums, the palettes and
// techniques they are made of, and the looks the checkpoints know); the
// artist hint suggests every artist the studio can steer. A hint field
// used to be a guessing game -- these lists say what the studio knows.
function makeAC(input, box, opts) {
  opts = opts || {};
  let words = [], counts = [], kinds = [], nsfw = [], items = [], sel = -1, asked = false;

  function tok() {
    const pos = input.selectionStart;
    const before = input.value.slice(0, pos);
    const start = Math.max(before.lastIndexOf(','), before.lastIndexOf('\\n')) + 1;
    let text = before.slice(start).replace(/^\\s+/, ''), at = '';
    // the '@' is booru artist syntax: it is the user's, so it survives
    if (opts.at && text.startsWith('@')) { at = '@'; text = text.slice(1); }
    return {start, text, at, pos};
  }
  function hide() { box.style.display = 'none'; items = []; sel = -1; }
  function render(matches) {
    if (!matches.length) return hide();
    items = matches.map(m => m[0]); sel = 0;
    box.innerHTML = '';
    matches.forEach((m, i) => {
      const d = document.createElement('div');
      const look = m[2] === 'look';
      d.className = (i === 0 ? 'sel' : '') + (look ? ' look' : '');
      // POST COUNT, not a rank -- danbooru shows counts here and so should we.
      // A 0 means the concept is not a booru tag (a checkpoint style, a
      // studio look): it still works, it is just not measured on booru.
      // A LOOK (2026-09-17) counts ARTISTS: the phrase as the artists'
      // descriptions say it -- it steers the artists and stays in the prompt.
      const n = m[1] || 0;
      const num = n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + 'k' : (n || '');
      const lbl = look ? `<span class="kind">look</span> ${num} artists` : num;
      d.innerHTML = `<span>${m[0].replace(/&/g,'&amp;').replace(/</g,'&lt;')}</span><span class="rank">${lbl}</span>`;
      d.onmousedown = e => { e.preventDefault(); accept(i); };
      box.appendChild(d);
    });
    box.style.display = 'block';
  }
  function update() {
    const q = tok().text.trim().toLowerCase();
    if (q.length < 2 || !words.length) return hide();
    // Match the exact name AND any word that starts with the query,
    // danbooru style ("pov" offers "pov", "pov hands", "futanari pov"),
    // sorted purely by post count with the exact match pinned first.
    const hit = [];
    // the looks that lean to nsfw artists are offered at nsfw and explicit only
    // a look is offered at the spice levels that allow what it says; auto reads
    // like sensitive (2026-09-17)
    const spiceEl = document.getElementById('spice');
    const RANK = {safe: 0, sensitive: 1, nsfw: 2, explicit: 3, auto: 1};
    const allowed = spiceEl ? (RANK[spiceEl.value] ?? 1) : 1;
    for (let i = 0; i < words.length; i++) {
      const t = words[i];
      if ((RANK[nsfw[i]] || 0) > allowed) continue;
      if (t === q) { hit.push([i, 2]); continue; }
      let w = false;
      if (t.startsWith(q)) w = true;
      else {
        let p = t.indexOf(q);
        while (p > 0) {
          const c = t[p - 1];
          if (c === " " || c === "_" || c === "-" || c === "(") { w = true; break; }
          p = t.indexOf(q, p + 1);
        }
      }
      if (w) hit.push([i, 1]);
    }
    hit.sort((a, b) => (b[1] - a[1]) || ((counts[b[0]] || 0) - (counts[a[0]] || 0)));
    render(hit.slice(0, 12).map(h => [words[h[0]], counts[h[0]] || 0, kinds[h[0]] || '']));
  }
  function accept(i) {
    if (i < 0 || i >= items.length) return;
    const t = tok();
    const after = input.value.slice(t.pos), head = input.value.slice(0, t.start);
    const pad = (head && !/\\s$/.test(head)) ? ' ' : '';   // keep ", tag" spacing tidy
    const insert = pad + t.at + items[i] + (after.trim().startsWith(',') ? '' : ', ');
    input.value = head + insert + after;
    const caret = (head + insert).length;
    input.setSelectionRange(caret, caret);
    hide(); input.focus();
  }
  function moveSel(d) {
    if (!items.length) return;
    const kids = box.children;
    kids[sel] && (kids[sel].className = '');
    sel = (sel + d + items.length) % items.length;
    kids[sel].className = 'sel';
    kids[sel].scrollIntoView({block:'nearest'});
  }
  // A HINT VOCABULARY IS FETCHED ON FIRST USE, not at boot: the artist
  // list is 48k names, and a user who never touches the field never pays
  // for it.
  function load() {
    if (asked || !opts.url) return;
    asked = true;
    fetch(opts.url).then(r => r.json()).then(j => {
      words = j.tags || []; counts = j.counts || []; kinds = j.kinds || []; nsfw = j.nsfw || []; update();
    }).catch(() => {});
  }
  if (opts.url) { input.addEventListener('focus', load); }
  input.addEventListener('input', update);
  input.addEventListener('blur', () => setTimeout(hide, 120));
  input.addEventListener('keydown', e => {
    const open = box.style.display === 'block';
    if (open && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) { e.preventDefault(); moveSel(e.key === 'ArrowDown' ? 1 : -1); return; }
    if (open && (e.key === 'Tab' || (e.key === 'Enter' && !e.ctrlKey))) { e.preventDefault(); accept(sel); return; }
    if (open && e.key === 'Escape') { e.preventDefault(); hide(); return; }
    if (e.key === 'Enter' && e.ctrlKey) { hide(); gen(); }
  });
  return {setVocab(w, c) { words = w || []; counts = c || []; }};
}

const TAG_AC = makeAC(document.getElementById('base'), document.getElementById('ac'));
makeAC(document.getElementById('styleHint'), document.getElementById('acStyle'),
       {url: '/api/vocab_style'});
makeAC(document.getElementById('artistHint'), document.getElementById('acArtist'),
       {url: '/api/vocab_artist', at: true});

function loadVocab() {
  // anima is the canonical resolution mode -- the illustrious tab is a
  // repackaging of the same decisions, so there is one vocabulary
  fetch('/api/vocab?mode=anima').then(r => r.json()).then(j => {
    TAG_AC.setVocab(j.tags, j.counts || []);
    const el = document.getElementById('vocabsrc');
    if (el) el.textContent = `${j.tags.length} ${j.source} tags`;
  });
}
fillPeriod();
loadVocab();

boot();
</script></div></body></html>"""


def load_base_vocab(source="danbooru"):
    """Layer 1 + 3 of the vocabulary, for ONE booru.

    Which booru depends on the mode being prompted. Illustrious is tagged the
    danbooru way; Anima's README asks for the gelbooru spelling wherever the two
    differ, so suggesting 'navel' while prompting Anima offers the wrong word.
    Gelbooru's own list is used for anima, danbooru's for illustrious.
    extra_tags.json applies to both — it is finetune vocabulary, not booru vocabulary.
    extra_tags.json    — prompt-craft vocabulary danbooru never had but finetunes
    were trained on anyway (quality tokens, rendering/photography/lighting language).
    The prompt corpus (layer 2) does not add vocabulary so much as show how these
    tags are actually USED, so it re-weights rather than defines.
    """
    base, known = Counter(), set()
    # LAYERED, NOT SWAPPED (the author's): anima's vocabulary is danbooru's
    # PLUS gelbooru -- gelbooru overrides the spelling where the two
    # differ (the _bridge dedup below) and adds its own tags on top.
    # Loading only the gelbooru harvest lost every danbooru tag the
    # harvest missed (19,777 vs the 25,000 cap).
    files = (["danbooru_tags.json", "gelbooru_tags.json"]
             if source in ("gelbooru", "combined") else ["danbooru_tags.json"])
    for name in files:
        p = _paths.data(name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                for tag, posts in json.load(f).items():
                    # log-ish compression: 8M-post '1girl' must not bury
                    # everything else
                    w = max(1, int(math.log10(max(posts, 10)) * 12))
                    base[tag] = max(base[tag], w)
                    known.add(tag)
    p = _paths.data("extra_tags.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for cat, blob in json.load(f).items():
                if not isinstance(blob, dict):
                    continue
                for tag, w in blob.get("tags", []):
                    t = pe.normalize_tag(tag)
                    # floor above the danbooru long tail so hand-picked prompt-craft
                    # vocabulary always survives the ranking cutoff
                    base[t] = max(base[t], 55 + int(w) * 6)
                    known.add(t)
    return base, known


# 8,000 was hiding two thirds of the usable vocabulary: danbooru carries
# 23,883 tags with 100+ posts, and the cap meant "futanari pov" (1,377),
# "pov breasts" (1,143) and "taker pov" (825) could not be typed at all.
#
# 25,000 then became the same problem again. After the two re-harvests the
# combined vocabulary is 38,393 and the cap was silently cutting 13,393
# suggestions -- including the whole (medium) family we had just gone to
# the trouble of recovering. The engine could emit them; the user could not
# type them. The whole list is 0.89 MB of JSON against 0.51 MB at the old
# cap, which is not a reason to hide a third of the vocabulary.
def build_vocab(banks, limit=60000, source="danbooru"):
    """Autocomplete vocabulary: the booru base plus the finetune extras
    (the v1 curated banks are gone, 2026-09-16)."""
    c, known = load_base_vocab(source)



    # Gelbooru carries BOTH spellings of many tags, so splitting the vocabulary is
    # not enough on its own — anima mode would still suggest 'navel' alongside
    # 'bellybutton'. Where the bridge says gelbooru prefers a form, drop the
    # danbooru form from the suggestions so the offered word is the right one.
    # 'combined' KEEPS BOTH SPELLINGS. There is no mode selector any more:
    # one prompt is written and then rendered for both models, so the
    # autocomplete must not decide the booru on the user's behalf. Either
    # form can be typed and bridge.spell_for() writes each tab in the
    # spelling that model has actually seen more. (The old 'gelbooru'
    # source still drops the danbooru form, for callers that really do
    # want one booru's vocabulary.)
    if source == "gelbooru":
        bridge = (banks.get("_bridge") or {})
        for dan, gel in bridge.items():
            if dan in c and gel in c:
                del c[dan]

    return [t for t, _ in c.most_common(limit)]


def vocab_counts(source="danbooru"):
    """tag -> real booru post count, for display in the autocomplete.
    anima layers gelbooru's counts over danbooru's (same union as the
    vocabulary itself); a tag both boorus carry shows the gelbooru
    number, the booru anima is prompted in."""
    names = (["danbooru_tags.json", "gelbooru_tags.json"]
             if source in ("gelbooru", "combined") else ["danbooru_tags.json"])
    out = {}
    for name in names:
        p = _paths.data(name)
        if os.path.exists(p):
            with open(p, encoding="utf-8-sig") as f:
                for tag, n in json.load(f).items():
                    out[pe.normalize_tag(tag)] = n
    return out




# THE HINT FIELDS GET THEIR OWN VOCABULARIES (2026-09-16). The brief
# autocompletes against the whole booru vocabulary, which is the wrong list
# for the style and artist hints: a user typing there is guessing at what
# the studio actually knows. These two lists say it outright -- the style
# axis's own vocabulary, and every artist the studio can steer.
def build_style_vocab(counts):
    """what the style hint can use: the style names the pool measures, the
    cultural/studio looks, the mediums, and the palette and technique tags
    those styles are made of. Post counts come from the booru table where
    the name is a booru tag; a pool name carries its own measured count.

    ONE ROW PER CONCEPT (the author's 2026-09-16, "are watercolor (medium) /
    watercolor / watercolour just aliases?" -- they are). Two tables here
    hold human spellings rather than tags: medium_pool's 'typed_surfaces'
    maps how people type a medium onto its canonical tag, and 'rider' is a
    rule, not a name. Those spellings were being offered as separate
    suggestions, so one medium ate three of the twelve rows. The list now
    takes the CANONICAL side of that map, skips the rule block, and runs
    everything through danbooru's alias table: an alias with no posts of
    its own collapses into the tag it aliases. A spelling gelbooru really
    carries keeps its row -- it has posts, and the engine resolves it.
    """
    c = {}

    def put(name, n):
        name = str(name).strip().lower()
        if not name or name.startswith("_"):
            return
        c[name] = max(c.get(name, 0), int(n or 0), int(counts.get(name, 0) or 0))

    try:
        pool = json.load(open(_paths.data("style_pool.json"), encoding="utf-8-sig"))
        for name, rec in pool.items():
            if not isinstance(rec, dict):
                continue
            put(name, rec.get("posts"))
            for field in ("techniques", "palette", "mediums"):
                for t in (rec.get(field) or {}):
                    put(t, 0)
    except Exception:
        pass
    try:                                    # the medium axis, section by section
        med = json.load(open(_paths.data("medium_pool.json"), encoding="utf-8-sig"))
        for name, rec in (med.get("roll") or {}).items():
            put(name, (rec or {}).get("posts") if isinstance(rec, dict) else 0)
        for name, n in (med.get("map_only") or {}).items():
            put(name, n if isinstance(n, int) else 0)
        # typed_surfaces is {how people type it: the tag} -- the tag is the
        # concept, the key is a spelling of it
        for spelling, tag in (med.get("typed_surfaces") or {}).items():
            if isinstance(tag, str):
                put(tag, 0)
        # rider is a rule whose value lists tags ('implies_traditional')
        for rule, tags in (med.get("rider") or {}).items():
            for t in (tags or []) if isinstance(tags, list) else []:
                put(t, 0)
    except Exception:
        pass
    try:                                    # studio and cultural looks
        cul = json.load(open(_paths.data("cultural_styles.json"), encoding="utf-8-sig"))
        for name, rec in (cul.get("pool") or {}).items():
            put(name, 0)
            if isinstance(rec, dict):
                for field in ("techniques", "palette", "mediums"):
                    for t in (rec.get(field) or {}):
                        put(t, 0)
    except Exception:
        pass
    try:                                    # looks the checkpoints know, not booru
        from promptstudio.library import external as _ext
        for name, rec in (_ext.load() or {}).items():
            if (rec or {}).get("kind") in ("style", "medium", "genre", "fashion"):
                put(name, 0)
    except Exception:
        pass
    c = collapse_aliases(c, counts)
    names = sorted(c, key=lambda t: (-c[t], t))
    return names, [c[t] for t in names]


def build_look_vocab(have=()):
    """the style hint's second kind (2026-09-17): the looks the artists'
    descriptions use -- 'loose linework', 'muted palette' -- with how many
    artists each describes and the spice level it needs (looks.vocabulary:
    at least 10 artists, duplicates combined, age and lettering screened). A look the style list already
    offers stays a style row. The source follows the vision data: the new
    signature sweep's fields as they arrive, the first sweep's traits until
    then."""
    try:
        from promptstudio.library import looks as _looks
        voc = _looks.vocabulary()
    except Exception:
        return [], [], []
    names = sorted((k for k in voc if k not in have), key=lambda k: (-voc[k]["artists"], k))
    return names, [voc[k]["artists"] for k in names], [voc[k].get("level") or "safe" for k in names]


_ALIASES = None


def aliases():
    """danbooru's alias table: antecedent -> the tag it resolves to"""
    global _ALIASES
    if _ALIASES is None:
        try:
            from promptstudio.engine.aliases import booru_aliases
            _ALIASES = dict(booru_aliases())
        except Exception:
            _ALIASES = {}
    return _ALIASES


def collapse_aliases(c, counts):
    """-> the same {name: count} with dead aliases folded into their tag.

    A name is folded only when BOTH booru tables give it no posts of its
    own: then it is a spelling, not a tag, and suggesting it hides a real
    concept. 'bellybutton' (873k posts on gelbooru) is NOT a dead alias --
    gelbooru carries it and anima is prompted in gelbooru's spelling --
    which is why the brief's own vocabulary is left alone.
    """
    al, out = aliases(), {}
    for name, n in c.items():
        tgt = al.get(name)
        if tgt and not int(counts.get(name, 0) or 0):
            out[tgt] = max(out.get(tgt, 0), int(counts.get(tgt, 0) or 0), int(n or 0))
        else:
            out[name] = max(out.get(name, 0), int(n or 0))
    return out


def build_artist_vocab():
    """every artist the studio can steer: the draw pool at its 100-post
    floor (already free of denied and unsteerable names), plus the artists
    the checkpoints know that booru never tagged. Below the floor there is
    no measured grounding, so those names are not suggested -- they can
    still be typed with '@', which bypasses the floor.

    One row per artist here too: danbooru's alias table folds a name with
    no posts of its own into the tag it aliases ('artgerm' is a name the
    checkpoints know, and danbooru's word for that artist is 'stanley
    lau', which the pool measures at 1,710 posts).
    """
    c = {}
    try:
        pool, draw = pe._artist_pool()
        for n in draw:
            c[str(n).lower()] = int(pool.get(n) or 0)
    except Exception:
        pass
    try:
        from promptstudio.library import external as _ext
        for name, rec in (_ext.load() or {}).items():
            if (rec or {}).get("kind") == "artist":
                c.setdefault(str(name).lower(), 0)
    except Exception:
        pass
    c = collapse_aliases(c, c)
    names = sorted(c, key=lambda n: (-c[n], n))
    return names, [c[n] for n in names]


# ONE GENERATION AT A TIME. llm_bridge drives a single llama-server;
# two prompts racing through it interleave completions and double latency.
GEN_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    banks = None
    vocab = []
    vocab_style, counts_style, kinds_style, nsfw_style = [], [], [], []
    vocab_artist, counts_artist = [], []

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html")
        elif self.path == "/api/status":
            st = dict(lb.STATUS)
            st["elapsed"] = round(time.time() - float(st.get("t0") or time.time()), 1)
            self._send(200, json.dumps(st))
            return
        elif self.path == "/api/info":
            H = Handler
            try:
                genres = lb.genre_menu()
            except Exception:
                genres = []
            self._send(200, json.dumps({
                "llm": llm_status(),
                "genres": genres,
                "install": INSTALL_ID,
            }))
        elif self.path.startswith("/api/vocab_style"):
            self._send(200, json.dumps({"tags": Handler.vocab_style,
                                        "counts": Handler.counts_style,
                                        "kinds": Handler.kinds_style,
                                        "nsfw": Handler.nsfw_style,
                                        "source": "styles, mediums, techniques; looks from the artists' descriptions"}))
        elif self.path.startswith("/api/vocab_artist"):
            self._send(200, json.dumps({"tags": Handler.vocab_artist,
                                        "counts": Handler.counts_artist,
                                        "source": "artists"}))
        elif self.path.startswith("/api/vocab"):
            # ONE VOCABULARY, BOTH BOORUS. The prompt is written once and
            # rendered for both models, so the suggestions are the union;
            # the per-model spelling is settled at render time.
            tags = Handler.vocab_all
            counts = Handler.counts_all
            self._send(200, json.dumps({
                "tags": tags,
                "counts": [counts.get(t, 0) for t in tags],
                "source": "danbooru + gelbooru"}))
        else:
            self._send(404, "{}")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        try:
            req = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            if not isinstance(req, dict):
                raise ValueError("not an object")
        except Exception as e:
            # a malformed body is answered, not dropped (2026-09-16)
            self._send(400, json.dumps({"error": "bad request: %s" % e}))
            return
        if self.path == "/api/invent":
            # "LET THE LLM INVENT IT": an idea for the brief, written by the
            # model from the controls alone. The brief is never sent, so
            # pressing again cannot hand back the idea it just wrote.
            if not llm_status():
                self._send(200, json.dumps(
                    {"error": "no language model: see README, "
                              "'Setting up the language model'"}))
                return
            try:
                with GEN_LOCK:
                    out = lb.invent_brief(
                        {"spice": req.get("spice"), "genre": req.get("genre"),
                         "period": req.get("period"),
                         "exclude": req.get("exclude")},
                        seed=None, avoid=req.get("avoid") or [])
            except Exception as e:
                self._send(200, json.dumps({"error": repr(e)[:300]}))
                return
            self._send(200, json.dumps(out))
            return
        if self.path == "/api/llm_release":
            # the button posted here since it was drawn; the handler never
            # served it (every click got a 404 and freed nothing, 2026-09-05)
            released = False
            try:
                from promptstudio.llm import server as llm_server
                llm_server.stop()
                llm_server.stop_embed()
                released = True
            except Exception as e:
                self._send(200, json.dumps({"released": False, "error": repr(e)[:200]}))
                return
            self._send(200, json.dumps({"released": released}))
            return
        if self.path == "/api/status":
            st = dict(lb.STATUS)
            st["elapsed"] = round(time.time() - float(st.get("t0") or time.time()), 1)
            self._send(200, json.dumps(st))
            return
        if self.path == "/api/cancel":
            # the flag stops the writer before its next model call and the
            # loop before its next variant; finished variants come back
            lb.CANCEL.set()
            self._send(200, json.dumps({"cancelling": True}))
            return
        if self.path != "/api/generate":
            self._send(404, "{}")
            return
        base = (req.get("base") or "").strip()   # empty = full creative mode
        # ANIMA IS THE CANONICAL RESOLUTION MODE. The selector is gone: one
        # scene is resolved and then packaged for both models (see the
        # dual-tab block in bridge.generate). Anima has to be the one that
        # decides, because the cast distributions are measured per booru
        # and the two models genuinely disagree about who is in the shot --
        # resolving as illustrious would change the picture, not the
        # wording. `mode` is still read so an older client keeps working.
        mode = req.get("mode", "anima")
        seed_raw = str(req.get("seed") or "").strip()
        seed = int(seed_raw) if seed_raw else random.randrange(1 << 30)
        opts = {
            "quality": req.get("quality", "standard"),
            # the genre dropdown: 'random' = the dice; a leaf drives the prompt
            "genre": req.get("genre") or "random",
            "period": req.get("period") or None,
            # genre / style / location / medium / scene are no longer
            # controls: each is resolved automatically (typed wins, else
            # generated with a measured nudge), so nothing is passed for
            # them. resolve_medium still honours an explicit False, which
            # is why it is simply absent rather than set.
            "appearance": req.get("appearance", "automatic"),
            "detail": req.get("detail", "standard"),
            "gen_characters": bool(req.get("gen_characters")),
            "gen_artists": bool(req.get("gen_artists")),
            # the no-LLM path: same pipeline, mechanical plan and picks
            "fast": bool(req.get("fast")),
            # what should not appear: the negative line, the plan prune,
            # the final tag gate (typed words still win)
            "exclude": req.get("exclude") or "",
        }
        # the artist hint rides on the brief; the style hint is read entry by
        # entry by the engine (2026-09-17: whole looks, not words for the scanners)
        if req.get("style_hint") and str(req.get("style_hint")).strip():
            opts["style_hint"] = str(req.get("style_hint")).strip()
        _h = req.get("artist_hint")
        if _h and str(_h).strip():
            base = (base + ", " if base.strip() else "") + str(_h).strip()
        results = []
        lb.CANCEL.clear()
        # THE LLM ENGINE WITHOUT A MODEL FALLS BACK TO THE FAST ENGINE
        # (2026-09-16, for the release): a user who picks 'llm' with no
        # llama.cpp or no model gets a prompt from the fast engine and a
        # note saying so, not an error card
        engine_note = None
        if not opts["fast"] and not llm_status():
            opts["fast"] = True
            engine_note = ("no language model available (see README, 'Setting up the "
                           "language model'): the fast engine answered")
        for i in range(max(1, min(10, int(req.get("variants", 1))))):
            if lb.CANCEL.is_set():
                results.append({"error": "cancelled", "seed": seed + i})
                break
            try:
                try:
                    with GEN_LOCK:
                        r = lb.generate(base, mode, req.get("spice", "auto"),
                                        dict(opts), seed + i)
                except RuntimeError as _e:
                    if opts["fast"] or "no LLM engine" not in str(_e):
                        raise
                    # the server died between the status check and the call
                    opts["fast"] = True
                    engine_note = "the language model could not start (see logs/llm_server.log): the fast engine answered"
                    with GEN_LOCK:
                        r = lb.generate(base, mode, req.get("spice", "auto"),
                                        dict(opts), seed + i)
                results.append({
                    "engine_note": engine_note,
                    "prompt": r["prompt"], "negative": r.get("negative"),
                    # both tabs: same scene, one packaging each
                    "prompts": r.get("prompts"),
                    "negatives": r.get("negatives"),
                    "seed": seed + i, "cast": r.get("cast"),
                    "why": r.get("why"), "refused": r.get("refused"),
                    "llm_calls": r.get("llm_calls"),
                    "coverage": r.get("coverage"),
                    "ledger_new": r.get("ledger_new"),
                    "dropped_unknown": r.get("dropped_unknown"),
                    "level": r.get("level"), "switched": r.get("switched"),
                    "unmapped": r.get("unmapped"),
                    "injected_for_level": r.get("injected_for_level"),
                    "stripped_by_verifier": r.get("stripped_by_verifier"),
                })
            except Exception as e:
                results.append({"error": f"{type(e).__name__}: {e}",
                                "seed": seed + i})
        self._send(200, json.dumps({"results": results}))


def main():
    Handler.banks = pe.load_all_banks()
    Handler.vocab = build_vocab(Handler.banks, source="danbooru")
    Handler.vocab_gel = build_vocab(Handler.banks, source="gelbooru")
    Handler.vocab_all = build_vocab(Handler.banks, source="combined")
    Handler.counts = vocab_counts("danbooru")
    Handler.counts_gel = vocab_counts("gelbooru")
    Handler.counts_all = vocab_counts("combined")
    Handler.vocab_style, Handler.counts_style = build_style_vocab(Handler.counts_all)
    Handler.kinds_style = ["style"] * len(Handler.vocab_style)
    Handler.nsfw_style = ["safe"] * len(Handler.vocab_style)
    _lv, _lc, _ln = build_look_vocab(set(Handler.vocab_style))
    Handler.vocab_style += _lv
    Handler.counts_style += _lc
    Handler.kinds_style += ["look"] * len(_lv)
    Handler.nsfw_style += _ln
    Handler.vocab_artist, Handler.counts_artist = build_artist_vocab()
    print(f"hint vocabularies: {len(Handler.vocab_style)} style words and looks, "
          f"{len(Handler.vocab_artist)} artists")
    print(f"autocomplete vocabulary: {len(Handler.vocab_all)} combined "
          f"(danbooru + gelbooru, either spelling typeable)")
    net = Handler.banks.get("_net")
    print(f"tag network: {len(net) if net else 0} rows, "
          f"{len(Handler.banks.get('_aliases') or {})} aliases")
    # a data file the code asked for and could not find is a silent
    # degradation everywhere downstream — say so at startup, once
    if _paths.MISSES:
        print("MISSING DATA FILES (loaders fell back to empty): "
              + ", ".join(sorted(_paths.MISSES)))
    # the engine is ours: llama-server, run from LLM/llama.cpp inside the
    # project, on the model named in llm_config.json.
    print("engine: " + {
        "own": "our llama-server is up",
        "own-idle": "our llama-server is up (idle, unloads after 10 min)",
        "ready-to-start": "built-in, auto-starts on first generate (~20s)",
    }.get(llm_status(), "NONE — no backend exe or model found; see paths.models_dir()"))
    # which model, so a config change is visible without reading the file
    try:
        from promptstudio.llm import config as _llm_config
        print("        " + _llm_config.describe())
    except Exception:
        pass
    # WARM-UP (the author's 2026-09-05: "the first generate takes 40s even in
    # no-llm mode"): the first call builds the engine's indexes (vocabulary
    # word index, retrieval, tag network rows, the tables). Build them
    # now, off the request thread, so the first click pays nothing.
    def _warm():
        try:
            with GEN_LOCK:
                lb.generate("a girl in a park", "anima", "safe",
                            {"fast": True, "quality": "standard", "appearance": "automatic",
                             "detail": "standard"}, seed=0)
            print("engine warmed (indexes built)")
        except Exception as e:
            print("warm-up skipped: %r" % (e,))
    threading.Thread(target=_warm, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"Bismuth Prompt Studio (v2) -> {url}   (Ctrl+C to stop)")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
