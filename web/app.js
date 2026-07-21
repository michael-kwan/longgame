(() => {
"use strict";

const B = JSON.parse(document.getElementById("bundle").textContent);
const { config, vocab, shapes, members, role_playrates: PLAY, stats, roster } = B;
const DIM = config.dim, NB = config.n_bins;
const CHAMPS = vocab.champions;          // champion i occupies model index i+1
const ROLES = vocab.roles, ROLE_UNK = ROLES.length;
const TIERS = vocab.tiers, QUEUES = vocab.queues;
const CENTERS = vocab.bin_centers, EDGES = vocab.edges_min;

/* ---------- weights (numeric core lives in infer.js) ---------- */

const ENS = LG.createEnsemble(B);
const predict = (st) => ENS.predict(st);

const pAbove = (probs, minutes) => {
  let s = 0;
  for (let i = 0; i < NB; i++) if (CENTERS[i] > minutes) s += probs[i];
  return s;
};
const pBelow = (probs, minutes) => {
  let s = 0;
  for (let i = 0; i < NB; i++) if (CENTERS[i] < minutes) s += probs[i];
  return s;
};

/* ---------- draft state ---------- */

const S = {
  bans: Array(10).fill(null),
  ally: ROLES.map((_, i) => ({ role: i, champ: null })),
  enemy: ROLES.map((_, i) => ({ role: i, champ: null })),
  tier: Math.max(0, TIERS.indexOf("EMERALD")),
  queue: Math.max(0, QUEUES.indexOf("FLEXRANKED")),
  onclock: 0,
  lambda: 0.5,
  support: 30,
  poolOnly: false,
  log: [],   // ordered lock events, so the trajectory is real history
};

function stateFrom(bans, ally, enemy) {
  const allyC = [], allyR = [], enemyC = [], enemyR = [];
  for (const s of ally) if (s.champ) { allyC.push(s.champ); allyR.push(s.role); }
  // Enemy roles are genuinely unknown during a draft (DESIGN.md §7).
  for (const s of enemy) if (s.champ) { enemyC.push(s.champ); enemyR.push(ROLE_UNK); }
  return { allyC, allyR, enemyC, enemyR, bans: bans.filter(Boolean), tier: S.tier, queue: S.queue };
}

const currentState = () => stateFrom(S.bans, S.ally, S.enemy);

function stateAfter(k) {
  const bans = [], ally = ROLES.map((_, i) => ({ role: i, champ: null })),
        enemy = ROLES.map((_, i) => ({ role: i, champ: null }));
  for (const e of S.log.slice(0, k)) {
    if (e.type === "ban") bans.push(e.champ);
    else if (e.type === "ally") ally[e.slot].champ = e.champ;
    else enemy[e.slot].champ = e.champ;
  }
  return stateFrom(bans, ally, enemy);
}

const taken = () => new Set([
  ...S.bans.filter(Boolean),
  ...S.ally.map((s) => s.champ).filter(Boolean),
  ...S.enemy.map((s) => s.champ).filter(Boolean),
]);

function logSet(type, slot, champ) {
  S.log = S.log.filter((e) => !(e.type === type && e.slot === slot));
  if (champ) S.log.push({ type, slot, champ });
}

/* ---------- roster ---------- */

const rosterByRole = {};
if (roster && roster.players) {
  for (const p of roster.players) {
    const r = p.main_role && ROLES.indexOf(p.main_role);
    if (r != null && r >= 0 && !rosterByRole[r]) rosterByRole[r] = p;
  }
}
const poolFor = (role) => {
  const p = rosterByRole[role];
  if (!p) return null;
  const idx = new Set();
  for (const e of p.pool) {
    const i = CHAMPS.indexOf(e.champion);
    if (i >= 0) idx.add(i + 1);
  }
  return idx;
};

/* ---------- recommendations ---------- */

function recommend() {
  const role = S.onclock;
  const used = taken();
  const pool = S.poolOnly ? poolFor(role) : null;
  const base = predict(currentState());
  const out = [];

  for (let ci = 1; ci <= CHAMPS.length; ci++) {
    if (used.has(ci)) continue;
    if ((PLAY[ci] ? PLAY[ci][role] : 0) < S.support) continue;   // support constraint
    if (pool && !pool.has(ci)) continue;                          // roster pool
    const ally = S.ally.map((s) => ({ ...s }));
    const slot = ally.findIndex((s) => s.role === role && !s.champ);
    if (slot < 0) continue;
    ally[slot].champ = ci;
    const p = predict(stateFrom(S.bans, ally, S.enemy));
    out.push({
      ci,
      name: CHAMPS[ci - 1],
      mean: p.mean,
      std: p.std,
      score: p.mean - S.lambda * p.std,     // pessimism (DESIGN.md §6)
      delta: p.mean - base.mean,
      games: PLAY[ci][role],
    });
  }
  out.sort((a, b) => b.score - a.score);
  return { base, list: out };
}

/* ---------- charts ---------- */

const tip = document.getElementById("tip");
function showTip(evt, html) {
  tip.innerHTML = html;
  tip.style.opacity = "1";
  const pad = 14;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  const r = tip.getBoundingClientRect();
  if (x + r.width > innerWidth - 8) x = evt.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = evt.clientY - r.height - pad;
  tip.style.left = x + "px";
  tip.style.top = y + "px";
}
const hideTip = () => { tip.style.opacity = "0"; };

const SVG = "http://www.w3.org/2000/svg";
const el = (n, attrs = {}) => {
  const e = document.createElementNS(SVG, n);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
};

function barPath(x, y, w, h, r) {
  r = Math.min(r, w / 2, Math.max(h, 0));
  if (h <= 0.5) return `M${x},${y + h} L${x + w},${y + h}`;
  return `M${x},${y + h} L${x},${y + r} Q${x},${y} ${x + r},${y} ` +
         `L${x + w - r},${y} Q${x + w},${y} ${x + w},${y + r} L${x + w},${y + h} Z`;
}

const binLabel = (i) =>
  i === 0 ? `< ${EDGES[0]}` : i === NB - 1 ? `${EDGES[EDGES.length - 1]}+` : `${EDGES[i - 1]}–${EDGES[i]}`;

function drawDistribution(probs, prev) {
  const svg = document.getElementById("dist");
  svg.textContent = "";
  const W = 520, H = 190, ml = 36, mr = 6, mt = 10, mb = 30;
  const pw = W - ml - mr, ph = H - mt - mb;
  let max = 0;
  for (let i = 0; i < NB; i++) max = Math.max(max, probs[i], prev ? prev[i] : 0);
  max = Math.max(max * 1.15, 0.05);
  const bw = pw / NB;
  const y = (v) => mt + ph - (v / max) * ph;

  for (let g = 0; g <= 4; g++) {
    const v = (max * g) / 4;
    svg.appendChild(el("line", { class: "gridline", x1: ml, x2: W - mr, y1: y(v), y2: y(v) }));
    const t = el("text", { class: "tick", x: ml - 6, y: y(v) + 3, "text-anchor": "end" });
    t.textContent = Math.round(v * 100) + "%";
    svg.appendChild(t);
  }

  if (prev) {
    let d = "";
    for (let i = 0; i < NB; i++) {
      const x0 = ml + i * bw, x1 = x0 + bw, yy = y(prev[i]);
      d += (i === 0 ? `M${x0},${yy}` : ` L${x0},${yy}`) + ` L${x1},${yy}`;
    }
    svg.appendChild(el("path", { class: "prev-outline", d }));
  }

  for (let i = 0; i < NB; i++) {
    const x = ml + i * bw + 1, w = bw - 2, yy = y(probs[i]);
    svg.appendChild(el("path", { class: "bar", d: barPath(x, yy, w, mt + ph - yy, 4) }));
    const hit = el("rect", { class: "hit", x: ml + i * bw, y: mt, width: bw, height: ph });
    hit.addEventListener("mousemove", (e) =>
      showTip(e, `<b>${binLabel(i)} min</b><br><span class="k">now</span> ${(probs[i] * 100).toFixed(1)}%` +
        (prev ? `<br><span class="k">before</span> ${(prev[i] * 100).toFixed(1)}%` : "")));
    hit.addEventListener("mouseleave", hideTip);
    svg.appendChild(hit);
  }

  svg.appendChild(el("line", { class: "axisline", x1: ml, x2: W - mr, y1: mt + ph, y2: mt + ph }));
  for (let i = 0; i < NB; i += 3) {
    const t = el("text", { class: "tick", x: ml + i * bw + bw / 2, y: H - mb + 14, "text-anchor": "middle" });
    t.textContent = i === 0 ? `<${EDGES[0]}` : String(EDGES[i - 1]);
    svg.appendChild(t);
  }
  const lab = el("text", { class: "tick", x: ml + pw / 2, y: H - 2, "text-anchor": "middle" });
  lab.textContent = "game length (minutes)";
  svg.appendChild(lab);
}

function drawTrajectory(points) {
  const svg = document.getElementById("traj");
  svg.textContent = "";
  const W = 520, H = 150, ml = 36, mr = 10, mt = 12, mb = 30;
  const pw = W - ml - mr, ph = H - mt - mb;
  if (points.length < 2) {
    const t = el("text", { class: "tick", x: W / 2, y: H / 2, "text-anchor": "middle" });
    t.textContent = "Lock in a ban or a pick to start the trajectory";
    svg.appendChild(t);
    return;
  }
  let lo = Infinity, hi = -Infinity;
  for (const p of points) { lo = Math.min(lo, p.mean); hi = Math.max(hi, p.mean); }
  const pad = Math.max((hi - lo) * 0.35, 0.4);
  lo -= pad; hi += pad;
  const x = (i) => ml + (points.length === 1 ? pw / 2 : (i / (points.length - 1)) * pw);
  const y = (v) => mt + ph - ((v - lo) / (hi - lo)) * ph;

  for (let g = 0; g <= 3; g++) {
    const v = lo + ((hi - lo) * g) / 3;
    svg.appendChild(el("line", { class: "gridline", x1: ml, x2: W - mr, y1: y(v), y2: y(v) }));
    const t = el("text", { class: "tick", x: ml - 6, y: y(v) + 3, "text-anchor": "end" });
    t.textContent = v.toFixed(1);
    svg.appendChild(t);
  }

  let d = "";
  points.forEach((p, i) => { d += (i ? " L" : "M") + x(i) + "," + y(p.mean); });
  svg.appendChild(el("path", { class: "traj-line", d }));
  points.forEach((p, i) => {
    svg.appendChild(el("circle", { class: "traj-dot", cx: x(i), cy: y(p.mean), r: 4 }));
  });

  const cross = el("line", { class: "crosshair", y1: mt, y2: mt + ph, x1: 0, x2: 0, opacity: 0 });
  svg.appendChild(cross);
  const hit = el("rect", { class: "hit", x: ml, y: mt, width: pw, height: ph });
  hit.addEventListener("mousemove", (e) => {
    const box = svg.getBoundingClientRect();
    const rel = ((e.clientX - box.left) / box.width) * W;
    let best = 0;
    points.forEach((_, i) => { if (Math.abs(x(i) - rel) < Math.abs(x(best) - rel)) best = i; });
    cross.setAttribute("x1", x(best)); cross.setAttribute("x2", x(best));
    cross.setAttribute("opacity", 1);
    const p = points[best];
    showTip(e, `<b>${p.label}</b><br><span class="k">E[duration]</span> ${p.mean.toFixed(2)} min` +
      (best ? `<br><span class="k">change</span> ${fmtDelta(p.mean - points[best - 1].mean)}` : ""));
  });
  hit.addEventListener("mouseleave", () => { hideTip(); cross.setAttribute("opacity", 0); });
  svg.appendChild(hit);

  points.forEach((p, i) => {
    if (i % Math.ceil(points.length / 6) && i !== points.length - 1) return;
    const t = el("text", { class: "tick", x: x(i), y: H - mb + 16, "text-anchor": "middle" });
    t.textContent = p.short;
    svg.appendChild(t);
  });
}

const fmtDelta = (d) =>
  (d >= 0 ? "+" : "") + d.toFixed(2) + " min";

/* ---------- render ---------- */

function renderRecs(base, list) {
  const box = document.getElementById("recs");
  if (!list.length) {
    box.innerHTML = `<p class="note">No legal candidates. Lower “min games in role”, turn off the
      roster pool, or pick a role that still has an open slot.</p>`;
    return;
  }
  const maxAbs = Math.max(...list.slice(0, 12).map((r) => Math.abs(r.delta)), 0.05);
  const rows = list.slice(0, 12).map((r, i) => {
    const w = (Math.abs(r.delta) / maxAbs) * 50;
    const pos = r.delta >= 0;
    const fill = pos
      ? `left:50%;width:${w}%;background:var(--series-1)`
      : `right:50%;width:${w}%;background:var(--neg)`;
    return `<tr>
      <td class="rank">${i + 1}</td>
      <td>${r.name}</td>
      <td class="num">${r.mean.toFixed(2)}</td>
      <td class="delta-cell"><div class="delta-bar"><div class="axis"></div>
        <div class="fill" style="${fill}"></div></div></td>
      <td class="num ${pos ? "up" : "down"}">${fmtDelta(r.delta)}</td>
      <td class="num spread">±${r.std.toFixed(2)}</td>
      <td class="num">${r.score.toFixed(2)}</td>
      <td class="num spread">${r.games}</td>
    </tr>`;
  }).join("");
  box.innerHTML = `<table>
    <thead><tr><th></th><th>Champion</th><th class="num">E[dur]</th><th></th>
      <th class="num">Δ</th><th class="num">spread</th><th class="num">score</th>
      <th class="num">games</th></tr></thead>
    <tbody>${rows}</tbody></table>
    <p class="note">Ranked by <b>E[duration] − ${S.lambda.toFixed(1)}×spread</b>. “Spread” is
    ensemble disagreement: high spread means the model is extrapolating, so pessimism pushes it down.
    “Games” is how often that champion was played in this role in the training data.</p>`;
}

function refresh() {
  const { base, list } = recommend();
  renderRecs(base, list);

  const prev = S.log.length ? predict(stateAfter(S.log.length - 1)).probs : null;
  drawDistribution(base.probs, prev);

  document.getElementById("heroMean").innerHTML =
    base.mean.toFixed(1) + '<span class="unit"> min</span>';
  document.getElementById("heroLong").innerHTML =
    (pAbove(base.probs, 35) * 100).toFixed(0) + '<span class="unit">%</span>';
  document.getElementById("heroShort").innerHTML =
    (pBelow(base.probs, 20) * 100).toFixed(0) + '<span class="unit">%</span>';

  const rows = Array.from({ length: NB }, (_, i) =>
    `<tr><td>${binLabel(i)} min</td><td class="num">${(base.probs[i] * 100).toFixed(1)}%</td></tr>`).join("");
  document.getElementById("distTable").innerHTML =
    `<table><thead><tr><th>Bucket</th><th class="num">Probability</th></tr></thead><tbody>${rows}</tbody></table>`;

  const pts = [{ label: "Empty draft", short: "start", mean: predict(stateAfter(0)).mean }];
  S.log.forEach((e, i) => {
    const who = e.type === "ban" ? "Ban" : e.type === "ally" ? "My " + ROLES[e.slot] : "Enemy";
    pts.push({
      label: `${who}: ${CHAMPS[e.champ - 1]}`,
      short: e.type === "ban" ? "ban" : e.type === "ally" ? ROLES[e.slot].slice(0, 3).toLowerCase() : "enemy",
      mean: predict(stateAfter(i + 1)).mean,
    });
  });
  drawTrajectory(pts);

  const total = pts.length > 1 ? pts[pts.length - 1].mean - pts[0].mean : 0;
  document.getElementById("trajNote").textContent =
    pts.length > 1
      ? `${S.log.length} champion${S.log.length === 1 ? "" : "s"} locked · net change ${fmtDelta(total)} versus an empty draft · ally win probability ${(base.win * 100).toFixed(0)}%`
      : "Each locked champion updates the estimate.";

  document.querySelectorAll(".slot").forEach((n) => n.classList.remove("oncall"));
  const open = S.ally.findIndex((s) => s.role === S.onclock && !s.champ);
  if (open >= 0) document.querySelector(`.slot[data-side="ally"][data-slot="${open}"]`)?.classList.add("oncall");

  document.getElementById("boardNote").textContent =
    `${S.bans.filter(Boolean).length} bans · ${S.ally.filter((s) => s.champ).length}/5 ally · ` +
    `${S.enemy.filter((s) => s.champ).length}/5 enemy locked. Enemy roles are left unknown to the model, as they are in a real draft.`;
}

/* ---------- inputs ---------- */

const nameToIdx = new Map(CHAMPS.map((c, i) => [c.toLowerCase(), i + 1]));

function champInput(onChange, placeholder) {
  const inp = document.createElement("input");
  inp.type = "text";
  inp.setAttribute("list", "champlist");
  inp.placeholder = placeholder;
  inp.autocomplete = "off";
  const commit = () => {
    const v = inp.value.trim().toLowerCase();
    if (!v) { onChange(null); inp.value = ""; return; }
    const idx = nameToIdx.get(v);
    if (idx) { onChange(idx); inp.value = CHAMPS[idx - 1]; }
    else { inp.value = ""; onChange(null); }
    refresh();
  };
  inp.addEventListener("change", commit);
  inp.addEventListener("blur", commit);
  return inp;
}

function build() {
  const dl = document.createElement("datalist");
  dl.id = "champlist";
  for (const c of CHAMPS) {
    const o = document.createElement("option");
    o.value = c;
    dl.appendChild(o);
  }
  document.body.appendChild(dl);

  const bansBox = document.getElementById("bans");
  for (let i = 0; i < 10; i++) {
    bansBox.appendChild(champInput((idx) => { S.bans[i] = idx; logSet("ban", i, idx); }, `ban ${i + 1}`));
  }

  for (const side of ["ally", "enemy"]) {
    const box = document.getElementById(side === "ally" ? "allySlots" : "enemySlots");
    S[side].forEach((slot, i) => {
      const row = document.createElement("div");
      row.className = "slot";
      row.dataset.side = side;
      row.dataset.slot = i;
      const lab = document.createElement("span");
      lab.className = "role";
      lab.textContent = ROLES[slot.role];
      row.appendChild(lab);
      row.appendChild(champInput((idx) => { slot.champ = idx; logSet(side, i, idx); }, "—"));
      box.appendChild(row);
    });
  }

  const tierSel = document.getElementById("tier");
  TIERS.forEach((t, i) => {
    const o = document.createElement("option");
    o.value = i; o.textContent = t[0] + t.slice(1).toLowerCase();
    if (i === S.tier) o.selected = true;
    tierSel.appendChild(o);
  });
  tierSel.addEventListener("change", () => { S.tier = +tierSel.value; refresh(); });

  const qSel = document.getElementById("queue");
  QUEUES.forEach((q, i) => {
    const o = document.createElement("option");
    o.value = i; o.textContent = q === "SOLORANKED" ? "Ranked solo/duo" : "Ranked flex";
    if (i === S.queue) o.selected = true;
    qSel.appendChild(o);
  });
  qSel.addEventListener("change", () => { S.queue = +qSel.value; refresh(); });

  const clock = document.getElementById("onclock");
  ROLES.forEach((r, i) => {
    const o = document.createElement("option");
    o.value = i;
    o.textContent = r[0] + r.slice(1).toLowerCase() + (rosterByRole[i] ? ` — ${rosterByRole[i].riot_id}` : "");
    clock.appendChild(o);
  });
  clock.addEventListener("change", () => { S.onclock = +clock.value; refresh(); });

  const lam = document.getElementById("lambda");
  const lamVal = document.getElementById("lambdaVal");
  lamVal.textContent = S.lambda.toFixed(1);
  lam.addEventListener("input", () => {
    S.lambda = +lam.value; lamVal.textContent = S.lambda.toFixed(1); refresh();
  });

  const sup = document.getElementById("support");
  const supVal = document.getElementById("supportVal");
  supVal.textContent = S.support;
  sup.addEventListener("input", () => {
    S.support = +sup.value; supVal.textContent = S.support; refresh();
  });

  const poolBtn = document.getElementById("poolonly");
  if (!roster || !roster.players || !roster.players.length) {
    poolBtn.disabled = true;
    poolBtn.textContent = "No roster loaded";
  } else {
    poolBtn.addEventListener("click", () => {
      S.poolOnly = !S.poolOnly;
      poolBtn.textContent = "Roster pool: " + (S.poolOnly ? "on" : "off");
      poolBtn.classList.toggle("primary", S.poolOnly);
      refresh();
    });
  }

  document.getElementById("reset").addEventListener("click", () => {
    S.bans.fill(null);
    S.ally.forEach((s) => (s.champ = null));
    S.enemy.forEach((s) => (s.champ = null));
    S.log = [];
    document.querySelectorAll('input[list="champlist"]').forEach((i) => (i.value = ""));
    refresh();
  });

  const chips = document.getElementById("chips");
  const r2 = stats.summary.r2, mae = stats.summary.mae;
  [
    `${stats.n_matches.toLocaleString()} matches`,
    `${ENS.nets.length}-model ensemble`,
    `R² ${r2 >= 0 ? "+" : ""}${r2.toFixed(3)}`,
    `MAE ${mae.toFixed(2)} min`,
  ].forEach((t) => {
    const c = document.createElement("span");
    c.className = "chip";
    c.textContent = t;
    chips.appendChild(c);
  });

  const rb = document.getElementById("rosterBox");
  if (roster && roster.players && roster.players.length) {
    rb.innerHTML = "<p class='note' style='margin-bottom:4px'>Roster loaded from OP.GG:</p>" +
      roster.players.map((p) =>
        `<p class="roster-line"><b>${p.riot_id}</b> · ${p.tier || "unranked"} · ${p.main_role || "?"} ·
         avg ${p.avg_minutes ?? "?"} min · pool ${p.pool.length}
         <span class="spread">(${p.pool.slice(0, 5).map((c) => c.champion).join(", ")})</span></p>`
      ).join("");
  }

  const phaseNote = () => {
    const ph = stats.phases || {};
    const keys = Object.keys(ph).sort((a, b) => a - b);
    if (!keys.length) return "";
    const cells = keys.map((k) =>
      `<td class="num">${ph[k].r2 >= 0 ? "+" : ""}${ph[k].r2.toFixed(4)}</td>`).join("");
    const heads = keys.map((k) => `<th class="num">${k}</th>`).join("");
    const late = ph[keys[keys.length - 1]].r2, early = ph[keys[0]].r2;
    return `<br><br><b>Where the signal actually is.</b> Held-out R² by how many champions are
      locked in:<table style="max-width:420px;margin-top:6px">
      <thead><tr><th>picks revealed</th>${heads}</tr></thead>
      <tbody><tr><td>R²</td>${cells}</tr></tbody></table>
      <span class="note">On an empty board the model is ${early < 0 ? "<b>worse than</b>" : "no better than"}
      simply predicting the average game — early recommendations are close to noise. It only earns
      its keep late in the draft (R² ${late >= 0 ? "+" : ""}${late.toFixed(4)} at a full board), which is
      also where the candidate spread widens to about a minute.</span>`;
  };

  const base = stats.baseline;
  document.getElementById("about").innerHTML =
    `Trained on <b>${stats.n_matches.toLocaleString()}</b> ranked games crawled from OP.GG
     (${stats.trained_at}). The model predicts a <i>distribution</i> over game length from the
     partial draft, then each candidate is scored by expected duration minus ensemble
     disagreement. Held-out MAE is <b>${mae.toFixed(2)} min</b> against a
     <b>${base.mean_mae.toFixed(2)} min</b> predict-the-mean baseline; R² is
     <b>${r2 >= 0 ? "+" : ""}${r2.toFixed(3)}</b>.
     <br><br>Draft explains only a small slice of why games run long — most of it is how the game is
     actually played. Treat a Δ of a minute or two as the real ceiling here, and note that the
     spread column is often larger than the gap between the top few candidates.
     ${phaseNote()}`;

  refresh();
}

build();
})();
