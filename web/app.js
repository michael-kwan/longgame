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

/* ---------- objectives ----------
   Measured on this dataset: champion identity + tier + queue + player history
   explain R^2 ~= 0.021 of duration in minutes, but give AUC 0.60 for
   "will this run past 35 minutes". The tail question is the better-posed one,
   so it is selectable — and it reads off the same distribution head. */

/* Minimum games a champion must have in a role to be recommended — the
   data-support guard of DESIGN.md §6. Fixed rather than exposed: dropping it to
   zero lets the model recommend champions it has never seen in that role. */
const SUPPORT_FLOOR = 30;

const OBJECTIVES = {
  mean:  { label: "E[duration]",   unit: "min",
           of: (probs) => { let s = 0; for (let i = 0; i < NB; i++) s += probs[i] * CENTERS[i]; return s; },
           fmt: (v) => v.toFixed(2), fmtDelta: (d) => (d >= 0 ? "+" : "") + d.toFixed(2) + " min" },
  p35:   { label: "P(> 35 min)",   unit: "%",
           of: (probs) => pAbove(probs, 35),
           fmt: (v) => (v * 100).toFixed(1), fmtDelta: (d) => (d >= 0 ? "+" : "") + (d * 100).toFixed(2) + " pp" },
  p40:   { label: "P(> 40 min)",   unit: "%",
           of: (probs) => pAbove(probs, 40),
           fmt: (v) => (v * 100).toFixed(1), fmtDelta: (d) => (d >= 0 ? "+" : "") + (d * 100).toFixed(2) + " pp" },
};

/** Objective value plus ensemble disagreement, both in the objective's own units. */
function score(pred) {
  const obj = OBJECTIVES[S.objective];
  const vals = pred.memberProbs.map(obj.of);
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  const varr = vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length;
  return { value: mean, spread: Math.sqrt(varr) };
}

/* ---------- draft state ---------- */

const S = {
  bans: Array(10).fill(null),
  ally: ROLES.map((_, i) => ({ role: i, champ: null })),
  enemy: ROLES.map((_, i) => ({ role: i, champ: null })),
  tier: Math.max(0, TIERS.indexOf("EMERALD")),
  queue: Math.max(0, QUEUES.indexOf("FLEXRANKED")),
  lambda: 0.5,
  support: SUPPORT_FLOOR,
  poolOnly: false,
  recommendAll: false,
  objective: "mean",
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
  // Slots are keyed by role, not by position, because the rows are draggable.
  const bans = [], ally = ROLES.map((_, i) => ({ role: i, champ: null })),
        enemy = ROLES.map((_, i) => ({ role: i, champ: null }));
  for (const e of S.log.slice(0, k)) {
    if (e.type === "ban") bans.push(e.champ);
    else if (e.type === "ally") ally[e.slot].champ = e.champ;
    else enemy[e.slot].champ = e.champ;
  }
  return stateFrom(bans, ally, enemy);
}

/** The next pick is the topmost unfilled row — that is what the ordering means. */
function onClockRole() {
  const open = S.ally.find((s) => !s.champ);
  return open ? open.role : null;
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

/** Every open ally role, in the order the rows are arranged. */
function openRoles() {
  return S.ally.filter((s) => !s.champ).map((s) => s.role);
}

function recommend(roleArg) {
  const role = roleArg === undefined ? onClockRole() : roleArg;
  if (role === null || role === undefined) {
    return { base: predict(currentState()), list: [], role: null };
  }
  const used = taken();
  const pool = S.poolOnly ? poolFor(role) : null;
  const base = predict(currentState());
  const baseScore = score(base);
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
    const s = score(p);
    out.push({
      ci,
      name: CHAMPS[ci - 1],
      mean: p.mean,
      value: s.value,
      std: s.spread,
      score: s.value - S.lambda * s.spread,   // pessimism (DESIGN.md §6)
      delta: s.value - baseScore.value,
      games: PLAY[ci][role],
    });
  }
  out.sort((a, b) => b.score - a.score);
  return { base, list: out, role };
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

function renderRecs(base, list, role) {
  const box = document.getElementById("recs");
  recsTitle("Best next pick");
  if (role === null) {
    box.innerHTML = `<p class="note">All five of your picks are locked. Clear one to see
      recommendations for it.</p>`;
    return;
  }
  if (!list.length) {
    box.innerHTML = `<p class="note">No legal candidates for
      <b>${ROLES[role]}</b>. Turn off the roster pool, or free up that row.</p>`;
    return;
  }
  const obj = OBJECTIVES[S.objective];
  const maxAbs = Math.max(...list.slice(0, 12).map((r) => Math.abs(r.delta)), 1e-4);
  const rows = list.slice(0, 12).map((r, i) => {
    const w = (Math.abs(r.delta) / maxAbs) * 50;
    const pos = r.delta >= 0;
    const fill = pos
      ? `left:50%;width:${w}%;background:var(--series-1)`
      : `right:50%;width:${w}%;background:var(--neg)`;
    return `<tr>
      <td class="rank">${i + 1}</td>
      <td>${r.name}</td>
      <td class="num">${obj.fmt(r.value)}</td>
      <td class="delta-cell"><div class="delta-bar"><div class="axis"></div>
        <div class="fill" style="${fill}"></div></div></td>
      <td class="num ${pos ? "up" : "down"}">${obj.fmtDelta(r.delta)}</td>
      <td class="num spread">±${obj.fmt(r.std)}</td>
      <td class="num">${obj.fmt(r.score)}</td>
      <td class="num spread">${r.games}</td>
    </tr>`;
  }).join("");
  box.innerHTML = `<table>
    <thead><tr><th></th><th>Champion</th><th class="num">${obj.label}</th><th></th>
      <th class="num">Δ</th><th class="num">spread</th><th class="num">score</th>
      <th class="num">games</th></tr></thead>
    <tbody>${rows}</tbody></table>
    <p class="note">For <b>${ROLES[role]}</b> — the topmost unfilled row on your team.
    Ranked by <b>${obj.label} − ${S.lambda.toFixed(1)}×spread</b>. “Spread” is
    ensemble disagreement: high spread means the model is extrapolating, so pessimism pushes it down.
    “Games” is how often that champion was played in this role in the training data.</p>`;
}

/* Because the model's optimum is separable (model/bestdraft.py: hill climbing
   never beats greedy), scoring each open role independently against the current
   board IS the joint optimum — no combinatorial search needed. */
function recsTitle(text) {
  const h = document.querySelector("#recs")?.closest(".card")?.querySelector("h2");
  if (h) h.textContent = text;
}

function renderAllRoles() {
  const box = document.getElementById("recs");
  recsTitle("Recommended picks — every open role");
  const roles = openRoles();
  if (!roles.length) {
    box.innerHTML = `<p class="note">All five of your picks are locked.</p>`;
    return;
  }
  const obj = OBJECTIVES[S.objective];
  const perRole = roles.map((role) => ({ role, ...recommend(role) }));

  const topCount = new Map();
  for (const { list } of perRole) {
    for (const r of list.slice(0, 4)) topCount.set(r.name, (topCount.get(r.name) || 0) + 1);
  }

  const blocks = perRole.map(({ role, list }, i) => {
    if (!list.length) {
      return `<div class="roleblock"><h4>${ROLES[role]}</h4>
        <p class="note">No legal candidates.</p></div>`;
    }
    const rows = list.slice(0, 4).map((r, j) => {
      const pos = r.delta >= 0;
      const dup = topCount.get(r.name) > 1 ? ` <span class="dup">also ${ROLES[
        perRole.find((pr) => pr.role !== role &&
          pr.list.slice(0, 4).some((x) => x.name === r.name))?.role] || ""}</span>` : "";
      return `<tr><td class="rank">${j + 1}</td><td>${r.name}${dup}</td>
        <td class="num">${obj.fmt(r.value)}</td>
        <td class="num ${pos ? "up" : "down"}">${obj.fmtDelta(r.delta)}</td></tr>`;
    }).join("");
    return `<div class="roleblock">
      <h4>${ROLES[role]} <span class="turnno">${i === 0 ? "next" : "open"}</span></h4>
      <table><tbody>${rows}</tbody></table></div>`;
  }).join("");

  box.innerHTML = `<div class="rolegrid">${blocks}</div>
    <p class="note">Best pick for each unfilled role, scored against the board as it
    stands and re-scored whenever either team locks something in. Ranked by
    <b>${obj.label} − ${S.lambda.toFixed(1)}×spread</b>. Because the model's optimum is
    separable, filling every role with its own best pick is also the best full draft —
    there is no combination to search for.</p>`;
}

function refresh() {
  const base = predict(currentState());
  if (S.recommendAll) renderAllRoles();
  else {
    const { list, role } = recommend();
    renderRecs(base, list, role);
  }

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
  const nextRole = onClockRole();
  if (nextRole !== null) {
    document.querySelector(`.slot[data-side="ally"][data-role="${nextRole}"]`)
      ?.classList.add("oncall");
  }
  renumber();

  document.getElementById("boardNote").textContent =
    `${S.bans.filter(Boolean).length} bans · ${S.ally.filter((s) => s.champ).length}/5 ally · ` +
    `${S.enemy.filter((s) => s.champ).length}/5 enemy locked. Enemy roles are left unknown to the model, as they are in a real draft.`;
}

/* ---------- drag to reorder ---------- */

/** Number the rows 1..5 so the pick order is legible at a glance. */
function renumber() {
  for (const side of ["ally", "enemy"]) {
    const box = document.getElementById(side === "ally" ? "allySlots" : "enemySlots");
    if (!box) continue;
    [...box.querySelectorAll(".slot")].forEach((row, i) => {
      const t = row.querySelector(".turn");
      if (t) t.textContent = String(i + 1);
    });
  }
}

/** Row order is the team's pick order; the model itself is order-invariant. */
function makeSortable(box, side) {
  let dragged = null;

  const rowBelow = (y) => {
    const rows = [...box.querySelectorAll(".slot:not(.dragging)")];
    return rows.find((r) => {
      const b = r.getBoundingClientRect();
      return y < b.top + b.height / 2;
    }) || null;
  };

  box.addEventListener("dragstart", (e) => {
    const row = e.target.closest(".slot");
    if (!row) return;
    dragged = row;
    row.classList.add("dragging");
    e.dataTransfer.effectAllowed = "move";
    // Firefox refuses to start a drag without payload.
    e.dataTransfer.setData("text/plain", row.dataset.role);
  });

  box.addEventListener("dragover", (e) => {
    if (!dragged) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
    const ref = rowBelow(e.clientY);
    if (ref === dragged) return;
    if (ref === null) box.appendChild(dragged);
    else box.insertBefore(dragged, ref);
  });

  box.addEventListener("drop", (e) => e.preventDefault());

  box.addEventListener("dragend", () => {
    if (!dragged) return;
    dragged.classList.remove("dragging");
    dragged = null;
    const order = [...box.querySelectorAll(".slot")].map((r) => +r.dataset.role);
    S[side] = order.map((role) => S[side].find((sl) => sl.role === role));
    refresh();
  });
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
    S[side].forEach((slot) => {
      const row = document.createElement("div");
      row.className = "slot";
      row.draggable = true;
      row.dataset.side = side;
      row.dataset.role = slot.role;

      const turn = document.createElement("span");
      turn.className = "turn";
      row.appendChild(turn);

      const grip = document.createElement("span");
      grip.className = "grip";
      grip.textContent = "⠿";
      grip.setAttribute("aria-hidden", "true");
      row.appendChild(grip);

      const lab = document.createElement("span");
      lab.className = "role";
      lab.textContent = ROLES[slot.role];
      row.appendChild(lab);

      row.appendChild(champInput(
        (idx) => { slot.champ = idx; logSet(side, slot.role, idx); }, "—"));
      box.appendChild(row);
    });
    makeSortable(box, side);
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

  const objSel = document.getElementById("objective");
  Object.entries(OBJECTIVES).forEach(([k, o]) => {
    const opt = document.createElement("option");
    opt.value = k; opt.textContent = o.label;
    objSel.appendChild(opt);
  });
  objSel.addEventListener("change", () => { S.objective = objSel.value; refresh(); });

  const lam = document.getElementById("lambda");
  const lamVal = document.getElementById("lambdaVal");
  lamVal.textContent = S.lambda.toFixed(1);
  lam.addEventListener("input", () => {
    S.lambda = +lam.value; lamVal.textContent = S.lambda.toFixed(1); refresh();
  });

  const recBtn = document.getElementById("recmode");
  recBtn.addEventListener("click", () => {
    S.recommendAll = !S.recommendAll;
    recBtn.textContent = "Recommend: " + (S.recommendAll ? "all open roles" : "next pick");
    recBtn.classList.toggle("primary", S.recommendAll);
    refresh();
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
  const c = document.createElement("span");
  c.className = "chip";
  c.textContent = `${stats.n_matches.toLocaleString()} matches`;
  chips.appendChild(c);

  const rb = document.getElementById("rosterBox");
  if (roster && roster.players && roster.players.length) {
    rb.innerHTML = "<p class='note' style='margin-bottom:4px'>Roster loaded from OP.GG:</p>" +
      roster.players.map((p) =>
        `<p class="roster-line"><b>${p.riot_id}</b> · ${p.tier || "unranked"} · ${p.main_role || "?"} ·
         avg ${p.avg_minutes ?? "?"} min · pool ${p.pool.length}
         <span class="spread">(${p.pool.slice(0, 5).map((c) => c.champion).join(", ")})</span></p>`
      ).join("");
  }

  // The methodology write-up lives in DESIGN.md / README.md, not on the page.
  refresh();
}

build();
})();
