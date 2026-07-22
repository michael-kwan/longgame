/* Numeric core of the DeepSets value model — the JS port of model/net.py.
   Kept separate from the UI so tests/parity.mjs can exercise exactly the code
   the page ships. Attaches to globalThis.LG. */
(() => {
"use strict";

function decode(b64) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}

/* tanh approximation of GELU; max abs error ~1e-3 vs torch's exact erf form,
   which is far below the noise floor of a model with R^2 in the hundredths. */
const gelu = (x) =>
  0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));

function linear(W, b, x, activate) {
  const [out, inn] = W.shape;
  const y = new Float32Array(out);
  const w = W.data;
  for (let o = 0; o < out; o++) {
    let s = b.data[o];
    const base = o * inn;
    for (let i = 0; i < inn; i++) s += w[base + i] * x[i];
    y[o] = activate ? gelu(s) : s;
  }
  return y;
}

function softmax(z) {
  let mx = -Infinity;
  for (const v of z) if (v > mx) mx = v;
  const p = new Float32Array(z.length);
  let s = 0;
  for (let i = 0; i < z.length; i++) { p[i] = Math.exp(z[i] - mx); s += p[i]; }
  for (let i = 0; i < p.length; i++) p[i] /= s;
  return p;
}

/* Win probability is NOT the network's win_head — that head scores 0.4926
   held-out accuracy, worse than chance, because the duration model is symmetric
   in the two teams and cannot represent "which side". This is a separate
   antisymmetric ridge (ally champions minus enemy champions), AUC ~0.529.
   Bootstrap members give it the same ensemble-spread semantics as the rest. */
function createWinModel(winModel) {
  if (!winModel || !winModel.members || !winModel.members.length) return null;
  const members = winModel.members.map((m) => ({ w: Float64Array.from(m.w), mu: m.mu }));
  return {
    cv: winModel.cv,
    /** P(ally side wins) per bootstrap member. */
    perMember(st) {
      return members.map(({ w, mu }) => {
        let s = mu;
        for (const c of st.allyC) s += w[c];
        for (const c of st.enemyC) s -= w[c];
        return Math.min(0.99, Math.max(0.01, s));
      });
    },
  };
}

function createEnsemble(bundle) {
  const DIM = bundle.config.dim;
  const NB = bundle.config.n_bins;
  const CENTERS = bundle.vocab.bin_centers;

  const nets = bundle.members.map((m) => {
    const flat = decode(m);
    const t = {};
    let off = 0;
    for (const [name, shape] of bundle.shapes) {
      const n = shape.reduce((a, b) => a * b, 1);
      t[name] = { data: flat.subarray(off, off + n), shape };
      off += n;
    }
    return t;
  });

  function forward(net, st) {
    const CE = net["champ.weight"].data, RE = net["role.weight"].data;
    const ally = new Float32Array(DIM), enemy = new Float32Array(DIM), bans = new Float32Array(DIM);

    for (let i = 0; i < st.allyC.length; i++) {
      const c = st.allyC[i] * DIM, r = st.allyR[i] * DIM;
      for (let d = 0; d < DIM; d++) ally[d] += CE[c + d] + RE[r + d];
    }
    for (let i = 0; i < st.enemyC.length; i++) {
      const c = st.enemyC[i] * DIM, r = st.enemyR[i] * DIM;
      for (let d = 0; d < DIM; d++) enemy[d] += CE[c + d] + RE[r + d];
    }
    for (let i = 0; i < st.bans.length; i++) {
      const c = st.bans[i] * DIM;
      for (let d = 0; d < DIM; d++) bans[d] += CE[c + d];
    }

    /* Mean-pool, matching DraftNet._pool — sums would make the feature scale
       grow as the draft fills. Empty groups stay at zero (divisor clamped to 1). */
    const nA = Math.max(st.allyC.length, 1);
    const nE = Math.max(st.enemyC.length, 1);
    const nB = Math.max(st.bans.length, 1);
    for (let d = 0; d < DIM; d++) { ally[d] /= nA; enemy[d] /= nE; bans[d] /= nB; }

    const tierE = net["tier.weight"].data, queueE = net["queue.weight"].data;
    const feats = new Float32Array(DIM * 4 + 3 + 16 + 8);
    let o = 0;
    for (let d = 0; d < DIM; d++) feats[o++] = ally[d];
    for (let d = 0; d < DIM; d++) feats[o++] = enemy[d];
    for (let d = 0; d < DIM; d++) feats[o++] = ally[d] * enemy[d];
    for (let d = 0; d < DIM; d++) feats[o++] = bans[d];
    feats[o++] = st.allyC.length / 5;
    feats[o++] = st.enemyC.length / 5;
    feats[o++] = st.bans.length / 10;
    for (let d = 0; d < 16; d++) feats[o++] = tierE[st.tier * 16 + d];
    for (let d = 0; d < 8; d++) feats[o++] = queueE[st.queue * 8 + d];

    let h = linear(net["trunk.0.weight"], net["trunk.0.bias"], feats, true);
    h = linear(net["trunk.3.weight"], net["trunk.3.bias"], h, true);
    const logits = linear(net["duration_head.weight"], net["duration_head.bias"], h, false);

    /* Champion main effects, shifted along the duration axis (model/net.py). */
    const EFF = net["champ_effect.weight"].data, DIR = net["bin_dir"].data;
    let effect = 0;
    for (let i = 0; i < st.allyC.length; i++) effect += EFF[st.allyC[i]];
    for (let i = 0; i < st.enemyC.length; i++) effect += EFF[st.enemyC[i]];
    for (let i = 0; i < logits.length; i++) logits[i] += effect * DIR[i];

    return { logits, win: linear(net["win_head.weight"], net["win_head.bias"], h, false)[0] };
  }

  /* Ensemble mean plus member disagreement — the pessimism signal of DESIGN.md §6. */
  function predict(st) {
    const probs = new Float32Array(NB);
    const means = [], memberProbs = [];
    let win = 0;
    for (const net of nets) {
      const { logits, win: wl } = forward(net, st);
      const p = softmax(logits);
      memberProbs.push(p);
      let m = 0;
      for (let i = 0; i < NB; i++) { probs[i] += p[i] / nets.length; m += p[i] * CENTERS[i]; }
      means.push(m);
      win += 1 / (1 + Math.exp(-wl)) / nets.length;
    }
    const mean = means.reduce((a, b) => a + b, 0) / means.length;
    const varr = means.reduce((a, b) => a + (b - mean) ** 2, 0) / means.length;
    return { probs, mean, std: Math.sqrt(varr), win, means, memberProbs };
  }

  return { nets, forward, predict, DIM, NB, CENTERS, winModel: createWinModel(bundle.win_model) };
}

globalThis.LG = { decode, gelu, linear, softmax, createEnsemble };
})();
