/* Runs the shipped JS inference core on fixed draft states and prints the
   results as JSON. tests/test_parity.py compares them against PyTorch.
   A silent mismatch here would make the page confidently wrong. */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");

globalThis.atob = (b64) => Buffer.from(b64, "base64").toString("binary");
new Function(readFileSync(join(root, "web", "infer.js"), "utf8"))();

const bundle = JSON.parse(readFileSync(join(root, "artifacts", "bundle.json"), "utf8"));
const ens = globalThis.LG.createEnsemble(bundle);
const states = JSON.parse(readFileSync(join(here, "fixtures", "parity_states.json"), "utf8"));

const out = states.map((st) => {
  const r = ens.predict(st);
  return { mean: r.mean, std: r.std, win: r.win, probs: Array.from(r.probs), means: r.means };
});
console.log(JSON.stringify(out));
