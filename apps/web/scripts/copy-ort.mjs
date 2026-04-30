#!/usr/bin/env node
/**
 * Copy ONNX Runtime Web's WASM + ESM glue into public/ort/.
 *
 * Run via: pnpm run prebuild  (or: pnpm postinstall)
 *
 * ORT must be served same-origin or its threaded WASM init fails. Bundlers
 * generally don't pick up the .wasm files automatically, so we mirror the
 * vendor dist into public/.
 */
import { mkdirSync, copyFileSync, existsSync, readdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");

// Find the ORT dist dir under pnpm's content-addressed store
const PNPM_STORE = join(ROOT, "node_modules", ".pnpm");
if (!existsSync(PNPM_STORE)) {
  console.error("[copy-ort] node_modules/.pnpm missing — run pnpm install first");
  process.exit(1);
}
const ortDir = readdirSync(PNPM_STORE).find((d) => d.startsWith("onnxruntime-web@"));
if (!ortDir) {
  console.error("[copy-ort] onnxruntime-web not found in pnpm store");
  process.exit(1);
}
const SRC = join(PNPM_STORE, ortDir, "node_modules", "onnxruntime-web", "dist");
const DST = join(ROOT, "public", "ort");
mkdirSync(DST, { recursive: true });

let n = 0;
for (const file of readdirSync(SRC)) {
  if (!file.endsWith(".wasm") && !file.endsWith(".mjs")) continue;
  copyFileSync(join(SRC, file), join(DST, file));
  n++;
}
console.log(`[copy-ort] copied ${n} files from ${ortDir} to public/ort/`);
