#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEMO_DIR="$ROOT_DIR/examples/wasm-demo"
OUT_DIR="${1:-$ROOT_DIR/dist}"

mkdir -p "$OUT_DIR"

GOOS=js GOARCH=wasm go build -o "$OUT_DIR/algofft.wasm" "$DEMO_DIR"
cp "$DEMO_DIR/index.html" "$DEMO_DIR/style.css" "$DEMO_DIR/app.js" "$DEMO_DIR/favicon.svg" "$OUT_DIR/"
cp "$(go env GOROOT)/lib/wasm/wasm_exec.js" "$OUT_DIR/"

printf "WASM demo built at %s\n" "$OUT_DIR"
