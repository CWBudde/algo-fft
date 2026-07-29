#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEMO_DIR="$ROOT_DIR/examples/wasm-demo"
OUT_DIR="${1:-$ROOT_DIR/dist}"

mkdir -p "$OUT_DIR"

GOOS=js GOARCH=wasm go build -o "$OUT_DIR/algofft.wasm" "$DEMO_DIR"

# Copy every static asset by glob rather than by name. Listing files explicitly
# meant each new page, script or stylesheet had to be remembered here, and
# forgetting one produced a broken GitHub Pages deploy with no build failure.
# Go sources are matched by none of these patterns, so nothing unwanted ships.
shopt -s nullglob
assets=("$DEMO_DIR"/*.html "$DEMO_DIR"/*.css "$DEMO_DIR"/*.js "$DEMO_DIR"/*.svg)
shopt -u nullglob

if [ ${#assets[@]} -eq 0 ]; then
    echo "error: no static assets found in $DEMO_DIR" >&2
    exit 1
fi

cp "${assets[@]}" "$OUT_DIR/"
cp "$(go env GOROOT)/lib/wasm/wasm_exec.js" "$OUT_DIR/"

printf "Copied %d static asset(s)\n" "${#assets[@]}"

printf "WASM demo built at %s\n" "$OUT_DIR"
