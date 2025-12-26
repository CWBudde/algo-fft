#!/usr/bin/env bash
set -euo pipefail

GOROOT="$(go env GOROOT)"
WASM_NODE="$GOROOT/lib/wasm/wasm_exec_node.js"

# Reduce environment size to avoid argv+env limits inside wasm_exec.js.
exec env -i \
	PATH="$PATH" \
	HOME="${HOME:-}" \
	TMPDIR="${TMPDIR:-/tmp}" \
	node --stack-size=8192 "$WASM_NODE" "$@"
