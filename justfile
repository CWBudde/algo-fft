# Build the library
build:
    go build -v ./...

# Run all tests
# -timeout is a backstop, not a target: the race detector's instrumentation
# makes the O(n^2) reference DFTs in internal/kernels far more expensive than
# the kernels they check, so that package sets its own size cap (see
# naiveReferenceRaceMaxSize). Go's 10m default left no headroom at all.
test:
    go test -v -race -count=1 -timeout=20m ./...

# Run benchmarks
bench:
    go test -bench=. -benchmem -run=^$ ./...

# Canary-gated codelet-candidate sweep for registry priority tuning.
# Measures only inside verified-quiet windows; see BENCHMARKS.md.
bench-gated *sizes:
    ./scripts/bench_gated.sh {{ sizes }}
    ./scripts/bench_gated_analyze.sh

# Measure every registered power-of-two candidate on this host and persist the
# winners. Import the resulting file before constructing estimate-mode plans.
tune OUTPUT="algofft-wisdom.txt" MAX="32768":
    go run ./cmd/tune -max {{ MAX }} -output {{ OUTPUT }}

# Run linters
lint:
    golangci-lint run

# Run linters and fix issues
lint-fix:
    golangci-lint run --fix

# Format code using treefmt
fmt:
    treefmt . --allow-missing-formatter

# Check if code is formatted
fmt-check:
    treefmt --allow-missing-formatter --fail-on-change

# Generate coverage report
cover:
    go test -coverprofile=coverage.txt -covermode=atomic ./...
    go tool cover -html=coverage.txt -o coverage.html

# Clean build artifacts
clean:
    rm -f coverage.txt coverage.html coverage_*.txt coverage_*.html
    rm -rf dist/
    find . -type f \( -name '*.test' -o -name '*.pprof' -o -name '*.o' \) -delete

# Run all checks (test, lint, coverage)
check: test lint cover check-deps

# Cross-compile for amd64 (SIMD is included in the default build)
build-amd64:
    GOOS=linux GOARCH=amd64 go build -v ./...

# Cross-compile for ARM64
build-arm64:
    GOOS=linux GOARCH=arm64 go build -v ./...

# Build WebAssembly target (js/wasm)
build-wasm:
    GOOS=js GOARCH=wasm go build -v ./...

# Run tests with the pure-Go fallback (no SIMD kernels)
test-purego:
    go test -tags "purego" -v -count=1 ./...

# Vet the SIMD build (asmdecl frame checks) on amd64, arm64, and 386
vet-arch:
    GOARCH=amd64 go vet ./...
    GOARCH=arm64 go vet ./...
    GOARCH=386 go vet ./...
    go vet -tags "purego" ./...

# Run WebAssembly tests in Node.js
test-wasm:
    GOOS=js GOARCH=wasm go test -exec="$(pwd)/scripts/wasm_exec_node_env.sh" -v -count=1 ./...

# Run WebAssembly tests for a single package
test-wasm-pkg pkg:
    GOOS=js GOARCH=wasm go test -exec="$(pwd)/scripts/wasm_exec_node_env.sh" -v -count=1 {{pkg}}

# Build the WebAssembly demo into ./dist
build-wasm-demo:
    ./scripts/build-wasm-demo.sh

# Build and run the WebAssembly demo locally
run-wasm-demo: build-wasm-demo
    @echo "Starting demo server at http://localhost:8090"
    python3 -m http.server -d dist 8090

# Run tests on ARM64 using QEMU (requires qemu-user-static)
test-arm64:
    #!/usr/bin/env bash
    if ! command -v qemu-aarch64-static &> /dev/null; then
        echo "Error: qemu-aarch64-static not found"
        echo "Install with: sudo apt-get install qemu-user-static binfmt-support"
        exit 1
    fi
    ALGOFFT_QEMU=1 GOOS=linux GOARCH=arm64 go test -exec="qemu-aarch64-static" -v -count=1 ./...

# Run benchmarks on ARM64 using QEMU (NOTE: performance not representative, correctness only)
bench-arm64:
    #!/usr/bin/env bash
    if ! command -v qemu-aarch64-static &> /dev/null; then
        echo "Error: qemu-aarch64-static not found"
        echo "Install with: sudo apt-get install qemu-user-static binfmt-support"
        exit 1
    fi
    @echo "NOTE: QEMU benchmarks are for correctness validation only, not performance measurement"
    GOOS=linux GOARCH=arm64 go test -exec="qemu-aarch64-static" -bench=. -benchmem -run=^$ ./...

# Build for both amd64 and arm64
build-all: build build-arm64
    @echo "Built for amd64 and arm64"

# Test on both amd64 and arm64
test-all: test test-arm64
    @echo "Tests passed on both architectures"

# Run all checks on both architectures
check-all: check test-arm64
    @echo "All checks passed on amd64 and arm64"

# Run SIMD verification tests
test-simd-verify:
    go test -v -run=TestSIMD ./internal/fft
    go test -v -run=TestAVX2 ./internal/fft
    go test -v -run=TestNEON ./internal/fft

# Run architecture-specific tests locally
test-arch:
    @echo "Running architecture-specific tests..."
    go test -v -count=1 ./...
    @echo "Verifying SIMD implementations..."
    just test-simd-verify

# Run stress tests (long-running, skip in short mode)
test-stress:
    go test -v -timeout=30m -run=Stress ./...

# Profile memory usage
profile-mem:
    go test -run=Stress -memprofile=mem.prof -timeout=30m ./...
    go tool pprof -http=:8080 mem.prof

# Profile CPU usage
profile-cpu:
    go test -run=Bench -cpuprofile=cpu.prof -bench=. ./...
    go tool pprof -http=:8080 cpu.prof

# Default target
default: build

fix:
    just lint-fix
    just fmt

# Are all github.com/cwbudde/* dependencies at their latest tags?
check-deps:
    ./scripts/release-guard.sh deps

# How much work is sitting on main past the latest tag?
check-unreleased:
    ./scripts/release-guard.sh unreleased

# Check every release precondition for VERSION without tagging anything.
release-check VERSION:
    ./scripts/release-guard.sh gate {{VERSION}}

# Tag VERSION: run the full gate, then create and push the annotated tag.
# Refuses on a dirty tree, stale siblings, a missing CHANGELOG section, or an
# incompatible API change the version does not signal. See AGENTS.md.
tag-release VERSION:
    ./scripts/release-guard.sh tag {{VERSION}}
