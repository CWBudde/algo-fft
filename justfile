# Build the library
build:
    go build -v ./...

# Run all tests
test:
    go test -v -race -count=1 ./...

# Run benchmarks
bench:
    go test -bench=. -benchmem -run=^$ ./...

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
    rm -f coverage.txt coverage.html

# Run all checks (test, lint, coverage)
check: test lint cover

# Cross-compile for ARM64
build-arm64:
    GOOS=linux GOARCH=arm64 go build -v ./...

# Build WebAssembly target (js/wasm)
build-wasm:
    GOOS=js GOARCH=wasm go build -v ./...

# Run WebAssembly tests in Node.js
test-wasm:
    GOOS=js GOARCH=wasm go test -exec="$(pwd)/scripts/wasm_exec_node_env.sh" -v -count=1 ./...

# Run WebAssembly tests for a single package
test-wasm-pkg pkg:
    GOOS=js GOARCH=wasm go test -exec="$(pwd)/scripts/wasm_exec_node_env.sh" -v -count=1 {{pkg}}

# Build the WebAssembly demo into ./dist
build-wasm-demo:
    ./scripts/build-wasm-demo.sh

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
