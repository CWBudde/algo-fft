# Repository Guidelines

## Project Structure & Module Organization

- Root package exposes the public API (see `plan.go`, `types.go`, `errors.go`).
- Core FFT implementation lives in `internal/fft/` with architecture-specific kernels (`*_amd64.go`, `*_arm64.go`) and assembly (`*.s`).
- Tests are colocated with code (`*_test.go` in root and `internal/fft/`).
- Supporting docs include `README.md`, `CONTRIBUTING.md`, and `PLAN.md`.

## Build, Test, and Development Commands

Use the `just` recipes defined in `justfile`:

- `just build` — compile all packages.
- `just test` — run unit tests with race detector.
- `just bench` — run benchmarks only.
- `just lint` / `just lint-fix` — run `golangci-lint` (optionally fix).
- `just fmt` — run `treefmt` (Go via `gofumpt` + `gci`, Markdown via `prettier`).
- `just cover` — generate `coverage.html` from `coverage.txt`.

## Coding Style & Naming Conventions

- Follow standard Go style; format with `gofumpt` and import ordering via `gci`.
- Use clear, descriptive names; keep functions focused and small.
- Add GoDoc comments for exported symbols.
- File naming: tests as `*_test.go`; architecture-specific files use `*_amd64.go`, `*_arm64.go`, and `.s` for assembly.

## Testing Guidelines

- Framework: Go `testing` package; tests are colocated with sources.
- Coverage target: aim for >80% (per `CONTRIBUTING.md`).
- Run tests with `just test`; coverage via `just cover`.
- Favor property-based and edge-case tests for new FFT behavior.

## Commit & Pull Request Guidelines

- Commit history is minimal; follow `CONTRIBUTING.md` style:
  - Short summary (<=50 chars), blank line, optional details wrapped at 72 chars.
  - Reference issues as needed (e.g., `#123`).
- PRs should include:
  - Clear description of changes and motivation.
  - Linked issues (if any).
  - Test results; include benchmarks for performance changes.

## Architecture Notes

- Plans are designed for zero-allocation steady-state transforms.
- SIMD optimizations are dispatched at runtime; keep hot paths allocation-free.
