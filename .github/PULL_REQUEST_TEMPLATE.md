# Description

<!-- What does this PR change, and why? Link related issues (e.g. Fixes #123). -->

## Checklist

- [ ] Commits follow [Conventional Commits](https://www.conventionalcommits.org/)
      (`feat:`, `fix:`, `docs:`, ...)
- [ ] `just check` passes locally (test + lint + coverage)
- [ ] `just fmt-check` passes (or run `just fmt`)
- [ ] New features include tests for **both** `complex64` and `complex128`
- [ ] Kernel/assembly changes: SIMD and pure-Go paths produce identical
      results (`just test-simd-verify`, `just test-purego`)
- [ ] Documentation updated (GoDoc comments, README/PLAN.md if applicable)

## Test results

<!-- Paste relevant `go test` output or CI links. -->

## Benchmarks

<!-- For performance changes: benchstat before/after comparison and the
     CPU/hardware used. Delete this section otherwise. -->
