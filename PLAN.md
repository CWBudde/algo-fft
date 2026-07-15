# PLAN.md — algofft Roadmap

This roadmap is the source of truth for status and direction. The v1.0
engineering work (Priorities 0–3 of the post-review roadmap) is **complete**;
the detailed item-by-item history is preserved in git (see the history of this
file). What remains here is a condensed record, the few carried-over open
items, and the post-v1.0 optimization backlog.

Design philosophy lives in `goal.md`; the component inventory is generated
into `docs/IMPLEMENTATION_INVENTORY.md` via `go generate ./internal/kernels/...`
(which runs `cmd/gencodelets -inventory <path>`).

---

## 1. Current Status (condensed)

All v1.0 Definition-of-Done items are green except the release tag itself:

- **P0 Correctness & build integrity** (2026-07): the SIMD build compiles and
  is CI-gated on amd64/386/arm64; ~3.9k lines of dead code deleted; all plan
  types are genuinely concurrency-safe (per-call scratch caches, `-race`
  tests); no silent wrong-answer paths (unschedulable radices panic, the
  scheduler/driver contract is validated); every registered codelet is
  verified forward-vs-`reference.NaiveDFT` per direction; docs/module-path/
  committed-binary issues fixed.
- **P1 Architecture hardening** (2026-07): kernel strategy is per-plan
  (`PlanOptions.Strategy`) with no process-global strategy state; tuning
  persists via the versioned, atomically-imported Wisdom cache (size +
  precision + CPU features). One deliberate global remains: the _default_
  Wisdom cache (`fft.DefaultWisdom`, mutated by `ImportWisdom*`/
  `ClearWisdom`) — per-consumer isolation is available via
  `PlanOptions.Wisdom`. Zero-allocation parity across 1D/2D/3D/ND/real/mixed-radix on
  both default and SIMD paths, locked in by `AllocsPerRun` guards. Plan-layer
  and DSP `*128` duplication collapsed into generics; dispatch de-duplicated;
  codelet **registration** is generated from a declarative table
  (`cmd/gencodelets`, ~164 entries).
- **P2 SIMD that ships** (2026-07): SIMD is on the **default build** behind
  runtime CPU detection (`-tags purego` is the opt-out; `-tags asm` a no-op).
  All known-incorrect kernels fixed (NEON complex64 bit-reversal/conjugation,
  NEON in-place copy-back corruption, x86 size-16 radix-16 aliasing); ~1,000
  `asmdecl` findings resolved and vet-gated. Kernel coverage: AVX2 (broad,
  incl. complex128 512–16384), SSE2/SSE3 (512/1024 both precisions), NEON
  complex64+complex128 size-specific sets, first AVX-512 tier (generic
  radix-2 + per-size codelets at 1024/4096/8192/16384 where it wins).
- **P3 API completeness & polish** (2026-07): no dead knobs or
  unimplemented options in the public API (the remaining `ErrNotImplemented`
  returns are intentional: the documented `FastPlan` "no codelet for this
  size" signal and unreachable defensive fallbacks); introspection parity
  (`Meta()`/`KernelStrategies()`/`Algorithms()` across plan types);
  `ForwardInPlace` naming unified; plan-reuse DSP types (`Convolver`,
  `Correlator`, `RealConvolver`); consistent `%w` error wrapping; GoDoc
  audit enforced by revive.
- **Testing & CI**: every arch matrix leg builds, vets, and tests both the
  default (SIMD) build and `-tags purego`; lint gate green; coverage gated
  at 90% (codecov); nightly benchmarks against a committed baseline;
  continuous fuzzing (PR + nightly) with committed regression corpus;
  property tests (Parseval/linearity/shift) across all dispatch families.

### Open items carried over

- [ ] **Tag `v1.0.0`**, GitHub release notes. _(Owner action — all
      engineering gates are green; issue/PR templates in place.)_
- [ ] **ARM64 NEON sizes 512+**: blocked on native ARM64 hardware — QEMU
      timings are not representative, so a size-specific 512/1024 kernel
      can't be benchmark-justified from CI today. Sizes 512/1024 are already
      NEON-served by the generic DIT kernels. Revisit with the native ARM64
      CI runner (Community backlog).
- [ ] **WASM SIMD**: blocked on toolchain — Go's `GOEXPERIMENT=simd`
      intrinsics (golang/go#73787) reached amd64 in Go 1.26 and Wasm/ARM64
      in the 1.27 RC, but remain experimental and this module targets
      Go 1.25. Revisit when the experiment graduates or the toolchain floor
      moves.
- AVX-512 follow-ups (higher radix, complex128, reclaiming 2048) are folded
  into P4.2 below.

---

## 2. Methodology for every P4 item

The P0–P2 discipline continues to apply to all optimization work:

1. **Measure baseline first** (`benchstat`, committed
   `benchmarks/baseline-*.txt` where relevant).
2. **Verify against reference**: forward-vs-`reference.NaiveDFT`, round-trip,
   and in-place for every new kernel/codelet (the P0.4 registry sweep gates
   registered codelets automatically).
3. **Both precisions** unless the item is precision-specific; **zero-alloc**
   guards extended to any new path.
4. **No regressions on the purego build** — algorithm-level wins must land in
   the generic code path too, not only in assembly.
5. Only land a per-size codelet if it **beats the incumbent in `benchstat`**
   on the target hardware tier (the AVX-512 codelet rule).

---

## Priority 4 — Optimized Algorithms Backlog

Ordered within each group by expected benefit/effort. Concrete code
references are to the current tree.

### P4.1 Algorithm-level upgrades (benefit all builds, including purego)

- [ ] **Bluestein padding: next 5-smooth size instead of next power of two.**
      Plan construction picks `bluesteinM = math.NextPowerOfTwo(2n-1)`
      (`plan.go`, `useBluestein` branch; kernels in
      `internal/kernels/bluestein.go` take m as given); the
      mixed-radix engine already handles any 2^a·3^b·5^c length, which is
      frequently much smaller (e.g. n=3000: m=8192 today vs m=6000 — and
      n=1000: 2048 vs 2000). Add `NextHighlyComposite` to `internal/math`,
      benchmark the crossover (a 5-smooth FFT is slightly slower per point
      than a radix-2 one), and pick per size at plan time — the choice is a
      pure function of n, so it can also be Wisdom-tuned.
- [ ] **Rader's algorithm for prime sizes.** Primes currently always pay
      Bluestein's ~4× padded-FFT cost. Rader maps a prime-p FFT to a cyclic
      convolution of length p−1; when p−1 is 5-smooth (e.g. 11, 31, 61, 101,
      151, 181, 241, 251…) this needs no padding at all, and otherwise pads
      far less than Bluestein. Implement as a new `KernelStrategy`
      (`KernelRader`) selected by the planner for primes with smooth p−1;
      validate vs reference and benchmark vs Bluestein per size.
- [ ] **Split-radix (conjugate-pair) kernels.** The core power-of-two paths
      are radix-2/4/mixed; split-radix cuts real operations ~25–33% vs
      radix-2 and is the classical best-known op count for 2^k. Start with a
      generic pure-Go split-radix DIT and benchmark against the tuned
      radix-4/Stockham paths at 32–4096 — it wins most cleanly at the sizes
      that lack hand-tuned codelets, and on the purego/WASM builds. Land as a
      `KernelStrategy` with auto-selection only where `benchstat` proves it.
- [ ] **Radix-8 stage for the generic DIT driver.** The generic driver
      currently composes radix-2/4 passes; a radix-8 stage cuts the pass
      count for 2^(3k) sizes and reduces twiddle loads per point. The radix-8
      infrastructure already exists for the size-512 codelet — generalize it
      into the mixed-radix scheduler's executable set.
- [ ] **Real-FFT for odd/multi-factor lengths + real-input Bluestein.**
      `NewPlanRealT` requires even n (pack method). Odd/arbitrary lengths
      currently force users through the complex path at 2× memory and flops.
      Support odd n via the complex fallback internally first, then evaluate
      a real-input Bluestein that exploits conjugate symmetry in the padded
      convolution.
- [ ] **Radix-7 / radix-11 butterflies for the mixed-radix engine.** Extends
      exact (non-Bluestein) coverage from 2^a·3^b·5^c to include factors 7
      and 11 (e.g. 448, 704, 1344, common in audio/comms block sizes).
      Butterfly count grows quickly with radix, so gate on a demonstrated
      win vs Bluestein at representative sizes before landing.

### P4.2 SIMD depth & breadth

- [ ] **FMA audit of the amd64 kernels.** Only 49 of 109 `.s` files under
      `internal/asm/amd64` use `VFMADD*`; the rest issue separate
      `VMULPS/VADDPS` chains. Convert the complex-multiply cores of the
      remaining AVX2 kernels to fused form (fewer uops, better accuracy —
      one rounding instead of two). Do it size-by-size with `benchstat` and
      the existing forward-vs-reference gates; expect the biggest wins on
      the twiddle-heavy generic radix-4 and Stockham kernels.
- [ ] **AVX-512 higher-radix / per-size-tuned variants** (carried over from
      P2.4). The shipped AVX-512 tier is generic radix-2; a radix-4 AVX-512
      kernel should widen the 1.2–2.4× gap and could reclaim size 2048 and
      the complex128 sizes where AVX2 codelets still win. Needs AVX-512
      CI/bench hardware for regression tracking.
- [ ] **ARMv8.3 FCMLA complex-arithmetic kernels.** `internal/cpu` detects
      only `HasNEON`; ARMv8.3's `FCMLA`/`FCADD` do a full complex
      multiply-accumulate in two instructions (vs 4 mul + 2 add today —
      `internal/asm/arm64/neon_complex_mul.s`). Add `HasFCMA`
      detection (HWCAP on Linux, sysctl on darwin), a `FCMLA` variant of the
      generic NEON butterfly, and runtime-dispatch it above plain NEON.
      Apple Silicon and Neoverse both support it. Blocked for benchmarking
      on the same native-ARM64-hardware item as NEON 512+.
- [ ] **SIMD the real-FFT recombination loop.** The pack step is already a
      `memcpy`, but the per-bin recombination
      (`plan_real_generic.go:240` / `:253`) is a scalar Go loop over
      `X[k] = A[k] − U[k]·(A[k]−B[k])` — one complex mul + conj + adds per
      bin, executed on every real transform of every size. An AVX2/NEON
      kernel (process 4–8 bins per iteration; the mirrored `B[k]` load is a
      reversed read) should noticeably cut small/medium real-FFT latency,
      where the recombination is a large fraction of total time. Same for
      the inverse pre-pass.
- [ ] **SSE2 tier breadth.** The non-AVX2 tier has tuned kernels only at
      512/1024; profile which other hot sizes (256, 2048, 4096) fall back to
      the generic path on SSE-only hardware and extend where `benchstat`
      justifies.

### P4.3 Memory & cache

- [ ] **Cache-blocked transpose for six-step/eight-step.**
      `internal/math/transpose.go` materializes an O(n²) swap-pair slice per
      matrix size (cached forever, ~storage of the matrix itself) and walks
      it in an order with no cache blocking. Replace with a tiled in-place
      transpose (e.g. 8×8 blocks, recursing or looping over tiles) — no
      index table at all, better locality, less resident memory. This
      directly speeds the six-step/eight-step large-size paths and the 2D
      plans. Follow up with a SIMD 8×8 complex tile kernel (AVX2
      `VPERM2F128`/`VUNPCK` pattern, NEON `TRN1/TRN2`).
- [ ] **Cache-blocked variants above L2.** For n where the working set
      exceeds L2 (≳2¹⁸ complex64), evaluate four-step/six-step with
      block sizes chosen from detected cache sizes (extend `internal/cpu`
      to expose L1d/L2 sizes) instead of fixed decompositions; Wisdom-tune
      the block choice.
- [ ] **Twiddle-table bandwidth reduction.** The packed-twiddle tables are
      per-plan precomputed; for large n they are a significant fraction of
      the working set. Evaluate (a) sharing tables between forward/inverse
      via conjugate-on-load (SIMD sign-flip is nearly free), and (b) the
      quarter-table symmetry (store n/8 entries + swizzle). Only worth it
      where measurements show the tables competing with data for cache.
- [ ] **SoA (split real/imag) layout exploration** (post-v1.0 API, from the
      Future list). Prototype internal SoA for one kernel family (e.g. the
      AVX-512 generic path, which currently spends shuffle uops
      de-interleaving) and measure; decide whether a v2 `PlanSoA` API is
      warranted before designing it.

### P4.4 Parallelism (opt-in, keep single-thread default)

- [ ] **Parallel batch execution.** `Plan.ForwardBatch`/`InverseBatch`
      (`plan.go:438,479`) run count transforms sequentially. Add
      `PlanOptions.Parallel`/`MaxWorkers` (default 1 = today's behavior):
      the batch loop fans out over a pre-created worker set with per-worker
      scratch from the existing resident-cache pattern, preserving
      zero-alloc-in-steady-state. Batch is embarrassingly parallel — this
      is the highest-value, lowest-risk parallel item.
- [ ] **Parallel 2D/3D/ND row-column passes.** Each axis pass is an
      independent batch of 1D transforms over rows/columns — reuse the
      P4.4 worker infrastructure. The transpose/gather steps stay serial
      initially. Gate on plan size (parallelism below ~256×256 is
      overhead-dominated); verify with the existing `-race` concurrent
      tests plus new parallel-enabled ones.
- [ ] **Parallel six-step for very large 1D.** The six-step algorithm is
      already a (transpose, batch-FFT, twiddle, transpose, batch-FFT)
      pipeline; run the inner batch-FFT stages on the worker pool for
      n ≳ 2²⁰. Depends on the cache-blocked transpose (P4.3) so the serial
      transpose doesn't dominate.

### P4.5 DSP-layer optimizations

- [ ] **Fast-size padding for one-shot convolution/correlation.**
      `convolveT` (`convolve.go`) plans at exactly `len(a)+len(b)-1` — a
      prime `convLen` silently pays full Bluestein cost. Linear convolution
      only needs a cyclic length ≥ convLen, so pad to the next 5-smooth
      (or power-of-two) size and truncate. Applies to `Convolve`,
      `Correlate`, `CrossCorrelate`, `AutoCorrelate`, the real variants,
      and the plan-reuse `Convolver`/`Correlator` types.
- [ ] **Buffer reuse in one-shot DSP helpers.** The one-shots allocate 5
      temporaries per call; route them through the pooled resident-cache
      scratch (as `Convolver` already does) so casual users get most of the
      steady-state performance without switching types.
- [ ] **Overlap-add/overlap-save streaming convolution.** For long-signal /
      short-kernel filtering (`len(b) ≪ len(a)`), one big FFT is
      asymptotically worse than block convolution with a plan of size
      ~4×len(b). Add `StreamingConvolver` (fixed kernel, chunked input) —
      this is both an API feature and the standard algorithmic fix for the
      current "FFT the whole signal" cost profile.

---

## Post-v1.0 Future (unchanged)

**Features**: DCT, Hilbert transform, STFT/spectrograms, audio/image examples,
Gonum ecosystem integration, optional GPU backends (kept out of the pure-Go
core).

**Community**: `CODE_OF_CONDUCT.md`, Dependabot, native ARM64 CI runner
(unblocks the NEON benchmarking items above).
