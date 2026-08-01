# Engineering history

Every round that has landed, one entry each. This is the detail that used to be
`PLAN.md` §1.1; `PLAN.md` now carries only the current status and the open work.

Dates are the month the round landed. Follow the links for the measurements and
the reasoning behind each.

**Foundations (2026-07).**

- **Correctness & build integrity.** SIMD build compiles and is CI-gated on
  amd64/386/arm64; ~3.9k lines of dead code deleted; all plan types
  concurrency-safe under `-race`; unschedulable radices panic rather than
  return a wrong answer; every registered codelet verified per-direction against
  `reference.NaiveDFT`.
- **Architecture consolidation.** One kernel contract (`CodeletFunc` returns
  `bool`; every call site honors the bail signal). One precision scheme —
  `NewPlan[T]`, `NewPlanReal[F, C]`, `PlanReal2D/3D[F, C]` — which closed the
  missing float64 real-2D/3D gap. One plan interface (`PlanInfo`). Public types
  owned by the root package, no aliases into `internal/*`. `Plan2D`/`Plan3D` are
  thin wrappers over `PlanND`. New leaf package `internal/registry` so `kernels`
  no longer registers _upward_ into `planner`. `Plan[T]` split into four
  `planExecutor` implementations (kernel/Bluestein/Rader/recursive), 40 fields →
  21, with the codelet binding cached as a zero-dispatch fast path because
  interface dispatch cost ~20 ns at n = 8. One deliberate global remains: the
  default Wisdom cache, with per-consumer isolation via `PlanOptions.Wisdom`.
- **complex128 kernel twins generated.** `cmd/genkernels` emits
  `<base>_c128.gen.go` from the complex64 sources (42 files, 108 functions);
  ~9.9k hand-written lines deleted. A pre-generation audit found 16 twins had
  drifted — the complex64 side had been optimized later — so generating them was
  itself a free win (1024/radix4 −27%/−18%). Deliberately still hand-written:
  the radix-3/5 c128 entry points and `dit_16384_radix4`'s c128 pair, whose
  `[16384]complex128` stage arrays exceed the compiler's 128 KiB stack limit.
- **Kernel strategy is per-plan** (`PlanOptions.Strategy`), no process-global
  state; tuning persists via the versioned Wisdom cache (v3; v2 files are
  rejected, not reinterpreted). Zero-allocation parity across
  1D/2D/3D/ND/real/mixed-radix on both default and SIMD paths, locked by
  `AllocsPerRun` guards.
- **Testing & CI.** Every arch matrix leg builds, vets and tests both the
  default and `purego` builds; lint green; coverage gated at 90%; nightly
  benchmarks against a committed baseline; continuous fuzzing with committed
  corpus; property tests (Parseval/linearity/shift) across all dispatch
  families.

**Algorithms (2026-07).** See [`MIXED_RADIX.md`](MIXED_RADIX.md).

- **Shape-aware Bluestein padding.** A single scalar penalty could never be
  right: a mixed-radix sub-FFT's cost per `m·log2(m)` spans ~7× on shape alone.
  The model is a whitelist (`padShapes`, `plan_padsize.go`), each shape admitted
  only above the pad size where it wins at **both** precisions: `3·2^(k-2)` from
  2^9, `15·2^(k-4)` from 2^13; `7·2^(k-3)` is dominated outright. End-to-end
  −15…−57% at the affected lengths.
- **Rader's algorithm**, gated on measured wins rather than on `p−1` being
  smooth. Power-of-two `p−1` wins 4–5×; 5-smooth `p−1 ≥ 96` with pow2 part ≥ 8
  wins 1.1–5.6×. Padded Rader for non-smooth `p−1` is intentionally skipped
  (pad ≥ `2p−3` vs Bluestein's `2p−1` is a wash).
- **Split-radix (conjugate-pair) DIT** with full strategy plumbing; beats the
  auto path at every power of two ≥ 256 on purego (+11–34%, 2.1× at 262144). No
  longer auto-selected anywhere after the square-rule re-measure.
- **Radix-8 stage for the generic DIT driver**, gated to the no-codelet path:
  geomean −16.9% across 32…12288 on purego, so it benefits purego, SSE-only
  amd64 and arm64 without touching AVX2 schedules.
- **Radix-7 / radix-11 butterflies** extend exact coverage to
  `2^a·3^b·5^c·7^d·11^e`.
- **Real-FFT for odd/multi-factor lengths.** `NewPlanReal*` accepts any n ≥ 2;
  odd lengths run an internal full-size complex FFT with DC-only spectrum
  validation.
- **The mixed-radix engine went from 0.20×/0.13× FFTW3 to today's figures.** The
  largest single cause was a **driver defect**, not a routing one: the AVX2
  drivers guarded codelet dispatch with `n > 1` while the scheduler requires
  `n > 5`, so every schedule ending in radix 2/3/4/5 sent each leaf through a
  full codelet call — 1225 dispatches per transform at n = 4900. Fused AVX2
  stage kernels for radix 3/5/7/11 then gave geomean −30%, with n = 704 dropping
  10244 → 2609 ns.

**Memory & cache (2026-07).**

- **Cache-blocked transpose** — the O(n²) swap-pair index table is gone;
  `math.TransposeSquare` tiles in place with edge 8. Transpose −70…−82%;
  six-step/eight-step −10…−23% at n ≥ 65536; square `Plan2D` −34.6% geomean.
- **Four-step** (`KernelFourStep`), the rectangular generalization of six-step,
  with the `n1×n2` split chosen by a cache-residency model over the L1d/L2 sizes
  `internal/cpu` detects. Beats split-radix at 2^21…2^23.
- **Power-of-two squares are no longer special-cased.** Measured across all
  candidate strategies at 2^18/2^20/2^22, both directions, precisions and
  builds: Stockham wins or ties every arm bar one. One dissenting arm is
  accepted knowingly (2^20 c128 forward prefers six-step) because a
  precision- and direction-blind rule cannot capture it.
- **Twiddle-table bandwidth.** Radix-4 stages conjugate on load, halving
  per-plan packed-table memory for free. Quarter-table symmetry was evaluated
  and **declined** for the scalar path — an L1-resident tiny-table experiment
  put the cache-footprint share at only ~4–8%, so octant-decode ALU would be a
  net loss. Revisit only inside SIMD kernels.

**The complex64 scalar-multiply defect (2026-07).** The single most valuable
finding of the comparative sweep, and the reason complex64 was _slower_ than
complex128 at 20 of 23 non-power-of-two lengths. **Go's compiler does not
implement scalar `complex64 * complex64` in single precision** — it widens all
four components to float64, multiplies in double precision and rounds back:
twelve instructions against six for the same expression on complex128. Only the
multiply promotes, so any FFT stage written as scalar Go is _structurally_ more
expensive in complex64. Powers of two hid it inside float32 SIMD codelets; the
arbitrary-length routes could not. Fixed in three rounds with
`math.MulComplex64` — arbitrary-length glue (c64/c128 ratio 1.18–1.27 → 0.90–0.98),
1378 products across 39 pure-Go codelet sources rewritten by a `go/types`-driven
tool (purego geomean **−24.4%** over the 8…16384 ladder), and 17 remaining
sites. Two independent hosts three generations apart agreed on the codelet
round's geomean to within 0.3% _and_ on which sizes did not move — the signature
of a code change rather than a measurement artifact. Accuracy cost is sub-ulp;
see [`PRECISION.md`](PRECISION.md). Standing rules from this round are
in `../PLAN.md` §2.3.

**SIMD kernels (2026-07).** Coverage: AVX2 broad in both precisions, SSE2/SSE3
tier to 32768, NEON size-specific 4–32768 both precisions, a first AVX-512 tier.

- **SIMD ships in the default build** behind runtime CPU detection (`-tags
purego` opts out; `-tags asm` is a no-op). All known-incorrect kernels fixed;
  ~1,000 `asmdecl` findings resolved and vet-gated.
- **The 256-bit radix-4 kernels** replaced a whole AVX2 codelet family that was
  XMM-width in VEX clothing — see [`AVX2_RADIX4.md`](AVX2_RADIX4.md).
  This is the change that made powers of two competitive with FFTW3.
- **Legacy-SSE encodings swept out of the AVX2/AVX-512 tree**: 4089 instructions
  converted to VEX across 59 files, then the last ten functions in a second
  round. End-to-end performance-neutral, but it uncovered six codelets carrying
  a fixed ~100 ns transition-penalty prologue regardless of transform size
  (`dit4_radix4_avx2` 102.9 → 5.0 ns) — which is what had mis-ranked them in the
  registry. The n = 64 cliff (0.36× FFTW3) was one legacy `MOVD AX, X8`.
  Method and traps are in [`AGENTS.md`](../AGENTS.md).
- **FMA audit, two passes.** Every AVX2 codelet the registry actually _selects_
  is now fused (97 sites in the second pass alone). The tier is gated on
  `HasAVX2 && HasFMA`, so an FMA-masked VM falls back instead of faulting.
  _Performance was never demonstrated_ — three benchstat attempts were swamped
  by thermal throttling; treat that pass as instruction-count and accuracy work.
  Remaining scope in §4.
- **AVX2 complex128 Stockham asm** for every Stockham-resolved c128 size above
  the codelet range: kernel-level −16…−50%, 65536 end-to-end 1.44 → 1.02 ms.
- **SSE tier extended to 16384/32768** in both precisions, emitted by a one-off
  generator validated by byte-reproducing the existing 4096/8192 files.
- **NEON ladder completed 4 → 32768** in both precisions across three subagent
  delegation rounds. Priorities are ladder-mirrored, **not** tuned — QEMU timing
  is meaningless (§6).
- **Real-FFT forward recombination in SIMD**: the kernel is 4.5–8× the scalar
  loop, `BenchmarkPlanRealForward` −34.7% geomean.

**Three unreachable fast paths (2026-07-29).** The same bug three times — a
correct, registered, faster path that nothing called, each behind a dispatch
decision whose stated justification had never been re-derived.

- **The size-384 path.** The AVX2 assembly for the c64 side had been written,
  assembled, declared and never called — and it was also _wrong_
  (`VINSERTPS $0x10` where `VMOVLHPS` was needed). Separately, both precisions'
  128-point sub-FFTs were bound to superseded kernels. Fixing the binding alone,
  with no new assembly, gave geomean **−58%**.
- **The six-step row FFTs** were still on the pre-radix-4 128-point kernel at
  six call sites — 320 ns where the generic kernel is 88. The swap also _removed_
  per-call work, because those files had been gathering their length-128 row
  table out of the caller's table on every transform. Geomean **−33%**.
- **The Bluestein sub-FFT never reached the registry**, so the default build was
  ~4% _slower_ than `-tags purego` at n = 1009 and 2003 — both builds ran the
  identical pure-Go kernel for ~96% of the work. See
  [`MIXED_RADIX.md`](MIXED_RADIX.md).
- **Packed Stockham was compiled out of SIMD builds** on the stated grounds that
  the codelet path superseded it. Every codelet is registered as `KernelDIT`, so
  the strategy check upstream had already excluded it; what the constant actually
  suppressed was the sizes with _no_ codelet, where the SIMD build fell through
  to a radix-2 Stockham kernel while the radix-4 route it had disabled was up to
  2.7× faster.

**Reporting and hygiene (2026-07-28 … 07-30).**

- **Plans report the route they take.** Non-power-of-two lengths used to name a
  strategy that never executed (1000 said `dit_fallback`; 2205/3600/12000/44100
  said `stockham`). `KernelMixedRadix` is now a first-class strategy resolved
  before anything else, and the rule is that the reported strategy always equals
  the executed one.
- **Test-vector blindness audit.** Four tests were checking nothing and one
  whole precision was uncovered above n = 16. See
  [`TESTING.md`](TESTING.md).
- **The race suite could not finish.** `internal/kernels` took 1499.7 s under
  `-race` against Go's 10-minute default, which neither `just test` nor CI
  overrode — a gate that had been red repo-wide. Details in `../CHANGELOG.md`.
- **Incumbent audits.** Every registered power-of-two size has now been ranked
  under the canary-gated sweep. Results, including the one mis-tuned row it
  found (complex128 at n = 8), are in
  [`CODELET_BENCHMARKS.md`](CODELET_BENCHMARKS.md).

**The radix sweep (2026-07-30 … 08-01).** Everything below closed while the
roadmap was being reorganised into phases; the open consequences carried
forward into `../PLAN.md` Phase 1.

- **Pruned the shadowed AVX2 codelet surplus — registry side (2026-07-30).**
  The incumbent audit turned up a large tail of AVX2 rows that nothing could
  ever select; **22 rows** were unregistered at a bar of 1.5× the size's winner
  (AVX2 tier 33 + 27 → 20 + 18 registrations; inventory 229 → 207). Three
  distinct causes. (1) `*_radix2_avx2` at n ≥ 16 is _structurally_ dominated —
  radix-2 makes `log2 n` passes to radix-4's half, and at complex128 a YMM holds
  only two elements, so there is no width left to recover; the complex128 rows
  at 16/32/64/128 losing to their own SSE2 twins, and n = 16 losing to pure Go,
  is the symptom. (2) The higher-radix rows were _implementation_-limited:
  `dit1024_radix32x32_avx2` is the slowest candidate at its size in both
  directions (8.1×/9.7× `dit1024_radix4_avx2`, slower even than its own pure-Go
  twin) because only one of its two stages is vectorised;
  `dit512_radix16x32_avx2`, `dit512_radix8_avx2` and `dit256_radix16_avx2` are
  the same shape of kernel. (3) The `sixstep` rows were a **stale crossover, not
  a bad kernel** — radix-4 got 2–4× faster and pushed the crossover up; the
  kernels stay for the forced-`KernelSixStep` route.
- **Tier-1 assembly deletion (2026-08-01): 15,191 lines, 24 files, no
  insertions.** A second-order reachability pass — for each 1:1 thunk in
  `internal/fft/asm_amd64.go`, is the _thunk_ called from non-test code? —
  reclassified nine `.s` files (12,983 lines) from "still called" to test-only.
  With them went `internal/kernels/avx2_wrappers.go`, the three unregistered
  AVX2 six-step files and their tests, and 20 `decl.go` declarations. Two
  findings the earlier audit had missed: the c128 generic radix-4 pair was
  unreachable because `forwardAVX2Complex128Asm` calls
  `ForwardAVX2Complex128Asm` directly with no radix cascade; and the AVX2
  six-step files were **never** the forced-six-step route —
  `simdTierServesStrategy` sends a forced `KernelSixStep` to the size-generic
  pure-Go `kernels.ForwardSixStepComplex64`.
- **Audited the `1f7977b` deletions (2026-08-01).** That commit removed ~15,200
  lines of AVX2 assembly alongside the pure-Go radix-16 ladder. Two files were
  restored and wired up: `avx2_f32_transpose{64x64,128x128}.s` (plain, fused
  transpose+twiddle, fused transpose+conj-twiddle), now reachable through the
  out-of-place transpose API in `internal/math` and covered by direct tests for
  the first time — all six symbols verified, the plain transpose bit-exact
  against a naive reference. `avx2_f64_generic_radix4_{even,odd}.s` was restored,
  measured, and kept unregistered behind `-tags fftprobe`. The other five are
  closed dead: `avx2_f32_size{512_radix16x32,512_radix8,256_radix16}.s` and
  `avx2_f64_size{128_radix2,256_radix2}.s`, none of which was a working 256-bit
  kernel (`Y`-vs-`X` operand census in
  [`CODELET_BENCHMARKS.md`](CODELET_BENCHMARKS.md#ruled-out-kernels-deleted-in-1f7977b-and-why)).
- **complex128 generic AVX2: radix-2 beats radix-4 _on this host_
  (2026-08-01).** The restored radix-4 pair was wired into
  `forwardAVX2Complex128Asm`/`inverseAVX2Complex128Asm` with the same
  radix-4 → radix-4-mixed → radix-2 preamble the complex64 twins use, verified
  against `reference.NaiveDFT128`, and confirmed by an instrumented run to
  actually fire. It then lost every size (forward 1.08–1.56, inverse 0.90–2.76
  at 64…8192). The same harness has complex64 radix-4 winning decisively
  (0.87 → 0.54 at 256…8192), so the protocol is not insensitive. Confounders
  ruled out: both precisions pass `nil` for `bitrev`, and the inverse penalty
  survives independently of the `1/n` pass the c128 radix-4 asm omits. Dispatch
  is back to radix-2; the `.s` files stay behind `-tags fftprobe`
  (`internal/fft/radix4_c128_probe_amd64.go`) pending a Xeon sweep.
- **Radix-8, pure Go first (2026-07-30).** Every kernel that had "shown" radix-8
  losing was broken somewhere unrelated to the algorithm:
  `avx2_f32_size512_radix8.s` has **1,905 X-register operands and zero Y
  instructions** despite a header promising 256-bit lanes;
  `avx2_f32_size512_radix16x32.s` likewise (3,862 X, zero Y);
  `avx2_f32_size256_radix16.s` is 256-bit but is a 16×16 matrix factorisation
  with two transposes through scratch, so it tests four-step; and
  `dit512_radix8_generic` spent a full complex multiply on `W_8^2 = −i`.
  `internal/kernels/radix8_generic.go` is the honest test — one size-generic
  ladder per precision over `8^k`, `2·8^k`, `4·8^k`, arch-neutral so the AVX2
  driver shares its twiddle layout and group-index table. **Forward geomean 0.87
  against the pure-Go incumbents**; thirteen rows promoted after the four
  held-back cells were re-measured against `1/n`-free incumbents (2026-07-30,
  46 of 48 groups accepted).
- **A size-generic AVX2 radix-8 stage (2026-07-30).**
  `internal/asm/amd64/avx2_f{32,64}_radix8.s`, ~1,000 lines rather than the 4–5k
  estimated, precisely because Phase 1 was built arch-neutral: the assembly
  derives no permutation table, no twiddle layout and no shape classifier of its
  own. Correct at every supported size in both precisions on the first run. Both
  anticipated risks were non-issues — register pressure closes exactly at 11
  live YMM, and stage 1 is not an 8×8 transpose but two independent 4×4
  transposes of 64-bit lanes interleaved 32 bytes apart. The eighth-root
  multiplies collapse into the rotation the butterfly already does
  (`W_8^1·p = c(p + (−i)p)`, `W_8^3·q = c((−i)q − q)`), so one loop body serves
  both directions. **Seven of sixteen cells won**; the complex64 column is
  decided entirely by the last stage's stride — every cell at ≤ 512 B wins,
  every cell at ≥ 4 KiB loses, no exceptions.
- **Blocking the wide radix-8 stages — tried, measured, reverted (2026-07-30).**
  Eight streams a multiple of 4 KiB apart map to one L1 set, which correlated
  perfectly with every AVX2 complex64 loss. The tiled version (gather 64
  butterflies into a contiguous stack tile, twiddling on the way in, butterfly
  in-tile, copy back, bit-identically) loses **every cell by 6.5–14%**, worst at
  n = 32768 — the cell the collision story predicted it would rescue. The
  general lesson: **a single FFT stage has no reuse to capture.** Each element is
  read once and written once, so blocking cannot remove traffic, only add the
  tile's extra read and write. What survives is that the stride rule is an
  empirical correlation whose _explanation_ is still open, and **capacity is now
  the better guess than conflicts**: at a 4 KiB element stride the stage's eight
  streams span 32 KiB — the whole of L1 — where at 512 B they span 4 KiB and stay
  resident. A capacity limit wants a decomposition that shrinks the span, not a
  tile.
- **The odd-exponent question is settled: it is the tail, not the radix
  (2026-08-01).** An odd-exponent length `n = 2·4^k` is also `8·4^(k-1)`, so the
  principled "specialise the odd exponent" kernel is the existing radix-8 ladder,
  which removes the radix-2 tail where radix-4 can at best fuse it. Swept at the
  last unmeasured cells (`GOOD=5216`, 16 passes, 42 °C, 95 accepted + 1 drift):
  nothing promotes. Two things carry forward. **The tail is the whole remaining
  prize** — `dit<N>_radix4_notail_avx2` measures 0.867–0.933 across all six
  groups, a 6.7–13.3% cost that neither fusion nor radix-8 recovers. And
  **deriving an even/odd split of the general radix-4 is not the lever**:
  `inverse` is tested once per call, `r4End`/`fuse` once per stage, and the hot
  loops carry no knob branches at all.
- **Radix-16: measured in pure Go, and closed (2026-08-01).** The same protocol
  that vindicated radix-8, one radix further up, to decide whether to spend
  assembly on it. `internal/kernels/radix16_generic.go` mirrors
  `radix8_generic.go` stage for stage so the comparison measures the radix and
  not the scaffolding; the butterfly is factored 4×4 (9 complex multiplies
  against the flat form's 120) and all four unrolled copies are pinned against
  the readable original by `TestRadix16LadderMatchesButterfly`. Sweep of
  2026-08-01, 18 groups × 16 passes, 282 accepted + 6 over gate: **not one cell
  wins**, ratios 1.018–1.356, while making 25–33% fewer passes at every size
  except 512. The pass advantage is real, is delivered, and is entirely consumed
  by the butterfly — diminishing passes (log2(n)/4 vs log2(n)/3 is 25% fewer,
  where radix-8 bought 33% over radix-4), growing twiddle cost (15 planes per
  stage against 7), growing gather cost (a 16×16 digit-reversed transpose against
  8×8, quadratic in the radix). This closes radix-16 for **every** instruction
  set: AVX2 has 16 YMM against 16 live streams, and AVX-512's 32 ZMM leave ~12
  scratch — structurally the losing position AVX2 radix-8 was measured in. A
  radix that cannot win where registers are free will not win where they are
  scarce. The ladder stays behind `-tags fftprobe`.
