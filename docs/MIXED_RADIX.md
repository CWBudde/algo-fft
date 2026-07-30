# The mixed-radix engine, Bluestein and Rader

Reference for the non-power-of-two routes: the mixed-radix scheduler and its
fused AVX2 stage kernels, the Bluestein pad model, and the Rader win gates.
`PLAN.md` §5 carries the open work and points here for background.

Moved out of `PLAN.md` on 2026-07-30.

## Where the code lives

| Piece                                                  | Files                                                                                                         |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| Scheduler and pure-Go recursion                        | `internal/fft/mixedradix.go`, `mixedradix_stage_twiddle.go`, `mixedradix_leaf_twiddle.go`                     |
| Butterflies                                            | `internal/kernels/radix{3,4,5,7,8,11}*.go` (radix 2 is inline in `mixedradix.go`)                             |
| Fused AVX2 stage kernels (radix 3/5/7/11, `span >= 4`) | `internal/fft/mixedradix_stage_asm_amd64.go`, `internal/asm/amd64/avx2_f{32,64}_mixedradix_stage{3,5,7,11}.s` |
| Codelet dispatch for whole sub-transforms              | `internal/fft/mixedradix_avx2.go` (`mixedRadixCodeletMinSize = 5`)                                            |
| Bluestein                                              | `plan_exec_bluestein.go`, `plan_padsize.go`, `internal/fft/bluestein.go`, `internal/kernels/bluestein.go`     |
| Rader                                                  | `internal/fft/rader.go`, `plan_exec_rader.go`                                                                 |

## The mixed-radix engine (2026-07)

Started the cycle at 0.20× FFTW3 for 5-smooth and 0.13× for 7/11-smooth
lengths — worse than Rader, which is the newer algorithm.

- **44100 (the canonical audio rate) went from losing to gonum to beating it.**
  Measuring both routes first, as the item required, exposed a **driver
  defect** rather than a routing one: the AVX2 mixed-radix drivers guarded
  codelet dispatch with `n > 1` while the scheduler requires `n > 5`, so every
  schedule ending in radix 2/3/4/5 sent each leaf through a full codelet call —
  strided twiddle gather, two `sync.Pool` round-trips and a `defer` — for a
  handful of butterflies. At n = 4900 that is 1225 dispatches per transform,
  and a profile put **32% of complex64 time inside
  `ForwardAVX2Size4Radix4Complex64Asm`**. Fixed by using the scheduler's own
  bound. Mixed-radix got 18–58% faster at every length measured. With the
  pathology gone the win gate collapsed to one criterion (mixed-radix wins
  where Bluestein's pad is ≥ ~2.5n), so the `pow2 ∈ {2,4} → lose` branch was
  simply deleted. Plan-level at 44100: c128 3.40 → 1.89 ms. One documented
  cost: 22050 c64 inverse regresses ~11% while c128 gains 24–36%; the gate sees
  neither precision nor direction and carving out one size is the overfitting
  that produced the rule being replaced.
- **Dispatch overhead removed from the leaves.** A profile at n = 1000 put only
  **1.9%** of runtime in the codelet assembly and ~40% in dispatch:
  `cpu.DetectFeatures` took two locks per call, `registry.Lookup` an RWMutex,
  and each leaf gathered a twiddle table into a pooled buffer the codelet then
  discarded. Features are now cached in an `atomic.Pointer`, the registry size
  map is copy-on-write behind one (also making the returned `*CodeletEntry`
  stable), and the leaf gather is gone. Geomean −15.0%. Then the leaf codelet
  resolution itself was hoisted out of the recursion — it can only ever match
  at a leaf, and every leaf of one transform has the same size — taking n=1000
  from 156 lookups per transform to 1, geomean −1.9%.
- **Fused AVX2 stage kernels for radix 3, 5, 7 and 11.** One pass: rows 1..r−1
  are multiplied by the stage table and stay in registers through the
  butterfly, never written back. The k index is the vector axis, so the
  butterfly needs no cross-lane movement; direction costs exactly one register
  (the XOR mask). Radix 3/5 gave geomean **−30%**; radix 7 gave **−30.7%** over
  the lengths it reaches (44100 −38…−42%, 2205 −20…−22%); radix 11 gave the
  largest single win of the round — n = 704 dropped **10244 → 2609 ns**
  (c64, −74.5%). Radix 11 no longer fits the register file, so its ten
  butterfly constants sit pre-broadcast in RODATA and are read as FMA memory
  operands, and five row-stride registers reach all ten rows via the SIB scale.
  Every kernel was verified against the definition-based stage reference at
  spans spanning the vector body and the Go tail, in both precisions and
  directions, and with `dst == input` aliasing.
- **The two-pass vectorized stage earns nothing on amd64** and was kept anyway.
  Re-derived on top of the hoist: `off` vs `full` is +50.9% but `nofused` vs
  `full` is +49.7% — the fused kernels carry all of it, and the two-pass form's
  −1.5% does not clear the run's own ±2% noise floor (n = 704 is a built-in
  null control, having no fused stage). Forced to the SSE2 tier it is a net
  **loss**. Retained because it is the only vectorized stage on NEON/WASM/
  purego, tiers this machine cannot measure.
- **`mixedRadixStageMinMuls` stays at 64.** The hypothesis that a kernel paying
  none of the two-pass form's fixed costs would break even sooner is
  **refuted**: 32 is +8.9%, 48 +1.5%, 128 +2.7%. The fused kernel did not shed
  the fixed cost so much as replace it — its prologue broadcasts up to six
  constants and derives up to six row offsets before the vector loop starts.
  Each arm regressed exactly the lengths whose stage set it changes, which is
  what makes that a result rather than noise.
- **Practical DSP lengths are in the internal benchmark set** (1000, 2205,
  3600, 12000, 44100, forward and inverse, both precisions). The lengths where
  the lead over gonum nearly vanished were exactly the ones a DSP user picks,
  and none of them were benchmarked in-tree — which is why the internal numbers
  looked healthy throughout.

44100 is now 786 µs against FFTW3's 185 µs (the 2026-07-29 re-measurement; the 236 µs
recorded here earlier was FFTW measured in a slower window). The gap has closed
from 8.0× to 4.3×.

## The Bluestein sub-FFT never reached the registry (2026-07-29)

The default build was ~4% _slower_ than `-tags purego` at n = 1009 and n = 2003
— the wrong direction for `PLAN.md` §2.1 rule 4, and recorded at the time as wanting a profile
on the theory that a codelet had been selected whose fixed overhead the pad size
could not amortize.

**No codelet had been selected.** `internal/fft/bluestein.go` splits on
`IsPowerOf2(m)`: a non-power-of-two pad runs the mixed-radix engine, which is
fully SIMD-dispatched, while a power-of-two pad called
`kernels.BluesteinConvolution` → `bluesteinSubForward` → the hardcoded size
switch in `internal/kernels/dit.go`, which never consults the codelet registry
and has no build tags. So at those two lengths **both builds ran the identical
pure-Go kernel for ~96% of the work**, and the measured deficit was the chirp
modulation's SIMD call overhead sitting on a path that got no benefit from it.
`dit2048_radix4_avx2`, `dit4096_radix4_avx2` and their complex128 twins were
registered and unreachable. n = 9973 pads to 24576 — not a power of two — which
is the only reason it went the right way.

That the split was on pad _shape_ rather than pad _size_ is what hid it: the two
regressing lengths were not a size class, they were the branch that had no
dispatch.

Fixed by binding the padded sub-FFT at plan time (`newBluesteinSubFFT` in
`plan.go`, `fft.BluesteinSubFFT`), reusing the machinery `kernelExecutor`
already uses: `planner.EstimatePlan` for the padded size, then
`prepareCodeletTwiddles` — which is the part a call-time fix could not have
done, since a codelet may need a prepared twiddle layout and preparing one is
plan-time work. The filter pointwise multiply moved to the SIMD
`ComplexMulArrayInPlace` the mixed-radix branch already used.

Measured on the i7-1255U, three interleaved rounds of two test binaries, medians
in ns/op (n = 9973 is the null control — it keeps the unbound route):

| cell              | before |  after | change |
| ----------------- | -----: | -----: | -----: |
| c64 1009 forward  |  32763 |   5951 |   5.5× |
| c64 1009 inverse  |  32344 |   6401 |   5.1× |
| c128 1009 forward |  32065 |  10885 |   2.9× |
| c64 2003 forward  |  90122 |  12481 |   7.2× |
| c64 2003 inverse  |  89702 |  12837 |   7.0× |
| c128 2003 forward | 103164 |  24916 |   4.1× |
| c64 9973 fwd/inv  | 224310 | 224168 |   1.00 |

The SIMD/purego ratio at 1009 went 0.96 → 5.9 and at 2003 0.96 → 7.6. That the
_baseline_ ratio was ~1.0 rather than the 0.86 originally recorded is itself the
confirmation: two builds measuring the same because they were running the same
code.

Two things worth keeping:

- **Bind only where the registry has a codelet.** The first version fell back to
  the strategy-dispatched kernels when no codelet existed, and that cost ~4% on
  purego at n = 1009 — the unbound size switch is hand-tuned per size, while
  `EstimatePlan`'s heuristic answers Stockham for everything above
  `ditAutoThreshold`, so a codelet-less pad of 2048 traded a tuned
  radix-4-then-2 DIT for plain Stockham. With the binding restricted to a real
  registry hit, purego is flat at 1009 and **15–18% faster** at 2003, where the
  generic codelet beats the switch.
- **A bail is a contract violation here, not a fallback.** The caller passes
  `dst` aliasing `x`, so a kernel that bailed part-way has already destroyed the
  input needed to re-run it. Plan construction verifies the binding end-to-end
  (`fft.VerifyBluesteinSub`) and degrades to the unbound route _there_; the
  transform-time path panics, like `mustMixedRadix`.

_Not done:_ Rader still passes `nil` — its length-(n−1) convolution takes the
same unbound route when n−1 is a power of two (5-smooth but non-power-of-two
n−1 already runs the SIMD mixed-radix engine). Same fix, separate measurement.
