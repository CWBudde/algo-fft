# Testing strategy and the test-vector blindness audit

Why the obvious FFT test vectors detect nothing, what the suite was changed to
use instead, and the tolerance conventions. `CONTRIBUTING.md` carries the
process; this file carries the reasoning.

Moved out of `PLAN.md` on 2026-07-30. The one-line rule ("test vectors
must not be blind") stays in `PLAN.md` §2.3.

## Test-vector blindness audit (2026-07-29)

An impulse (`x[0] = 1`) is the most common FFT test vector and one of the
weakest: its spectrum is all-ones, so every twiddle multiplies a zero and every
permutation of the output is still all-ones. Wrong twiddles and wrong bin
ordering — the two ways this library has actually been wrong — both pass, and so
do Parseval and linearity, because linearity holds for any linear operator
including the wrong one. That combination hid a wrong-spectrum bug in the
recursive `complex128` path at every size ≥ 1024.

The suite was audited for that pattern. Four tests were checking nothing, and
one whole precision was uncovered above `n = 16`:

- **The recursive path was still impulse-only.**
  `TestRecursiveFFTCorrectness` (512…16384) and `TestRecursiveFFTSmallSizes`
  drive an impulse; `TestRecursiveFFTComplex128` uses a ramp at one size. New
  `internal/transform/recursive_broadband_test.go` adds the companions: a
  broadband signal compared bin-by-bin against `internal/reference` at
  512…4096, and — for 8192/16384, where the O(n²) reference costs 1.1 s and
  4.3 s — a sum of complex exponentials whose spectrum is exact in closed form,
  which costs O(n) and still catches both blind classes (a wrong twiddle leaks
  into the zero bins, a wrong ordering moves the spikes). Both precisions.
- **`complex128` plans had no broadband reference check above n = 16.**
  `TestPlannerModesMatchReference` sweeps `complex64` to 16384, but the wider
  precision was covered only at `n ≤ 16` and at non-power-of-two lengths — the
  gap the recursive bug lived in. New `plan_broadband_reference_test.go` sweeps
  every forceable strategy (auto, DIT, Stockham, split-radix, recursive,
  six-step, four-step, Bluestein) at 256/1024/4096 against the reference,
  forward and inverse, both precisions. 72 subtests.
- **`TestBluesteinHelper` asserted only "output is not all zeros".** No wrong
  filter, twiddle table or bin order could fail it. It now runs the full
  Bluestein assembly (pre-chirp, cyclic convolution, post-chirp — the chirp
  multiplies live in `internal/fft`, so the test does them inline) and compares
  against the naive DFT at n = 3/5/12.
- **The mixed-radix engine was tested at one size with one ramp.**
  `internal/fft/mixedradix_test.go` now sweeps 15 lengths covering every radix
  the scheduler emits (2/3/4/5/7/8/11) with a broadband signal, both
  precisions, forward against the reference and round-trip.
- **`testPrecisionComparison` measured precision on two bins.** Its input was a
  real sine at bin 5 — a lattice frequency, so the spectrum is two nonzero bins
  and n−2 zeros, and the relative-difference loop skipped everything under
  1e-10. Now broadband, bounded relative to the spectrum peak.
- **`testLargePrecision` measured precision on an impulse** at 65536…262144,
  where "max error" was structurally 0. Now the closed-form multi-tone
  spectrum; the tolerance tightened from 1e-11·n to 5e-14·n and still holds
  with ~40× headroom.
- Smaller: the four-step fallback test, `TestClone_Forward`,
  `TestClone_InPlaceRoundTrip`, `TestPlan_Clone*` all asserted properties an
  impulse cannot distinguish (a clone that lost its parent's prepared twiddle
  layout still returns all-ones), and now use broadband inputs.

Two constructions matter for reuse. The multi-tone reference must reduce
`k·j mod n` before scaling to radians — at n = 262144 the unreduced angle
reaches ~1.6e6 rad, where `cmplx.Exp`'s argument reduction alone costs ~1e-10 of
phase and the "exact" expected spectrum stops being exact (this is what made
the first tightened tolerance fail). And `complex64` results are compared
against `reference.NaiveDFTWide`, which accumulates in float64, so the
comparison is not limited by the reference's own rounding.

The new tests were mutation-checked by perturbing the shared twiddle table
(`internal/math.ComputeTwiddleFactors`) by one part in 1e4. Every broadband and
multi-tone test that routes through that table fails; `TestForward_Impulse_2048`
passes, which is the point. The clone tests at n = 128/256 also pass under that
mutation, because those plans bind a codelet carrying its own tables — the
mutation does not reach them, so their coverage rests on the reference
comparison rather than on this check.

One byproduct: the sweep first failed only in a full-package run and passed
under `-run`. `internal/fft/measure_codelet_test.go` registers **identity** stub
codelets at n = 1155 and n = 1331, and the codelet registry is process-global and
append-only, so every later transform at those lengths copies its input to its
output and reports success. Any correctness test that picks one of those lengths
is silently vacuous. The sweep avoids them and both files now say why.

Clean by construction and left alone: `internal/kernels/codelet_reference_all_test.go`
(impulse at a nonzero position, whose spectrum is a full twiddle sequence, plus
tones and a random-input naive check to 2048), the Rader tests (random input,
bin-by-bin against the reference), plan-level Bluestein (`i² + i·j`) and
`TestPrecisionRoundTripSweep` (random). The remaining impulse uses are in tests
of plan lifecycle, pooling and allocation counts, where the input is irrelevant.
