package kernels

import (
	"fmt"
	"math"
	"math/cmplx"
	"runtime"
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/planner"
	"github.com/cwbudde/algo-fft/internal/reference"
)

// These sweeps validate every registered codelet against independent ground
// truth, direction by direction. The round-trip sweep alone cannot catch a
// compensating forward/inverse bug (a wrong forward whose inverse undoes the
// same mistake passes round-trip), so forward and inverse are each compared
// against analytically known spectra for all sizes, plus a naive-DFT check on
// random input for sizes where the O(n²) reference is affordable.

// naiveReferenceMaxSize bounds the O(n²) naive-DFT random-input check.
const naiveReferenceMaxSize = 2048

// referencePattern pairs a time-domain input with its known spectrum.
type referencePattern[T Complex] struct {
	name     string
	input    []T
	spectrum []T
}

// impulsePattern returns an impulse at position pos; its DFT is the pure
// twiddle sequence X[k] = exp(-2πi·k·pos/n), computed in closed form.
func impulsePattern[T Complex](n, pos int) referencePattern[T] {
	input := make([]T, n)
	input[pos] = complexFrom[T](1, 0)

	spectrum := make([]T, n)
	for k := range n {
		angle := -2 * math.Pi * float64(k) * float64(pos) / float64(n)
		spectrum[k] = complexFrom[T](math.Cos(angle), math.Sin(angle))
	}

	return referencePattern[T]{
		name:     fmt.Sprintf("impulse_%d", pos),
		input:    input,
		spectrum: spectrum,
	}
}

// tonePattern returns the complex exponential x[m] = exp(2πi·bin·m/n); its
// DFT is n·δ[k−bin], computed in closed form.
func tonePattern[T Complex](n, bin int) referencePattern[T] {
	input := make([]T, n)
	for m := range n {
		angle := 2 * math.Pi * float64(bin) * float64(m) / float64(n)
		input[m] = complexFrom[T](math.Cos(angle), math.Sin(angle))
	}

	spectrum := make([]T, n)
	spectrum[bin] = complexFrom[T](float64(n), 0)

	return referencePattern[T]{
		name:     fmt.Sprintf("tone_%d", bin),
		input:    input,
		spectrum: spectrum,
	}
}

func complexFrom[T Complex](re, im float64) T {
	var zero T
	if _, ok := any(zero).(complex64); ok {
		return any(complex(float32(re), float32(im))).(T)
	}

	return any(complex(re, im)).(T)
}

// referencePatterns64 assembles the per-size reference set for complex64.
func referencePatterns64(n int) []referencePattern[complex64] {
	patterns := []referencePattern[complex64]{impulsePattern[complex64](n, 1%n)}
	if n > 3 {
		patterns = append(patterns, tonePattern[complex64](n, 3))
	}

	if n <= naiveReferenceMaxSize {
		input := make([]complex64, n)
		for i := range input {
			input[i] = complex(float32(i%7-3), float32(i%5-2))
		}

		patterns = append(patterns, referencePattern[complex64]{
			name:     "random_naive",
			input:    input,
			spectrum: reference.NaiveDFT(input),
		})
	}

	return patterns
}

// referencePatterns128 assembles the per-size reference set for complex128.
func referencePatterns128(n int) []referencePattern[complex128] {
	patterns := []referencePattern[complex128]{impulsePattern[complex128](n, 1%n)}
	if n > 3 {
		patterns = append(patterns, tonePattern[complex128](n, 3))
	}

	if n <= naiveReferenceMaxSize {
		input := make([]complex128, n)
		for i := range input {
			input[i] = complex(float64(i%7-3), float64(i%5-2))
		}

		patterns = append(patterns, referencePattern[complex128]{
			name:     "random_naive",
			input:    input,
			spectrum: reference.NaiveDFT128(input),
		})
	}

	return patterns
}

// maxNormalizedError returns max|got-want| / max(1, max|want|).
func maxNormalizedError[T Complex](got, want []T) float64 {
	maxErr := 0.0
	maxRef := 1.0

	for i := range want {
		err := cmplx.Abs(complex128(got[i]) - complex128(want[i]))
		if err > maxErr {
			maxErr = err
		}

		mag := cmplx.Abs(complex128(want[i]))
		if mag > maxRef {
			maxRef = mag
		}
	}

	return maxErr / maxRef
}

// TestForwardInverseAllCodeletsVsReference64 checks every runnable complex64
// codelet against independent reference spectra in both directions.
func TestForwardInverseAllCodeletsVsReference64(t *testing.T) {
	t.Parallel()

	features := cpu.DetectFeatures()

	checked := 0

	for _, size := range Registry64.Sizes() {
		patterns := referencePatterns64(size)

		for _, entry := range Registry64.GetAllForSize(size) {
			if entry.Priority < 0 || !cpuSupportsLevel(features, entry.SIMDLevel) {
				continue
			}

			checked++

			t.Run(fmt.Sprintf("size%d/%s", size, entry.Signature), func(t *testing.T) {
				t.Parallel()
				testCodeletVsReference64(t, &entry, patterns)
			})
		}
	}

	if checked == 0 {
		t.Fatal("no runnable codelets found in Registry64 — registry sweep is vacuous")
	}
}

// TestForwardInverseAllCodeletsVsReference128 is the complex128 counterpart.
func TestForwardInverseAllCodeletsVsReference128(t *testing.T) {
	t.Parallel()

	features := cpu.DetectFeatures()

	checked := 0

	for _, size := range Registry128.Sizes() {
		patterns := referencePatterns128(size)

		for _, entry := range Registry128.GetAllForSize(size) {
			if entry.Priority < 0 || !cpuSupportsLevel(features, entry.SIMDLevel) {
				continue
			}

			checked++

			t.Run(fmt.Sprintf("size%d/%s", size, entry.Signature), func(t *testing.T) {
				t.Parallel()
				testCodeletVsReference128(t, &entry, patterns)
			})
		}
	}

	if checked == 0 {
		t.Fatal("no runnable codelets found in Registry128 — registry sweep is vacuous")
	}
}

func testCodeletVsReference64(t *testing.T, entry *planner.CodeletEntry[complex64], patterns []referencePattern[complex64]) {
	t.Helper()

	size := entry.Size
	twiddle := ComputeTwiddleFactors[complex64](size)
	twiddleForward, twiddleInverse, forwardBacking, inverseBacking := prepareCodeletTwiddles64(size, twiddle, entry)

	const tol = 2e-3

	dst := make([]complex64, size)
	scratch := make([]complex64, size)
	src := make([]complex64, size)

	for _, pattern := range patterns {
		if entry.Forward != nil {
			copy(src, pattern.input)
			clear(dst)
			entry.Forward(dst, src, twiddleForward, scratch)

			if err := maxNormalizedError(dst, pattern.spectrum); err > tol {
				t.Errorf("%s forward: normalized error %v exceeds %v", pattern.name, err, tol)
			}
		}

		if entry.Inverse != nil {
			copy(src, pattern.spectrum)
			clear(dst)
			entry.Inverse(dst, src, twiddleInverse, scratch)

			if err := maxNormalizedError(dst, pattern.input); err > tol {
				t.Errorf("%s inverse: normalized error %v exceeds %v", pattern.name, err, tol)
			}
		}
	}

	runtime.KeepAlive(forwardBacking)
	runtime.KeepAlive(inverseBacking)
}

func testCodeletVsReference128(t *testing.T, entry *planner.CodeletEntry[complex128], patterns []referencePattern[complex128]) {
	t.Helper()

	size := entry.Size
	twiddle := ComputeTwiddleFactors[complex128](size)
	twiddleForward, twiddleInverse, forwardBacking, inverseBacking := prepareCodeletTwiddles128(size, twiddle, entry)

	const tol = 1e-9

	dst := make([]complex128, size)
	scratch := make([]complex128, size)
	src := make([]complex128, size)

	for _, pattern := range patterns {
		if entry.Forward != nil {
			copy(src, pattern.input)
			clear(dst)
			entry.Forward(dst, src, twiddleForward, scratch)

			if err := maxNormalizedError(dst, pattern.spectrum); err > tol {
				t.Errorf("%s forward: normalized error %v exceeds %v", pattern.name, err, tol)
			}
		}

		if entry.Inverse != nil {
			copy(src, pattern.spectrum)
			clear(dst)
			entry.Inverse(dst, src, twiddleInverse, scratch)

			if err := maxNormalizedError(dst, pattern.input); err > tol {
				t.Errorf("%s inverse: normalized error %v exceeds %v", pattern.name, err, tol)
			}
		}
	}

	runtime.KeepAlive(forwardBacking)
	runtime.KeepAlive(inverseBacking)
}

// TestCodeletRegistrySignaturesUnique asserts every registered codelet carries
// a unique, non-empty signature per size, so sweep failures are attributable.
func TestCodeletRegistrySignaturesUnique(t *testing.T) {
	t.Parallel()

	for _, size := range Registry64.Sizes() {
		seen := map[string]bool{}

		for _, entry := range Registry64.GetAllForSize(size) {
			if entry.Signature == "" {
				t.Errorf("complex64 size %d: entry with empty signature", size)
			}

			if seen[entry.Signature] {
				t.Errorf("complex64 size %d: duplicate signature %q", size, entry.Signature)
			}

			seen[entry.Signature] = true
		}
	}

	for _, size := range Registry128.Sizes() {
		seen := map[string]bool{}

		for _, entry := range Registry128.GetAllForSize(size) {
			if entry.Signature == "" {
				t.Errorf("complex128 size %d: entry with empty signature", size)
			}

			if seen[entry.Signature] {
				t.Errorf("complex128 size %d: duplicate signature %q", size, entry.Signature)
			}

			seen[entry.Signature] = true
		}
	}
}
