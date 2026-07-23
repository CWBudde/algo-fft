package transform

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/reference"
	"github.com/cwbudde/algo-fft/internal/registry"
)

// splitStrategy builds a hand-rolled one-level decomposition (size =
// radix × subSize with codelet leaves) so tests can target a specific
// combine path regardless of what PlanDecomposition would score best.
func splitStrategy(size, radix int) *DecomposeStrategy {
	subSize := size / radix

	return &DecomposeStrategy{
		Size:        size,
		SplitFactor: radix,
		SubSize:     subSize,
		NumSubs:     radix,
		Recursive: &DecomposeStrategy{
			Size:       subSize,
			UseCodelet: true,
		},
	}
}

// TestRecursiveInverseRadix2Split exercises the radix-2 conjugate combine
// (combineRadix2Conj) via a forced 2-way split.
func TestRecursiveInverseRadix2Split(t *testing.T) {
	t.Parallel()

	features := cpu.DetectFeatures()
	strategy := splitStrategy(512, 2)

	src := randomComplex64(512, 0x2C0B1)
	twiddle := TwiddleFactorsRecursive[complex64](strategy)
	scratch := make([]complex64, ScratchSizeRecursive(strategy))

	forward := make([]complex64, 512)
	recursiveForward(forward, src, strategy, twiddle, scratch, registry.Registry64, features)

	inverse := make([]complex64, 512)
	recursiveInverse(inverse, forward, strategy, twiddle, scratch, registry.Registry64, features)

	assertComplex64Close(t, inverse, src, 1e-3)
}

// TestRecursiveInverseGeneralRadixSplit exercises the general conjugate
// combine (combineGeneralConj, radix outside 2/4/8) via a forced 16-way
// split, for both precisions (covering both scaleComplexSlice branches).
func TestRecursiveInverseGeneralRadixSplit(t *testing.T) {
	t.Parallel()

	features := cpu.DetectFeatures()
	strategy := splitStrategy(256, 16)

	t.Run("complex64", func(t *testing.T) {
		t.Parallel()

		src := randomComplex64(256, 0x6E11E)
		twiddle := TwiddleFactorsRecursive[complex64](strategy)
		scratch := make([]complex64, ScratchSizeRecursive(strategy))

		forward := make([]complex64, 256)
		recursiveForward(forward, src, strategy, twiddle, scratch, registry.Registry64, features)

		want := reference.NaiveDFT(src)
		assertComplex64Close(t, forward, want, 1e-3)

		inverse := make([]complex64, 256)
		recursiveInverse(inverse, forward, strategy, twiddle, scratch, registry.Registry64, features)

		assertComplex64Close(t, inverse, src, 1e-3)
	})

	t.Run("complex128", func(t *testing.T) {
		t.Parallel()

		src := randomComplex128(256, 0x6E12E)
		twiddle := TwiddleFactorsRecursive[complex128](strategy)
		scratch := make([]complex128, ScratchSizeRecursive(strategy))

		forward := make([]complex128, 256)
		recursiveForward(forward, src, strategy, twiddle, scratch, registry.Registry128, features)

		want := reference.NaiveDFT128(src)
		assertComplex128Close(t, forward, want, 1e-9)

		inverse := make([]complex128, 256)
		recursiveInverse(inverse, forward, strategy, twiddle, scratch, registry.Registry128, features)

		assertComplex128Close(t, inverse, src, 1e-9)
	})
}

// TestRecursiveInverseDITFallbackLeaf exercises the generic-DIT leaf
// fallback (ditInverse) by using leaves of size 2, below the smallest
// registered codelet.
func TestRecursiveInverseDITFallbackLeaf(t *testing.T) {
	t.Parallel()

	features := cpu.DetectFeatures()
	strategy := splitStrategy(4, 2)

	src := randomComplex64(4, 0xD17)
	twiddle := TwiddleFactorsRecursive[complex64](strategy)
	scratch := make([]complex64, ScratchSizeRecursive(strategy))

	forward := make([]complex64, 4)
	recursiveForward(forward, src, strategy, twiddle, scratch, registry.Registry64, features)

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, forward, want, 1e-4)

	inverse := make([]complex64, 4)
	recursiveInverse(inverse, forward, strategy, twiddle, scratch, registry.Registry64, features)

	assertComplex64Close(t, inverse, src, 1e-4)
}

func TestDecomposeStrategyString(t *testing.T) {
	t.Parallel()

	leaf := &DecomposeStrategy{Size: 64, UseCodelet: true}
	if got := leaf.String(); got != "Codelet" {
		t.Errorf("codelet String() = %q, want %q", got, "Codelet")
	}

	split := splitStrategy(256, 4)
	if got := split.String(); got != "Split-4" {
		t.Errorf("split String() = %q, want %q", got, "Split-4")
	}
}
