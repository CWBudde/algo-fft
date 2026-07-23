package transform

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/planner"
	"github.com/cwbudde/algo-fft/internal/reference"
)

// bailingRegistry returns a registry whose only codelet models a kernel that
// bails without doing any work (e.g. an undersized-slice guard). The recursive
// executor must detect this and fall back to the generic DIT path instead of
// silently returning wrong output. Regression test for PLAN.md A0.
func bailingRegistry(size int) *planner.CodeletRegistry[complex64] {
	reg := planner.NewCodeletRegistry[complex64]()
	reg.Register(planner.CodeletEntry[complex64]{
		Size: size,
		Forward: func(dst, src, twiddle, scratch []complex64) bool {
			return false // bails: does no work
		},
		Inverse: func(dst, src, twiddle, scratch []complex64) bool {
			return false // bails: does no work
		},
		Signature: "bailing_test_codelet",
	})

	return reg
}

func rampInput(n int) []complex64 {
	input := make([]complex64, n)
	for i := range input {
		input[i] = complex(float32(i%5)*0.25+0.5, float32(i%3)*0.125-0.25)
	}

	return input
}

func TestRecursiveForward_BailingCodeletFallsBack(t *testing.T) {
	t.Parallel()

	features := cpu.DetectFeatures()

	// n=8 exercises the codelet leaf directly; n=16 exercises a composite
	// split whose leaves hit the bailing codelet.
	for _, n := range []int{8, 16} {
		t.Run(formatSize(n), func(t *testing.T) {
			t.Parallel()

			strategy := PlanDecomposition(n, []int{8}, 32768)
			twiddle := TwiddleFactorsRecursive[complex64](strategy)
			scratch := make([]complex64, ScratchSizeRecursive(strategy))
			input := rampInput(n)
			output := make([]complex64, n)

			recursiveForward(output, input, strategy, twiddle, scratch, bailingRegistry(8), features)

			expected := reference.NaiveDFT(input)

			err := compareComplexSlices(output, expected, 1e-4)
			if err != nil {
				t.Errorf("bailing codelet produced silent wrong output instead of falling back: %v", err)
			}
		})
	}
}

func TestRecursiveInverse_BailingCodeletFallsBack(t *testing.T) {
	t.Parallel()

	features := cpu.DetectFeatures()

	for _, n := range []int{8, 16} {
		t.Run(formatSize(n), func(t *testing.T) {
			t.Parallel()

			strategy := PlanDecomposition(n, []int{8}, 32768)
			twiddle := TwiddleFactorsRecursive[complex64](strategy)
			scratch := make([]complex64, ScratchSizeRecursive(strategy))
			input := rampInput(n)
			freq := reference.NaiveDFT(input)
			output := make([]complex64, n)

			recursiveInverse(output, freq, strategy, twiddle, scratch, bailingRegistry(8), features)

			err := compareComplexSlices(output, input, 1e-4)
			if err != nil {
				t.Errorf("bailing codelet produced silent wrong output instead of falling back: %v", err)
			}
		})
	}
}
