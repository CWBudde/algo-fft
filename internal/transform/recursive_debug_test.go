package transform

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/reference"
)

// TestRecursiveDebug_Size512 - Minimal test to diagnose size-512 codelet failure.
func TestRecursiveDebug_Size512(t *testing.T) {
	size := 512
	codeletSizes := []int{4, 8, 16, 32, 64, 128, 256, 512}
	cacheSize := 32768

	features := cpu.DetectFeatures()
	strategy := PlanDecomposition(size, codeletSizes, cacheSize)

	// EVIDENCE GATHERING: Log strategy details
	t.Logf("=== Strategy for size %d ===", size)
	t.Logf("  UseCodelet: %v", strategy.UseCodelet)
	t.Logf("  SplitFactor: %d", strategy.SplitFactor)
	t.Logf("  SubSize: %d", strategy.SubSize)
	t.Logf("  Size: %d", strategy.Size)

	// EVIDENCE: Check if codelet exists
	codelet := Registry64.Lookup(size, features)
	if codelet == nil {
		t.Errorf("CRITICAL: No codelet found for size %d with features %+v", size, features)
		t.Logf("Available sizes: %v", Registry64.GetAvailableSizes(features))
	} else {
		t.Logf("Codelet found: %s (SIMD: %v, Priority: %d)",
			codelet.Signature, codelet.SIMDLevel, codelet.Priority)
	}

	// Create simple input: impulse at index 0
	input := make([]complex64, size)
	input[0] = complex(1, 0)

	// Generate twiddles
	twiddle := TwiddleFactorsRecursive[complex64](strategy)
	t.Logf("Generated %d twiddle factors", len(twiddle))

	numToShow := 5
	if len(twiddle) < numToShow {
		numToShow = len(twiddle)
	}

	t.Logf("First few twiddles: %v", twiddle[:numToShow])

	// Allocate output and scratch
	output := make([]complex64, size)
	scratch := make([]complex64, ScratchSizeRecursive(strategy))

	// EVIDENCE: Check input before transform
	t.Logf("Input[0] = %v", input[0])
	t.Logf("Output[0] before = %v", output[0])

	// Execute recursive FFT
	recursiveForward(output, input, strategy, twiddle, scratch, Registry64, features)

	// EVIDENCE: Check output after transform
	t.Logf("Output[0] after = %v", output[0])
	t.Logf("Output[1] after = %v", output[1])

	// Check if output is all zeros
	allZero := true

	for i, v := range output {
		if v != 0 {
			allZero = false

			t.Logf("First non-zero output at index %d: %v", i, v)

			break
		}
	}

	if allZero {
		t.Error("CRITICAL: Output is all zeros!")
	}

	// Compare with reference
	expected := reference.NaiveDFT(input)
	t.Logf("Expected[0] = %v", expected[0])
	t.Logf("Expected[1] = %v", expected[1])

	// For impulse input, all DFT outputs should be 1
	if real(expected[0]) < 0.99 || real(expected[0]) > 1.01 {
		t.Errorf("Reference DFT is wrong! Expected[0] = %v, should be ~1", expected[0])
	}
}

// TestRecursiveDebug_DITComparison - Test DIT codelet directly vs recursive call
// TestRecursiveDebug_Size1024 - Minimal test to diagnose size-1024 recursive failure.
func TestRecursiveDebug_Size1024(t *testing.T) {
	t.Parallel()

	size := 1024
	codeletSizes := []int{4, 8, 16, 32, 64, 128, 256, 512}
	cacheSize := 32768

	features := cpu.DetectFeatures()
	strategy := PlanDecomposition(size, codeletSizes, cacheSize)

	t.Logf("=== Strategy for size %d ===", size)
	t.Logf("  UseCodelet: %v", strategy.UseCodelet)
	t.Logf("  SplitFactor: %d", strategy.SplitFactor)
	t.Logf("  SubSize: %d", strategy.SubSize)
	t.Logf("  Depth: %d", strategy.Depth())

	// Create simple input: impulse at index 0
	input := make([]complex64, size)
	input[0] = complex(1, 0)

	// Generate twiddles
	twiddle := TwiddleFactorsRecursive[complex64](strategy)
	t.Logf("Generated %d twiddle factors for recursive decomposition", len(twiddle))

	// EVIDENCE: What should the twiddle structure be?
	expectedCombineTwiddles := strategy.SplitFactor * strategy.SubSize
	expectedSubTwiddles := strategy.SplitFactor * strategy.SubSize // Each sub is size 256, needs 256 twiddles
	expectedTotal := expectedCombineTwiddles + expectedSubTwiddles
	t.Logf("Expected structure: %d combine twiddles + %d sub twiddles = %d total",
		expectedCombineTwiddles, expectedSubTwiddles, expectedTotal)

	// Allocate output and scratch
	output := make([]complex64, size)
	scratch := make([]complex64, ScratchSizeRecursive(strategy))

	// Execute recursive FFT
	recursiveForward(output, input, strategy, twiddle, scratch, Registry64, features)

	// Check output
	t.Logf("Output[0] = %v (expected: (1+0i))", output[0])
	t.Logf("Output[1] = %v (expected: (1+0i))", output[1])

	// Compare with reference
	expected := reference.NaiveDFT(input)
	t.Logf("Expected[0] = %v", expected[0])
	t.Logf("Expected[1] = %v", expected[1])

	// Check first few values
	maxDiff := float32(0)

	for i := 0; i < 10 && i < len(output); i++ {
		diff := cmplx64Abs(output[i] - expected[i])
		if diff > maxDiff {
			maxDiff = diff
		}

		if diff > 1e-5 {
			t.Logf("Index %d: got %v, expected %v, diff=%v", i, output[i], expected[i], diff)
		}
	}

	t.Logf("Max difference in first 10 values: %v", maxDiff)
}

func TestRecursiveDebug_Size1024_Combine(t *testing.T) {
	size := 1024
	codeletSizes := []int{4, 8, 16, 32, 64, 128, 256, 512}
	cacheSize := 32768

	features := cpu.DetectFeatures()
	strategy := PlanDecomposition(size, codeletSizes, cacheSize)

	if strategy == nil || strategy.UseCodelet {
		t.Fatalf("expected recursive strategy for size %d, got %#v", size, strategy)
	}

	radix := strategy.SplitFactor
	subSize := strategy.SubSize

	input := make([]complex64, size)
	input[0] = complex(1, 0)

	twiddle := TwiddleFactorsRecursive[complex64](strategy)

	blockSize := radix * subSize
	if len(twiddle) < blockSize {
		t.Fatalf("twiddle buffer too small: got %d, need at least %d", len(twiddle), blockSize)
	}

	combineBlock := twiddle[:blockSize]
	offset := blockSize

	subResults := make([][]complex64, radix)
	for i := range radix {
		subResults[i] = make([]complex64, subSize)

		subInput := make([]complex64, subSize)
		for j := range subSize {
			subInput[j] = input[i+j*radix]
		}

		subTwiddle := twiddle[offset : offset+subSize]
		offset += subSize

		codelet := Registry64.Lookup(subSize, features)
		if codelet != nil {
			codelet.Forward(subResults[i], subInput, subTwiddle, make([]complex64, subSize))
		} else {
			ditForward(subResults[i], subInput, subTwiddle, make([]complex64, subSize))
		}
	}

	output := make([]complex64, size)

	switch radix {
	case 2:
		tw := combineBlock[subSize : 2*subSize]
		combineRadix2(output, subResults[0], subResults[1], tw)
	case 4:
		tw1 := combineBlock[subSize : 2*subSize]
		tw2 := combineBlock[2*subSize : 3*subSize]
		tw3 := combineBlock[3*subSize : 4*subSize]
		combineRadix4(output, subResults[0], subResults[1], subResults[2], subResults[3], tw1, tw2, tw3)
	case 8:
		twiddles := splitTwiddleBlock(combineBlock, radix, subSize)
		combineRadix8(output, subResults, twiddles)
	default:
		twiddles := splitTwiddleBlock(combineBlock, radix, subSize)
		combineGeneral(output, subResults, twiddles, radix)
	}

	expected := reference.NaiveDFT(input)

	err := compareComplexSlices(output, expected, 1e-5)
	if err != nil {
		t.Errorf("manual combine mismatch: %v", err)
	}

	maxDiff := float32(0)

	var maxIndex int

	for i := range output {
		diff := cmplx64Abs(output[i] - expected[i])
		if diff > maxDiff {
			maxDiff = diff
			maxIndex = i
		}
	}

	t.Logf("manual combine max diff=%v at index %d (got=%v want=%v)",
		maxDiff, maxIndex, output[maxIndex], expected[maxIndex])

	recursiveOutput := make([]complex64, size)
	recursiveForward(recursiveOutput, input, strategy, twiddle, make([]complex64, ScratchSizeRecursive(strategy)), Registry64, features)

	err = compareComplexSlices(recursiveOutput, output, 1e-6)
	if err != nil {
		t.Errorf("recursive output mismatch vs manual combine: %v", err)
	}
}

func TestRecursiveDebug_DITComparison(t *testing.T) {
	t.Parallel()

	size := 512
	features := cpu.DetectFeatures()

	// Create input
	input := make([]complex64, size)
	input[0] = complex(1, 0)

	// Method 1: Call DIT directly
	output1 := make([]complex64, size)
	twiddle1 := ComputeTwiddleFactors[complex64](size)
	scratch1 := make([]complex64, size)

	ditForward(output1, input, twiddle1, scratch1)

	t.Logf("DIT Direct: output[0] = %v", output1[0])

	// Check if DIT output is all zeros
	allZero1 := true

	for _, v := range output1 {
		if v != 0 {
			allZero1 = false
			break
		}
	}

	if allZero1 {
		t.Error("DIT direct call produces all zeros!")
	}

	// Method 2: Call via codelet registry
	codelet := Registry64.Lookup(size, features)
	if codelet == nil {
		t.Fatal("No codelet for size 512")
	}

	output2 := make([]complex64, size)
	twiddle2 := ComputeTwiddleFactors[complex64](size)
	scratch2 := make([]complex64, size)

	codelet.Forward(output2, input, twiddle2, scratch2)

	t.Logf("Codelet call: output[0] = %v", output2[0])

	// Check if codelet output is all zeros
	allZero2 := true

	for _, v := range output2 {
		if v != 0 {
			allZero2 = false
			break
		}
	}

	if allZero2 {
		t.Error("Codelet call produces all zeros!")
	}

	// Compare the two methods
	if output1[0] != output2[0] {
		t.Errorf("DIT direct vs codelet mismatch: %v != %v", output1[0], output2[0])
	}
}

func TestRecursiveDebug_InverseSize512(t *testing.T) {
	size := 512
	features := cpu.DetectFeatures()

	input := make([]complex64, size)
	for i := range input {
		input[i] = complex(float32(i), float32(i*2))
	}

	strategy := PlanDecomposition(size, []int{4, 8, 16, 32, 64, 128, 256, 512}, 32768)
	twiddle := TwiddleFactorsRecursive[complex64](strategy)

	forward := make([]complex64, size)
	inverse := make([]complex64, size)
	recursiveForward(forward, input, strategy, twiddle, make([]complex64, ScratchSizeRecursive(strategy)), Registry64, features)
	recursiveInverse(inverse, forward, strategy, twiddle, make([]complex64, ScratchSizeRecursive(strategy)), Registry64, features)

	err := compareComplexSlices(inverse, input, 1e-3)
	if err != nil {
		t.Errorf("recursive round-trip mismatch: %v", err)
	}

	codelet := Registry64.Lookup(size, features)
	if codelet == nil {
		t.Fatal("no codelet for size 512")
	}

	forwardCodelet := make([]complex64, size)
	inverseCodelet := make([]complex64, size)
	twiddleDirect := ComputeTwiddleFactors[complex64](size)
	scratch := make([]complex64, size)

	codelet.Forward(forwardCodelet, input, twiddleDirect, scratch)
	codelet.Inverse(inverseCodelet, forwardCodelet, twiddleDirect, scratch)

	err = compareComplexSlices(inverseCodelet, input, 1e-3)
	if err != nil {
		t.Errorf("codelet round-trip mismatch: %v", err)
	}

	maxDiffCodelet := float32(0)

	var maxIndexCodelet int

	for i := range inverseCodelet {
		diff := cmplx64Abs(inverseCodelet[i] - input[i])
		if diff > maxDiffCodelet {
			maxDiffCodelet = diff
			maxIndexCodelet = i
		}
	}

	t.Logf("codelet vs input max diff=%v at index %d (got=%v want=%v)",
		maxDiffCodelet, maxIndexCodelet, inverseCodelet[maxIndexCodelet], input[maxIndexCodelet])

	maxDiff := float32(0)

	var maxIndex int

	for i := range inverse {
		diff := cmplx64Abs(inverse[i] - inverseCodelet[i])
		if diff > maxDiff {
			maxDiff = diff
			maxIndex = i
		}
	}

	t.Logf("recursive vs codelet max diff=%v at index %d", maxDiff, maxIndex)
}
