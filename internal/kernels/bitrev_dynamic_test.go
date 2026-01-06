package kernels

import (
	"fmt"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/cpu"
	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// Dynamic bitrev test tolerances.
const (
	dynBitrevTol64  = 1e-4
	dynBitrevTol128 = 1e-10
)

// TestDynamicBitrevUsage_AllCodelets64 tests all registered complex64 codelets
// to verify they dynamically use the passed bitrev array.
func TestDynamicBitrevUsage_AllCodelets64(t *testing.T) {
	t.Parallel()

	features := cpu.DetectFeatures()
	sizes := Registry64.Sizes()

	for _, size := range sizes {
		codelets := Registry64.GetAllForSize(size)
		for _, codelet := range codelets {
			codelet := codelet // capture

			// Skip codelets that don't have a bitrev function (size 4)
			if codelet.BitrevFunc == nil {
				continue
			}

			// Skip codelets that require unsupported CPU features
			if !cpuSupportsCodelet(features, codelet.SIMDLevel) {
				continue
			}

			// Skip very large sizes in short mode (naive DFT is O(n²))
			if testing.Short() && size > 1024 {
				continue
			}

			t.Run(fmt.Sprintf("size_%d/%s", size, codelet.Signature), func(t *testing.T) {
				t.Parallel()
				testCodeletDynamicBitrev64(t, size, codelet)
			})
		}
	}
}

// TestDynamicBitrevUsage_AllCodelets128 tests all registered complex128 codelets.
func TestDynamicBitrevUsage_AllCodelets128(t *testing.T) {
	t.Parallel()

	features := cpu.DetectFeatures()
	sizes := Registry128.Sizes()

	for _, size := range sizes {
		codelets := Registry128.GetAllForSize(size)
		for _, codelet := range codelets {
			codelet := codelet // capture

			// Skip codelets that don't have a bitrev function (size 4)
			if codelet.BitrevFunc == nil {
				continue
			}

			// Skip codelets that require unsupported CPU features
			if !cpuSupportsCodelet(features, codelet.SIMDLevel) {
				continue
			}

			// Skip very large sizes in short mode
			if testing.Short() && size > 1024 {
				continue
			}

			t.Run(fmt.Sprintf("size_%d/%s", size, codelet.Signature), func(t *testing.T) {
				t.Parallel()
				testCodeletDynamicBitrev128(t, size, codelet)
			})
		}
	}
}

// testCodeletDynamicBitrev64 tests a single complex64 codelet for dynamic bitrev usage.
func testCodeletDynamicBitrev64(t *testing.T, size int, codelet CodeletEntry[complex64]) {
	t.Helper()

	// Generate test input
	input := randomComplex64(size, uint64(0xB17E0000)+uint64(size))

	// Get normal bitrev for this codelet
	normalBitrev := codelet.BitrevFunc(size)
	if normalBitrev == nil {
		t.Skip("codelet has no bitrev function")
	}

	// Create scrambled bitrev by swapping adjacent pairs
	scrambledBitrev := make([]int, size)
	for i := range size {
		if i%2 == 0 && i+1 < size {
			scrambledBitrev[i] = normalBitrev[i+1]
			scrambledBitrev[i+1] = normalBitrev[i]
		}
	}

	// Build source that produces same logical sequence when accessed with scrambledBitrev
	src := make([]complex64, size)
	for i := range size {
		src[scrambledBitrev[i]] = input[normalBitrev[i]]
	}

	// Allocate buffers
	dst := make([]complex64, size)
	scratch := make([]complex64, size)
	twiddle := ComputeTwiddleFactors[complex64](size)

	// Run forward transform with scrambled bitrev
	codelet.Forward(dst, src, twiddle, scratch, scrambledBitrev)

	// Expected: FFT of input using naive DFT
	expected := reference.NaiveDFT(input)

	// Tolerance scales with size
	tol := dynBitrevTol64 * float64(size) / 16.0
	if tol < dynBitrevTol64 {
		tol = dynBitrevTol64
	}
	if tol > 1e-2 {
		tol = 1e-2
	}

	// Verify results match
	assertComplex64Close(t, dst, expected, tol)

	// Additional check: using wrong bitrev should give different result (for non-trivial sizes)
	if size > 4 {
		dst2 := make([]complex64, size)
		codelet.Forward(dst2, src, twiddle, scratch, normalBitrev)

		// dst2 should NOT match dst (since we used wrong bitrev for the scrambled src)
		matches := true
		for i := range size {
			if dst[i] != dst2[i] {
				matches = false
				break
			}
		}
		if matches {
			t.Errorf("Using different bitrev arrays produced identical results - codelet %s may not be using bitrev dynamically", codelet.Signature)
		}
	}
}

// testCodeletDynamicBitrev128 tests a single complex128 codelet for dynamic bitrev usage.
func testCodeletDynamicBitrev128(t *testing.T, size int, codelet CodeletEntry[complex128]) {
	t.Helper()

	// Generate test input
	input := randomComplex128(size, uint64(0xB17E0000)+uint64(size))

	// Get normal bitrev for this codelet
	normalBitrev := codelet.BitrevFunc(size)
	if normalBitrev == nil {
		t.Skip("codelet has no bitrev function")
	}

	// Create scrambled bitrev by swapping adjacent pairs
	scrambledBitrev := make([]int, size)
	for i := range size {
		if i%2 == 0 && i+1 < size {
			scrambledBitrev[i] = normalBitrev[i+1]
			scrambledBitrev[i+1] = normalBitrev[i]
		}
	}

	// Build source that produces same logical sequence when accessed with scrambledBitrev
	src := make([]complex128, size)
	for i := range size {
		src[scrambledBitrev[i]] = input[normalBitrev[i]]
	}

	// Allocate buffers
	dst := make([]complex128, size)
	scratch := make([]complex128, size)
	twiddle := ComputeTwiddleFactors[complex128](size)

	// Run forward transform with scrambled bitrev
	codelet.Forward(dst, src, twiddle, scratch, scrambledBitrev)

	// Expected: FFT of input using naive DFT
	expected := reference.NaiveDFT128(input)

	// Tolerance scales with size
	tol := dynBitrevTol128 * float64(size) / 16.0
	if tol < dynBitrevTol128 {
		tol = dynBitrevTol128
	}
	if tol > 1e-6 {
		tol = 1e-6
	}

	// Verify results match
	assertComplex128Close(t, dst, expected, tol)

	// Additional check: using wrong bitrev should give different result
	if size > 4 {
		dst2 := make([]complex128, size)
		codelet.Forward(dst2, src, twiddle, scratch, normalBitrev)

		matches := true
		for i := range size {
			if dst[i] != dst2[i] {
				matches = false
				break
			}
		}
		if matches {
			t.Errorf("Using different bitrev arrays produced identical results - codelet %s may not be using bitrev dynamically", codelet.Signature)
		}
	}
}

// cpuSupportsCodelet checks if the CPU supports the required SIMD level for a codelet.
func cpuSupportsCodelet(features cpu.Features, level SIMDLevel) bool {
	switch level {
	case SIMDNone:
		return true
	case SIMDSSE2:
		return features.HasSSE2
	case SIMDAVX2:
		return features.HasAVX2
	case SIMDAVX512:
		return features.HasAVX512
	case SIMDNEON:
		return features.HasNEON
	default:
		return false
	}
}

// TestDynamicBitrevUsage_Size16Radix4_Complex64 verifies that the size-16 radix-4 kernel
// dynamically uses the passed bitrev array rather than hardcoding indices.
// This is essential for composing larger FFTs from smaller kernels (e.g., six-step algorithm).
//
// The test verifies kernels use br[i] to index into src, not hardcoded i values.
// We pass a permuted input where the data at bitrev positions has been pre-arranged,
// and verify the kernel produces the correct DFT result.
func TestDynamicBitrevUsage_Size16Radix4_Complex64(t *testing.T) {
	t.Parallel()

	const n = 16

	// Create random input
	input1 := randomComplex64(n, 0xABCDEF01)

	// Standard radix-4 bitrev for size 16
	normalBitrev := mathpkg.ComputeBitReversalIndicesRadix4(n)

	// Create a "scrambled" bitrev that's different from normal
	// Swap pairs: [0,8,4,12,...] -> [8,0,12,4,...]
	scrambledBitrev := make([]int, n)
	for i := range n {
		// Swap adjacent pairs in bitrev
		if i%2 == 0 && i+1 < n {
			scrambledBitrev[i] = normalBitrev[i+1]
			scrambledBitrev[i+1] = normalBitrev[i]
		}
	}

	// Build a source array that, when accessed with scrambledBitrev,
	// produces the same logical sequence as input1 accessed with normalBitrev
	src := make([]complex64, n)
	for i := range n {
		// src[scrambledBitrev[i]] should equal input1[normalBitrev[i]]
		// So when kernel does src[br[i]] with scrambledBitrev, it gets input1[normalBitrev[i]]
		src[scrambledBitrev[i]] = input1[normalBitrev[i]]
	}

	// Now FFT(src) with scrambledBitrev should equal FFT(input1) with normalBitrev
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT16Radix4Complex64(dst, src, twiddle, scratch, scrambledBitrev) {
		t.Fatal("forwardDIT16Radix4Complex64 with scrambled bitrev failed")
	}

	// Compute expected using naive DFT on input1
	expected := reference.NaiveDFT(input1)

	// Verify results match
	assertComplex64Close(t, dst, expected, dynBitrevTol64)

	// Additional check: using wrong bitrev should give different result
	dst2 := make([]complex64, n)
	if !forwardDIT16Radix4Complex64(dst2, src, twiddle, scratch, normalBitrev) {
		t.Fatal("forwardDIT16Radix4Complex64 with normal bitrev failed")
	}

	// dst2 should NOT match expected (since we used wrong bitrev for the scrambled src)
	matches := true
	for i := range n {
		if dst[i] != dst2[i] {
			matches = false
			break
		}
	}
	if matches {
		t.Error("Using different bitrev arrays produced identical results - kernel may not be using bitrev dynamically")
	}
}

// TestDynamicBitrevUsage_Size16Radix4_Complex128 verifies dynamic bitrev usage for complex128.
func TestDynamicBitrevUsage_Size16Radix4_Complex128(t *testing.T) {
	t.Parallel()

	const n = 16

	input1 := randomComplex128(n, 0xABCDEF02)
	normalBitrev := mathpkg.ComputeBitReversalIndicesRadix4(n)

	// Create scrambled bitrev by swapping adjacent pairs
	scrambledBitrev := make([]int, n)
	for i := range n {
		if i%2 == 0 && i+1 < n {
			scrambledBitrev[i] = normalBitrev[i+1]
			scrambledBitrev[i+1] = normalBitrev[i]
		}
	}

	// Build source that produces same logical sequence when accessed with scrambledBitrev
	src := make([]complex128, n)
	for i := range n {
		src[scrambledBitrev[i]] = input1[normalBitrev[i]]
	}

	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT16Radix4Complex128(dst, src, twiddle, scratch, scrambledBitrev) {
		t.Fatal("forwardDIT16Radix4Complex128 with scrambled bitrev failed")
	}

	expected := reference.NaiveDFT128(input1)
	assertComplex128Close(t, dst, expected, dynBitrevTol128)
}

// TestDynamicBitrevUsage_Size16Radix2_Complex64 verifies dynamic bitrev usage for radix-2.
func TestDynamicBitrevUsage_Size16Radix2_Complex64(t *testing.T) {
	t.Parallel()

	const n = 16

	input1 := randomComplex64(n, 0xABCDEF03)
	normalBitrev := mathpkg.ComputeBitReversalIndices(n) // Radix-2

	// Create scrambled bitrev by swapping adjacent pairs
	scrambledBitrev := make([]int, n)
	for i := range n {
		if i%2 == 0 && i+1 < n {
			scrambledBitrev[i] = normalBitrev[i+1]
			scrambledBitrev[i+1] = normalBitrev[i]
		}
	}

	// Build source that produces same logical sequence when accessed with scrambledBitrev
	src := make([]complex64, n)
	for i := range n {
		src[scrambledBitrev[i]] = input1[normalBitrev[i]]
	}

	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT16Complex64(dst, src, twiddle, scratch, scrambledBitrev) {
		t.Fatal("forwardDIT16Complex64 with scrambled bitrev failed")
	}

	expected := reference.NaiveDFT(input1)
	assertComplex64Close(t, dst, expected, dynBitrevTol64)
}

// TestDynamicBitrevUsage_Size64Radix4_Complex64 verifies dynamic bitrev usage for size-64.
// Larger sizes are important to ensure the pattern holds beyond small test cases.
func TestDynamicBitrevUsage_Size64Radix4_Complex64(t *testing.T) {
	t.Parallel()

	const n = 64

	input1 := randomComplex64(n, 0xABCDEF04)
	normalBitrev := mathpkg.ComputeBitReversalIndicesRadix4(n)

	// Create scrambled bitrev by swapping adjacent pairs
	scrambledBitrev := make([]int, n)
	for i := range n {
		if i%2 == 0 && i+1 < n {
			scrambledBitrev[i] = normalBitrev[i+1]
			scrambledBitrev[i+1] = normalBitrev[i]
		}
	}

	// Build source that produces same logical sequence when accessed with scrambledBitrev
	src := make([]complex64, n)
	for i := range n {
		src[scrambledBitrev[i]] = input1[normalBitrev[i]]
	}

	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT64Radix4Complex64(dst, src, twiddle, scratch, scrambledBitrev) {
		t.Fatal("forwardDIT64Radix4Complex64 with scrambled bitrev failed")
	}

	expected := reference.NaiveDFT(input1)

	// Slightly higher tolerance for larger sizes
	assertComplex64Close(t, dst, expected, 5e-4)
}

// TestDynamicBitrevUsage_ZeroOffset verifies that normal bitrev still works.
// This is a sanity check that our scrambled test isn't passing for the wrong reasons.
func TestDynamicBitrevUsage_ZeroOffset(t *testing.T) {
	t.Parallel()

	const n = 16

	src := randomComplex64(n, 0xABCDEF05)
	bitrev := mathpkg.ComputeBitReversalIndicesRadix4(n)

	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT16Radix4Complex64(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT16Radix4Complex64 with normal bitrev failed")
	}

	expected := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, expected, dynBitrevTol64)
}

// TestDynamicBitrevUsage_IdentityBitrev tests that passing identity permutation [0,1,2,...]
// produces different results than the correct bitrev, proving the kernel uses the array.
func TestDynamicBitrevUsage_IdentityBitrev(t *testing.T) {
	t.Parallel()

	const n = 16

	src := randomComplex64(n, 0xABCDEF06)
	normalBitrev := mathpkg.ComputeBitReversalIndicesRadix4(n)
	identityBitrev := mathpkg.ComputeIdentityIndices(n)

	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	// FFT with correct bitrev
	dstCorrect := make([]complex64, n)
	if !forwardDIT16Radix4Complex64(dstCorrect, src, twiddle, scratch, normalBitrev) {
		t.Fatal("forwardDIT16Radix4Complex64 with normal bitrev failed")
	}

	// FFT with identity bitrev (wrong for radix-4)
	dstIdentity := make([]complex64, n)
	if !forwardDIT16Radix4Complex64(dstIdentity, src, twiddle, scratch, identityBitrev) {
		t.Fatal("forwardDIT16Radix4Complex64 with identity bitrev failed")
	}

	// Results should differ (proving bitrev is actually used)
	matches := true
	for i := range n {
		if dstCorrect[i] != dstIdentity[i] {
			matches = false
			break
		}
	}

	if matches {
		t.Error("Identity and normal bitrev produced identical results - kernel may not be using bitrev")
	}

	// The correct one should match naive DFT
	expected := reference.NaiveDFT(src)
	assertComplex64Close(t, dstCorrect, expected, dynBitrevTol64)
}
