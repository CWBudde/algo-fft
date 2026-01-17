package kernels

import (
	"fmt"
	"math/cmplx"
	"runtime"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/cpu"
	"github.com/MeKo-Christian/algo-fft/internal/memory"
	"github.com/MeKo-Christian/algo-fft/internal/planner"
)

// TestRoundTripAllCodelets64 tests that every registered complex64 codelet passes
// the forward -> inverse roundtrip test. This ensures all radix variants work correctly.
func TestRoundTripAllCodelets64(t *testing.T) {
	t.Parallel()

	sizes := Registry64.Sizes()
	features := cpu.DetectFeatures()

	for _, size := range sizes {
		entries := Registry64.GetAllForSize(size)
		for _, entry := range entries {
			// Skip disabled codelets (negative priority)
			if entry.Priority < 0 {
				continue
			}

			// Skip codelets that require unsupported CPU features
			if !cpuSupportsLevel(features, entry.SIMDLevel) {
				continue
			}

			t.Run(fmt.Sprintf("size%d/%s", size, entry.Signature), func(t *testing.T) {
				t.Parallel()
				testRoundTripCodelet64(t, &entry)
			})
		}
	}
}

// TestRoundTripAllCodelets128 tests that every registered complex128 codelet passes
// the forward -> inverse roundtrip test.
func TestRoundTripAllCodelets128(t *testing.T) {
	t.Parallel()

	sizes := Registry128.Sizes()
	features := cpu.DetectFeatures()

	for _, size := range sizes {
		entries := Registry128.GetAllForSize(size)
		for _, entry := range entries {
			// Skip disabled codelets (negative priority)
			if entry.Priority < 0 {
				continue
			}

			// Skip codelets that require unsupported CPU features
			if !cpuSupportsLevel(features, entry.SIMDLevel) {
				continue
			}

			t.Run(fmt.Sprintf("size%d/%s", size, entry.Signature), func(t *testing.T) {
				t.Parallel()
				testRoundTripCodelet128(t, &entry)
			})
		}
	}
}

func testRoundTripCodelet64(t *testing.T, entry *planner.CodeletEntry[complex64]) {
	t.Helper()

	size := entry.Size

	// Prepare buffers
	original := make([]complex64, size)
	src := make([]complex64, size)
	freq := make([]complex64, size)
	roundtrip := make([]complex64, size)
	twiddle := ComputeTwiddleFactors[complex64](size)
	scratch := make([]complex64, size)

	twiddleForward, twiddleInverse, forwardBacking, inverseBacking := prepareCodeletTwiddles64(size, twiddle, entry)

	// Initialize with test pattern: impulse at various positions
	testPatterns := []struct {
		name string
		init func([]complex64)
	}{
		{"impulse_0", func(data []complex64) {
			clear(data)
			data[0] = 1
		}},
		{"impulse_mid", func(data []complex64) {
			clear(data)
			data[size/2] = 1
		}},
		{"random", func(data []complex64) {
			for i := range data {
				data[i] = complex(float32(i%7-3), float32(i%5-2))
			}
		}},
	}

	for _, pattern := range testPatterns {
		pattern.init(original)
		copy(src, original)

		// Forward transform
		entry.Forward(freq, src, twiddleForward, scratch)

		// Inverse transform
		entry.Inverse(roundtrip, freq, twiddleInverse, scratch)

		// Compare roundtrip with original
		maxErr := float64(0)

		for i := range size {
			err := cmplx.Abs(complex128(roundtrip[i] - original[i]))
			if err > maxErr {
				maxErr = err
			}
		}

		// Tolerance scales with size due to accumulated floating-point errors
		tol := 1e-4 * float64(size) / 8
		if tol < 1e-4 {
			tol = 1e-4
		}

		if tol > 0.1 {
			tol = 0.1
		}

		if maxErr > tol {
			t.Errorf("%s: max roundtrip error %v exceeds tolerance %v", pattern.name, maxErr, tol)
		}
	}

	runtime.KeepAlive(forwardBacking)
	runtime.KeepAlive(inverseBacking)
}

func testRoundTripCodelet128(t *testing.T, entry *planner.CodeletEntry[complex128]) {
	t.Helper()

	size := entry.Size

	// Prepare buffers
	original := make([]complex128, size)
	src := make([]complex128, size)
	freq := make([]complex128, size)
	roundtrip := make([]complex128, size)
	twiddle := ComputeTwiddleFactors[complex128](size)
	scratch := make([]complex128, size)

	twiddleForward, twiddleInverse, forwardBacking, inverseBacking := prepareCodeletTwiddles128(size, twiddle, entry)

	// Initialize with test pattern: impulse at various positions
	testPatterns := []struct {
		name string
		init func([]complex128)
	}{
		{"impulse_0", func(data []complex128) {
			clear(data)
			data[0] = 1
		}},
		{"impulse_mid", func(data []complex128) {
			clear(data)
			data[size/2] = 1
		}},
		{"random", func(data []complex128) {
			for i := range data {
				data[i] = complex(float64(i%7-3), float64(i%5-2))
			}
		}},
	}

	for _, pattern := range testPatterns {
		pattern.init(original)
		copy(src, original)

		// Forward transform
		entry.Forward(freq, src, twiddleForward, scratch)

		// Inverse transform
		entry.Inverse(roundtrip, freq, twiddleInverse, scratch)

		// Compare roundtrip with original
		maxErr := float64(0)

		for i := range size {
			err := cmplx.Abs(roundtrip[i] - original[i])
			if err > maxErr {
				maxErr = err
			}
		}

		// Tolerance for complex128 is tighter due to higher precision
		tol := 1e-10 * float64(size) / 8
		if tol < 1e-10 {
			tol = 1e-10
		}

		if tol > 1e-6 {
			tol = 1e-6
		}

		if maxErr > tol {
			t.Errorf("%s: max roundtrip error %v exceeds tolerance %v", pattern.name, maxErr, tol)
		}
	}

	runtime.KeepAlive(forwardBacking)
	runtime.KeepAlive(inverseBacking)
}

// TestInPlaceAllCodelets64 tests that every registered complex64 codelet works
// correctly when dst == src (in-place operation).
func TestInPlaceAllCodelets64(t *testing.T) {
	t.Parallel()

	sizes := Registry64.Sizes()
	features := cpu.DetectFeatures()

	for _, size := range sizes {
		entries := Registry64.GetAllForSize(size)
		for _, entry := range entries {
			// Skip disabled codelets (negative priority)
			if entry.Priority < 0 {
				continue
			}

			// Skip codelets that require unsupported CPU features
			if !cpuSupportsLevel(features, entry.SIMDLevel) {
				continue
			}

			t.Run(fmt.Sprintf("size%d/%s", size, entry.Signature), func(t *testing.T) {
				t.Parallel()
				testInPlaceCodelet64(t, &entry)
			})
		}
	}
}

// TestInPlaceAllCodelets128 tests that every registered complex128 codelet works
// correctly when dst == src (in-place operation).
func TestInPlaceAllCodelets128(t *testing.T) {
	t.Parallel()

	sizes := Registry128.Sizes()
	features := cpu.DetectFeatures()

	for _, size := range sizes {
		entries := Registry128.GetAllForSize(size)
		for _, entry := range entries {
			// Skip disabled codelets (negative priority)
			if entry.Priority < 0 {
				continue
			}

			// Skip codelets that require unsupported CPU features
			if !cpuSupportsLevel(features, entry.SIMDLevel) {
				continue
			}

			t.Run(fmt.Sprintf("size%d/%s", size, entry.Signature), func(t *testing.T) {
				t.Parallel()
				testInPlaceCodelet128(t, &entry)
			})
		}
	}
}

func testInPlaceCodelet64(t *testing.T, entry *planner.CodeletEntry[complex64]) {
	t.Helper()

	size := entry.Size

	// Prepare buffers
	original := make([]complex64, size)
	outOfPlace := make([]complex64, size)
	inPlace := make([]complex64, size)
	twiddle := ComputeTwiddleFactors[complex64](size)
	scratch := make([]complex64, size)

	twiddleForward, twiddleInverse, forwardBacking, inverseBacking := prepareCodeletTwiddles64(size, twiddle, entry)

	// Initialize with deterministic random-like pattern
	for i := range original {
		original[i] = complex(float32(i%7-3), float32(i%5-2))
	}

	// Out-of-place forward transform
	copy(outOfPlace, original)
	entry.Forward(outOfPlace, original, twiddleForward, scratch)

	// In-place forward transform (dst == src)
	copy(inPlace, original)
	entry.Forward(inPlace, inPlace, twiddleForward, scratch)

	// Compare in-place vs out-of-place
	maxErr := float64(0)

	for i := range size {
		err := cmplx.Abs(complex128(inPlace[i] - outOfPlace[i]))
		if err > maxErr {
			maxErr = err
		}
	}

	// In-place should match out-of-place exactly (or very close)
	tol := 1e-6
	if maxErr > tol {
		t.Errorf("forward in-place differs from out-of-place: max error %v exceeds %v", maxErr, tol)
	}

	// Also test inverse in-place
	freqOOP := make([]complex64, size)
	copy(freqOOP, outOfPlace)

	resultOOP := make([]complex64, size)
	entry.Inverse(resultOOP, freqOOP, twiddleInverse, scratch)

	freqIP := make([]complex64, size)
	copy(freqIP, outOfPlace)
	entry.Inverse(freqIP, freqIP, twiddleInverse, scratch)

	maxErr = 0

	for i := range size {
		err := cmplx.Abs(complex128(freqIP[i] - resultOOP[i]))
		if err > maxErr {
			maxErr = err
		}
	}

	if maxErr > tol {
		t.Errorf("inverse in-place differs from out-of-place: max error %v exceeds %v", maxErr, tol)
	}

	runtime.KeepAlive(forwardBacking)
	runtime.KeepAlive(inverseBacking)
}

func testInPlaceCodelet128(t *testing.T, entry *planner.CodeletEntry[complex128]) {
	t.Helper()

	size := entry.Size

	// Prepare buffers
	original := make([]complex128, size)
	outOfPlace := make([]complex128, size)
	inPlace := make([]complex128, size)
	twiddle := ComputeTwiddleFactors[complex128](size)
	scratch := make([]complex128, size)

	twiddleForward, twiddleInverse, forwardBacking, inverseBacking := prepareCodeletTwiddles128(size, twiddle, entry)

	// Initialize with deterministic random-like pattern
	for i := range original {
		original[i] = complex(float64(i%7-3), float64(i%5-2))
	}

	// Out-of-place forward transform
	copy(outOfPlace, original)
	entry.Forward(outOfPlace, original, twiddleForward, scratch)

	// In-place forward transform (dst == src)
	copy(inPlace, original)
	entry.Forward(inPlace, inPlace, twiddleForward, scratch)

	// Compare in-place vs out-of-place
	maxErr := float64(0)

	for i := range size {
		err := cmplx.Abs(inPlace[i] - outOfPlace[i])
		if err > maxErr {
			maxErr = err
		}
	}

	// In-place should match out-of-place exactly (or very close)
	tol := 1e-12
	if maxErr > tol {
		t.Errorf("forward in-place differs from out-of-place: max error %v exceeds %v", maxErr, tol)
	}

	// Also test inverse in-place
	freqOOP := make([]complex128, size)
	copy(freqOOP, outOfPlace)

	resultOOP := make([]complex128, size)
	entry.Inverse(resultOOP, freqOOP, twiddleInverse, scratch)

	freqIP := make([]complex128, size)
	copy(freqIP, outOfPlace)
	entry.Inverse(freqIP, freqIP, twiddleInverse, scratch)

	maxErr = 0

	for i := range size {
		err := cmplx.Abs(freqIP[i] - resultOOP[i])
		if err > maxErr {
			maxErr = err
		}
	}

	if maxErr > tol {
		t.Errorf("inverse in-place differs from out-of-place: max error %v exceeds %v", maxErr, tol)
	}

	runtime.KeepAlive(forwardBacking)
	runtime.KeepAlive(inverseBacking)
}

func prepareCodeletTwiddles64(
	size int,
	base []complex64,
	entry *planner.CodeletEntry[complex64],
) (forward, inverse []complex64, forwardBacking, inverseBacking []byte) {
	forward = base
	inverse = base

	if entry.TwiddleSize == nil || entry.PrepareTwiddle == nil {
		return forward, inverse, nil, nil
	}

	twiddleLen := entry.TwiddleSize(size)
	if twiddleLen <= 0 {
		return forward, inverse, nil, nil
	}

	forward, forwardBacking = memory.AllocAlignedComplex64(twiddleLen)
	inverse, inverseBacking = memory.AllocAlignedComplex64(twiddleLen)

	entry.PrepareTwiddle(size, false, forward)
	entry.PrepareTwiddle(size, true, inverse)

	return forward, inverse, forwardBacking, inverseBacking
}

func prepareCodeletTwiddles128(
	size int,
	base []complex128,
	entry *planner.CodeletEntry[complex128],
) (forward, inverse []complex128, forwardBacking, inverseBacking []byte) {
	forward = base
	inverse = base

	if entry.TwiddleSize == nil || entry.PrepareTwiddle == nil {
		return forward, inverse, nil, nil
	}

	twiddleLen := entry.TwiddleSize(size)
	if twiddleLen <= 0 {
		return forward, inverse, nil, nil
	}

	forward, forwardBacking = memory.AllocAlignedComplex128(twiddleLen)
	inverse, inverseBacking = memory.AllocAlignedComplex128(twiddleLen)

	entry.PrepareTwiddle(size, false, forward)
	entry.PrepareTwiddle(size, true, inverse)

	return forward, inverse, forwardBacking, inverseBacking
}

// cpuSupportsLevel checks if the current CPU supports the required SIMD level.
func cpuSupportsLevel(features cpu.Features, level planner.SIMDLevel) bool {
	switch level {
	case planner.SIMDNone:
		return true
	case planner.SIMDSSE2:
		return features.HasSSE2
	case planner.SIMDAVX2:
		return features.HasAVX2
	case planner.SIMDAVX512:
		return features.HasAVX512
	case planner.SIMDNEON:
		return features.HasNEON
	default:
		return false
	}
}
