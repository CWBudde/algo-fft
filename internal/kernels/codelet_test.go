package kernels

import (
	"runtime"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/cpu"
)

func TestCodeletRegistryLookup(t *testing.T) {
	t.Parallel()
	// Test complex64 registry
	t.Run("complex64", func(t *testing.T) {
		t.Parallel()

		features := cpu.Features{HasSSE2: true}

		// Should find codelets for sizes 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
		sizes := []int{4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}
		for _, size := range sizes {
			entry := Registry64.Lookup(size, features)
			if entry == nil {
				t.Errorf("expected codelet for size %d, got nil", size)
				continue
			}

			if entry.Size != size {
				t.Errorf("expected size %d, got %d", size, entry.Size)
			}

			if entry.Forward == nil {
				t.Errorf("expected Forward codelet for size %d, got nil", size)
			}

			if entry.Inverse == nil {
				t.Errorf("expected Inverse codelet for size %d, got nil", size)
			}

			if entry.Signature == "" {
				t.Errorf("expected non-empty Signature for size %d", size)
			}
		}
	})

	// Test complex128 registry
	t.Run("complex128", func(t *testing.T) {
		t.Parallel()

		features := cpu.Features{HasSSE2: true}

		sizes := []int{4, 8, 16, 32, 64, 128, 256}
		for _, size := range sizes {
			entry := Registry128.Lookup(size, features)
			if entry == nil {
				t.Errorf("expected codelet for size %d, got nil", size)
				continue
			}

			if entry.Size != size {
				t.Errorf("expected size %d, got %d", size, entry.Size)
			}
		}
	})
}

func TestCodeletRegistryNoMatch(t *testing.T) {
	t.Parallel()

	features := cpu.Features{HasSSE2: true}

	// Sizes without codelets should return nil
	noCodeletSizes := []int{2}
	for _, size := range noCodeletSizes {
		entry := Registry64.Lookup(size, features)
		if entry != nil {
			t.Errorf("expected nil for size %d, got %v", size, entry)
		}
	}
}

func TestCodeletRegistrySizes(t *testing.T) {
	t.Parallel()

	sizes := Registry64.Sizes()
	has384 := false

	for _, size := range sizes {
		if size == 384 {
			has384 = true
			break
		}
	}

	expected := map[int]bool{4: true, 8: true, 16: true, 32: true, 64: true, 128: true, 256: true, 512: true, 1024: true, 2048: true, 4096: true, 8192: true, 16384: true}
	expectedCount := 13

	if has384 {
		expected[384] = true
		expectedCount = 14
	}

	if len(sizes) != expectedCount {
		t.Errorf("expected %d registered sizes, got %d", expectedCount, len(sizes))
	}

	// Check that all expected sizes are present
	for _, size := range sizes {
		if !expected[size] {
			t.Errorf("unexpected size %d in registry", size)
		}
	}
}

func TestCodeletRegistryLookupBySignature(t *testing.T) {
	t.Parallel()
	// Updated signature format: dit{size}_radix{radix}_{simd}
	entry := Registry64.LookupBySignature(64, "dit64_radix2_generic")
	if entry == nil {
		t.Fatal("expected to find dit64_radix2_generic, got nil")
	}

	if entry.Size != 64 {
		t.Errorf("expected size 64, got %d", entry.Size)
	}

	if entry.Signature != "dit64_radix2_generic" {
		t.Errorf("expected signature dit64_radix2_generic, got %s", entry.Signature)
	}

	// Non-existent signature
	entry = Registry64.LookupBySignature(64, "nonexistent")
	if entry != nil {
		t.Errorf("expected nil for nonexistent signature, got %v", entry)
	}
}

func TestGetRegistry(t *testing.T) {
	t.Parallel()

	reg64 := GetRegistry[complex64]()
	if reg64 != Registry64 {
		t.Error("GetRegistry[complex64]() should return Registry64")
	}

	reg128 := GetRegistry[complex128]()
	if reg128 != Registry128 {
		t.Error("GetRegistry[complex128]() should return Registry128")
	}
}

func TestSIMDLevelString(t *testing.T) {
	t.Parallel()

	tests := []struct {
		level    SIMDLevel
		expected string
	}{
		{SIMDNone, "generic"},
		{SIMDSSE2, "sse2"},
		{SIMDAVX2, "avx2"},
		{SIMDAVX512, "avx512"},
		{SIMDNEON, "neon"},
		{SIMDLevel(99), "unknown"},
	}

	for _, tt := range tests {
		got := tt.level.String()
		if got != tt.expected {
			t.Errorf("SIMDLevel(%d).String() = %q, want %q", tt.level, got, tt.expected)
		}
	}
}

//nolint:gocognit
func TestCodeletFunctional(t *testing.T) {
	// Test that codelets produce correct results
	features := cpu.Features{HasSSE2: true}

	t.Run("forward_8", func(t *testing.T) {
		t.Parallel()

		entry := Registry64.Lookup(8, features)
		if entry == nil {
			t.Skip("no codelet for size 8")
		}

		src := make([]complex64, 8)
		dst := make([]complex64, 8)
		twiddle := ComputeTwiddleFactors[complex64](8)
		scratch := make([]complex64, 8)

		codeletTwiddle, _, forwardBacking, inverseBacking := prepareCodeletTwiddles64(8, twiddle, entry)
		defer runtime.KeepAlive(forwardBacking)
		defer runtime.KeepAlive(inverseBacking)

		// Initialize with impulse
		src[0] = 1

		entry.Forward(dst, src, codeletTwiddle, scratch)

		// FFT of impulse should be all ones
		for i, v := range dst {
			if real(v) < 0.99 || real(v) > 1.01 || imag(v) < -0.01 || imag(v) > 0.01 {
				t.Errorf("dst[%d] = %v, expected ~1+0i", i, v)
			}
		}
	})

	t.Run("inverse_8", func(t *testing.T) {
		t.Parallel()

		entry := Registry64.Lookup(8, features)
		if entry == nil {
			t.Skip("no codelet for size 8")
		}

		src := make([]complex64, 8)
		dst := make([]complex64, 8)
		twiddle := ComputeTwiddleFactors[complex64](8)
		scratch := make([]complex64, 8)

		_, codeletTwiddle, forwardBacking, inverseBacking := prepareCodeletTwiddles64(8, twiddle, entry)
		defer runtime.KeepAlive(forwardBacking)
		defer runtime.KeepAlive(inverseBacking)

		// Initialize with all ones (FFT of impulse)
		for i := range src {
			src[i] = 1
		}

		entry.Inverse(dst, src, codeletTwiddle, scratch)

		// IFFT should give impulse at index 0 (~1+0i)
		if real(dst[0]) < 0.99 || real(dst[0]) > 1.01 {
			t.Errorf("dst[0] = %v, expected ~1+0i", dst[0])
		}

		if imag(dst[0]) < -0.01 || imag(dst[0]) > 0.01 {
			t.Errorf("dst[0] = %v, imaginary part should be ~0", dst[0])
		}

		for i := 1; i < 8; i++ {
			if real(dst[i]) > 0.01 || real(dst[i]) < -0.01 || imag(dst[i]) > 0.01 || imag(dst[i]) < -0.01 {
				t.Errorf("dst[%d] = %v, expected ~0+0i", i, dst[i])
			}
		}
	})

	t.Run("forward_512", func(t *testing.T) {
		t.Parallel()

		entry := Registry64.Lookup(512, features)
		if entry == nil {
			t.Skip("no codelet for size 512")
		}

		src := make([]complex64, 512)
		dst := make([]complex64, 512)
		twiddle := ComputeTwiddleFactors[complex64](512)
		scratch := make([]complex64, 512)

		codeletTwiddle, _, forwardBacking, inverseBacking := prepareCodeletTwiddles64(512, twiddle, entry)
		defer runtime.KeepAlive(forwardBacking)
		defer runtime.KeepAlive(inverseBacking)

		// Initialize with impulse
		src[0] = 1

		entry.Forward(dst, src, codeletTwiddle, scratch)

		// FFT of impulse should be all ones
		for i, v := range dst {
			if real(v) < 0.99 || real(v) > 1.01 || imag(v) < -0.01 || imag(v) > 0.01 {
				t.Errorf("dst[%d] = %v, expected ~1+0i", i, v)
			}
		}
	})

	t.Run("inverse_512", func(t *testing.T) {
		t.Parallel()

		entry := Registry64.Lookup(512, features)
		if entry == nil {
			t.Skip("no codelet for size 512")
		}

		src := make([]complex64, 512)
		dst := make([]complex64, 512)
		twiddle := ComputeTwiddleFactors[complex64](512)
		scratch := make([]complex64, 512)

		_, codeletTwiddle, forwardBacking, inverseBacking := prepareCodeletTwiddles64(512, twiddle, entry)
		defer runtime.KeepAlive(forwardBacking)
		defer runtime.KeepAlive(inverseBacking)

		// Initialize with all ones (FFT of impulse)
		for i := range src {
			src[i] = 1
		}

		entry.Inverse(dst, src, codeletTwiddle, scratch)

		// IFFT should give impulse at index 0 (~1+0i)
		if real(dst[0]) < 0.99 || real(dst[0]) > 1.01 {
			t.Errorf("dst[0] = %v, expected ~1+0i", dst[0])
		}

		if imag(dst[0]) < -0.01 || imag(dst[0]) > 0.01 {
			t.Errorf("dst[0] = %v, imaginary part should be ~0", dst[0])
		}

		for i := 1; i < 512; i++ {
			if real(dst[i]) > 0.01 || real(dst[i]) < -0.01 || imag(dst[i]) > 0.01 || imag(dst[i]) < -0.01 {
				t.Errorf("dst[%d] = %v, expected ~0+0i", i, dst[i])
			}
		}
	})
}
