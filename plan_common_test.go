package algofft

import (
	"errors"
	"testing"
)

func TestComplexTypeName(t *testing.T) {
	if got := complexTypeName[complex64](); got != "complex64" {
		t.Errorf("complexTypeName[complex64]() = %q, want %q", got, "complex64")
	}

	if got := complexTypeName[complex128](); got != "complex128" {
		t.Errorf("complexTypeName[complex128]() = %q, want %q", got, "complex128")
	}
}

func TestValidateDstSrc(t *testing.T) {
	dst := make([]complex64, 4)
	src := make([]complex64, 4)

	if err := validateDstSrc(dst, src, 4, 4); err != nil {
		t.Errorf("valid slices: got %v, want nil", err)
	}

	if err := validateDstSrc([]complex64(nil), src, 4, 4); !errors.Is(err, ErrNilSlice) {
		t.Errorf("nil dst: got %v, want ErrNilSlice", err)
	}

	if err := validateDstSrc(dst, []complex64(nil), 4, 4); !errors.Is(err, ErrNilSlice) {
		t.Errorf("nil src: got %v, want ErrNilSlice", err)
	}

	if err := validateDstSrc(dst, src, 8, 4); !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("short dst: got %v, want ErrLengthMismatch", err)
	}

	if err := validateDstSrc(dst, src, 4, 8); !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("short src: got %v, want ErrLengthMismatch", err)
	}
}

// TestValidateDstSrc_MixedTypes covers the real-FFT case where dst and src
// have different element types and lengths.
func TestValidateDstSrc_MixedTypes(t *testing.T) {
	dst := make([]complex64, 5)
	src := make([]float32, 8)

	if err := validateDstSrc(dst, src, 5, 8); err != nil {
		t.Errorf("valid mixed slices: got %v, want nil", err)
	}

	if err := validateDstSrc(dst, src, 5, 9); !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("wrong src length: got %v, want ErrLengthMismatch", err)
	}
}

// TestCheapestPaddedLength_MatchesBluesteinPadSize pins the shared pad cost
// model to the Bluestein pad choice: bluesteinPadSize(n) must equal the shared
// helper applied to the minimum cyclic length 2n-1.
func TestCheapestPaddedLength_MatchesBluesteinPadSize(t *testing.T) {
	for _, n := range []int{2, 3, 7, 11, 17, 101, 1009, 4099, 65537} {
		want := bluesteinPadSize(n)
		if got := cheapestPaddedLength(2*n - 1); got != want {
			t.Errorf("cheapestPaddedLength(2*%d-1) = %d, want bluesteinPadSize(%d) = %d", n, got, n, want)
		}
	}
}

// TestFastConvolutionLength_UsesSharedCostModel pins fastConvolutionLength to
// the shared cost model for lengths that are neither exact powers of two nor
// mixed-radix eligible.
func TestFastConvolutionLength_UsesSharedCostModel(t *testing.T) {
	for _, convLen := range []int{7, 11, 127, 1009, 4099} {
		got := fastConvolutionLength(convLen)
		if got < convLen {
			t.Fatalf("fastConvolutionLength(%d) = %d < input", convLen, got)
		}
		// Exact lengths must be kept unchanged.
		if got == convLen {
			continue
		}

		if want := cheapestPaddedLength(convLen); got != want {
			t.Errorf("fastConvolutionLength(%d) = %d, want cheapestPaddedLength = %d", convLen, got, want)
		}
	}
}
