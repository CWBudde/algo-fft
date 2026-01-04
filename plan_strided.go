package algofft

import (
	"github.com/MeKo-Christian/algo-fft/internal/fft"
	m "github.com/MeKo-Christian/algo-fft/internal/math"
)

// ForwardStrided computes the forward FFT on strided input/output data.
//
// The stride parameter specifies the distance between consecutive elements.
// For example, stride=numCols transforms a matrix column in row-major storage.
//
// Returns ErrNilSlice if dst or src is nil.
// Returns ErrInvalidStride if stride < 1 or overflows index computation.
// Returns ErrLengthMismatch if slices are too short for the given stride.
func (p *Plan[T]) ForwardStrided(dst, src []T, stride int) error {
	return p.transformStrided(dst, src, stride, false)
}

// InverseStrided computes the inverse FFT on strided input/output data.
//
// Returns ErrNilSlice if dst or src is nil.
// Returns ErrInvalidStride if stride < 1 or overflows index computation.
// Returns ErrLengthMismatch if slices are too short for the given stride.
func (p *Plan[T]) InverseStrided(dst, src []T, stride int) error {
	return p.transformStrided(dst, src, stride, true)
}

// TransformStrided computes either forward or inverse FFT based on the inverse flag.
// This is a convenience wrapper over ForwardStrided/InverseStrided.
func (p *Plan[T]) TransformStrided(dst, src []T, stride int, inverse bool) error {
	return p.transformStrided(dst, src, stride, inverse)
}

func (p *Plan[T]) transformStrided(dst, src []T, stride int, inverse bool) error {
	err := p.validateStridedSlices(dst, src, stride)
	if err != nil {
		return err
	}

	if stride == 1 {
		if inverse {
			return p.Inverse(dst[:p.n], src[:p.n])
		}

		return p.Forward(dst[:p.n], src[:p.n])
	}

	// Use optimized strided DIT only if:
	// - Size is power of 2
	// - Not using Bluestein's algorithm
	// - dst != src (not in-place)
	// - The bitrev is standard radix-2 (strided DIT requires radix-2 bit-reversal)
	canUseStridedDIT := m.IsPowerOf2(p.n) &&
		p.kernelStrategy != fft.KernelBluestein &&
		!sameSliceStrided(dst, src) &&
		isRadix2BitRev(p.bitrev, p.n)

	//nolint:nestif
	if canUseStridedDIT {
		if inverse {
			if fft.InverseStridedDIT(dst, src, p.twiddle, p.bitrev, stride, p.n) {
				return nil
			}
		} else {
			if fft.ForwardStridedDIT(dst, src, p.twiddle, p.bitrev, stride, p.n) {
				return nil
			}
		}
	}

	buffer := p.stridedScratch[:p.n]
	for i := range p.n {
		buffer[i] = src[i*stride]
	}

	if inverse {
		err := p.Inverse(buffer, buffer)
		if err != nil {
			return err
		}
	} else {
		err := p.Forward(buffer, buffer)
		if err != nil {
			return err
		}
	}

	for i := range p.n {
		dst[i*stride] = buffer[i]
	}

	return nil
}

func (p *Plan[T]) validateStridedSlices(dst, src []T, stride int) error {
	if dst == nil || src == nil {
		return ErrNilSlice
	}

	if stride < 1 {
		return ErrInvalidStride
	}

	if p.n == 0 {
		return ErrLengthMismatch
	}

	if stride == 1 {
		if len(dst) < p.n || len(src) < p.n {
			return ErrLengthMismatch
		}

		return nil
	}

	maxInt := int(^uint(0) >> 1)

	maxIndex := p.n - 1
	if maxIndex > (maxInt-1)/stride {
		return ErrInvalidStride
	}

	required := 1 + maxIndex*stride
	if len(dst) < required || len(src) < required {
		return ErrLengthMismatch
	}

	return nil
}

func sameSliceStrided[T any](a, b []T) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}

	return &a[0] == &b[0]
}

// isRadix2BitRev checks if bitrev is standard radix-2 bit-reversal.
// For radix-2, bitrev[1] == n/2 (the bit-reversal of index 1).
// This is used to skip the strided DIT optimization when the Plan
// uses a non-radix-2 codelet (e.g., radix-4 or mixed-radix).
func isRadix2BitRev(bitrev []int, n int) bool {
	if n < 2 || len(bitrev) < 2 {
		return false
	}
	// For standard radix-2 bit-reversal, index 1 maps to n/2
	// e.g., for n=8: bitrev = [0, 4, 2, 6, 1, 5, 3, 7], so bitrev[1] = 4 = 8/2
	return bitrev[1] == n/2
}
