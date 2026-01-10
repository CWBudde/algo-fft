package kernels

import mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"

// Pre-computed bit-reversal indices for multiple sizes/algorithms.
//
//nolint:gochecknoglobals
var (
	bitrevSize8Radix2     = mathpkg.ComputeBitReversalIndices(8)
	bitrevSize8Radix4     = mathpkg.ComputeBitReversalIndicesMixed24(8)
	bitrevSize16Radix2    = mathpkg.ComputeBitReversalIndices(16)
	bitrevSize16Radix4    = mathpkg.ComputeBitReversalIndicesRadix4(16)
	bitrevSize32Radix2    = mathpkg.ComputeBitReversalIndices(32)
	bitrevSize64Radix2    = mathpkg.ComputeBitReversalIndices(64)
	bitrevSize64Radix4    = mathpkg.ComputeBitReversalIndicesRadix4(64)
	bitrevSize128Radix2   = mathpkg.ComputeBitReversalIndices(128)
	bitrevSize128Radix4   = mathpkg.ComputeBitReversalIndicesRadix4(128)
	bitrevSize256Radix2   = mathpkg.ComputeBitReversalIndices(256)
	bitrevSize256Radix4   = mathpkg.ComputeBitReversalIndicesRadix4(256)
	bitrevSize512Radix2   = mathpkg.ComputeBitReversalIndices(512)
	bitrevSize512Radix8   = mathpkg.ComputeBitReversalIndicesRadix8(512)
	bitrevSize512Mixed24  = mathpkg.ComputeBitReversalIndicesMixed24(512)
	bitrevSize1024Radix2  = mathpkg.ComputeBitReversalIndices(1024)
	bitrevSize1024Radix4  = mathpkg.ComputeBitReversalIndicesRadix4(1024)
	bitrevSize2048Radix2  = mathpkg.ComputeBitReversalIndices(2048)
	bitrevSize2048Mixed24 = mathpkg.ComputeBitReversalIndicesMixed24(2048)
	bitrevSize4096Radix2  = mathpkg.ComputeBitReversalIndices(4096)
	bitrevSize4096Radix4  = mathpkg.ComputeBitReversalIndicesRadix4(4096)
	bitrevSize8192Radix2  = mathpkg.ComputeBitReversalIndices(8192)
	bitrevSize8192Mixed24 = mathpkg.ComputeBitReversalIndicesMixed24(8192)
	bitrevSize16384Radix2 = mathpkg.ComputeBitReversalIndices(16384)
	bitrevSize16384Radix4 = mathpkg.ComputeBitReversalIndicesRadix4(16384)
)

//nolint:cyclop
func forwardDITComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	switch len(src) {
	case 8:
		return forwardDIT8Complex64(dst, src, twiddle, scratch)
	case 16:
		// Use faster radix-4 implementation (12-15% faster than radix-2)
		return forwardDIT16Radix4Complex64(dst, src, twiddle, scratch)
	case 32:
		return forwardDIT32Complex64(dst, src, twiddle, scratch)
	case 64:
		return forwardDIT64Radix4Complex64(dst, src, twiddle, scratch)
	case 128:
		return forwardDIT128Complex64(dst, src, twiddle, scratch)
	case 256:
		return forwardDIT256Complex64(dst, src, twiddle, scratch)
	case 512:
		return forwardDIT512Complex64(dst, src, twiddle, scratch)
	case 1024:
		// Try radix-32x32 first (usually faster for large N)
		if forwardDIT1024Mixed32x32Complex64(dst, src, twiddle, scratch) {
			return true
		}
		// Fallback to optimized radix-4
		return forwardDIT1024Radix4Complex64(dst, src, twiddle, scratch)
	case 2048:
		return forwardDIT2048Mixed24Complex64(dst, src, twiddle, scratch)
	case 4096:
		if forwardDIT4096SixStepComplex64(dst, src, twiddle, scratch) {
			return true
		}
		return forwardDIT4096Radix4Complex64(dst, src, twiddle, scratch)
	}

	n := len(src)
	if isPowerOf4(n) {
		if forwardRadix4Complex64(dst, src, twiddle, scratch, bitrev) {
			return true
		}
	} else if IsPowerOf2(n) {
		if forwardMixedRadix24Complex64(dst, src, twiddle, scratch, bitrev) {
			return true
		}
	}

	return ditForward[complex64](dst, src, twiddle, scratch, bitrev)
}

//nolint:cyclop
func inverseDITComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	switch len(src) {
	case 8:
		return inverseDIT8Complex64(dst, src, twiddle, scratch)
	case 16:
		// Use faster radix-4 implementation (12-15% faster than radix-2)
		return inverseDIT16Radix4Complex64(dst, src, twiddle, scratch)
	case 32:
		return inverseDIT32Complex64(dst, src, twiddle, scratch)
	case 64:
		return inverseDIT64Radix4Complex64(dst, src, twiddle, scratch)
	case 128:
		return inverseDIT128Complex64(dst, src, twiddle, scratch)
	case 256:
		return inverseDIT256Complex64(dst, src, twiddle, scratch)
	case 512:
		return inverseDIT512Complex64(dst, src, twiddle, scratch)
	case 1024:
		// Try radix-32x32 first
		if inverseDIT1024Mixed32x32Complex64(dst, src, twiddle, scratch) {
			return true
		}
		// Fallback to optimized radix-4
		return inverseDIT1024Radix4Complex64(dst, src, twiddle, scratch)
	case 2048:
		return inverseDIT2048Mixed24Complex64(dst, src, twiddle, scratch)
	case 4096:
		if inverseDIT4096SixStepComplex64(dst, src, twiddle, scratch) {
			return true
		}
		return inverseDIT4096Radix4Complex64(dst, src, twiddle, scratch)
	}

	n := len(src)
	if isPowerOf4(n) {
		if inverseRadix4Complex64(dst, src, twiddle, scratch, bitrev) {
			return true
		}
	} else if IsPowerOf2(n) {
		if inverseMixedRadix24Complex64(dst, src, twiddle, scratch, bitrev) {
			return true
		}
	}

	return ditInverseComplex64(dst, src, twiddle, scratch, bitrev)
}

func forwardDITComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	switch len(src) {
	case 8:
		return forwardDIT8Complex128(dst, src, twiddle, scratch)
	case 16:
		// Use faster radix-4 implementation (12-15% faster than radix-2)
		return forwardDIT16Radix4Complex128(dst, src, twiddle, scratch)
	case 32:
		return forwardDIT32Complex128(dst, src, twiddle, scratch)
	case 64:
		return forwardDIT64Radix4Complex128(dst, src, twiddle, scratch)
	case 128:
		return forwardDIT128Complex128(dst, src, twiddle, scratch)
	case 256:
		return forwardDIT256Complex128(dst, src, twiddle, scratch)
	case 512:
		return forwardDIT512Complex128(dst, src, twiddle, scratch)
	case 1024:
		// Try radix-32x32 first
		if forwardDIT1024Mixed32x32Complex128(dst, src, twiddle, scratch) {
			return true
		}
		// Fallback to optimized radix-4
		return forwardDIT1024Radix4Complex128(dst, src, twiddle, scratch)
	case 2048:
		return forwardDIT2048Mixed24Complex128(dst, src, twiddle, scratch)
	case 4096:
		if forwardDIT4096SixStepComplex128(dst, src, twiddle, scratch) {
			return true
		}
		return forwardDIT4096Radix4Complex128(dst, src, twiddle, scratch)
	}

	if forwardRadix4Complex128(dst, src, twiddle, scratch, bitrev) {
		return true
	}

	return ditForward[complex128](dst, src, twiddle, scratch, bitrev)
}

func inverseDITComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	switch len(src) {
	case 8:
		return inverseDIT8Complex128(dst, src, twiddle, scratch)
	case 16:
		// Use faster radix-4 implementation (12-15% faster than radix-2)
		return inverseDIT16Radix4Complex128(dst, src, twiddle, scratch)
	case 32:
		return inverseDIT32Complex128(dst, src, twiddle, scratch)
	case 64:
		return inverseDIT64Radix4Complex128(dst, src, twiddle, scratch)
	case 128:
		return inverseDIT128Complex128(dst, src, twiddle, scratch)
	case 256:
		return inverseDIT256Complex128(dst, src, twiddle, scratch)
	case 512:
		return inverseDIT512Complex128(dst, src, twiddle, scratch)
	case 1024:
		// Try radix-32x32 first
		if inverseDIT1024Mixed32x32Complex128(dst, src, twiddle, scratch) {
			return true
		}
		// Fallback to optimized radix-4
		return inverseDIT1024Radix4Complex128(dst, src, twiddle, scratch)
	case 2048:
		return inverseDIT2048Mixed24Complex128(dst, src, twiddle, scratch)
	case 4096:
		if inverseDIT4096SixStepComplex128(dst, src, twiddle, scratch) {
			return true
		}
		return inverseDIT4096Radix4Complex128(dst, src, twiddle, scratch)
	}

	if inverseRadix4Complex128(dst, src, twiddle, scratch, bitrev) {
		return true
	}

	return ditInverseComplex128(dst, src, twiddle, scratch, bitrev)
}

//nolint:cyclop
func ditForward[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	n := len(src)
	if n == 0 {
		return true
	}

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n {
		return false
	}

	if n == 1 {
		dst[0] = src[0]
		return true
	}

	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	work = work[:n]
	src = src[:n]
	twiddle = twiddle[:n]
	bitrev = bitrev[:n]

	for i := range n {
		work[i] = src[bitrev[i]]
	}

	for size := 2; size <= n; size <<= 1 {
		half := size >> 1

		step := n / size
		for base := 0; base < n; base += size {
			block := work[base : base+size]

			for j := range half {
				tw := twiddle[j*step]
				a, b := butterfly2(block[j], block[j+half], tw)
				block[j] = a
				block[j+half] = b
			}
		}
	}

	if !workIsDst {
		copy(dst, work)
	}

	return true
}

//nolint:cyclop
func ditInverse[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	n := len(src)
	if n == 0 {
		return true
	}

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n {
		return false
	}

	if n == 1 {
		dst[0] = src[0]
		return true
	}

	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	work = work[:n]
	src = src[:n]
	twiddle = twiddle[:n]
	bitrev = bitrev[:n]

	for i := range n {
		work[i] = src[bitrev[i]]
	}

	for size := 2; size <= n; size <<= 1 {
		half := size >> 1

		step := n / size
		for base := 0; base < n; base += size {
			block := work[base : base+size]

			for j := range half {
				tw := conj(twiddle[j*step])
				a, b := butterfly2(block[j], block[j+half], tw)
				block[j] = a
				block[j+half] = b
			}
		}
	}

	if !workIsDst {
		copy(dst, work)
	}

	scale := complexFromFloat64[T](1.0/float64(n), 0)
	for i := range dst {
		dst[i] *= scale
	}

	return true
}

//nolint:cyclop
func ditInverseComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	n := len(src)
	if n == 0 {
		return true
	}

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n {
		return false
	}

	if n == 1 {
		dst[0] = src[0]
		return true
	}

	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	work = work[:n]
	src = src[:n]
	twiddle = twiddle[:n]
	bitrev = bitrev[:n]

	for i := range n {
		work[i] = src[bitrev[i]]
	}

	for size := 2; size <= n; size <<= 1 {
		half := size >> 1

		step := n / size
		for base := 0; base < n; base += size {
			block := work[base : base+size]

			for j := range half {
				tw := twiddle[j*step]
				tw = complex(real(tw), -imag(tw))
				a, b := butterfly2(block[j], block[j+half], tw)
				block[j] = a
				block[j+half] = b
			}
		}
	}

	if !workIsDst {
		copy(dst, work)
	}

	scale := complex(float32(1.0/float64(n)), 0)
	for i := range dst {
		dst[i] *= scale
	}

	return true
}

//nolint:cyclop
func ditInverseComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	n := len(src)
	if n == 0 {
		return true
	}

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n {
		return false
	}

	if n == 1 {
		dst[0] = src[0]
		return true
	}

	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	work = work[:n]
	src = src[:n]
	twiddle = twiddle[:n]
	bitrev = bitrev[:n]

	for i := range n {
		work[i] = src[bitrev[i]]
	}

	for size := 2; size <= n; size <<= 1 {
		half := size >> 1

		step := n / size
		for base := 0; base < n; base += size {
			block := work[base : base+size]

			for j := range half {
				tw := twiddle[j*step]
				tw = complex(real(tw), -imag(tw))
				a, b := butterfly2(block[j], block[j+half], tw)
				block[j] = a
				block[j+half] = b
			}
		}
	}

	if !workIsDst {
		copy(dst, work)
	}

	scale := complex(1.0/float64(n), 0)
	for i := range dst {
		dst[i] *= scale
	}

	return true
}

func butterfly2[T Complex](a, b, w T) (T, T) {
	t := w * b
	return a + t, a - t
}

// Public exports for internal/fft re-export.
func DITForward[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	return ditForward(dst, src, twiddle, scratch, bitrev)
}

func DITInverse[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	return ditInverse(dst, src, twiddle, scratch, bitrev)
}

// Precision-specific exports.
var (
	ForwardDITComplex64  = forwardDITComplex64
	InverseDITComplex64  = inverseDITComplex64
	ForwardDITComplex128 = forwardDITComplex128
	InverseDITComplex128 = inverseDITComplex128
)

// Butterfly2 performs a radix-2 butterfly operation.
func Butterfly2[T Complex](a, b, w T) (T, T) {
	return butterfly2(a, b, w)
}

// Internal wrappers for sizes that now handle bit-reversal internally.
func forwardDIT8Complex64(dst, src, twiddle, scratch []complex64) bool {
	return forwardDIT8Radix8Complex64(dst, src, twiddle, scratch)
}

func inverseDIT8Complex64(dst, src, twiddle, scratch []complex64) bool {
	return inverseDIT8Radix8Complex64(dst, src, twiddle, scratch)
}

func forwardDIT8Complex128(dst, src, twiddle, scratch []complex128) bool {
	return forwardDIT8Radix8Complex128(dst, src, twiddle, scratch)
}

func inverseDIT8Complex128(dst, src, twiddle, scratch []complex128) bool {
	return inverseDIT8Radix8Complex128(dst, src, twiddle, scratch)
}

func forwardDIT16Complex64(dst, src, twiddle, scratch []complex64) bool {
	return forwardDIT16Radix4Complex64(dst, src, twiddle, scratch)
}

func inverseDIT16Complex64(dst, src, twiddle, scratch []complex64) bool {
	return inverseDIT16Radix4Complex64(dst, src, twiddle, scratch)
}

func forwardDIT16Complex128(dst, src, twiddle, scratch []complex128) bool {
	return forwardDIT16Radix4Complex128(dst, src, twiddle, scratch)
}

func inverseDIT16Complex128(dst, src, twiddle, scratch []complex128) bool {
	return inverseDIT16Radix4Complex128(dst, src, twiddle, scratch)
}

// Size-specific DIT exports for benchmarks and tests.
var (
	// Size 4.
	ForwardDIT4Radix4Complex64 = func(dst, src, twiddle, scratch []complex64) bool {
		return forwardDIT4Radix4Complex64(dst, src, twiddle, scratch)
	}
	InverseDIT4Radix4Complex64 = func(dst, src, twiddle, scratch []complex64) bool {
		return inverseDIT4Radix4Complex64(dst, src, twiddle, scratch)
	}
	// Size 8.
	ForwardDIT8Complex64       = forwardDIT8Complex64
	InverseDIT8Complex64       = inverseDIT8Complex64
	ForwardDIT8Radix8Complex64 = forwardDIT8Complex64
	InverseDIT8Radix8Complex64 = inverseDIT8Complex64
	ForwardDIT8Radix2Complex64 = func(dst, src, twiddle, scratch []complex64) bool {
		return forwardDIT8Radix2Complex64(dst, src, twiddle, scratch)
	}
	InverseDIT8Radix2Complex64 = func(dst, src, twiddle, scratch []complex64) bool {
		return inverseDIT8Radix2Complex64(dst, src, twiddle, scratch)
	}
	ForwardDIT8Radix4Complex64 = func(dst, src, twiddle, scratch []complex64) bool {
		return forwardDIT8Radix4Complex64(dst, src, twiddle, scratch)
	}
	InverseDIT8Radix4Complex64 = func(dst, src, twiddle, scratch []complex64) bool {
		return inverseDIT8Radix4Complex64(dst, src, twiddle, scratch)
	}
	// Size 16.
	ForwardDIT16Complex64       = forwardDIT16Complex64
	InverseDIT16Complex64       = inverseDIT16Complex64
	ForwardDIT16Radix4Complex64 = forwardDIT16Complex64
	InverseDIT16Radix4Complex64 = inverseDIT16Complex64
	ForwardDIT16Radix2Complex64 = func(dst, src, twiddle, scratch []complex64) bool {
		return forwardDIT16Radix2Complex64(dst, src, twiddle, scratch)
	}
	InverseDIT16Radix2Complex64 = func(dst, src, twiddle, scratch []complex64) bool {
		return inverseDIT16Radix2Complex64(dst, src, twiddle, scratch)
	}
	// Size 32.
	ForwardDIT32Complex64 = func(dst, src, twiddle, scratch []complex64) bool {
		return forwardDIT32Complex64(dst, src, twiddle, scratch)
	}
	InverseDIT32Complex64 = func(dst, src, twiddle, scratch []complex64) bool {
		return inverseDIT32Complex64(dst, src, twiddle, scratch)
	}
	// Size 64.
	ForwardDIT64Complex64       = forwardDIT64Complex64
	InverseDIT64Complex64       = inverseDIT64Complex64
	ForwardDIT64Radix4Complex64 = forwardDIT64Radix4Complex64
	InverseDIT64Radix4Complex64 = inverseDIT64Radix4Complex64
	// Size 128.
	ForwardDIT128Complex64 = forwardDIT128Complex64
	InverseDIT128Complex64 = inverseDIT128Complex64
	// Size 256.
	ForwardDIT256Complex64       = forwardDIT256Complex64
	InverseDIT256Complex64       = inverseDIT256Complex64
	ForwardDIT256Radix4Complex64 = forwardDIT256Radix4Complex64
	InverseDIT256Radix4Complex64 = inverseDIT256Radix4Complex64
	// Size 512.
	ForwardDIT512Complex64        = forwardDIT512Complex64
	InverseDIT512Complex64        = inverseDIT512Complex64
	ForwardDIT512Mixed24Complex64 = forwardDIT512Mixed24Complex64
	InverseDIT512Mixed24Complex64 = inverseDIT512Mixed24Complex64

	// Complex128 variants.
	ForwardDIT4Radix4Complex128 = func(dst, src, twiddle, scratch []complex128) bool {
		return forwardDIT4Radix4Complex128(dst, src, twiddle, scratch)
	}
	InverseDIT4Radix4Complex128 = func(dst, src, twiddle, scratch []complex128) bool {
		return inverseDIT4Radix4Complex128(dst, src, twiddle, scratch)
	}
	// Size 8.
	ForwardDIT8Complex128       = forwardDIT8Complex128
	InverseDIT8Complex128       = inverseDIT8Complex128
	ForwardDIT8Radix8Complex128 = forwardDIT8Complex128
	InverseDIT8Radix8Complex128 = inverseDIT8Complex128
	ForwardDIT8Radix2Complex128 = func(dst, src, twiddle, scratch []complex128) bool {
		return forwardDIT8Radix2Complex128(dst, src, twiddle, scratch)
	}
	InverseDIT8Radix2Complex128 = func(dst, src, twiddle, scratch []complex128) bool {
		return inverseDIT8Radix2Complex128(dst, src, twiddle, scratch)
	}
	ForwardDIT8Radix4Complex128 = func(dst, src, twiddle, scratch []complex128) bool {
		return forwardDIT8Radix4Complex128(dst, src, twiddle, scratch)
	}
	InverseDIT8Radix4Complex128 = func(dst, src, twiddle, scratch []complex128) bool {
		return inverseDIT8Radix4Complex128(dst, src, twiddle, scratch)
	}
	// Size 16.
	ForwardDIT16Complex128       = forwardDIT16Complex128
	InverseDIT16Complex128       = inverseDIT16Complex128
	ForwardDIT16Radix4Complex128 = forwardDIT16Complex128
	InverseDIT16Radix4Complex128 = inverseDIT16Complex128
	ForwardDIT16Radix2Complex128 = func(dst, src, twiddle, scratch []complex128) bool {
		return forwardDIT16Radix2Complex128(dst, src, twiddle, scratch)
	}
	InverseDIT16Radix2Complex128 = func(dst, src, twiddle, scratch []complex128) bool {
		return inverseDIT16Radix2Complex128(dst, src, twiddle, scratch)
	}
	ForwardDIT32Complex128 = func(dst, src, twiddle, scratch []complex128) bool {
		return forwardDIT32Complex128(dst, src, twiddle, scratch)
	}
	InverseDIT32Complex128 = func(dst, src, twiddle, scratch []complex128) bool {
		return inverseDIT32Complex128(dst, src, twiddle, scratch)
	}
	ForwardDIT64Complex128         = forwardDIT64Complex128
	InverseDIT64Complex128         = inverseDIT64Complex128
	ForwardDIT64Radix4Complex128   = forwardDIT64Radix4Complex128
	InverseDIT64Radix4Complex128   = inverseDIT64Radix4Complex128
	ForwardDIT128Complex128        = forwardDIT128Complex128
	InverseDIT128Complex128        = inverseDIT128Complex128
	ForwardDIT256Complex128        = forwardDIT256Complex128
	InverseDIT256Complex128        = inverseDIT256Complex128
	ForwardDIT256Radix4Complex128  = forwardDIT256Radix4Complex128
	InverseDIT256Radix4Complex128  = inverseDIT256Radix4Complex128
	ForwardDIT512Complex128        = forwardDIT512Complex128
	InverseDIT512Complex128        = inverseDIT512Complex128
	ForwardDIT512Mixed24Complex128 = forwardDIT512Mixed24Complex128
	InverseDIT512Mixed24Complex128 = inverseDIT512Mixed24Complex128
)
