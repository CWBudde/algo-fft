package kernels

import mathpkg "github.com/cwbudde/algo-fft/internal/math"

// Pre-computed bit-reversal indices for multiple sizes/algorithms.
//
//nolint:gochecknoglobals
var (
	bitrevSize16Radix2         = mathpkg.ComputeBitReversalIndices(16)
	bitrevSize64Radix4         = mathpkg.ComputeBitReversalIndicesRadix4(64)
	bitrevSize256Radix2        = mathpkg.ComputeBitReversalIndices(256)
	bitrevSize256Radix4        = mathpkg.ComputeBitReversalIndicesRadix4(256)
	bitrevSize512Radix2        = mathpkg.ComputeBitReversalIndices(512)
	bitrevSize512Radix8        = mathpkg.ComputeBitReversalIndicesRadix8(512)
	bitrevSize512Radix4Then2   = mathpkg.ComputeBitReversalIndicesRadix4Then2(512)
	bitrevSize1024Radix4       = mathpkg.ComputeBitReversalIndicesRadix4(1024)
	bitrevSize2048Radix4Then2  = mathpkg.ComputeBitReversalIndicesRadix4Then2(2048)
	bitrevSize4096Radix4       = mathpkg.ComputeBitReversalIndicesRadix4(4096)
	bitrevSize8192Radix4Then2  = mathpkg.ComputeBitReversalIndicesRadix4Then2(8192)
	bitrevSize16384Radix4      = mathpkg.ComputeBitReversalIndicesRadix4(16384)
	bitrevSize32768Radix4Then2 = mathpkg.ComputeBitReversalIndicesRadix4Then2(32768)
)

func forwardDITComplex64(dst, src, twiddle, scratch []complex64) bool {
	switch len(src) {
	case 8:
		return forwardDIT8Complex64(dst, src, twiddle, scratch)
	case 16:
		// Use faster radix-4 implementation (12-15% faster than radix-2)
		return forwardDIT16Radix4Complex64(dst, src, twiddle, scratch)
	case 32:
		return forwardDIT32Radix2Complex64(dst, src, twiddle, scratch)
	case 64:
		return forwardDIT64Radix4Complex64(dst, src, twiddle, scratch)
	case 128:
		return forwardDIT128Radix2Complex64(dst, src, twiddle, scratch)
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
		return forwardDIT2048Radix4Then2Complex64(dst, src, twiddle, scratch)
	case 4096:
		if forwardDIT4096SixStepComplex64(dst, src, twiddle, scratch) {
			return true
		}

		return forwardDIT4096Radix4Complex64(dst, src, twiddle, scratch)
	}

	n := len(src)
	if isPowerOf4(n) {
		if forwardRadix4Complex64(dst, src, twiddle, scratch) {
			return true
		}
	} else if IsPowerOf2(n) {
		if forwardRadix4Then2Complex64(dst, src, twiddle, scratch) {
			return true
		}
	}

	return ditForwardComplex64(dst, src, twiddle, scratch)
}

func inverseDITComplex64(dst, src, twiddle, scratch []complex64) bool {
	switch len(src) {
	case 8:
		return inverseDIT8Complex64(dst, src, twiddle, scratch)
	case 16:
		// Use faster radix-4 implementation (12-15% faster than radix-2)
		return inverseDIT16Radix4Complex64(dst, src, twiddle, scratch)
	case 32:
		return inverseDIT32Radix2Complex64(dst, src, twiddle, scratch)
	case 64:
		return inverseDIT64Radix4Complex64(dst, src, twiddle, scratch)
	case 128:
		return inverseDIT128Radix2Complex64(dst, src, twiddle, scratch)
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
		return inverseDIT2048Radix4Then2Complex64(dst, src, twiddle, scratch)
	case 4096:
		if inverseDIT4096SixStepComplex64(dst, src, twiddle, scratch) {
			return true
		}

		return inverseDIT4096Radix4Complex64(dst, src, twiddle, scratch)
	}

	n := len(src)
	if isPowerOf4(n) {
		if inverseRadix4Complex64(dst, src, twiddle, scratch) {
			return true
		}
	} else if IsPowerOf2(n) {
		if inverseRadix4Then2Complex64(dst, src, twiddle, scratch) {
			return true
		}
	}

	return ditInverseComplex64(dst, src, twiddle, scratch)
}

func ditForward[T Complex](dst, src, twiddle, scratch []T) bool {
	return ditForwardBitrev(dst, src, twiddle, scratch, nil)
}

// ditForwardBitrev is ditForward with an optional precomputed bit-reversal
// table. Passing nil (or a short slice) computes the table locally; hot paths
// that transform repeatedly at the same size (e.g. Bluestein convolutions)
// should pass the plan's precomputed table to stay allocation-free.
//

func ditForwardBitrev[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	n := len(src)
	if n == 0 {
		return true
	}

	if len(dst) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	if n == 1 {
		dst[0] = src[0]
		return true
	}

	if !mathpkg.IsPowerOf2(n) {
		return false
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

	if len(bitrev) < n {
		// Compute bit-reversal indices locally for fallback
		bitrev = mathpkg.ComputeBitReversalIndices(n)
	}

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

// ditForwardComplex64 is the monomorphized twin of ditForward, mirroring
// ditInverseComplex64. It exists so the complex64 fallback path multiplies in
// single precision (see butterfly2Complex64); the generic instantiation cannot.
func ditForwardComplex64(dst, src, twiddle, scratch []complex64) bool {
	n := len(src)
	if n == 0 {
		return true
	}

	if len(dst) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	if n == 1 {
		dst[0] = src[0]
		return true
	}

	if !mathpkg.IsPowerOf2(n) {
		return false
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

	bitrev := mathpkg.ComputeBitReversalIndices(n)

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
				a, b := butterfly2Complex64(block[j], block[j+half], tw)
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

func ditInverse[T Complex](dst, src, twiddle, scratch []T) bool {
	return ditInverseBitrev(dst, src, twiddle, scratch, nil)
}

// ditInverseBitrev is ditInverse with an optional precomputed bit-reversal
// table; see ditForwardBitrev.
//

func ditInverseBitrev[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	n := len(src)
	if n == 0 {
		return true
	}

	if len(dst) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	if n == 1 {
		dst[0] = src[0]
		return true
	}

	if !mathpkg.IsPowerOf2(n) {
		return false
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

	if len(bitrev) < n {
		bitrev = mathpkg.ComputeBitReversalIndices(n)
	}

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

func ditInverseComplex64(dst, src, twiddle, scratch []complex64) bool {
	n := len(src)
	if n == 0 {
		return true
	}

	if len(dst) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	if n == 1 {
		dst[0] = src[0]
		return true
	}

	if !mathpkg.IsPowerOf2(n) {
		return false
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

	bitrev := mathpkg.ComputeBitReversalIndices(n)

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
				a, b := butterfly2Complex64(block[j], block[j+half], tw)
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
		dst[i] = mathpkg.MulComplex64(dst[i], scale)
	}

	return true
}

func butterfly2[T Complex](a, b, w T) (T, T) {
	t := w * b
	return a + t, a - t
}

// butterfly2Complex64 is the monomorphized twin of butterfly2. The generic
// form cannot avoid the float32->float64 promotion Go emits for scalar
// complex64 multiplication (see mathpkg.MulComplex64), because `w * b` there
// has type T; the complex64 codelets call this instead.
func butterfly2Complex64(a, b, w complex64) (complex64, complex64) {
	t := mathpkg.MulComplex64(w, b)
	return a + t, a - t
}

// Public exports for internal/fft re-export.
func DITForward[T Complex](dst, src, twiddle, scratch []T) bool {
	return ditForward(dst, src, twiddle, scratch)
}

func DITInverse[T Complex](dst, src, twiddle, scratch []T) bool {
	return ditInverse(dst, src, twiddle, scratch)
}

// DITForwardBitrev is DITForward with a caller-supplied bit-reversal table.
// Passing a table of at least len(src) entries keeps the call allocation-free;
// a nil or short slice falls back to computing the permutation locally.
func DITForwardBitrev[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	return ditForwardBitrev(dst, src, twiddle, scratch, bitrev)
}

// DITInverseBitrev is DITInverse with a caller-supplied bit-reversal table.
// See DITForwardBitrev.
func DITInverseBitrev[T Complex](dst, src, twiddle, scratch []T, bitrev []int) bool {
	return ditInverseBitrev(dst, src, twiddle, scratch, bitrev)
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

func forwardDIT16Complex64(dst, src, twiddle, scratch []complex64) bool {
	return forwardDIT16Radix4Complex64(dst, src, twiddle, scratch)
}

func inverseDIT16Complex64(dst, src, twiddle, scratch []complex64) bool {
	return inverseDIT16Radix4Complex64(dst, src, twiddle, scratch)
}

// Size-specific DIT exports for benchmarks and tests.
var (
	// Size 4.
	ForwardDIT4Radix4Complex64 = forwardDIT4Radix4Complex64
	InverseDIT4Radix4Complex64 = inverseDIT4Radix4Complex64
	// Size 8.
	ForwardDIT8Complex64       = forwardDIT8Complex64
	InverseDIT8Complex64       = inverseDIT8Complex64
	ForwardDIT8Radix8Complex64 = forwardDIT8Complex64
	InverseDIT8Radix8Complex64 = inverseDIT8Complex64
	ForwardDIT8Radix2Complex64 = forwardDIT8Radix2Complex64
	InverseDIT8Radix2Complex64 = inverseDIT8Radix2Complex64
	ForwardDIT8Radix4Complex64 = forwardDIT8Radix4Complex64
	InverseDIT8Radix4Complex64 = inverseDIT8Radix4Complex64
	// Size 16.
	ForwardDIT16Complex64       = forwardDIT16Complex64
	InverseDIT16Complex64       = inverseDIT16Complex64
	ForwardDIT16Radix4Complex64 = forwardDIT16Complex64
	InverseDIT16Radix4Complex64 = inverseDIT16Complex64
	ForwardDIT16Radix2Complex64 = forwardDIT16Radix2Complex64
	InverseDIT16Radix2Complex64 = inverseDIT16Radix2Complex64
	// Size 32.
	ForwardDIT32Complex64 = forwardDIT32Radix2Complex64
	InverseDIT32Complex64 = inverseDIT32Radix2Complex64
	// Size 64.
	ForwardDIT64Complex64       = forwardDIT64Radix2Complex64
	InverseDIT64Complex64       = inverseDIT64Radix2Complex64
	ForwardDIT64Radix4Complex64 = forwardDIT64Radix4Complex64
	InverseDIT64Radix4Complex64 = inverseDIT64Radix4Complex64
	// Size 128.
	ForwardDIT128Complex64 = forwardDIT128Radix2Complex64
	InverseDIT128Complex64 = inverseDIT128Radix2Complex64
	// Size 256.
	ForwardDIT256Complex64       = forwardDIT256Complex64
	InverseDIT256Complex64       = inverseDIT256Complex64
	ForwardDIT256Radix4Complex64 = forwardDIT256Radix4Complex64
	InverseDIT256Radix4Complex64 = inverseDIT256Radix4Complex64
	// Size 512.
	ForwardDIT512Complex64            = forwardDIT512Complex64
	InverseDIT512Complex64            = inverseDIT512Complex64
	ForwardDIT512Radix4Then2Complex64 = forwardDIT512Radix4Then2Complex64
	InverseDIT512Radix4Then2Complex64 = inverseDIT512Radix4Then2Complex64

	// Complex128 variants.
	ForwardDIT4Radix4Complex128 = forwardDIT4Radix4Complex128
	InverseDIT4Radix4Complex128 = inverseDIT4Radix4Complex128
	// Size 8.
	ForwardDIT8Complex128       = forwardDIT8Complex128
	InverseDIT8Complex128       = inverseDIT8Complex128
	ForwardDIT8Radix8Complex128 = forwardDIT8Complex128
	InverseDIT8Radix8Complex128 = inverseDIT8Complex128
	ForwardDIT8Radix2Complex128 = forwardDIT8Radix2Complex128
	InverseDIT8Radix2Complex128 = inverseDIT8Radix2Complex128
	ForwardDIT8Radix4Complex128 = forwardDIT8Radix4Complex128
	InverseDIT8Radix4Complex128 = inverseDIT8Radix4Complex128
	// Size 16.
	ForwardDIT16Complex128             = forwardDIT16Complex128
	InverseDIT16Complex128             = inverseDIT16Complex128
	ForwardDIT16Radix4Complex128       = forwardDIT16Complex128
	InverseDIT16Radix4Complex128       = inverseDIT16Complex128
	ForwardDIT16Radix2Complex128       = forwardDIT16Radix2Complex128
	InverseDIT16Radix2Complex128       = inverseDIT16Radix2Complex128
	ForwardDIT32Complex128             = forwardDIT32Radix2Complex128
	InverseDIT32Complex128             = inverseDIT32Radix2Complex128
	ForwardDIT64Complex128             = forwardDIT64Radix2Complex128
	InverseDIT64Complex128             = inverseDIT64Radix2Complex128
	ForwardDIT64Radix4Complex128       = forwardDIT64Radix4Complex128
	InverseDIT64Radix4Complex128       = inverseDIT64Radix4Complex128
	ForwardDIT128Complex128            = forwardDIT128Radix2Complex128
	InverseDIT128Complex128            = inverseDIT128Radix2Complex128
	ForwardDIT256Complex128            = forwardDIT256Complex128
	InverseDIT256Complex128            = inverseDIT256Complex128
	ForwardDIT256Radix4Complex128      = forwardDIT256Radix4Complex128
	InverseDIT256Radix4Complex128      = inverseDIT256Radix4Complex128
	ForwardDIT512Complex128            = forwardDIT512Complex128
	InverseDIT512Complex128            = inverseDIT512Complex128
	ForwardDIT512Radix4Then2Complex128 = forwardDIT512Radix4Then2Complex128
	InverseDIT512Radix4Then2Complex128 = inverseDIT512Radix4Then2Complex128
)
