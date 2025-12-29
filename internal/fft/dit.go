package fft

// Pre-computed radix-4 bit-reversal indices for size 16
var bitrevSize16Radix4 = ComputeBitReversalIndicesRadix4(16)

func forwardDITComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	switch len(src) {
	case 8:
		return forwardDIT8Complex64(dst, src, twiddle, scratch, bitrev)
	case 16:
		// Use faster radix-4 implementation (12-15% faster than radix-2)
		return forwardDIT16Radix4Complex64(dst, src, twiddle, scratch, bitrevSize16Radix4)
	case 32:
		return forwardDIT32Complex64(dst, src, twiddle, scratch, bitrev)
	case 64:
		return forwardDIT64Complex64(dst, src, twiddle, scratch, bitrev)
	case 128:
		return forwardDIT128Complex64(dst, src, twiddle, scratch, bitrev)
	case 256:
		return forwardDIT256Complex64(dst, src, twiddle, scratch, bitrev)
	case 512:
		return forwardDIT512Complex64(dst, src, twiddle, scratch, bitrev)
	}

	if forwardRadix4Complex64(dst, src, twiddle, scratch, bitrev) {
		return true
	}

	return ditForward[complex64](dst, src, twiddle, scratch, bitrev)
}

func inverseDITComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	switch len(src) {
	case 8:
		return inverseDIT8Complex64(dst, src, twiddle, scratch, bitrev)
	case 16:
		// Use faster radix-4 implementation (12-15% faster than radix-2)
		return inverseDIT16Radix4Complex64(dst, src, twiddle, scratch, bitrevSize16Radix4)
	case 32:
		return inverseDIT32Complex64(dst, src, twiddle, scratch, bitrev)
	case 64:
		return inverseDIT64Complex64(dst, src, twiddle, scratch, bitrev)
	case 128:
		return inverseDIT128Complex64(dst, src, twiddle, scratch, bitrev)
	case 256:
		return inverseDIT256Complex64(dst, src, twiddle, scratch, bitrev)
	case 512:
		return inverseDIT512Complex64(dst, src, twiddle, scratch, bitrev)
	}

	if inverseRadix4Complex64(dst, src, twiddle, scratch, bitrev) {
		return true
	}

	return ditInverseComplex64(dst, src, twiddle, scratch, bitrev)
}

func forwardDITComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	switch len(src) {
	case 8:
		return forwardDIT8Complex128(dst, src, twiddle, scratch, bitrev)
	case 16:
		// Use faster radix-4 implementation (12-15% faster than radix-2)
		return forwardDIT16Radix4Complex128(dst, src, twiddle, scratch, bitrevSize16Radix4)
	case 32:
		return forwardDIT32Complex128(dst, src, twiddle, scratch, bitrev)
	case 64:
		return forwardDIT64Complex128(dst, src, twiddle, scratch, bitrev)
	case 128:
		return forwardDIT128Complex128(dst, src, twiddle, scratch, bitrev)
	case 256:
		return forwardDIT256Complex128(dst, src, twiddle, scratch, bitrev)
	case 512:
		return forwardDIT512Complex128(dst, src, twiddle, scratch, bitrev)
	}

	if forwardRadix4Complex128(dst, src, twiddle, scratch, bitrev) {
		return true
	}

	return ditForward[complex128](dst, src, twiddle, scratch, bitrev)
}

func inverseDITComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	switch len(src) {
	case 8:
		return inverseDIT8Complex128(dst, src, twiddle, scratch, bitrev)
	case 16:
		// Use faster radix-4 implementation (12-15% faster than radix-2)
		return inverseDIT16Radix4Complex128(dst, src, twiddle, scratch, bitrevSize16Radix4)
	case 32:
		return inverseDIT32Complex128(dst, src, twiddle, scratch, bitrev)
	case 64:
		return inverseDIT64Complex128(dst, src, twiddle, scratch, bitrev)
	case 128:
		return inverseDIT128Complex128(dst, src, twiddle, scratch, bitrev)
	case 256:
		return inverseDIT256Complex128(dst, src, twiddle, scratch, bitrev)
	case 512:
		return inverseDIT512Complex128(dst, src, twiddle, scratch, bitrev)
	}

	if inverseRadix4Complex128(dst, src, twiddle, scratch, bitrev) {
		return true
	}

	return ditInverseComplex128(dst, src, twiddle, scratch, bitrev)
}

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
