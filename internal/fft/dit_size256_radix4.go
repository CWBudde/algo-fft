package fft

// forwardDIT256Radix4Complex64 computes a 256-point forward FFT using the
// radix-4 Decimation-in-Time (DIT) algorithm for complex64 data.
// The algorithm performs 4 stages of radix-4 butterfly operations (log4(256) = 4).
// Returns false if any slice is too small.
func forwardDIT256Radix4Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 256

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Use scratch buffer for in-place transforms to avoid aliasing issues
	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	// Bounds hint for compiler optimization
	work = work[:n]
	src = src[:n]
	twiddle = twiddle[:n]
	bitrev = bitrev[:n]

	// Bit-reversal permutation (base-4 bit-reversal for radix-4 FFT)
	// For radix-4, we reverse pairs of bits instead of single bits
	for i := range n {
		work[i] = src[bitReversalRadix4(i, 4)] // 4 stages = 4 digit positions in base-4
	}

	// Stage 1: 64 radix-4 butterflies, stride=4, process groups of size 4
	// Twiddle step: n/4 = 64
	for base := 0; base < n; base += 4 {
		idx0 := base
		idx1 := base + 1
		idx2 := base + 2
		idx3 := base + 3

		// For stage 1, j=0 always, so w1=w2=w3=twiddle[0]=1
		// Optimize by skipping twiddle multiplies
		a0 := work[idx0]
		a1 := work[idx1]
		a2 := work[idx2]
		a3 := work[idx3]

		y0, y1, y2, y3 := butterfly4Forward(a0, a1, a2, a3)

		work[idx0] = y0
		work[idx1] = y1
		work[idx2] = y2
		work[idx3] = y3
	}

	// Stage 2: 16 radix-4 butterflies, stride=16, process groups of size 16
	// Twiddle step: n/16 = 16
	for base := 0; base < n; base += 16 {
		for j := range 4 {
			idx0 := base + j
			idx1 := base + j + 4
			idx2 := base + j + 8
			idx3 := base + j + 12

			w1 := twiddle[j*16]
			w2 := twiddle[2*j*16]
			w3 := twiddle[3*j*16]

			a0 := work[idx0]
			a1 := w1 * work[idx1]
			a2 := w2 * work[idx2]
			a3 := w3 * work[idx3]

			y0, y1, y2, y3 := butterfly4Forward(a0, a1, a2, a3)

			work[idx0] = y0
			work[idx1] = y1
			work[idx2] = y2
			work[idx3] = y3
		}
	}

	// Stage 3: 4 radix-4 butterflies, stride=64, process groups of size 64
	// Twiddle step: n/64 = 4
	for base := 0; base < n; base += 64 {
		for j := range 16 {
			idx0 := base + j
			idx1 := base + j + 16
			idx2 := base + j + 32
			idx3 := base + j + 48

			w1 := twiddle[j*4]
			w2 := twiddle[2*j*4]
			w3 := twiddle[3*j*4]

			a0 := work[idx0]
			a1 := w1 * work[idx1]
			a2 := w2 * work[idx2]
			a3 := w3 * work[idx3]

			y0, y1, y2, y3 := butterfly4Forward(a0, a1, a2, a3)

			work[idx0] = y0
			work[idx1] = y1
			work[idx2] = y2
			work[idx3] = y3
		}
	}

	// Stage 4: 1 radix-4 butterfly, stride=256, process entire array
	// Twiddle step: n/256 = 1
	for j := range 64 {
		idx0 := j
		idx1 := j + 64
		idx2 := j + 128
		idx3 := j + 192

		w1 := twiddle[j]
		w2 := twiddle[2*j]
		w3 := twiddle[3*j]

		a0 := work[idx0]
		a1 := w1 * work[idx1]
		a2 := w2 * work[idx2]
		a3 := w3 * work[idx3]

		y0, y1, y2, y3 := butterfly4Forward(a0, a1, a2, a3)

		work[idx0] = y0
		work[idx1] = y1
		work[idx2] = y2
		work[idx3] = y3
	}

	// Copy back if we used scratch buffer
	if !workIsDst {
		copy(dst, work)
	}

	return true
}

// inverseDIT256Radix4Complex64 computes a 256-point inverse FFT using the
// radix-4 Decimation-in-Time (DIT) algorithm for complex64 data.
// Uses conjugated twiddle factors and applies 1/N scaling at the end.
// Returns false if any slice is too small.
func inverseDIT256Radix4Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 256

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Use scratch buffer for in-place transforms to avoid aliasing issues
	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	// Bounds hint for compiler optimization
	work = work[:n]
	src = src[:n]
	twiddle = twiddle[:n]
	bitrev = bitrev[:n]

	// Bit-reversal permutation (base-4 bit-reversal for radix-4 FFT)
	for i := range n {
		work[i] = src[bitReversalRadix4(i, 4)]
	}

	// Stage 1: 64 radix-4 butterflies, stride=4
	for base := 0; base < n; base += 4 {
		idx0 := base
		idx1 := base + 1
		idx2 := base + 2
		idx3 := base + 3

		a0 := work[idx0]
		a1 := work[idx1]
		a2 := work[idx2]
		a3 := work[idx3]

		y0, y1, y2, y3 := butterfly4Inverse(a0, a1, a2, a3)

		work[idx0] = y0
		work[idx1] = y1
		work[idx2] = y2
		work[idx3] = y3
	}

	// Stage 2: 16 radix-4 butterflies, stride=16
	for base := 0; base < n; base += 16 {
		for j := range 4 {
			idx0 := base + j
			idx1 := base + j + 4
			idx2 := base + j + 8
			idx3 := base + j + 12

			// Conjugate twiddle factors for inverse transform
			w1 := conj(twiddle[j*16])
			w2 := conj(twiddle[2*j*16])
			w3 := conj(twiddle[3*j*16])

			a0 := work[idx0]
			a1 := w1 * work[idx1]
			a2 := w2 * work[idx2]
			a3 := w3 * work[idx3]

			y0, y1, y2, y3 := butterfly4Inverse(a0, a1, a2, a3)

			work[idx0] = y0
			work[idx1] = y1
			work[idx2] = y2
			work[idx3] = y3
		}
	}

	// Stage 3: 4 radix-4 butterflies, stride=64
	for base := 0; base < n; base += 64 {
		for j := range 16 {
			idx0 := base + j
			idx1 := base + j + 16
			idx2 := base + j + 32
			idx3 := base + j + 48

			w1 := conj(twiddle[j*4])
			w2 := conj(twiddle[2*j*4])
			w3 := conj(twiddle[3*j*4])

			a0 := work[idx0]
			a1 := w1 * work[idx1]
			a2 := w2 * work[idx2]
			a3 := w3 * work[idx3]

			y0, y1, y2, y3 := butterfly4Inverse(a0, a1, a2, a3)

			work[idx0] = y0
			work[idx1] = y1
			work[idx2] = y2
			work[idx3] = y3
		}
	}

	// Stage 4: 1 radix-4 butterfly, stride=256
	for j := range 64 {
		idx0 := j
		idx1 := j + 64
		idx2 := j + 128
		idx3 := j + 192

		w1 := conj(twiddle[j])
		w2 := conj(twiddle[2*j])
		w3 := conj(twiddle[3*j])

		a0 := work[idx0]
		a1 := w1 * work[idx1]
		a2 := w2 * work[idx2]
		a3 := w3 * work[idx3]

		y0, y1, y2, y3 := butterfly4Inverse(a0, a1, a2, a3)

		work[idx0] = y0
		work[idx1] = y1
		work[idx2] = y2
		work[idx3] = y3
	}

	// Copy back if we used scratch buffer
	if !workIsDst {
		copy(dst, work)
	}

	// Apply 1/N scaling for inverse transform
	scale := complex(1.0/float32(n), 0)
	for i := range n {
		dst[i] *= scale
	}

	return true
}

// bitReversalRadix4 performs base-4 bit-reversal for radix-4 FFT
// For a value with 'digits' base-4 digits, reverse the digit order
func bitReversalRadix4(x, digits int) int {
	result := 0
	for range digits {
		result = (result << 2) | (x & 0x3) // Extract 2 bits and shift result
		x >>= 2
	}
	return result
}

// ComputeBitReversalIndicesRadix4 precomputes radix-4 bit-reversal indices for size n.
// n must be a power of 4.
func ComputeBitReversalIndicesRadix4(n int) []int {
	if n <= 0 || (n&(n-1)) != 0 {
		return nil // Not a power of 2
	}

	// Calculate number of base-4 digits
	digits := 0
	temp := n
	for temp > 1 {
		if temp&0x3 != 0 {
			return nil // Not a power of 4
		}
		digits++
		temp >>= 2
	}

	indices := make([]int, n)
	for i := 0; i < n; i++ {
		indices[i] = bitReversalRadix4(i, digits)
	}
	return indices
}
