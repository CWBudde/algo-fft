package fft

// forwardMixedRadix24Complex64 computes a forward FFT using mixed-radix-2/4
// Decimation-in-Time (DIT) algorithm for complex64 data.
//
// This function is optimized for power-of-2 sizes with ODD log2 exponents
// (e.g., 8, 32, 128, 512, 2048, 8192) which cannot use pure radix-4.
//
// Algorithm:
//   - Stage 1: ONE radix-2 stage (standard DIT butterfly)
//   - Stages 2+: Pure radix-4 stages (reusing butterfly4Forward)
//
// This reduces the total number of stages significantly:
//   - Size 512 (2^9): 9 radix-2 stages → 1 radix-2 + 4 radix-4 = 5 total
//   - Size 2048 (2^11): 11 radix-2 → 1 radix-2 + 5 radix-4 = 6 total
//
// Expected speedup: 30-40% over pure radix-2 for affected sizes.
//
// Returns false if any slice is too small or if size is not power-of-2.
func forwardMixedRadix24Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	n := len(src)

	// Validate inputs
	if n == 0 {
		return true
	}

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n {
		return false
	}

	// Only works for power-of-2 sizes
	if !IsPowerOf2(n) {
		return false
	}

	if n == 1 {
		dst[0] = src[0]
		return true
	}

	log2n := log2(n)

	// If even log2, delegate to pure radix-4
	if log2n%2 == 0 {
		return radix4Forward[complex64](dst, src, twiddle, scratch, bitrev)
	}

	// Mixed-radix-2/4 path for odd log2

	// Determine working buffer (handle dst == src case)
	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	// Bounds hints for compiler optimization
	work = work[:n]
	src = src[:n]
	twiddle = twiddle[:n]
	bitrev = bitrev[:n]

	// Apply standard DIT bit-reversal permutation
	for i := range n {
		work[i] = src[bitrev[i]]
	}

	// Stage 1: ONE radix-2 stage (W^0 = 1, no twiddle multiplication needed)
	// This reduces the problem to power-of-4
	for base := 0; base < n; base += 2 {
		a := work[base]
		b := work[base+1]
		work[base] = a + b
		work[base+1] = a - b
	}

	// Stages 2+: Pure radix-4 butterfly stages
	// After the radix-2 stage, we have (log2n - 1) remaining bits
	// which is now even, so we can use (log2n - 1) / 2 radix-4 stages
	numRadix4Stages := (log2n - 1) / 2

	for stage := 0; stage < numRadix4Stages; stage++ {
		// Size for this radix-4 stage: 4, 16, 64, 256, 1024...
		size := 4 << (stage * 2)
		quarter := size / 4
		step := n / size

		for base := 0; base < n; base += size {
			for j := range quarter {
				idx0 := base + j
				idx1 := idx0 + quarter
				idx2 := idx1 + quarter
				idx3 := idx2 + quarter

				// Load twiddle factors for radix-4 butterfly
				w1 := twiddle[j*step]
				w2 := twiddle[2*j*step]
				w3 := twiddle[3*j*step]

				// Apply twiddle factors
				a0 := work[idx0]
				a1 := w1 * work[idx1]
				a2 := w2 * work[idx2]
				a3 := w3 * work[idx3]

				// Radix-4 butterfly (from radix4.go)
				y0, y1, y2, y3 := butterfly4Forward(a0, a1, a2, a3)

				// Store results
				work[idx0] = y0
				work[idx1] = y1
				work[idx2] = y2
				work[idx3] = y3
			}
		}
	}

	// Copy result back if we used scratch buffer
	if !workIsDst {
		copy(dst, work)
	}

	return true
}

// inverseMixedRadix24Complex64 computes an inverse FFT using mixed-radix-2/4
// Decimation-in-Time (DIT) algorithm for complex64 data.
//
// This function is optimized for power-of-2 sizes with ODD log2 exponents
// (e.g., 8, 32, 128, 512, 2048, 8192) which cannot use pure radix-4.
//
// Algorithm:
//   - Stage 1: ONE radix-2 stage (standard DIT butterfly)
//   - Stages 2+: Pure radix-4 stages (reusing butterfly4Inverse)
//   - Final: 1/N scaling
//
// This reduces the total number of stages significantly:
//   - Size 512 (2^9): 9 radix-2 stages → 1 radix-2 + 4 radix-4 = 5 total
//   - Size 2048 (2^11): 11 radix-2 → 1 radix-2 + 5 radix-4 = 6 total
//
// Expected speedup: 30-40% over pure radix-2 for affected sizes.
//
// Returns false if any slice is too small or if size is not power-of-2.
func inverseMixedRadix24Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	n := len(src)

	// Validate inputs
	if n == 0 {
		return true
	}

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n {
		return false
	}

	// Only works for power-of-2 sizes
	if !IsPowerOf2(n) {
		return false
	}

	if n == 1 {
		dst[0] = src[0]
		return true
	}

	log2n := log2(n)

	// If even log2, delegate to pure radix-4
	if log2n%2 == 0 {
		return radix4Inverse[complex64](dst, src, twiddle, scratch, bitrev)
	}

	// Mixed-radix-2/4 path for odd log2

	// Determine working buffer (handle dst == src case)
	work := dst
	workIsDst := true

	if sameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	// Bounds hints for compiler optimization
	work = work[:n]
	src = src[:n]
	twiddle = twiddle[:n]
	bitrev = bitrev[:n]

	// Apply standard DIT bit-reversal permutation
	for i := range n {
		work[i] = src[bitrev[i]]
	}

	// Stage 1: ONE radix-2 stage (W^0 = 1, no twiddle multiplication needed)
	// This reduces the problem to power-of-4
	for base := 0; base < n; base += 2 {
		a := work[base]
		b := work[base+1]
		work[base] = a + b
		work[base+1] = a - b
	}

	// Stages 2+: Pure radix-4 butterfly stages with conjugated twiddles
	// After the radix-2 stage, we have (log2n - 1) remaining bits
	// which is now even, so we can use (log2n - 1) / 2 radix-4 stages
	numRadix4Stages := (log2n - 1) / 2

	for stage := 0; stage < numRadix4Stages; stage++ {
		// Size for this radix-4 stage: 4, 16, 64, 256, 1024...
		size := 4 << (stage * 2)
		quarter := size / 4
		step := n / size

		for base := 0; base < n; base += size {
			for j := range quarter {
				idx0 := base + j
				idx1 := idx0 + quarter
				idx2 := idx1 + quarter
				idx3 := idx2 + quarter

				// Load and conjugate twiddle factors for inverse transform
				w1 := twiddle[j*step]
				w2 := twiddle[2*j*step]
				w3 := twiddle[3*j*step]

				w1 = complex(real(w1), -imag(w1))
				w2 = complex(real(w2), -imag(w2))
				w3 = complex(real(w3), -imag(w3))

				// Apply twiddle factors
				a0 := work[idx0]
				a1 := w1 * work[idx1]
				a2 := w2 * work[idx2]
				a3 := w3 * work[idx3]

				// Radix-4 inverse butterfly (from radix4.go)
				y0, y1, y2, y3 := butterfly4Inverse(a0, a1, a2, a3)

				// Store results
				work[idx0] = y0
				work[idx1] = y1
				work[idx2] = y2
				work[idx3] = y3
			}
		}
	}

	// Copy result back if we used scratch buffer
	if !workIsDst {
		copy(dst, work)
	}

	// Apply 1/N scaling for inverse transform
	scale := complex(float32(1.0/float64(n)), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}
