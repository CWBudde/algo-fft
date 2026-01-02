package kernels

// forwardDIT32MixedRadix24Complex64 computes a size-32 FFT using a mixed
// radix-4/4/2 decomposition (two radix-4 stages, then one radix-2 stage).
func forwardDIT32MixedRadix24Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 32
	if len(dst) < n || len(src) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n {
		return false
	}

	work := dst[:n]
	workIsDst := true
	if &dst[0] == &src[0] {
		work = scratch[:n]
		workIsDst = false
	}

	br := bitrev[:n]
	s := src[:n]
	for i := range n {
		work[i] = s[br[i]]
	}

	// Stage 1+2 fused: radix-4, size=4 (no twiddles).
	for base := 0; base < n; base += 4 {
		a0 := work[base+0]
		a1 := work[base+1]
		a2 := work[base+2]
		a3 := work[base+3]

		y0, y1, y2, y3 := butterfly4Forward(a0, a1, a2, a3)

		work[base+0] = y0
		work[base+1] = y1
		work[base+2] = y2
		work[base+3] = y3
	}

	// Stage 3+4 fused: radix-4, size=16 (twiddle step = 2).
	for base := 0; base < n; base += 16 {
		for j := 0; j < 4; j++ {
			idx0 := base + j
			idx1 := idx0 + 4
			idx2 := idx1 + 4
			idx3 := idx2 + 4

			a0 := work[idx0]
			a1 := twiddle[j*2] * work[idx1]
			a2 := twiddle[j*4] * work[idx2]
			a3 := twiddle[j*6] * work[idx3]

			y0, y1, y2, y3 := butterfly4Forward(a0, a1, a2, a3)

			work[idx0] = y0
			work[idx1] = y1
			work[idx2] = y2
			work[idx3] = y3
		}
	}

	// Final stage: radix-2, size=32.
	for j := 0; j < 16; j++ {
		idx0 := j
		idx1 := j + 16
		a0 := work[idx0]
		t := twiddle[j] * work[idx1]
		work[idx0] = a0 + t
		work[idx1] = a0 - t
	}

	if !workIsDst {
		copy(dst[:n], work)
	}

	return true
}

// inverseDIT32MixedRadix24Complex64 computes the inverse size-32 FFT using the
// same radix-4/4/2 decomposition and applies 1/N scaling.
func inverseDIT32MixedRadix24Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 32
	if len(dst) < n || len(src) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n {
		return false
	}

	work := dst[:n]
	workIsDst := true
	if &dst[0] == &src[0] {
		work = scratch[:n]
		workIsDst = false
	}

	br := bitrev[:n]
	s := src[:n]
	for i := range n {
		work[i] = s[br[i]]
	}

	// Stage 1+2 fused: radix-4, size=4 (no twiddles).
	for base := 0; base < n; base += 4 {
		a0 := work[base+0]
		a1 := work[base+1]
		a2 := work[base+2]
		a3 := work[base+3]

		y0, y1, y2, y3 := butterfly4Inverse(a0, a1, a2, a3)

		work[base+0] = y0
		work[base+1] = y1
		work[base+2] = y2
		work[base+3] = y3
	}

	// Stage 3+4 fused: radix-4, size=16 (twiddle step = 2).
	for base := 0; base < n; base += 16 {
		for j := 0; j < 4; j++ {
			idx0 := base + j
			idx1 := idx0 + 4
			idx2 := idx1 + 4
			idx3 := idx2 + 4

			a0 := work[idx0]
			a1 := conj(twiddle[j*2]) * work[idx1]
			a2 := conj(twiddle[j*4]) * work[idx2]
			a3 := conj(twiddle[j*6]) * work[idx3]

			y0, y1, y2, y3 := butterfly4Inverse(a0, a1, a2, a3)

			work[idx0] = y0
			work[idx1] = y1
			work[idx2] = y2
			work[idx3] = y3
		}
	}

	// Final stage: radix-2, size=32.
	for j := 0; j < 16; j++ {
		idx0 := j
		idx1 := j + 16
		a0 := work[idx0]
		t := conj(twiddle[j]) * work[idx1]
		work[idx0] = a0 + t
		work[idx1] = a0 - t
	}

	if !workIsDst {
		copy(dst[:n], work)
	}

	scale := complexFromFloat64[complex64](1.0/float64(n), 0)
	for i := range n {
		dst[i] *= scale
	}

	return true
}

// forwardDIT32MixedRadix24Complex128 computes a size-32 FFT using a mixed
// radix-4/4/2 decomposition (two radix-4 stages, then one radix-2 stage).
func forwardDIT32MixedRadix24Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 32
	if len(dst) < n || len(src) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n {
		return false
	}

	work := dst[:n]
	workIsDst := true
	if &dst[0] == &src[0] {
		work = scratch[:n]
		workIsDst = false
	}

	br := bitrev[:n]
	s := src[:n]
	for i := range n {
		work[i] = s[br[i]]
	}

	// Stage 1+2 fused: radix-4, size=4 (no twiddles).
	for base := 0; base < n; base += 4 {
		a0 := work[base+0]
		a1 := work[base+1]
		a2 := work[base+2]
		a3 := work[base+3]

		y0, y1, y2, y3 := butterfly4Forward(a0, a1, a2, a3)

		work[base+0] = y0
		work[base+1] = y1
		work[base+2] = y2
		work[base+3] = y3
	}

	// Stage 3+4 fused: radix-4, size=16 (twiddle step = 2).
	for base := 0; base < n; base += 16 {
		for j := 0; j < 4; j++ {
			idx0 := base + j
			idx1 := idx0 + 4
			idx2 := idx1 + 4
			idx3 := idx2 + 4

			a0 := work[idx0]
			a1 := twiddle[j*2] * work[idx1]
			a2 := twiddle[j*4] * work[idx2]
			a3 := twiddle[j*6] * work[idx3]

			y0, y1, y2, y3 := butterfly4Forward(a0, a1, a2, a3)

			work[idx0] = y0
			work[idx1] = y1
			work[idx2] = y2
			work[idx3] = y3
		}
	}

	// Final stage: radix-2, size=32.
	for j := 0; j < 16; j++ {
		idx0 := j
		idx1 := j + 16
		a0 := work[idx0]
		t := twiddle[j] * work[idx1]
		work[idx0] = a0 + t
		work[idx1] = a0 - t
	}

	if !workIsDst {
		copy(dst[:n], work)
	}

	return true
}

// inverseDIT32MixedRadix24Complex128 computes the inverse size-32 FFT using the
// same radix-4/4/2 decomposition and applies 1/N scaling.
func inverseDIT32MixedRadix24Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 32
	if len(dst) < n || len(src) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n {
		return false
	}

	work := dst[:n]
	workIsDst := true
	if &dst[0] == &src[0] {
		work = scratch[:n]
		workIsDst = false
	}

	br := bitrev[:n]
	s := src[:n]
	for i := range n {
		work[i] = s[br[i]]
	}

	// Stage 1+2 fused: radix-4, size=4 (no twiddles).
	for base := 0; base < n; base += 4 {
		a0 := work[base+0]
		a1 := work[base+1]
		a2 := work[base+2]
		a3 := work[base+3]

		y0, y1, y2, y3 := butterfly4Inverse(a0, a1, a2, a3)

		work[base+0] = y0
		work[base+1] = y1
		work[base+2] = y2
		work[base+3] = y3
	}

	// Stage 3+4 fused: radix-4, size=16 (twiddle step = 2).
	for base := 0; base < n; base += 16 {
		for j := 0; j < 4; j++ {
			idx0 := base + j
			idx1 := idx0 + 4
			idx2 := idx1 + 4
			idx3 := idx2 + 4

			a0 := work[idx0]
			a1 := conj(twiddle[j*2]) * work[idx1]
			a2 := conj(twiddle[j*4]) * work[idx2]
			a3 := conj(twiddle[j*6]) * work[idx3]

			y0, y1, y2, y3 := butterfly4Inverse(a0, a1, a2, a3)

			work[idx0] = y0
			work[idx1] = y1
			work[idx2] = y2
			work[idx3] = y3
		}
	}

	// Final stage: radix-2, size=32.
	for j := 0; j < 16; j++ {
		idx0 := j
		idx1 := j + 16
		a0 := work[idx0]
		t := conj(twiddle[j]) * work[idx1]
		work[idx0] = a0 + t
		work[idx1] = a0 - t
	}

	if !workIsDst {
		copy(dst[:n], work)
	}

	scale := complexFromFloat64[complex128](1.0/float64(n), 0)
	for i := range n {
		dst[i] *= scale
	}

	return true
}
