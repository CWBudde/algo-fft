package fft

// forwardDIT16Radix4Complex64 computes a 16-point forward FFT using the
// radix-4 Decimation-in-Time (DIT) algorithm for complex64 data.
// This uses 2 stages of radix-4 butterflies instead of 4 stages of radix-2.
// Fully unrolled for maximum performance.
// Returns false if any slice is too small.
func forwardDIT16Radix4Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 16

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Bounds hints
	br := bitrev[:n]
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 4 radix-4 butterflies with fused bit-reversal
	// No twiddle multiplies (all W^0 = 1)
	var stage1 [16]complex64

	for base := 0; base < n; base += 4 {
		// Load with bit-reversal
		a0 := s[br[base]]
		a1 := s[br[base+1]]
		a2 := s[br[base+2]]
		a3 := s[br[base+3]]

		// Inline radix-4 butterfly
		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[base] = t0 + t2
		stage1[base+2] = t0 - t2
		// mulNegI(t3) = complex(imag(t3), -real(t3))
		stage1[base+1] = t1 + complex(imag(t3), -real(t3))
		// mulI(t3) = complex(-imag(t3), real(t3))
		stage1[base+3] = t1 + complex(-imag(t3), real(t3))
	}

	// Stage 2: 1 group × 4 butterflies
	// Write directly to output or scratch to avoid aliasing
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}
	work = work[:n]

	// For j=0..3, we process 4 butterflies
	// Each butterfly combines indices: j, j+4, j+8, j+12
	// With twiddle factors: W^(0*j), W^(1*j), W^(2*j), W^(3*j)
	for j := range 4 {
		w1 := tw[j]
		w2 := tw[2*j]
		w3 := tw[3*j]

		idx0 := j
		idx1 := j + 4
		idx2 := j + 8
		idx3 := j + 12

		a0 := stage1[idx0]
		a1 := w1 * stage1[idx1]
		a2 := w2 * stage1[idx2]
		a3 := w3 * stage1[idx3]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work[idx0] = t0 + t2
		work[idx2] = t0 - t2
		work[idx1] = t1 + complex(imag(t3), -real(t3))
		work[idx3] = t1 + complex(-imag(t3), real(t3))
	}

	// Copy result back if we used scratch buffer
	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}

// inverseDIT16Radix4Complex64 computes a 16-point inverse FFT using the
// radix-4 Decimation-in-Time (DIT) algorithm for complex64 data.
// Uses conjugated twiddle factors (negated imaginary parts) and applies
// 1/N scaling at the end. Fully unrolled for maximum performance.
// Returns false if any slice is too small.
func inverseDIT16Radix4Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	const n = 16

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Bounds hints
	br := bitrev[:n]
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 4 radix-4 butterflies with fused bit-reversal
	// No twiddle multiplies (all W^0 = 1)
	var stage1 [16]complex64

	for base := 0; base < n; base += 4 {
		// Load with bit-reversal
		a0 := s[br[base]]
		a1 := s[br[base+1]]
		a2 := s[br[base+2]]
		a3 := s[br[base+3]]

		// Inline radix-4 butterfly
		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[base] = t0 + t2
		stage1[base+2] = t0 - t2
		// For inverse: mulI instead of mulNegI
		stage1[base+1] = t1 + complex(-imag(t3), real(t3))
		stage1[base+3] = t1 + complex(imag(t3), -real(t3))
	}

	// Stage 2: 1 group × 4 butterflies with conjugated twiddles
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}
	work = work[:n]

	// For j=0..3, process 4 butterflies with conjugated twiddles
	for j := range 4 {
		w1 := complex(real(tw[j]), -imag(tw[j]))
		w2 := complex(real(tw[2*j]), -imag(tw[2*j]))
		w3 := complex(real(tw[3*j]), -imag(tw[3*j]))

		idx0 := j
		idx1 := j + 4
		idx2 := j + 8
		idx3 := j + 12

		a0 := stage1[idx0]
		a1 := w1 * stage1[idx1]
		a2 := w2 * stage1[idx2]
		a3 := w3 * stage1[idx3]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work[idx0] = t0 + t2
		work[idx2] = t0 - t2
		work[idx1] = t1 + complex(-imag(t3), real(t3))
		work[idx3] = t1 + complex(imag(t3), -real(t3))
	}

	// Copy result back if we used scratch buffer
	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	// Apply 1/N scaling for inverse transform
	scale := complex(float32(1.0/float64(n)), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}

// forwardDIT16Radix4Complex128 computes a 16-point forward FFT using the
// radix-4 Decimation-in-Time (DIT) algorithm for complex128 data.
// This uses 2 stages of radix-4 butterflies instead of 4 stages of radix-2.
// Fully unrolled for maximum performance.
// Returns false if any slice is too small.
func forwardDIT16Radix4Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 16

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Bounds hints
	br := bitrev[:n]
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 4 radix-4 butterflies with fused bit-reversal
	var stage1 [16]complex128

	for base := 0; base < n; base += 4 {
		// Load with bit-reversal
		a0 := s[br[base]]
		a1 := s[br[base+1]]
		a2 := s[br[base+2]]
		a3 := s[br[base+3]]

		// Inline radix-4 butterfly
		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[base] = t0 + t2
		stage1[base+2] = t0 - t2
		stage1[base+1] = t1 + complex(imag(t3), -real(t3))
		stage1[base+3] = t1 + complex(-imag(t3), real(t3))
	}

	// Stage 2: 1 group × 4 butterflies
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}
	work = work[:n]

	// For j=0..3, we process 4 butterflies
	for j := range 4 {
		w1 := tw[j]
		w2 := tw[2*j]
		w3 := tw[3*j]

		idx0 := j
		idx1 := j + 4
		idx2 := j + 8
		idx3 := j + 12

		a0 := stage1[idx0]
		a1 := w1 * stage1[idx1]
		a2 := w2 * stage1[idx2]
		a3 := w3 * stage1[idx3]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work[idx0] = t0 + t2
		work[idx2] = t0 - t2
		work[idx1] = t1 + complex(imag(t3), -real(t3))
		work[idx3] = t1 + complex(-imag(t3), real(t3))
	}

	// Copy result back if we used scratch buffer
	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	return true
}

// inverseDIT16Radix4Complex128 computes a 16-point inverse FFT using the
// radix-4 Decimation-in-Time (DIT) algorithm for complex128 data.
// Uses conjugated twiddle factors and applies 1/N scaling at the end.
// Fully unrolled for maximum performance.
// Returns false if any slice is too small.
func inverseDIT16Radix4Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	const n = 16

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(bitrev) < n || len(src) < n {
		return false
	}

	// Bounds hints
	br := bitrev[:n]
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 4 radix-4 butterflies with fused bit-reversal
	var stage1 [16]complex128

	for base := 0; base < n; base += 4 {
		// Load with bit-reversal
		a0 := s[br[base]]
		a1 := s[br[base+1]]
		a2 := s[br[base+2]]
		a3 := s[br[base+3]]

		// Inline radix-4 butterfly
		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[base] = t0 + t2
		stage1[base+2] = t0 - t2
		// For inverse: mulI instead of mulNegI
		stage1[base+1] = t1 + complex(-imag(t3), real(t3))
		stage1[base+3] = t1 + complex(imag(t3), -real(t3))
	}

	// Stage 2: 1 group × 4 butterflies with conjugated twiddles
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}
	work = work[:n]

	// For j=0..3, process 4 butterflies with conjugated twiddles
	for j := range 4 {
		w1 := complex(real(tw[j]), -imag(tw[j]))
		w2 := complex(real(tw[2*j]), -imag(tw[2*j]))
		w3 := complex(real(tw[3*j]), -imag(tw[3*j]))

		idx0 := j
		idx1 := j + 4
		idx2 := j + 8
		idx3 := j + 12

		a0 := stage1[idx0]
		a1 := w1 * stage1[idx1]
		a2 := w2 * stage1[idx2]
		a3 := w3 * stage1[idx3]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work[idx0] = t0 + t2
		work[idx2] = t0 - t2
		work[idx1] = t1 + complex(-imag(t3), real(t3))
		work[idx3] = t1 + complex(imag(t3), -real(t3))
	}

	// Copy result back if we used scratch buffer
	if &work[0] != &dst[0] {
		copy(dst, work)
	}

	// Apply 1/N scaling for inverse transform
	scale := complex(1.0/float64(n), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}
