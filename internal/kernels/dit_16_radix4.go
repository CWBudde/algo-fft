package kernels

// forwardDIT16Radix4Complex64 computes a 16-point forward FFT using the
// radix-4 Decimation-in-Time (DIT) algorithm for complex64 data.
// This uses 2 stages of radix-4 butterflies instead of 4 stages of radix-2.
// Fully unrolled for maximum performance.
// Returns false if any slice is too small.
func forwardDIT16Radix4Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 16

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	// Bounds hints
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 4 radix-4 butterflies with fused bit-reversal (FULLY UNROLLED)
	var stage1 [16]complex64

	// Butterfly 0: indices 0,1,2,3
	{
		a0 := s[0]
		a1 := s[4]
		a2 := s[8]
		a3 := s[12]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[0] = t0 + t2
		stage1[2] = t0 - t2
		stage1[1] = t1 + complex(imag(t3), -real(t3))
		stage1[3] = t1 + complex(-imag(t3), real(t3))
	}

	// Butterfly 1: indices 4,5,6,7
	{
		a0 := s[1]
		a1 := s[5]
		a2 := s[9]
		a3 := s[13]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[4] = t0 + t2
		stage1[6] = t0 - t2
		stage1[5] = t1 + complex(imag(t3), -real(t3))
		stage1[7] = t1 + complex(-imag(t3), real(t3))
	}

	// Butterfly 2: indices 8,9,10,11
	{
		a0 := s[2]
		a1 := s[6]
		a2 := s[10]
		a3 := s[14]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[8] = t0 + t2
		stage1[10] = t0 - t2
		stage1[9] = t1 + complex(imag(t3), -real(t3))
		stage1[11] = t1 + complex(-imag(t3), real(t3))
	}

	// Butterfly 3: indices 12,13,14,15
	{
		a0 := s[3]
		a1 := s[7]
		a2 := s[11]
		a3 := s[15]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[12] = t0 + t2
		stage1[14] = t0 - t2
		stage1[13] = t1 + complex(imag(t3), -real(t3))
		stage1[15] = t1 + complex(-imag(t3), real(t3))
	}

	// Stage 2: 4 radix-4 butterflies (FULLY UNROLLED)
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	// Butterfly 0: j=0, indices 0,4,8,12 with twiddles W^0, W^0
	{
		w1 := tw[0]
		w2 := tw[0]
		w3 := tw[0]

		a0 := stage1[0]
		a1 := w1 * stage1[4]
		a2 := w2 * stage1[8]
		a3 := w3 * stage1[12]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work[0] = t0 + t2
		work[8] = t0 - t2
		work[4] = t1 + complex(imag(t3), -real(t3))
		work[12] = t1 + complex(-imag(t3), real(t3))
	}

	// Butterfly 1: j=1, indices 1,5,9,13 with twiddles W^1, W^2, W^3
	{
		w1 := tw[1]
		w2 := tw[2]
		w3 := tw[3]

		a0 := stage1[1]
		a1 := w1 * stage1[5]
		a2 := w2 * stage1[9]
		a3 := w3 * stage1[13]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work[1] = t0 + t2
		work[9] = t0 - t2
		work[5] = t1 + complex(imag(t3), -real(t3))
		work[13] = t1 + complex(-imag(t3), real(t3))
	}

	// Butterfly 2: j=2, indices 2,6,10,14 with twiddles W^2, W^4, W^6
	{
		w1 := tw[2]
		w2 := tw[4]
		w3 := tw[6]

		a0 := stage1[2]
		a1 := w1 * stage1[6]
		a2 := w2 * stage1[10]
		a3 := w3 * stage1[14]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work[2] = t0 + t2
		work[10] = t0 - t2
		work[6] = t1 + complex(imag(t3), -real(t3))
		work[14] = t1 + complex(-imag(t3), real(t3))
	}

	// Butterfly 3: j=3, indices 3,7,11,15 with twiddles W^3, W^6, W^9
	{
		w1 := tw[3]
		w2 := tw[6]
		w3 := tw[9]

		a0 := stage1[3]
		a1 := w1 * stage1[7]
		a2 := w2 * stage1[11]
		a3 := w3 * stage1[15]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work[3] = t0 + t2
		work[11] = t0 - t2
		work[7] = t1 + complex(imag(t3), -real(t3))
		work[15] = t1 + complex(-imag(t3), real(t3))
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
//
//nolint:funlen
func inverseDIT16Radix4Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 16

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	// Bounds hints
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 4 radix-4 butterflies with fused bit-reversal (FULLY UNROLLED)
	var stage1 [16]complex64

	// Butterfly 0: indices 0,1,2,3
	{
		a0 := s[0]
		a1 := s[4]
		a2 := s[8]
		a3 := s[12]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[0] = t0 + t2
		stage1[2] = t0 - t2
		// For inverse: mulI instead of mulNegI
		stage1[1] = t1 + complex(-imag(t3), real(t3))
		stage1[3] = t1 + complex(imag(t3), -real(t3))
	}

	// Butterfly 1: indices 4,5,6,7
	{
		a0 := s[1]
		a1 := s[5]
		a2 := s[9]
		a3 := s[13]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[4] = t0 + t2
		stage1[6] = t0 - t2
		stage1[5] = t1 + complex(-imag(t3), real(t3))
		stage1[7] = t1 + complex(imag(t3), -real(t3))
	}

	// Butterfly 2: indices 8,9,10,11
	{
		a0 := s[2]
		a1 := s[6]
		a2 := s[10]
		a3 := s[14]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[8] = t0 + t2
		stage1[10] = t0 - t2
		stage1[9] = t1 + complex(-imag(t3), real(t3))
		stage1[11] = t1 + complex(imag(t3), -real(t3))
	}

	// Butterfly 3: indices 12,13,14,15
	{
		a0 := s[3]
		a1 := s[7]
		a2 := s[11]
		a3 := s[15]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[12] = t0 + t2
		stage1[14] = t0 - t2
		stage1[13] = t1 + complex(-imag(t3), real(t3))
		stage1[15] = t1 + complex(imag(t3), -real(t3))
	}

	// Stage 2: 4 radix-4 butterflies with conjugated twiddles (FULLY UNROLLED)
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	// Butterfly 0: j=0, indices 0,4,8,12
	{
		w1 := complex(real(tw[0]), -imag(tw[0]))
		w2 := complex(real(tw[0]), -imag(tw[0]))
		w3 := complex(real(tw[0]), -imag(tw[0]))

		a0 := stage1[0]
		a1 := w1 * stage1[4]
		a2 := w2 * stage1[8]
		a3 := w3 * stage1[12]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work[0] = t0 + t2
		work[8] = t0 - t2
		work[4] = t1 + complex(-imag(t3), real(t3))
		work[12] = t1 + complex(imag(t3), -real(t3))
	}

	// Butterfly 1: j=1, indices 1,5,9,13
	{
		w1 := complex(real(tw[1]), -imag(tw[1]))
		w2 := complex(real(tw[2]), -imag(tw[2]))
		w3 := complex(real(tw[3]), -imag(tw[3]))

		a0 := stage1[1]
		a1 := w1 * stage1[5]
		a2 := w2 * stage1[9]
		a3 := w3 * stage1[13]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work[1] = t0 + t2
		work[9] = t0 - t2
		work[5] = t1 + complex(-imag(t3), real(t3))
		work[13] = t1 + complex(imag(t3), -real(t3))
	}

	// Butterfly 2: j=2, indices 2,6,10,14
	{
		w1 := complex(real(tw[2]), -imag(tw[2]))
		w2 := complex(real(tw[4]), -imag(tw[4]))
		w3 := complex(real(tw[6]), -imag(tw[6]))

		a0 := stage1[2]
		a1 := w1 * stage1[6]
		a2 := w2 * stage1[10]
		a3 := w3 * stage1[14]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work[2] = t0 + t2
		work[10] = t0 - t2
		work[6] = t1 + complex(-imag(t3), real(t3))
		work[14] = t1 + complex(imag(t3), -real(t3))
	}

	// Butterfly 3: j=3, indices 3,7,11,15
	{
		w1 := complex(real(tw[3]), -imag(tw[3]))
		w2 := complex(real(tw[6]), -imag(tw[6]))
		w3 := complex(real(tw[9]), -imag(tw[9]))

		a0 := stage1[3]
		a1 := w1 * stage1[7]
		a2 := w2 * stage1[11]
		a3 := w3 * stage1[15]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work[3] = t0 + t2
		work[11] = t0 - t2
		work[7] = t1 + complex(-imag(t3), real(t3))
		work[15] = t1 + complex(imag(t3), -real(t3))
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
func forwardDIT16Radix4Complex128(dst, src, twiddle, scratch []complex128) bool {
	const n = 16

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	// Bounds hints
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 4 radix-4 butterflies with fused bit-reversal (FULLY UNROLLED)
	var stage1 [16]complex128

	// Butterfly 0: indices 0,1,2,3
	{
		a0 := s[0]
		a1 := s[4]
		a2 := s[8]
		a3 := s[12]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[0] = t0 + t2
		stage1[2] = t0 - t2
		stage1[1] = t1 + complex(imag(t3), -real(t3))
		stage1[3] = t1 + complex(-imag(t3), real(t3))
	}

	// Butterfly 1: indices 4,5,6,7
	{
		a0 := s[1]
		a1 := s[5]
		a2 := s[9]
		a3 := s[13]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[4] = t0 + t2
		stage1[6] = t0 - t2
		stage1[5] = t1 + complex(imag(t3), -real(t3))
		stage1[7] = t1 + complex(-imag(t3), real(t3))
	}

	// Butterfly 2: indices 8,9,10,11
	{
		a0 := s[2]
		a1 := s[6]
		a2 := s[10]
		a3 := s[14]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[8] = t0 + t2
		stage1[10] = t0 - t2
		stage1[9] = t1 + complex(imag(t3), -real(t3))
		stage1[11] = t1 + complex(-imag(t3), real(t3))
	}

	// Butterfly 3: indices 12,13,14,15
	{
		a0 := s[3]
		a1 := s[7]
		a2 := s[11]
		a3 := s[15]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[12] = t0 + t2
		stage1[14] = t0 - t2
		stage1[13] = t1 + complex(imag(t3), -real(t3))
		stage1[15] = t1 + complex(-imag(t3), real(t3))
	}

	// Stage 2: 4 radix-4 butterflies (FULLY UNROLLED)
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	// Butterfly 0: j=0, indices 0,4,8,12
	{
		w1 := tw[0]
		w2 := tw[0]
		w3 := tw[0]

		a0 := stage1[0]
		a1 := w1 * stage1[4]
		a2 := w2 * stage1[8]
		a3 := w3 * stage1[12]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work[0] = t0 + t2
		work[8] = t0 - t2
		work[4] = t1 + complex(imag(t3), -real(t3))
		work[12] = t1 + complex(-imag(t3), real(t3))
	}

	// Butterfly 1: j=1, indices 1,5,9,13
	{
		w1 := tw[1]
		w2 := tw[2]
		w3 := tw[3]

		a0 := stage1[1]
		a1 := w1 * stage1[5]
		a2 := w2 * stage1[9]
		a3 := w3 * stage1[13]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work[1] = t0 + t2
		work[9] = t0 - t2
		work[5] = t1 + complex(imag(t3), -real(t3))
		work[13] = t1 + complex(-imag(t3), real(t3))
	}

	// Butterfly 2: j=2, indices 2,6,10,14
	{
		w1 := tw[2]
		w2 := tw[4]
		w3 := tw[6]

		a0 := stage1[2]
		a1 := w1 * stage1[6]
		a2 := w2 * stage1[10]
		a3 := w3 * stage1[14]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work[2] = t0 + t2
		work[10] = t0 - t2
		work[6] = t1 + complex(imag(t3), -real(t3))
		work[14] = t1 + complex(-imag(t3), real(t3))
	}

	// Butterfly 3: j=3, indices 3,7,11,15
	{
		w1 := tw[3]
		w2 := tw[6]
		w3 := tw[9]

		a0 := stage1[3]
		a1 := w1 * stage1[7]
		a2 := w2 * stage1[11]
		a3 := w3 * stage1[15]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work[3] = t0 + t2
		work[11] = t0 - t2
		work[7] = t1 + complex(imag(t3), -real(t3))
		work[15] = t1 + complex(-imag(t3), real(t3))
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
//
//nolint:funlen
func inverseDIT16Radix4Complex128(dst, src, twiddle, scratch []complex128) bool {
	const n = 16

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
		return false
	}

	// Bounds hints
	s := src[:n]
	tw := twiddle[:n]

	// Stage 1: 4 radix-4 butterflies with fused bit-reversal (FULLY UNROLLED)
	var stage1 [16]complex128

	// Butterfly 0: indices 0,1,2,3
	{
		a0 := s[0]
		a1 := s[4]
		a2 := s[8]
		a3 := s[12]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[0] = t0 + t2
		stage1[2] = t0 - t2
		stage1[1] = t1 + complex(-imag(t3), real(t3))
		stage1[3] = t1 + complex(imag(t3), -real(t3))
	}

	// Butterfly 1: indices 4,5,6,7
	{
		a0 := s[1]
		a1 := s[5]
		a2 := s[9]
		a3 := s[13]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[4] = t0 + t2
		stage1[6] = t0 - t2
		stage1[5] = t1 + complex(-imag(t3), real(t3))
		stage1[7] = t1 + complex(imag(t3), -real(t3))
	}

	// Butterfly 2: indices 8,9,10,11
	{
		a0 := s[2]
		a1 := s[6]
		a2 := s[10]
		a3 := s[14]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[8] = t0 + t2
		stage1[10] = t0 - t2
		stage1[9] = t1 + complex(-imag(t3), real(t3))
		stage1[11] = t1 + complex(imag(t3), -real(t3))
	}

	// Butterfly 3: indices 12,13,14,15
	{
		a0 := s[3]
		a1 := s[7]
		a2 := s[11]
		a3 := s[15]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		stage1[12] = t0 + t2
		stage1[14] = t0 - t2
		stage1[13] = t1 + complex(-imag(t3), real(t3))
		stage1[15] = t1 + complex(imag(t3), -real(t3))
	}

	// Stage 2: 4 radix-4 butterflies with conjugated twiddles (FULLY UNROLLED)
	work := dst
	if &dst[0] == &src[0] {
		work = scratch
	}

	work = work[:n]

	// Butterfly 0: j=0, indices 0,4,8,12
	{
		w1 := complex(real(tw[0]), -imag(tw[0]))
		w2 := complex(real(tw[0]), -imag(tw[0]))
		w3 := complex(real(tw[0]), -imag(tw[0]))

		a0 := stage1[0]
		a1 := w1 * stage1[4]
		a2 := w2 * stage1[8]
		a3 := w3 * stage1[12]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work[0] = t0 + t2
		work[8] = t0 - t2
		work[4] = t1 + complex(-imag(t3), real(t3))
		work[12] = t1 + complex(imag(t3), -real(t3))
	}

	// Butterfly 1: j=1, indices 1,5,9,13
	{
		w1 := complex(real(tw[1]), -imag(tw[1]))
		w2 := complex(real(tw[2]), -imag(tw[2]))
		w3 := complex(real(tw[3]), -imag(tw[3]))

		a0 := stage1[1]
		a1 := w1 * stage1[5]
		a2 := w2 * stage1[9]
		a3 := w3 * stage1[13]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work[1] = t0 + t2
		work[9] = t0 - t2
		work[5] = t1 + complex(-imag(t3), real(t3))
		work[13] = t1 + complex(imag(t3), -real(t3))
	}

	// Butterfly 2: j=2, indices 2,6,10,14
	{
		w1 := complex(real(tw[2]), -imag(tw[2]))
		w2 := complex(real(tw[4]), -imag(tw[4]))
		w3 := complex(real(tw[6]), -imag(tw[6]))

		a0 := stage1[2]
		a1 := w1 * stage1[6]
		a2 := w2 * stage1[10]
		a3 := w3 * stage1[14]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work[2] = t0 + t2
		work[10] = t0 - t2
		work[6] = t1 + complex(-imag(t3), real(t3))
		work[14] = t1 + complex(imag(t3), -real(t3))
	}

	// Butterfly 3: j=3, indices 3,7,11,15
	{
		w1 := complex(real(tw[3]), -imag(tw[3]))
		w2 := complex(real(tw[6]), -imag(tw[6]))
		w3 := complex(real(tw[9]), -imag(tw[9]))

		a0 := stage1[3]
		a1 := w1 * stage1[7]
		a2 := w2 * stage1[11]
		a3 := w3 * stage1[15]

		t0 := a0 + a2
		t1 := a0 - a2
		t2 := a1 + a3
		t3 := a1 - a3

		work[3] = t0 + t2
		work[11] = t0 - t2
		work[7] = t1 + complex(-imag(t3), real(t3))
		work[15] = t1 + complex(imag(t3), -real(t3))
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
