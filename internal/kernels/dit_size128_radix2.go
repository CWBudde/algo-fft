package kernels

import mathpkg "github.com/cwbudde/algo-fft/internal/math"

// bitrev128Radix2 pre-computes bit-reversal indices for a 128-point radix-2 FFT.
//
//nolint:gochecknoglobals
var bitrev128Radix2 = mathpkg.ComputeBitReversalIndices(128)

// forwardDIT128Complex64 computes a 128-point forward FFT using the
// Decimation-in-Time (DIT) Cooley-Tukey algorithm for complex64 data.
// The algorithm performs 7 stages of butterfly operations (log2(128) = 7).
// Returns false if any slice is too small.
func forwardDIT128Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 128

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
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

	// Stage 1: 64 radix-2 butterflies, stride=2, no twiddles (W^0 = 1)
	// Fuse bit-reversal permutation with stage 1 to save one full pass over work.
	for base := 0; base < n; base += 2 {
		a := src[bitrev128Radix2[base]]
		b := src[bitrev128Radix2[base+1]]
		work[base] = a + b
		work[base+1] = a - b
	}

	// Stage 2: 32 radix-2 butterflies, stride=4
	for base := 0; base < n; base += 4 {
		for j := range 2 {
			tw := twiddle[j*32]
			a, b := butterfly2(work[base+j], work[base+j+2], tw)
			work[base+j] = a
			work[base+j+2] = b
		}
	}

	// Stage 3: 16 radix-2 butterflies, stride=8
	for base := 0; base < n; base += 8 {
		for j := range 4 {
			tw := twiddle[j*16]
			a, b := butterfly2(work[base+j], work[base+j+4], tw)
			work[base+j] = a
			work[base+j+4] = b
		}
	}

	// Stage 4: 8 radix-2 butterflies, stride=16
	for base := 0; base < n; base += 16 {
		for j := range 8 {
			tw := twiddle[j*8]
			a, b := butterfly2(work[base+j], work[base+j+8], tw)
			work[base+j] = a
			work[base+j+8] = b
		}
	}

	// Stage 5: 4 radix-2 butterflies, stride=32
	for base := 0; base < n; base += 32 {
		for j := range 16 {
			tw := twiddle[j*4]
			a, b := butterfly2(work[base+j], work[base+j+16], tw)
			work[base+j] = a
			work[base+j+16] = b
		}
	}

	// Stage 6: 2 radix-2 butterflies, stride=64
	for base := 0; base < n; base += 64 {
		for j := range 32 {
			tw := twiddle[j*2]
			a, b := butterfly2(work[base+j], work[base+j+32], tw)
			work[base+j] = a
			work[base+j+32] = b
		}
	}

	// Stage 7: 1 radix-2 butterfly, stride=128 (full array)
	for j := range 64 {
		tw := twiddle[j]
		a, b := butterfly2(work[j], work[j+64], tw)
		work[j] = a
		work[j+64] = b
	}

	// Copy result back if we used scratch buffer
	if !workIsDst {
		copy(dst, work)
	}

	return true
}

// inverseDIT128Complex64 computes a 128-point inverse FFT using the
// Decimation-in-Time (DIT) algorithm for complex64 data.
// Uses conjugated twiddle factors (negated imaginary parts) and applies
// 1/N scaling at the end. Returns false if any slice is too small.
func inverseDIT128Complex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 128

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
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

	// Stage 1: 64 radix-2 butterflies, stride=2, no twiddles (W^0 = 1)
	// Fuse bit-reversal permutation with stage 1 to save one full pass over work.
	for base := 0; base < n; base += 2 {
		a := src[bitrev128Radix2[base]]
		b := src[bitrev128Radix2[base+1]]
		work[base] = a + b
		work[base+1] = a - b
	}

	// Stage 2: 32 radix-2 butterflies, stride=4
	for base := 0; base < n; base += 4 {
		for j := range 2 {
			tw := twiddle[j*32]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+2], tw)
			work[base+j] = a
			work[base+j+2] = b
		}
	}

	// Stage 3: 16 radix-2 butterflies, stride=8
	for base := 0; base < n; base += 8 {
		for j := range 4 {
			tw := twiddle[j*16]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+4], tw)
			work[base+j] = a
			work[base+j+4] = b
		}
	}

	// Stage 4: 8 radix-2 butterflies, stride=16
	for base := 0; base < n; base += 16 {
		for j := range 8 {
			tw := twiddle[j*8]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+8], tw)
			work[base+j] = a
			work[base+j+8] = b
		}
	}

	// Stage 5: 4 radix-2 butterflies, stride=32
	for base := 0; base < n; base += 32 {
		for j := range 16 {
			tw := twiddle[j*4]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+16], tw)
			work[base+j] = a
			work[base+j+16] = b
		}
	}

	// Stage 6: 2 radix-2 butterflies, stride=64
	for base := 0; base < n; base += 64 {
		for j := range 32 {
			tw := twiddle[j*2]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+32], tw)
			work[base+j] = a
			work[base+j+32] = b
		}
	}

	// Stage 7: 1 radix-2 butterfly, stride=128 (full array)
	for j := range 64 {
		tw := twiddle[j]
		tw = complex(real(tw), -imag(tw))
		a, b := butterfly2(work[j], work[j+64], tw)
		work[j] = a
		work[j+64] = b
	}

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

// forwardDIT128Complex128 computes a 128-point forward FFT using the
// Decimation-in-Time (DIT) Cooley-Tukey algorithm for complex128 data.
// The algorithm performs 7 stages of butterfly operations (log2(128) = 7).
// Returns false if any slice is too small.
func forwardDIT128Complex128(dst, src, twiddle, scratch []complex128) bool {
	const n = 128

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
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

	// Stage 1: 64 radix-2 butterflies, stride=2, no twiddles (W^0 = 1)
	// Fuse bit-reversal permutation with stage 1 to save one full pass over work.
	for base := 0; base < n; base += 2 {
		a := src[bitrev128Radix2[base]]
		b := src[bitrev128Radix2[base+1]]
		work[base] = a + b
		work[base+1] = a - b
	}

	// Stage 2: 32 radix-2 butterflies, stride=4
	for base := 0; base < n; base += 4 {
		for j := range 2 {
			tw := twiddle[j*32]
			a, b := butterfly2(work[base+j], work[base+j+2], tw)
			work[base+j] = a
			work[base+j+2] = b
		}
	}

	// Stage 3: 16 radix-2 butterflies, stride=8
	for base := 0; base < n; base += 8 {
		for j := range 4 {
			tw := twiddle[j*16]
			a, b := butterfly2(work[base+j], work[base+j+4], tw)
			work[base+j] = a
			work[base+j+4] = b
		}
	}

	// Stage 4: 8 radix-2 butterflies, stride=16
	for base := 0; base < n; base += 16 {
		for j := range 8 {
			tw := twiddle[j*8]
			a, b := butterfly2(work[base+j], work[base+j+8], tw)
			work[base+j] = a
			work[base+j+8] = b
		}
	}

	// Stage 5: 4 radix-2 butterflies, stride=32
	for base := 0; base < n; base += 32 {
		for j := range 16 {
			tw := twiddle[j*4]
			a, b := butterfly2(work[base+j], work[base+j+16], tw)
			work[base+j] = a
			work[base+j+16] = b
		}
	}

	// Stage 6: 2 radix-2 butterflies, stride=64
	for base := 0; base < n; base += 64 {
		for j := range 32 {
			tw := twiddle[j*2]
			a, b := butterfly2(work[base+j], work[base+j+32], tw)
			work[base+j] = a
			work[base+j+32] = b
		}
	}

	// Stage 7: 1 radix-2 butterfly, stride=128 (full array)
	for j := range 64 {
		tw := twiddle[j]
		a, b := butterfly2(work[j], work[j+64], tw)
		work[j] = a
		work[j+64] = b
	}

	// Copy result back if we used scratch buffer
	if !workIsDst {
		copy(dst, work)
	}

	return true
}

// inverseDIT128Complex128 computes a 128-point inverse FFT using the
// Decimation-in-Time (DIT) algorithm for complex128 data.
// Uses conjugated twiddle factors (negated imaginary parts) and applies
// 1/N scaling at the end. Returns false if any slice is too small.
func inverseDIT128Complex128(dst, src, twiddle, scratch []complex128) bool {
	const n = 128

	if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
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

	// Stage 1: 64 radix-2 butterflies, stride=2, no twiddles (W^0 = 1)
	// Fuse bit-reversal permutation with stage 1 to save one full pass over work.
	for base := 0; base < n; base += 2 {
		a := src[bitrev128Radix2[base]]
		b := src[bitrev128Radix2[base+1]]
		work[base] = a + b
		work[base+1] = a - b
	}

	// Stage 2: 32 radix-2 butterflies, stride=4
	for base := 0; base < n; base += 4 {
		for j := range 2 {
			tw := twiddle[j*32]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+2], tw)
			work[base+j] = a
			work[base+j+2] = b
		}
	}

	// Stage 3: 16 radix-2 butterflies, stride=8
	for base := 0; base < n; base += 8 {
		for j := range 4 {
			tw := twiddle[j*16]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+4], tw)
			work[base+j] = a
			work[base+j+4] = b
		}
	}

	// Stage 4: 8 radix-2 butterflies, stride=16
	for base := 0; base < n; base += 16 {
		for j := range 8 {
			tw := twiddle[j*8]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+8], tw)
			work[base+j] = a
			work[base+j+8] = b
		}
	}

	// Stage 5: 4 radix-2 butterflies, stride=32
	for base := 0; base < n; base += 32 {
		for j := range 16 {
			tw := twiddle[j*4]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+16], tw)
			work[base+j] = a
			work[base+j+16] = b
		}
	}

	// Stage 6: 2 radix-2 butterflies, stride=64
	for base := 0; base < n; base += 64 {
		for j := range 32 {
			tw := twiddle[j*2]
			tw = complex(real(tw), -imag(tw))
			a, b := butterfly2(work[base+j], work[base+j+32], tw)
			work[base+j] = a
			work[base+j+32] = b
		}
	}

	// Stage 7: 1 radix-2 butterfly, stride=128 (full array)
	for j := range 64 {
		tw := twiddle[j]
		tw = complex(real(tw), -imag(tw))
		a, b := butterfly2(work[j], work[j+64], tw)
		work[j] = a
		work[j+64] = b
	}

	// Copy result back if we used scratch buffer
	if !workIsDst {
		copy(dst, work)
	}

	// Apply 1/N scaling for inverse transform
	scale := complex(1.0/float64(n), 0)
	for i := range dst[:n] {
		dst[i] *= scale
	}

	return true
}
