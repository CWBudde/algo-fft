package kernels

import (
	"github.com/cwbudde/algo-fft/internal/math"
)

// This file does not implement an eight-step FFT. Its transform bodies are
// sixstep.go's with the names changed — same perfect-square rejection, same two
// math.TransposeSquare-bracketed Stockham row passes, same twiddle stage. A
// normalised diff of the two files (2026-08-02) leaves only the `stdmath`
// import and the fillRowTwiddle/intSqrt helpers that sixstep.go hosts for both.
//
// The consequence is that KernelEightStep is a second name for KernelSixStep,
// not a second algorithm, and benchmarking one against the other measures
// noise. That is why the eight-step family's matrix verdict is `untested`
// rather than a measured loss (PLAN.md §2.2: a poor implementation disqualifies
// the file, not the algorithm) and why BenchmarkStepCrossover carries no
// eight-step arm. See docs/CODELET_BENCHMARKS.md, "Eight-step is six-step".
//
// Writing a real eight-step — or retiring the strategy enum — is tracked
// separately in PLAN.md §1.2.

// ForwardEightStepComplex64 performs a forward eight-step FFT on complex64 data.
//
// It is currently six-step under another name — see the file comment in
// eightstep.go.
func ForwardEightStepComplex64(dst, src, twiddle, scratch []complex64) bool {
	return eightStepForward[complex64](dst, src, twiddle, scratch)
}

// InverseEightStepComplex64 performs an inverse eight-step FFT on complex64 data.
//
// It is currently six-step under another name — see the file comment in
// eightstep.go.
func InverseEightStepComplex64(dst, src, twiddle, scratch []complex64) bool {
	return eightStepInverse[complex64](dst, src, twiddle, scratch)
}

func eightStepForward[T Complex](dst, src, twiddle, scratch []T) bool {
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

	m := intSqrt(n)
	if m*m != n {
		return false
	}

	if sameSlice(dst, src) {
		copy(scratch, src)
		src = scratch
	}

	data := dst
	if !sameSlice(dst, src) {
		copy(dst, src)
	}

	math.TransposeSquare(data, m)

	rowTwiddle := scratch[:m]
	rowScratch := scratch[m : 2*m]
	fillRowTwiddle(rowTwiddle, twiddle, n/m)

	for r := range m {
		row := data[r*m : (r+1)*m]
		if !stockhamForward(row, row, rowTwiddle, rowScratch) {
			return false
		}
	}

	math.TransposeSquare(data, m)

	for i := range m {
		for j := range m {
			data[i*m+j] *= twiddle[(i*j)%n]
		}
	}

	for r := range m {
		row := data[r*m : (r+1)*m]
		if !stockhamForward(row, row, rowTwiddle, rowScratch) {
			return false
		}
	}

	math.TransposeSquare(data, m)

	return true
}

func eightStepInverse[T Complex](dst, src, twiddle, scratch []T) bool {
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

	m := intSqrt(n)
	if m*m != n {
		return false
	}

	if sameSlice(dst, src) {
		copy(scratch, src)
		src = scratch
	}

	data := dst
	if !sameSlice(dst, src) {
		copy(dst, src)
	}

	math.TransposeSquare(data, m)

	rowTwiddle := scratch[:m]
	rowScratch := scratch[m : 2*m]
	fillRowTwiddle(rowTwiddle, twiddle, n/m)

	for r := range m {
		row := data[r*m : (r+1)*m]
		if !stockhamInverse(row, row, rowTwiddle, rowScratch) {
			return false
		}
	}

	math.TransposeSquare(data, m)

	for i := range m {
		for j := range m {
			data[i*m+j] *= conj(twiddle[(i*j)%n])
		}
	}

	for r := range m {
		row := data[r*m : (r+1)*m]
		if !stockhamInverse(row, row, rowTwiddle, rowScratch) {
			return false
		}
	}

	math.TransposeSquare(data, m)

	return true
}
