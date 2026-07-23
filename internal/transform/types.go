package transform

import (
	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/kernels"
	m "github.com/cwbudde/algo-fft/internal/math"
)

// Complex is a type alias for the complex number constraint.
type Complex = fftypes.Complex

// Helper functions from math package.
func ComputeTwiddleFactors[T Complex](n int) []T {
	return m.ComputeTwiddleFactors[T](n)
}

var ComputeBitReversalIndices = m.ComputeBitReversalIndices

func conj[T Complex](val T) T {
	return m.Conj[T](val)
}

// Helper functions from kernels package.
func ditForward[T Complex](dst, src, twiddle, scratch []T) bool {
	return kernels.DITForward(dst, src, twiddle, scratch)
}

func ditInverse[T Complex](dst, src, twiddle, scratch []T) bool {
	return kernels.DITInverse(dst, src, twiddle, scratch)
}

func sameSlice[T any](a, b []T) bool {
	return kernels.SameSlice(a, b)
}

func IsPowerOf2(n int) bool {
	return m.IsPowerOf2(n)
}

// stockhamPackedEnabled is defined in stockham_packed_toggle_*.go files

// Helper functions for tests.
var (
	forwardStockhamComplex64  = kernels.ForwardStockhamComplex64
	forwardStockhamComplex128 = kernels.ForwardStockhamComplex128
)

// Test helper functions (defined in test files in kernels package)
// These need to be redefined here or the test files need to import kernels test helpers
