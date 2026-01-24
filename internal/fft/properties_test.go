package fft

import (
	"fmt"
	"math"
	"math/cmplx"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/cpu"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// TestFFTLinearity verifies that FFT is a linear operation:
// FFT(a*x + b*y) = a*FFT(x) + b*FFT(y)
func TestFFTLinearity(t *testing.T) {
	t.Parallel()

	sizes := []int{8, 12, 16, 32, 60, 64, 128, 256, 1024}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			t.Parallel()

			t.Run("complex64", func(t *testing.T) {
				testLinearity64(t, n)
			})
			t.Run("complex128", func(t *testing.T) {
				testLinearity128(t, n)
			})
		})
	}
}

func testLinearity64(t *testing.T, n int) {
	t.Helper()

	features := cpu.DetectFeatures()
	kernels := SelectKernels[complex64](features)

	// Generate random inputs
	x := randomComplex64(n, 12345)
	y := randomComplex64(n, 67890)

	// Scalars
	a := complex(float32(2.5), float32(1.3))
	b := complex(float32(-1.7), float32(0.8))

	// Compute a*x + b*y
	combined := make([]complex64, n)
	for i := range n {
		combined[i] = a*x[i] + b*y[i]
	}

	// FFT of combined
	twiddle := ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	fftCombined := make([]complex64, n)
	if !kernels.Forward(fftCombined, combined, twiddle, scratch) {
		t.Skip("Kernel not available")
	}

	// FFT of x and y separately
	fftX := make([]complex64, n)
	fftY := make([]complex64, n)
	scratch = make([]complex64, n)
	if !kernels.Forward(fftX, x, twiddle, scratch) {
		t.Skip("Kernel not available")
	}
	scratch = make([]complex64, n)
	if !kernels.Forward(fftY, y, twiddle, scratch) {
		t.Skip("Kernel not available")
	}

	// Compute a*FFT(x) + b*FFT(y)
	expected := make([]complex64, n)
	for i := range n {
		expected[i] = a*fftX[i] + b*fftY[i]
	}

	// Verify linearity
	assertComplex64SliceClose(t, fftCombined, expected, n)
}

func testLinearity128(t *testing.T, n int) {
	t.Helper()

	features := cpu.DetectFeatures()
	kernels := SelectKernels[complex128](features)

	x := randomComplex128(n, 12345)
	y := randomComplex128(n, 67890)

	a := complex(2.5, 1.3)
	b := complex(-1.7, 0.8)

	combined := make([]complex128, n)
	for i := range n {
		combined[i] = a*x[i] + b*y[i]
	}

	twiddle := ComputeTwiddleFactors[complex128](n)
	scratch := make([]complex128, n)
	fftCombined := make([]complex128, n)
	if !kernels.Forward(fftCombined, combined, twiddle, scratch) {
		t.Skip("Kernel not available")
	}

	fftX := make([]complex128, n)
	fftY := make([]complex128, n)
	scratch = make([]complex128, n)
	if !kernels.Forward(fftX, x, twiddle, scratch) {
		t.Skip("Kernel not available")
	}
	scratch = make([]complex128, n)
	if !kernels.Forward(fftY, y, twiddle, scratch) {
		t.Skip("Kernel not available")
	}

	expected := make([]complex128, n)
	for i := range n {
		expected[i] = a*fftX[i] + b*fftY[i]
	}

	assertComplex128SliceClose(t, fftCombined, expected, n)
}

// TestFFTParseval verifies Parseval's theorem:
// sum(|x|²) = (1/n) * sum(|FFT(x)|²)
func TestFFTParseval(t *testing.T) {
	t.Parallel()

	sizes := []int{8, 16, 32, 64, 128, 256, 1024}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			t.Parallel()

			t.Run("complex64", func(t *testing.T) {
				testParseval64(t, n)
			})
			t.Run("complex128", func(t *testing.T) {
				testParseval128(t, n)
			})
		})
	}
}

func testParseval64(t *testing.T, n int) {
	t.Helper()

	features := cpu.DetectFeatures()
	kernels := SelectKernels[complex64](features)

	src := randomComplex64(n, 11111)

	// Compute energy in time domain
	var timeEnergy float64
	for _, v := range src {
		timeEnergy += float64(real(v)*real(v) + imag(v)*imag(v))
	}

	// FFT
	twiddle := ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	dst := make([]complex64, n)
	if !kernels.Forward(dst, src, twiddle, scratch) {
		t.Skip("Kernel not available")
	}

	// Compute energy in frequency domain
	var freqEnergy float64
	for _, v := range dst {
		freqEnergy += float64(real(v)*real(v) + imag(v)*imag(v))
	}
	freqEnergy /= float64(n)

	// Verify Parseval's theorem
	relError := math.Abs(timeEnergy-freqEnergy) / math.Max(timeEnergy, freqEnergy)
	if relError > 1e-4 {
		t.Errorf("Parseval's theorem violated: time=%v, freq=%v, relError=%e", timeEnergy, freqEnergy, relError)
	}
}

func testParseval128(t *testing.T, n int) {
	t.Helper()

	features := cpu.DetectFeatures()
	kernels := SelectKernels[complex128](features)

	src := randomComplex128(n, 11111)

	var timeEnergy float64
	for _, v := range src {
		timeEnergy += real(v)*real(v) + imag(v)*imag(v)
	}

	twiddle := ComputeTwiddleFactors[complex128](n)
	scratch := make([]complex128, n)
	dst := make([]complex128, n)
	if !kernels.Forward(dst, src, twiddle, scratch) {
		t.Skip("Kernel not available")
	}

	var freqEnergy float64
	for _, v := range dst {
		freqEnergy += real(v)*real(v) + imag(v)*imag(v)
	}
	freqEnergy /= float64(n)

	relError := math.Abs(timeEnergy-freqEnergy) / math.Max(timeEnergy, freqEnergy)
	if relError > 1e-10 {
		t.Errorf("Parseval's theorem violated: time=%v, freq=%v, relError=%e", timeEnergy, freqEnergy, relError)
	}
}

// TestFFTRoundTrip verifies that IFFT(FFT(x)) ≈ x
func TestFFTRoundTrip(t *testing.T) {
	t.Parallel()

	sizes := []int{8, 12, 16, 32, 60, 64, 128, 256, 512, 1024, 2048}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			t.Parallel()

			t.Run("complex64", func(t *testing.T) {
				testRoundTrip64(t, n)
			})
			t.Run("complex128", func(t *testing.T) {
				testRoundTrip128(t, n)
			})
		})
	}
}

func testRoundTrip64(t *testing.T, n int) {
	t.Helper()

	features := cpu.DetectFeatures()
	kernels := SelectKernels[complex64](features)

	src := randomComplex64(n, 99999)

	// Forward FFT
	twiddle := ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	fwd := make([]complex64, n)
	if !kernels.Forward(fwd, src, twiddle, scratch) {
		t.Skip("Kernel not available")
	}

	// Inverse FFT
	scratch = make([]complex64, n)
	dst := make([]complex64, n)
	if !kernels.Inverse(dst, fwd, twiddle, scratch) {
		t.Skip("Kernel not available")
	}

	// Verify round-trip
	assertComplex64SliceClose(t, dst, src, n)
}

func testRoundTrip128(t *testing.T, n int) {
	t.Helper()

	features := cpu.DetectFeatures()
	kernels := SelectKernels[complex128](features)

	src := randomComplex128(n, 99999)

	twiddle := ComputeTwiddleFactors[complex128](n)
	scratch := make([]complex128, n)
	fwd := make([]complex128, n)
	if !kernels.Forward(fwd, src, twiddle, scratch) {
		t.Skip("Kernel not available")
	}

	scratch = make([]complex128, n)
	dst := make([]complex128, n)
	if !kernels.Inverse(dst, fwd, twiddle, scratch) {
		t.Skip("Kernel not available")
	}

	assertComplex128SliceClose(t, dst, src, n)
}

// TestFFTShiftTheorem verifies the shift theorem:
// If y[k] = x[(k-m) mod n], then FFT(y)[k] = FFT(x)[k] * exp(-2πikm/n)
func TestFFTShiftTheorem(t *testing.T) {
	t.Parallel()

	sizes := []int{8, 16, 32, 64, 128}
	shifts := []int{1, 2, 3}

	for _, n := range sizes {
		for _, m := range shifts {
			t.Run(fmt.Sprintf("n=%d/m=%d", n, m), func(t *testing.T) {
				t.Parallel()

				t.Run("complex64", func(t *testing.T) {
					testShiftTheorem64(t, n, m)
				})
				t.Run("complex128", func(t *testing.T) {
					testShiftTheorem128(t, n, m)
				})
			})
		}
	}
}

func testShiftTheorem64(t *testing.T, n, m int) {
	t.Helper()

	features := cpu.DetectFeatures()
	kernels := SelectKernels[complex64](features)

	x := randomComplex64(n, 77777)

	// Create shifted version: y[k] = x[(k-m) mod n]
	y := make([]complex64, n)
	for k := range n {
		y[k] = x[(k-m+n)%n]
	}

	// FFT of both
	twiddle := ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	fftX := make([]complex64, n)
	if !kernels.Forward(fftX, x, twiddle, scratch) {
		t.Skip("Kernel not available")
	}

	scratch = make([]complex64, n)
	fftY := make([]complex64, n)
	if !kernels.Forward(fftY, y, twiddle, scratch) {
		t.Skip("Kernel not available")
	}

	// Verify shift theorem: FFT(y)[k] = FFT(x)[k] * exp(-2πikm/n)
	expected := make([]complex64, n)
	for k := range n {
		phase := -2 * math.Pi * float64(k*m) / float64(n)
		shift := complex(float32(math.Cos(phase)), float32(math.Sin(phase)))
		expected[k] = fftX[k] * shift
	}

	assertComplex64SliceClose(t, fftY, expected, n)
}

func testShiftTheorem128(t *testing.T, n, m int) {
	t.Helper()

	features := cpu.DetectFeatures()
	kernels := SelectKernels[complex128](features)

	x := randomComplex128(n, 77777)

	y := make([]complex128, n)
	for k := range n {
		y[k] = x[(k-m+n)%n]
	}

	twiddle := ComputeTwiddleFactors[complex128](n)
	scratch := make([]complex128, n)
	fftX := make([]complex128, n)
	if !kernels.Forward(fftX, x, twiddle, scratch) {
		t.Skip("Kernel not available")
	}

	scratch = make([]complex128, n)
	fftY := make([]complex128, n)
	if !kernels.Forward(fftY, y, twiddle, scratch) {
		t.Skip("Kernel not available")
	}

	expected := make([]complex128, n)
	for k := range n {
		phase := -2 * math.Pi * float64(k*m) / float64(n)
		shift := complex(math.Cos(phase), math.Sin(phase))
		expected[k] = fftX[k] * shift
	}

	assertComplex128SliceClose(t, fftY, expected, n)
}

// TestFFTAgainstReference verifies FFT output matches naive DFT
func TestFFTAgainstReference(t *testing.T) {
	t.Parallel()

	sizes := []int{4, 8, 12, 16, 20, 32, 60, 64, 128, 256}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			t.Parallel()

			t.Run("complex64", func(t *testing.T) {
				testAgainstReference64(t, n)
			})
			t.Run("complex128", func(t *testing.T) {
				testAgainstReference128(t, n)
			})
		})
	}
}

func testAgainstReference64(t *testing.T, n int) {
	t.Helper()

	features := cpu.DetectFeatures()
	kernels := SelectKernels[complex64](features)

	src := randomComplex64(n, 55555)

	// FFT
	twiddle := ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	dst := make([]complex64, n)
	if !kernels.Forward(dst, src, twiddle, scratch) {
		t.Skip("Kernel not available")
	}

	// Reference
	want := reference.NaiveDFT(src)

	assertComplex64SliceClose(t, dst, want, n)
}

func testAgainstReference128(t *testing.T, n int) {
	t.Helper()

	features := cpu.DetectFeatures()
	kernels := SelectKernels[complex128](features)

	src := randomComplex128(n, 55555)

	twiddle := ComputeTwiddleFactors[complex128](n)
	scratch := make([]complex128, n)
	dst := make([]complex128, n)
	if !kernels.Forward(dst, src, twiddle, scratch) {
		t.Skip("Kernel not available")
	}

	want := reference.NaiveDFT128(src)

	assertComplex128SliceClose(t, dst, want, n)
}

// TestFFTSymmetry verifies that FFT of real input has conjugate symmetry
func TestFFTRealInputSymmetry(t *testing.T) {
	t.Parallel()

	sizes := []int{8, 16, 32, 64, 128}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			t.Parallel()

			t.Run("complex64", func(t *testing.T) {
				testRealInputSymmetry64(t, n)
			})
			t.Run("complex128", func(t *testing.T) {
				testRealInputSymmetry128(t, n)
			})
		})
	}
}

func testRealInputSymmetry64(t *testing.T, n int) {
	t.Helper()

	features := cpu.DetectFeatures()
	kernels := SelectKernels[complex64](features)

	// Generate real-valued input
	src := make([]complex64, n)
	for i := range n {
		src[i] = complex(float32(i), 0)
	}

	// FFT
	twiddle := ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	dst := make([]complex64, n)
	if !kernels.Forward(dst, src, twiddle, scratch) {
		t.Skip("Kernel not available")
	}

	// Verify conjugate symmetry: FFT[k] = conj(FFT[n-k])
	for k := 1; k < n/2; k++ {
		conjSym := complex(real(dst[n-k]), -imag(dst[n-k]))
		if cmplx.Abs(complex128(dst[k]-conjSym)) > testTol64 {
			t.Errorf("Symmetry violation at k=%d: FFT[%d]=%v, conj(FFT[%d])=%v",
				k, k, dst[k], n-k, conjSym)
		}
	}
}

func testRealInputSymmetry128(t *testing.T, n int) {
	t.Helper()

	features := cpu.DetectFeatures()
	kernels := SelectKernels[complex128](features)

	src := make([]complex128, n)
	for i := range n {
		src[i] = complex(float64(i), 0)
	}

	twiddle := ComputeTwiddleFactors[complex128](n)
	scratch := make([]complex128, n)
	dst := make([]complex128, n)
	if !kernels.Forward(dst, src, twiddle, scratch) {
		t.Skip("Kernel not available")
	}

	for k := 1; k < n/2; k++ {
		conjSym := complex(real(dst[n-k]), -imag(dst[n-k]))
		if cmplx.Abs(dst[k]-conjSym) > testTol128 {
			t.Errorf("Symmetry violation at k=%d: FFT[%d]=%v, conj(FFT[%d])=%v",
				k, k, dst[k], n-k, conjSym)
		}
	}
}
