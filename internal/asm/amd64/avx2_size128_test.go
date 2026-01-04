//go:build amd64 && asm && !purego

package amd64

import (
	"fmt"
	"math"
	"testing"
)

func TestTranspose128x128_Basic(t *testing.T) {
	const n = 16384
	const m = 128

	src := make([]complex64, n)
	dst := make([]complex64, n)

	// Fill with test pattern: src[row][col] = row*1000 + col
	for row := 0; row < m; row++ {
		for col := 0; col < m; col++ {
			src[row*m+col] = complex(float32(row*1000+col), 0)
		}
	}

	if !Transpose128x128Complex64AVX2Asm(dst, src) {
		t.Fatal("Transpose returned false")
	}

	// Verify: dst[col][row] should equal src[row][col]
	errors := 0
	for row := 0; row < m; row++ {
		for col := 0; col < m; col++ {
			expected := src[row*m+col]
			got := dst[col*m+row]
			if got != expected {
				if errors < 10 {
					t.Errorf("Error at [%d][%d]: expected %v, got %v", row, col, expected, got)
				}
				errors++
			}
		}
	}
	if errors > 0 {
		t.Errorf("Total %d errors", errors)
	}
}

func TestTranspose128x128_SelfInverse(t *testing.T) {
	const n = 16384
	const m = 128

	src := make([]complex64, n)
	tmp := make([]complex64, n)
	dst := make([]complex64, n)

	// Fill with random-ish data
	for i := range src {
		src[i] = complex(float32(i)*0.1, float32(i)*0.2)
	}

	// Transpose twice should give back the original
	if !Transpose128x128Complex64AVX2Asm(tmp, src) {
		t.Fatal("First transpose returned false")
	}
	if !Transpose128x128Complex64AVX2Asm(dst, tmp) {
		t.Fatal("Second transpose returned false")
	}

	for i := range src {
		if dst[i] != src[i] {
			t.Errorf("At %d: expected %v, got %v", i, src[i], dst[i])
			if i > 10 {
				break
			}
		}
	}
}

func TestTransposeTwiddle128x128_Correctness(t *testing.T) {
	const n = 16384
	const m = 128

	src := make([]complex64, n)
	dst := make([]complex64, n)
	twiddle := make([]complex64, n)

	// Fill with test pattern
	for i := range src {
		src[i] = complex(float32(i%100)*0.01, float32(i%50)*0.01)
	}

	// Fill twiddles with W_n^k = exp(-2*pi*i*k/n)
	// For simplicity, use 1+0i so we can verify transpose works
	for i := range twiddle {
		twiddle[i] = complex(1, 0)
	}

	if !TransposeTwiddle128x128Complex64AVX2Asm(dst, src, twiddle) {
		t.Fatal("TransposeTwiddle returned false")
	}

	// With twiddle = 1, this should be identical to plain transpose
	// dst[i,j] = src[j,i] * 1 = src[j,i]
	errors := 0
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			expected := src[j*m+i]
			got := dst[i*m+j]
			if got != expected {
				if errors < 10 {
					t.Errorf("At (%d,%d): expected %v, got %v", i, j, expected, got)
				}
				errors++
			}
		}
	}
	if errors > 0 {
		t.Errorf("Total %d errors", errors)
	}
}

func TestTransposeTwiddle128x128_WithRealTwiddles(t *testing.T) {
	const n = 16384
	const m = 128

	src := make([]complex64, n)
	dstAsm := make([]complex64, n)
	dstRef := make([]complex64, n)
	twiddle := make([]complex64, n)

	// Fill with random-ish pattern
	for i := range src {
		src[i] = complex(float32(i%100)*0.01-0.5, float32(i%73)*0.01-0.3)
	}

	// Fill twiddles with actual W_n^k = exp(-2*pi*i*k/n)
	for k := 0; k < n; k++ {
		angle := -2.0 * 3.141592653589793 * float64(k) / float64(n)
		twiddle[k] = complex(float32(math.Cos(angle)), float32(math.Sin(angle)))
	}

	// Compute reference in Go
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			srcVal := src[j*m+i]                    // src[j,i]
			twIdx := (i * j) % n                    // twiddle index
			twVal := twiddle[twIdx]                 // twiddle factor
			dstRef[i*m+j] = srcVal * twVal          // dst[i,j] = src[j,i] * tw
		}
	}

	// Compute with assembly
	if !TransposeTwiddle128x128Complex64AVX2Asm(dstAsm, src, twiddle) {
		t.Fatal("TransposeTwiddle returned false")
	}

	// Compare
	maxErr := float32(0)
	maxErrIdx := 0
	for i := range n {
		re := real(dstAsm[i]) - real(dstRef[i])
		im := imag(dstAsm[i]) - imag(dstRef[i])
		err := float32(math.Sqrt(float64(re*re + im*im)))
		if err > maxErr {
			maxErr = err
			maxErrIdx = i
		}
	}

	t.Logf("Max error: %e at index %d", maxErr, maxErrIdx)
	if maxErr > 1e-6 {
		row := maxErrIdx / m
		col := maxErrIdx % m
		t.Errorf("Asm vs Ref mismatch: maxErr=%e at [%d,%d] (asm=%v, ref=%v)",
			maxErr, row, col, dstAsm[maxErrIdx], dstRef[maxErrIdx])
	}
}

func TestTransposeTwiddleConj128x128_WithRealTwiddles(t *testing.T) {
	const n = 16384
	const m = 128

	src := make([]complex64, n)
	dstAsm := make([]complex64, n)
	dstRef := make([]complex64, n)
	twiddle := make([]complex64, n)

	// Fill with random-ish pattern
	for i := range src {
		src[i] = complex(float32(i%100)*0.01-0.5, float32(i%73)*0.01-0.3)
	}

	// Fill twiddles with actual W_n^k = exp(-2*pi*i*k/n)
	for k := 0; k < n; k++ {
		angle := -2.0 * 3.141592653589793 * float64(k) / float64(n)
		twiddle[k] = complex(float32(math.Cos(angle)), float32(math.Sin(angle)))
	}

	// Compute reference in Go with conjugate twiddle
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			srcVal := src[j*m+i]                          // src[j,i]
			twIdx := (i * j) % n                          // twiddle index
			twVal := complex(real(twiddle[twIdx]), -imag(twiddle[twIdx])) // conj(twiddle)
			dstRef[i*m+j] = srcVal * twVal                // dst[i,j] = src[j,i] * conj(tw)
		}
	}

	// Compute with assembly
	if !TransposeTwiddleConj128x128Complex64AVX2Asm(dstAsm, src, twiddle) {
		t.Fatal("TransposeTwiddleConj returned false")
	}

	// Compare
	maxErr := float32(0)
	maxErrIdx := 0
	for i := range n {
		re := real(dstAsm[i]) - real(dstRef[i])
		im := imag(dstAsm[i]) - imag(dstRef[i])
		err := float32(math.Sqrt(float64(re*re + im*im)))
		if err > maxErr {
			maxErr = err
			maxErrIdx = i
		}
	}

	t.Logf("Max error: %e at index %d", maxErr, maxErrIdx)
	if maxErr > 1e-6 {
		row := maxErrIdx / m
		col := maxErrIdx % m
		t.Errorf("Asm vs Ref mismatch: maxErr=%e at [%d,%d] (asm=%v, ref=%v)",
			maxErr, row, col, dstAsm[maxErrIdx], dstRef[maxErrIdx])
	}
}

func TestSize128FFT_Correctness(t *testing.T) {
	const n = 128

	src := make([]complex64, n)
	dstFFT := make([]complex64, n)
	twiddle := make([]complex64, n)
	scratch := make([]complex64, n)
	bitrev := make([]int, n)

	// Fill with simple test data
	for i := range src {
		src[i] = complex(float32(i), 0)
	}

	// Compute bit-reversal (radix-2)
	nbits := 7 // log2(128)
	for i := 0; i < n; i++ {
		rev := 0
		for b := 0; b < nbits; b++ {
			if (i>>b)&1 != 0 {
				rev |= 1 << (nbits - 1 - b)
			}
		}
		bitrev[i] = rev
	}

	// Compute twiddle factors
	for k := 0; k < n; k++ {
		angle := -2.0 * 3.141592653589793 * float64(k) / float64(n)
		re := float32(math.Cos(angle))
		im := float32(math.Sin(angle))
		twiddle[k] = complex(re, im)
	}

	// Test forward FFT
	if !ForwardAVX2Size128Complex64Asm(dstFFT, src, twiddle, scratch, bitrev) {
		t.Fatal("ForwardAVX2Size128Complex64Asm returned false")
	}

	// DC component should be sum of all inputs = 0+1+2+...+127 = 128*127/2 = 8128
	dcExpected := float32(n * (n - 1) / 2)
	dcGot := real(dstFFT[0])
	if diff := abs32(dcGot - dcExpected); diff > 1 {
		t.Errorf("DC component: expected %v, got %v (diff %v)", dcExpected, dcGot, diff)
	}

	t.Logf("DC component: %v (expected %v)", dcGot, dcExpected)
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// TestSixStep16384Simple tests the six-step algorithm step by step
func TestSixStep16384Simple(t *testing.T) {
	const n = 16384
	const m = 128

	// Create impulse at position 0
	src := make([]complex64, n)
	src[0] = complex(1, 0)

	dst := make([]complex64, n)
	twiddle := make([]complex64, n)
	scratch := make([]complex64, n)
	bitrev := make([]int, n)

	// Compute twiddle factors for size n
	for k := 0; k < n; k++ {
		angle := -2.0 * 3.141592653589793 * float64(k) / float64(n)
		twiddle[k] = complex(float32(math.Cos(angle)), float32(math.Sin(angle)))
	}

	// Compute bit-reversal for size n
	nbits := 14 // log2(16384)
	for i := 0; i < n; i++ {
		rev := 0
		for b := 0; b < nbits; b++ {
			if (i>>b)&1 != 0 {
				rev |= 1 << (nbits - 1 - b)
			}
		}
		bitrev[i] = rev
	}

	// Step 1: Transpose src -> scratch
	if !Transpose128x128Complex64AVX2Asm(scratch, src) {
		t.Fatal("Step 1 transpose failed")
	}

	// Check: After transpose of impulse at [0,0], the element should still be at [0,0]
	if scratch[0] != complex(1, 0) {
		t.Errorf("After step 1: expected (1,0) at [0], got %v", scratch[0])
	}

	// Step 2: Row FFTs
	// Compute row twiddles (stride by m)
	rowTwiddle := make([]complex64, m)
	for k := 0; k < m; k++ {
		rowTwiddle[k] = twiddle[k*m]
	}

	// Compute row bit-reversal for size m
	rowBitrev := make([]int, m)
	rowNbits := 7 // log2(128)
	for i := 0; i < m; i++ {
		rev := 0
		for b := 0; b < rowNbits; b++ {
			if (i>>b)&1 != 0 {
				rev |= 1 << (rowNbits - 1 - b)
			}
		}
		rowBitrev[i] = rev
	}

	rowScratch := make([]complex64, m)
	work := scratch // Use scratch as our working buffer

	for r := 0; r < m; r++ {
		row := work[r*m : (r+1)*m]
		if !ForwardAVX2Size128Complex64Asm(row, row, rowTwiddle, rowScratch, rowBitrev) {
			t.Fatalf("Row FFT %d failed", r)
		}
	}

	// Check: After row FFTs, row 0 should be all ones (DFT of impulse = all ones)
	t.Logf("After step 2, first 4 of row 0: %v %v %v %v", work[0], work[1], work[2], work[3])

	// Step 3+4: Transpose + twiddle
	if !TransposeTwiddle128x128Complex64AVX2Asm(dst, work, twiddle) {
		t.Fatal("TransposeTwiddle failed")
	}

	t.Logf("After step 3+4, first 4: %v %v %v %v", dst[0], dst[1], dst[2], dst[3])

	// Step 5: Row FFTs
	for r := 0; r < m; r++ {
		row := dst[r*m : (r+1)*m]
		if !ForwardAVX2Size128Complex64Asm(row, row, rowTwiddle, rowScratch, rowBitrev) {
			t.Fatalf("Row FFT %d (step 5) failed", r)
		}
	}

	t.Logf("After step 5, first 4: %v %v %v %v", dst[0], dst[1], dst[2], dst[3])

	// Step 6: Final transpose
	if !Transpose128x128Complex64AVX2Asm(work, dst) {
		t.Fatal("Step 6 transpose failed")
	}
	copy(dst[:n], work)

	// For impulse input, FFT should give all ones
	t.Logf("Final first 4: %v %v %v %v", dst[0], dst[1], dst[2], dst[3])

	// Check that all outputs are close to 1
	errors := 0
	for i := range dst {
		if diff := cmplx64Abs(dst[i] - complex(1, 0)); diff > 0.01 {
			if errors < 5 {
				t.Errorf("Output[%d] = %v, expected ~(1,0), diff=%v", i, dst[i], diff)
			}
			errors++
		}
	}
	if errors > 0 {
		t.Errorf("Total %d errors", errors)
	} else {
		t.Log("Impulse response correct - all outputs ~1")
	}
}

func cmplx64Abs(c complex64) float32 {
	r, i := real(c), imag(c)
	return float32(math.Sqrt(float64(r*r + i*i)))
}

func TestDumpFirstBlock(t *testing.T) {
	const n = 16384
	const m = 128

	src := make([]complex64, n)
	dst := make([]complex64, n)

	// Fill with identifiable pattern
	for row := 0; row < m; row++ {
		for col := 0; col < m; col++ {
			src[row*m+col] = complex(float32(row), float32(col))
		}
	}

	Transpose128x128Complex64AVX2Asm(dst, src)

	// Print first 4x4 block of src and dst
	fmt.Println("Source first 4x4:")
	for row := 0; row < 4; row++ {
		for col := 0; col < 4; col++ {
			fmt.Printf("(%v,%v) ", real(src[row*m+col]), imag(src[row*m+col]))
		}
		fmt.Println()
	}

	fmt.Println("Dest first 4x4:")
	for row := 0; row < 4; row++ {
		for col := 0; col < 4; col++ {
			fmt.Printf("(%v,%v) ", real(dst[row*m+col]), imag(dst[row*m+col]))
		}
		fmt.Println()
	}

	// For transpose: dst[col,row] = src[row,col]
	// So dst[0,0] = src[0,0], dst[1,0] = src[0,1], dst[0,1] = src[1,0]
	fmt.Println("\nVerifying transpose pattern:")
	for row := 0; row < 4; row++ {
		for col := 0; col < 4; col++ {
			srcVal := src[row*m+col]
			dstVal := dst[col*m+row]
			match := srcVal == dstVal
			fmt.Printf("src[%d,%d]=%v -> dst[%d,%d]=%v (match=%v)\n",
				row, col, srcVal, col, row, dstVal, match)
		}
	}
}
