//go:build amd64 && fft_asm && !purego

package fft

import (
	"fmt"
	"math"
	"testing"
)

func TestDebugAVX2_RandomInput(t *testing.T) {
	n := 16
	seed := uint64(12345)

	// Create deterministic "random" input
	src := make([]complex64, n)
	for i := 0; i < n; i++ {
		seed = seed*6364136223846793005 + 1442695040888963407
		re := float32(seed>>32) / float32(1<<32) * 2 - 1
		seed = seed*6364136223846793005 + 1442695040888963407
		im := float32(seed>>32) / float32(1<<32) * 2 - 1
		src[i] = complex(re, im)
	}

	dst := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrev := ComputeBitReversalIndices(n)

	t.Logf("Input:")
	for i := 0; i < n; i++ {
		t.Logf("  src[%d] = %v", i, src[i])
	}

	// Call AVX2 kernel
	ok := forwardAVX2Complex64(dst, src, twiddle, scratch, bitrev)
	if !ok {
		t.Fatal("forwardAVX2Complex64 returned false")
	}

	// Compare with pure Go
	goDst := make([]complex64, n)
	goScratch := make([]complex64, n)
	ditForward[complex64](goDst, src, twiddle, goScratch, bitrev)

	t.Logf("\nResults:")
	maxRelErr := float32(0)
	for i := 0; i < n; i++ {
		avx2Val := dst[i]
		goVal := goDst[i]

		// Compute relative error
		diff := avx2Val - goVal
		mag := float32(math.Sqrt(float64(real(goVal)*real(goVal) + imag(goVal)*imag(goVal))))
		if mag < 1e-10 {
			mag = 1
		}
		relErr := float32(math.Sqrt(float64(real(diff)*real(diff) + imag(diff)*imag(diff)))) / mag
		if relErr > maxRelErr {
			maxRelErr = relErr
		}

		if relErr > 1e-5 {
			t.Logf("  [%d]: AVX2=%v, Go=%v, relErr=%e", i, avx2Val, goVal, relErr)
		}
	}

	t.Logf("\nMax relative error: %e", maxRelErr)
	if maxRelErr > 1e-5 {
		t.Fail()
	}
}

func TestDebugAVX2_SimpleSize16(t *testing.T) {
	n := 16

	// Create simple input: impulse at position 0
	src := make([]complex64, n)
	src[0] = 1 + 0i

	dst := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrev := ComputeBitReversalIndices(n)

	t.Logf("Input: src[0]=%v", src[0])
	t.Logf("Twiddle factors:")
	for i := 0; i < 4; i++ {
		t.Logf("  twiddle[%d] = %v", i, twiddle[i])
	}
	t.Logf("Bit-reversal indices:")
	for i := 0; i < n; i++ {
		t.Logf("  bitrev[%d] = %d", i, bitrev[i])
	}

	// Call AVX2 kernel
	ok := forwardAVX2Complex64(dst, src, twiddle, scratch, bitrev)
	if !ok {
		t.Fatal("forwardAVX2Complex64 returned false")
	}

	t.Logf("AVX2 output:")
	for i := 0; i < n; i++ {
		t.Logf("  dst[%d] = %v", i, dst[i])
	}

	// Compare with pure Go
	goDst := make([]complex64, n)
	goScratch := make([]complex64, n)
	ditForward[complex64](goDst, src, twiddle, goScratch, bitrev)

	t.Logf("Pure Go output:")
	for i := 0; i < n; i++ {
		t.Logf("  goDst[%d] = %v", i, goDst[i])
	}

	// For impulse at 0, FFT should give all 1s
	t.Logf("\nExpected: all 1+0i (impulse at DC)")

	mismatch := false
	for i := 0; i < n; i++ {
		if dst[i] != goDst[i] {
			t.Logf("MISMATCH at [%d]: AVX2=%v, Go=%v", i, dst[i], goDst[i])
			mismatch = true
		}
	}

	if mismatch {
		t.Fail()
	}
}

func TestDebugAVX2_SimpleBitReversal(t *testing.T) {
	n := 16

	// Test bit-reversal only
	src := make([]complex64, n)
	for i := 0; i < n; i++ {
		src[i] = complex(float32(i), 0)
	}

	dst := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrev := ComputeBitReversalIndices(n)

	// After bit-reversal, dst should contain src[bitrev[i]]
	for i := 0; i < n; i++ {
		dst[i] = src[bitrev[i]]
	}

	t.Logf("Expected after bit-reversal:")
	for i := 0; i < n; i++ {
		t.Logf("  dst[%d] = %v (from src[%d])", i, dst[i], bitrev[i])
	}

	_ = twiddle
	_ = scratch
}

func TestDebugAVX2_StageByStage(t *testing.T) {
	n := 16
	seed := uint64(12345)

	// Create deterministic "random" input
	src := make([]complex64, n)
	for i := 0; i < n; i++ {
		seed = seed*6364136223846793005 + 1442695040888963407
		re := float32(seed>>32) / float32(1<<32) * 2 - 1
		seed = seed*6364136223846793005 + 1442695040888963407
		im := float32(seed>>32) / float32(1<<32) * 2 - 1
		src[i] = complex(re, im)
	}

	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	// Pure Go stages manually
	goWork := make([]complex64, n)
	for i := 0; i < n; i++ {
		goWork[i] = src[bitrev[i]]
	}

	t.Logf("After bit-reversal (Go):")
	for i := 0; i < n; i++ {
		t.Logf("  goWork[%d] = %v", i, goWork[i])
	}

	for size := 2; size <= n; size <<= 1 {
		half := size >> 1
		step := n / size
		t.Logf("\n=== Stage size=%d, half=%d, step=%d ===", size, half, step)

		for base := 0; base < n; base += size {
			for j := 0; j < half; j++ {
				idx1 := base + j
				idx2 := idx1 + half
				tw := twiddle[j*step]

				a := goWork[idx1]
				b := goWork[idx2]
				t_ := tw * b

				goWork[idx1] = a + t_
				goWork[idx2] = a - t_

				if base == 0 && j < 4 {
					t.Logf("base=%d j=%d: tw=%v, a=%v, b=%v", base, j, tw, a, b)
					t.Logf("  t=%v, a'=%v, b'=%v", t_, a+t_, a-t_)
				}
			}
		}
	}

	t.Logf("\nFinal Go result:")
	for i := 0; i < n; i++ {
		t.Logf("  goWork[%d] = %v", i, goWork[i])
	}

	// Now AVX2 result
	avxDst := make([]complex64, n)
	scratch := make([]complex64, n)
	forwardAVX2Complex64(avxDst, src, twiddle, scratch, bitrev)

	t.Logf("\nAVX2 result:")
	for i := 0; i < n; i++ {
		t.Logf("  avxDst[%d] = %v", i, avxDst[i])
	}

	// Compare
	t.Logf("\nDifferences:")
	for i := 0; i < n; i++ {
		if goWork[i] != avxDst[i] {
			t.Logf("  [%d]: Go=%v, AVX2=%v", i, goWork[i], avxDst[i])
		}
	}
}

func TestDebugAVX2_Stage1Only(t *testing.T) {
	// Test first butterfly stage (size=2) only
	n := 16

	src := make([]complex64, n)
	for i := 0; i < n; i++ {
		src[i] = complex(float32(i), float32(i%3))
	}

	// Apply bit-reversal
	bitrev := ComputeBitReversalIndices(n)
	work := make([]complex64, n)
	for i := 0; i < n; i++ {
		work[i] = src[bitrev[i]]
	}

	twiddle := ComputeTwiddleFactors[complex64](n)

	// Stage 1: size=2, half=1, step=8
	// For each pair (i, i+1), compute:
	//   a' = a + w*b
	//   b' = a - w*b
	// where w = twiddle[0] = 1+0i for j=0
	t.Logf("After bit-reversal:")
	for i := 0; i < n; i++ {
		t.Logf("  work[%d] = %v", i, work[i])
	}

	// Manual stage 1
	size := 2
	half := 1
	step := n / size // 8
	for base := 0; base < n; base += size {
		for j := 0; j < half; j++ {
			idx1 := base + j
			idx2 := idx1 + half
			tw := twiddle[j*step]

			a := work[idx1]
			b := work[idx2]
			t_ := tw * b

			work[idx1] = a + t_
			work[idx2] = a - t_

			t.Logf("Stage1: base=%d, j=%d, idx1=%d, idx2=%d, tw=%v", base, j, idx1, idx2, tw)
			t.Logf("  a=%v, b=%v, t=%v", a, b, t_)
			t.Logf("  work[%d]=%v, work[%d]=%v", idx1, work[idx1], idx2, work[idx2])
		}
	}

	fmt.Println("Done")
}
