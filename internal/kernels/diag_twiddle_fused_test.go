//go:build amd64 && asm && !purego

package kernels

import (
	"math/cmplx"
	"os"
	"testing"

	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	m "github.com/MeKo-Christian/algo-fft/internal/math"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// TestDiagFusedTwiddleForwardTile validates the forward fused-twiddle mapping
// for the (rb=0, cb=0) tile when the diagnostic hook is enabled.
func makeDiagInput() []complex128 {
	const n = 256
	src := make([]complex128, n)
	for r := 0; r < 16; r++ {
		for c := 0; c < 16; c++ {
			src[r*16+c] = complex(float64(r+1), float64((c+1)*(r+1)))
		}
	}
	return src
}

func TestDiagFusedTwiddleForwardTile(t *testing.T) {
	if os.Getenv("ALGOFFT_DIAG_TWIDDLE") != "1" {
		t.Skip("set ALGOFFT_DIAG_TWIDDLE=1 to enable the fused-twiddle diagnostic test")
	}

	const n = 256
	src := makeDiagInput()

	twiddle := make([]complex128, twiddleSize256Radix16AVX2(n))
	prepareTwiddle256Radix16AVX2(n, false, twiddle)

	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	_ = amd64.ForwardAVX2Size256Radix16Complex128Asm(dst, src, twiddle, scratch)

	baseTwiddle := m.ComputeTwiddleFactors[complex128](n)
	colFFT := make([][]complex128, 4)
	for col := 0; col < 4; col++ {
		column := make([]complex128, 16)
		for row := 0; row < 16; row++ {
			column[row] = src[row*16+col]
		}
		colFFT[col] = reference.NaiveDFT128(column)
	}

	const tol = 1e-9
	for row := 0; row < 4; row++ {
		for col := 0; col < 4; col++ {
			expected := colFFT[col][row] * baseTwiddle[row*col]
			got := dst[row*16+col]
			if cmplx.Abs(got-expected) > tol {
				t.Fatalf("tile[%d,%d] = %v, want %v", row, col, got, expected)
			}
		}
	}
}

// TestDiagFusedTwiddleForwardTileCb1 validates the forward fused-twiddle mapping
// for the (rb=0, cb=1) tile when the diagnostic hook is enabled.
func TestDiagFusedTwiddleForwardTileCb1(t *testing.T) {
	if os.Getenv("ALGOFFT_DIAG_TWIDDLE") != "1" || os.Getenv("ALGOFFT_DIAG_TWIDDLE_TILE") != "1" {
		t.Skip("set ALGOFFT_DIAG_TWIDDLE=1 and ALGOFFT_DIAG_TWIDDLE_TILE=1 to enable this test")
	}

	const n = 256
	src := makeDiagInput()

	twiddle := make([]complex128, twiddleSize256Radix16AVX2(n))
	prepareTwiddle256Radix16AVX2(n, false, twiddle)

	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	_ = amd64.ForwardAVX2Size256Radix16Complex128Asm(dst, src, twiddle, scratch)

	baseTwiddle := m.ComputeTwiddleFactors[complex128](n)
	colFFT := make([][]complex128, 4)
	for col := 4; col < 8; col++ {
		column := make([]complex128, 16)
		for row := 0; row < 16; row++ {
			column[row] = src[row*16+col]
		}
		colFFT[col-4] = reference.NaiveDFT128(column)
	}

	const tol = 1e-9
	for row := 0; row < 4; row++ {
		for col := 4; col < 8; col++ {
			expected := colFFT[col-4][row] * baseTwiddle[row*col]
			got := dst[row*16+col]
			if cmplx.Abs(got-expected) > tol {
				t.Fatalf("tile[%d,%d] = %v, want %v", row, col, got, expected)
			}
		}
	}
}

// TestDiagFusedTwiddleForwardTileCb1Mapped validates the cb1 tile against the
// transpose-out mapping (row_in = cb*4+r, col_in = rb*4+c).
func TestDiagFusedTwiddleForwardTileCb1Mapped(t *testing.T) {
	if os.Getenv("ALGOFFT_DIAG_TWIDDLE") != "1" || os.Getenv("ALGOFFT_DIAG_TWIDDLE_TILE") != "1" || os.Getenv("ALGOFFT_DIAG_TWIDDLE_MAP") != "1" {
		t.Skip("set ALGOFFT_DIAG_TWIDDLE=1 ALGOFFT_DIAG_TWIDDLE_TILE=1 ALGOFFT_DIAG_TWIDDLE_MAP=1")
	}

	const n = 256
	src := makeDiagInput()

	twiddle := make([]complex128, twiddleSize256Radix16AVX2(n))
	prepareTwiddle256Radix16AVX2(n, false, twiddle)

	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	_ = amd64.ForwardAVX2Size256Radix16Complex128Asm(dst, src, twiddle, scratch)

	baseTwiddle := m.ComputeTwiddleFactors[complex128](n)
	colFFT := make([][]complex128, 4)
	for col := 0; col < 4; col++ {
		column := make([]complex128, 16)
		for row := 0; row < 16; row++ {
			column[row] = src[row*16+col]
		}
		colFFT[col] = reference.NaiveDFT128(column)
	}

	const tol = 1e-9
	for row := 0; row < 4; row++ {
		rowIn := 4 + row
		for col := 4; col < 8; col++ {
			colIn := col - 4
			expected := colFFT[colIn][rowIn] * baseTwiddle[rowIn*colIn]
			got := dst[row*16+col]
			if cmplx.Abs(got-expected) > tol {
				t.Fatalf("tile[%d,%d] = %v, want %v", row, col, got, expected)
			}
		}
	}
}

// TestDiagTransposeForwardTileCb1NoTwiddle validates the transpose-out mapping
// for the (rb=0, cb=1) tile without twiddle.
func TestDiagTransposeForwardTileCb1NoTwiddle(t *testing.T) {
	if os.Getenv("ALGOFFT_DIAG_TWIDDLE") != "1" || os.Getenv("ALGOFFT_DIAG_TWIDDLE_TILE") != "1" || os.Getenv("ALGOFFT_DIAG_TWIDDLE_NOTW") != "1" {
		t.Skip("set ALGOFFT_DIAG_TWIDDLE=1 ALGOFFT_DIAG_TWIDDLE_TILE=1 ALGOFFT_DIAG_TWIDDLE_NOTW=1")
	}

	const n = 256
	src := makeDiagInput()

	twiddle := make([]complex128, twiddleSize256Radix16AVX2(n))
	prepareTwiddle256Radix16AVX2(n, false, twiddle)

	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	_ = amd64.ForwardAVX2Size256Radix16Complex128Asm(dst, src, twiddle, scratch)

	colFFT := make([][]complex128, 4)
	for col := 4; col < 8; col++ {
		column := make([]complex128, 16)
		for row := 0; row < 16; row++ {
			column[row] = src[row*16+col]
		}
		colFFT[col-4] = reference.NaiveDFT128(column)
	}

	const tol = 1e-9
	for row := 0; row < 4; row++ {
		for col := 4; col < 8; col++ {
			expected := colFFT[col-4][row]
			got := dst[row*16+col]
			if cmplx.Abs(got-expected) > tol {
				t.Fatalf("tile[%d,%d] = %v, want %v", row, col, got, expected)
			}
		}
	}
}

// TestDiagTransposeMappingCb1 reports which (row,col) from the column FFTs
// appear in the (rb=0, cb=1) tile when no twiddle is applied.
func TestDiagTransposeMappingCb1(t *testing.T) {
	if os.Getenv("ALGOFFT_DIAG_TWIDDLE") != "1" || os.Getenv("ALGOFFT_DIAG_TWIDDLE_TILE") != "1" || os.Getenv("ALGOFFT_DIAG_TWIDDLE_NOTW") != "1" || os.Getenv("ALGOFFT_DIAG_MAP") != "1" {
		t.Skip("set ALGOFFT_DIAG_TWIDDLE=1 ALGOFFT_DIAG_TWIDDLE_TILE=1 ALGOFFT_DIAG_TWIDDLE_NOTW=1 ALGOFFT_DIAG_MAP=1")
	}

	const n = 256
	src := makeDiagInput()

	twiddle := make([]complex128, twiddleSize256Radix16AVX2(n))
	prepareTwiddle256Radix16AVX2(n, false, twiddle)

	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	_ = amd64.ForwardAVX2Size256Radix16Complex128Asm(dst, src, twiddle, scratch)

	colFFT := make([][]complex128, 16)
	for col := 0; col < 16; col++ {
		column := make([]complex128, 16)
		for row := 0; row < 16; row++ {
			column[row] = src[row*16+col]
		}
		colFFT[col] = reference.NaiveDFT128(column)
	}

	const tol = 1e-9
	for row := 0; row < 4; row++ {
		for col := 4; col < 8; col++ {
			got := dst[row*16+col]
			found := false
			for srcCol := 0; srcCol < 16 && !found; srcCol++ {
				for srcRow := 0; srcRow < 16; srcRow++ {
					expected := colFFT[srcCol][srcRow]
					if cmplx.Abs(got-expected) <= tol {
						t.Logf("tile[%d,%d] matches col=%d row=%d value=%v", row, col, srcCol, srcRow, got)
						found = true
						break
					}
				}
			}
			if !found {
				t.Fatalf("tile[%d,%d] value %v not matched", row, col, got)
			}
		}
	}
}
