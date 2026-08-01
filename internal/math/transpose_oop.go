package math

// TransposeSquareOutOfPlace writes the transpose of the n×n row-major matrix
// src into dst: dst[i*n+j] = src[j*n+i]. dst and src must each hold at least
// n*n elements and must not alias. Unlike TransposeSquare this is a plain
// copy, not a swap, so it is correct for any n (including 0 and 1) and needs
// no blocking to stay correct — the blocking below is purely a cache
// optimization, matching transposeSquareBlocked's tile edge.
func TransposeSquareOutOfPlace[T any](dst, src []T, n int) {
	if n <= 0 {
		return
	}

	if n <= transposeSmallCutoff {
		for i := range n {
			for j := range n {
				dst[i*n+j] = src[j*n+i]
			}
		}

		return
	}

	for bi := 0; bi < n; bi += transposeBlock {
		iEnd := min(bi+transposeBlock, n)

		for bj := 0; bj < n; bj += transposeBlock {
			jEnd := min(bj+transposeBlock, n)

			for i := bi; i < iEnd; i++ {
				for j := bj; j < jEnd; j++ {
					dst[i*n+j] = src[j*n+i]
				}
			}
		}
	}
}

// TransposeSquareTwiddleComplex64 computes the fused transpose + twiddle
// multiply used by the six-step FFT's steps 3-4:
//
//	dst[i,j] = src[j,i] * twiddle[(i*j) % (n*n)]
//
// dst and src must each hold at least n*n elements and must not alias.
// twiddle must hold at least n*n elements. Uses MulComplex64 to keep the
// multiply in single precision, matching the AVX2 asm's arithmetic.
func TransposeSquareTwiddleComplex64(dst, src, twiddle []complex64, n int) {
	if n <= 0 {
		return
	}

	nn := n * n

	if n <= transposeSmallCutoff {
		for i := range n {
			for j := range n {
				dst[i*n+j] = MulComplex64(src[j*n+i], twiddle[(i*j)%nn])
			}
		}

		return
	}

	for bi := 0; bi < n; bi += transposeBlock {
		iEnd := min(bi+transposeBlock, n)

		for bj := 0; bj < n; bj += transposeBlock {
			jEnd := min(bj+transposeBlock, n)

			for i := bi; i < iEnd; i++ {
				for j := bj; j < jEnd; j++ {
					dst[i*n+j] = MulComplex64(src[j*n+i], twiddle[(i*j)%nn])
				}
			}
		}
	}
}

// TransposeSquareTwiddleConjComplex64 computes the fused transpose +
// conjugate-twiddle multiply used by the six-step FFT's inverse steps 3-4:
//
//	dst[i,j] = src[j,i] * conj(twiddle[(i*j) % (n*n)])
//
// dst and src must each hold at least n*n elements and must not alias.
// twiddle must hold at least n*n elements.
func TransposeSquareTwiddleConjComplex64(dst, src, twiddle []complex64, n int) {
	if n <= 0 {
		return
	}

	nn := n * n

	if n <= transposeSmallCutoff {
		for i := range n {
			for j := range n {
				w := twiddle[(i*j)%nn]
				dst[i*n+j] = MulComplex64(src[j*n+i], complex(real(w), -imag(w)))
			}
		}

		return
	}

	for bi := 0; bi < n; bi += transposeBlock {
		iEnd := min(bi+transposeBlock, n)

		for bj := 0; bj < n; bj += transposeBlock {
			jEnd := min(bj+transposeBlock, n)

			for i := bi; i < iEnd; i++ {
				for j := bj; j < jEnd; j++ {
					w := twiddle[(i*j)%nn]
					dst[i*n+j] = MulComplex64(src[j*n+i], complex(real(w), -imag(w)))
				}
			}
		}
	}
}
