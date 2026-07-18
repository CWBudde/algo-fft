package math

// transposeBlock is the tile edge used by TransposeSquare. Small tiles
// win here because the strided (column-major) stream touches one page per
// row: an 8-row tile stays within the TLB and L1 even for 1024×1024
// complex128 matrices. Measured across block sizes 4–64 on AVX2 hardware,
// 8 is at or near the optimum for both precisions at every size from
// 64×64 to 1024×1024 (16+ falls off a cliff beyond 512×512).
const transposeBlock = 8

// transposeSmallCutoff is the largest n handled by the unblocked walk: at
// n ≤ 32 the whole matrix fits in L1 for both precisions, so tiling only
// adds loop overhead (measured ~20% at 16×16/32×32 complex64).
const transposeSmallCutoff = 32

// TransposeSquare transposes an n×n matrix stored in row-major order in
// place. data must hold at least n*n elements.
func TransposeSquare[T any](data []T, n int) {
	if n <= 1 {
		return
	}

	if n <= transposeSmallCutoff {
		transposeSquareBlocked(data, n, n)
		return
	}

	transposeSquareBlocked(data, n, transposeBlock)
}

// transposeSquareBlocked transposes with a cache-blocked walk: tiles of
// block×block elements are swapped pairwise across the diagonal, so both
// the row-major and column-major streams stay resident while a tile pair
// is processed.
func transposeSquareBlocked[T any](data []T, n, block int) {
	for bi := 0; bi < n; bi += block {
		iEnd := min(bi+block, n)

		// Diagonal tile: swap the strictly-lower triangle with the upper.
		for i := bi + 1; i < iEnd; i++ {
			for j := bi; j < i; j++ {
				data[i*n+j], data[j*n+i] = data[j*n+i], data[i*n+j]
			}
		}

		// Off-diagonal tiles: swap tile (bi,bj) with its mirror (bj,bi).
		for bj := bi + block; bj < n; bj += block {
			jEnd := min(bj+block, n)

			for i := bi; i < iEnd; i++ {
				row := i * n
				for j := bj; j < jEnd; j++ {
					data[row+j], data[j*n+i] = data[j*n+i], data[row+j]
				}
			}
		}
	}
}
