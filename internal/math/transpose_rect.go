package math

// TransposeRect transposes a rows×cols matrix stored row-major in src into
// dst, which receives the cols×rows transpose: dst[c*rows+r] = src[r*cols+c].
// dst and src must not overlap; both must hold at least rows*cols elements.
//
// The walk is tiled with the same tile edge as TransposeSquare so the strided
// stream stays within the TLB and L1 (see transposeBlock).
func TransposeRect[T any](dst, src []T, rows, cols int) {
	if rows <= 0 || cols <= 0 {
		return
	}

	// A single row or column is the same sequence in both layouts.
	if rows == 1 || cols == 1 {
		copy(dst[:rows*cols], src[:rows*cols])
		return
	}

	if rows*cols <= transposeSmallCutoff*transposeSmallCutoff {
		transposeRectBlocked(dst, src, rows, cols, rows*cols)
		return
	}

	transposeRectBlocked(dst, src, rows, cols, transposeBlock)
}

// transposeRectBlocked copies tiles of block×block elements so that both the
// source (row-major) and destination (column-major) streams stay resident
// while a tile is processed.
func transposeRectBlocked[T any](dst, src []T, rows, cols, block int) {
	for br := 0; br < rows; br += block {
		rEnd := min(br+block, rows)

		for bc := 0; bc < cols; bc += block {
			cEnd := min(bc+block, cols)

			for r := br; r < rEnd; r++ {
				srcRow := r * cols
				for c := bc; c < cEnd; c++ {
					dst[c*rows+r] = src[srcRow+c]
				}
			}
		}
	}
}
