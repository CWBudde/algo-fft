package fft

// PackedTwiddles stores twiddle factors packed by radix and stage.
// StageOffsets marks the starting index in Values for each stage.
type PackedTwiddles[T Complex] struct {
	Radix        int
	StageOffsets []int
	Values       []T
}

// ComputePackedTwiddles precomputes twiddles for radix-r stages.
// It returns nil when inputs are invalid or the radix is unsupported.
func ComputePackedTwiddles[T Complex](n, radix int, twiddle []T) *PackedTwiddles[T] {
	if n <= 0 || len(twiddle) < n {
		return nil
	}

	if radix < 2 || (radix&(radix-1)) != 0 {
		return nil
	}

	packed := &PackedTwiddles[T]{
		Radix:        radix,
		StageOffsets: make([]int, 0),
		Values:       make([]T, 0),
	}

	for size := radix; size <= n; size *= radix {
		step := n / size
		stageOffset := len(packed.Values)
		packed.StageOffsets = append(packed.StageOffsets, stageOffset)

		span := size / radix
		for j := range span {
			base := j * step
			for k := 1; k < radix; k++ {
				idx := (k * base) % n
				packed.Values = append(packed.Values, twiddle[idx])
			}
		}
	}

	if len(packed.StageOffsets) == 0 {
		return nil
	}

	return packed
}
