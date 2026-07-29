package transform

// PackedTwiddles stores twiddle factors packed by radix and stage.
// StageOffsets marks the starting index in Values for each stage.
type PackedTwiddles[T Complex] struct {
	Radix        int
	StageOffsets []int
	Values       []T
}

// PackedTwiddleLen reports the exact number of values ComputePackedTwiddles
// produces for (n, radix), and the number of stages. It is the closed form of
// that function's loop, so the two cannot drift.
//
// The table is large: for radix 4 it holds n-1 values when n is a power of
// four and n/2-1 when n is twice one — i.e. roughly a second full twiddle
// table. At n = 2^22 complex128 that is 64 MiB on top of the plain table, which
// is why the route that uses it is gated on a measured win rather than enabled
// wherever it is merely correct.
func PackedTwiddleLen(n, radix int) (int, int) {
	if n <= 0 || radix < 2 {
		return 0, 0
	}

	values, stages := 0, 0

	for size := radix; size <= n; size *= radix {
		values += (size / radix) * (radix - 1)
		stages++
	}

	return values, stages
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

	// Preallocated exactly: appending from a zero-capacity slice reallocated
	// ~log_radix(n) times and left the final capacity up to ~2x the used
	// length, which at these sizes is tens of MiB of transient garbage.
	values, stages := PackedTwiddleLen(n, radix)

	packed := &PackedTwiddles[T]{
		Radix:        radix,
		StageOffsets: make([]int, 0, stages),
		Values:       make([]T, 0, values),
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
