package transform

import (
	"testing"

	imath "github.com/cwbudde/algo-fft/internal/math"
)

// TestPackedTwiddleLenMatchesBuild pins the closed form against the loop it
// describes. If they drift, ComputePackedTwiddles silently reallocates again
// and the documented memory figure stops being true.
func TestPackedTwiddleLenMatchesBuild(t *testing.T) {
	t.Parallel()

	for _, radix := range []int{2, 4} {
		for n := radix; n <= 1<<16; n *= 2 {
			twiddle := imath.ComputeTwiddleFactors[complex64](n)

			packed := ComputePackedTwiddles[complex64](n, radix, twiddle)
			if packed == nil {
				t.Fatalf("n=%d radix=%d: ComputePackedTwiddles returned nil", n, radix)
			}

			values, stages := PackedTwiddleLen(n, radix)

			if values != len(packed.Values) {
				t.Errorf("n=%d radix=%d: PackedTwiddleLen values = %d, built %d", n, radix, values, len(packed.Values))
			}

			if stages != len(packed.StageOffsets) {
				t.Errorf("n=%d radix=%d: PackedTwiddleLen stages = %d, built %d", n, radix, stages, len(packed.StageOffsets))
			}

			// The preallocation must be exact, not merely sufficient: an
			// over-estimate would quietly double the table's footprint.
			if cap(packed.Values) != values {
				t.Errorf("n=%d radix=%d: cap(Values) = %d, want exactly %d", n, radix, cap(packed.Values), values)
			}
		}
	}
}

// TestPackedTwiddleLenRadix4ClosedForm checks the figure quoted in the docs:
// n-1 values for a power of four, n/2-1 for twice one.
func TestPackedTwiddleLenRadix4ClosedForm(t *testing.T) {
	t.Parallel()

	tests := []struct{ n, want int }{
		{n: 4, want: 3},       // 4^1
		{n: 16, want: 15},     // 4^2
		{n: 1024, want: 1023}, // 4^5
		{n: 8, want: 3},       // 2*4^1
		{n: 32, want: 15},     // 2*4^2
		{n: 2048, want: 1023}, // 2*4^5
		{n: 1 << 20, want: (1 << 20) - 1},
		{n: 1 << 21, want: (1 << 20) - 1},
	}

	for _, tt := range tests {
		if got, _ := PackedTwiddleLen(tt.n, 4); got != tt.want {
			t.Errorf("PackedTwiddleLen(%d, 4) = %d, want %d", tt.n, got, tt.want)
		}
	}
}
