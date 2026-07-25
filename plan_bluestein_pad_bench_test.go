package algofft

import (
	"strconv"
	"testing"
)

// padModelSizes are Bluestein-routed lengths whose padded sub-FFT size the
// shape-aware pad model changes, plus a control it leaves alone. Each has a
// non-smooth n-1, so the planner cannot route it to Rader or the mixed-radix
// engine directly.
//
//nolint:gochecknoglobals // benchmark input table
var padModelSizes = []struct {
	n   int
	pad int // pad chosen by the shape-aware model (was the next power of two)
}{
	{n: 677, pad: 1536},   // was 2048
	{n: 1009, pad: 2048},  // control: unchanged
	{n: 2531, pad: 6144},  // was 8192
	{n: 3079, pad: 7680},  // was 8192
	{n: 4099, pad: 12288}, // was 16384
	{n: 6151, pad: 15360}, // was 16384
	{n: 8209, pad: 24576}, // was 32768
}

// BenchmarkBluesteinPadModel measures the user-visible effect of the pad model:
// a full forward transform at each Bluestein-routed length, once with the
// shape-aware pad ("model") and once with the previous
// always-the-next-power-of-two choice ("pow2").
//
// Both arms run in one process, adjacent per size, because the pad is fixed at
// plan construction: emptying padShapes around NewPlan is enough to build the
// power-of-two baseline, and the transform loop reads nothing global.
// Interleaving this way keeps thermal drift from biasing one arm — on this
// laptop an all-A-then-all-B comparison does not reproduce.
func BenchmarkBluesteinPadModel(b *testing.B) {
	b.Run("complex64", func(b *testing.B) { benchPadModel[complex64](b) })
	b.Run("complex128", func(b *testing.B) { benchPadModel[complex128](b) })
}

func benchPadModel[T Complex](b *testing.B) {
	b.Helper()

	for _, tc := range padModelSizes {
		for _, withModel := range []bool{true, false} {
			arm := "model"
			if !withModel {
				arm = "pow2"
			}

			b.Run("n"+strconv.Itoa(tc.n)+"/"+arm, func(b *testing.B) {
				plan := padModelPlan[T](b, tc.n, withModel)

				src := make([]T, tc.n)
				for i := range src {
					src[i] = T(complex(float64(i%7)-3, float64(i%5)-2))
				}

				dst := make([]T, tc.n)

				if err := plan.Forward(dst, src); err != nil {
					b.Fatalf("warm-up Forward failed: %v", err)
				}

				b.ReportAllocs()
				b.SetBytes(int64(tc.n) * int64(complexWidth[T]()))
				b.ResetTimer()

				for range b.N {
					_ = plan.Forward(dst, src)
				}
			})
		}
	}
}

// padModelPlan builds the plan for one arm. With withModel false the pad model
// is disabled for the duration of plan construction, which is what fixes the
// padded sub-FFT size, so the plan comes out with the power-of-two pad.
func padModelPlan[T Complex](b *testing.B, n int, withModel bool) *Plan[T] {
	b.Helper()

	if !withModel {
		saved := padShapes
		padShapes = nil

		defer func() { padShapes = saved }()
	}

	plan, err := NewPlan[T](n)
	if err != nil {
		b.Fatalf("NewPlan(%d) failed: %v", n, err)
	}

	if plan.Algorithm() != "bluestein" {
		b.Fatalf("n=%d uses %q, want bluestein", n, plan.Algorithm())
	}

	return plan
}

// complexWidth returns the byte width of one complex sample of T.
func complexWidth[T Complex]() int {
	var zero T

	if _, ok := any(zero).(complex64); ok {
		return 8
	}

	return 16
}
