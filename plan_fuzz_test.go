package algofft

import (
	"testing"

	m "github.com/cwbudde/algo-fft/internal/math"
)

func FuzzRoundTripComplex64(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8})
	f.Add([]byte{0})
	f.Add([]byte{9, 10, 11, 12, 13, 14, 15, 16})

	f.Fuzz(func(t *testing.T, data []byte) {
		n := m.NextPowerOfTwo(len(data))
		if n == 0 || n > 1024 {
			return
		}

		plan, err := NewPlanT[complex64](n)
		if err != nil {
			t.Fatalf("NewPlan(%d) returned error: %v", n, err)
		}

		src := make([]complex64, n)
		for i := range src {
			src[i] = complex(float32(data[i%len(data)]), float32(i))
		}

		freq := make([]complex64, n)

		err = plan.Forward(freq, src)
		if err != nil {
			t.Fatalf("Forward() returned error: %v", err)
		}

		out := make([]complex64, n)

		err = plan.Inverse(out, freq)
		if err != nil {
			t.Fatalf("Inverse() returned error: %v", err)
		}

		for i := range src {
			if absComplex64(out[i]-src[i]) > 1e-3 {
				t.Fatalf("out[%d] = %v, want %v", i, out[i], src[i])
			}
		}
	})
}

func FuzzDeterministicForward(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8})

	f.Fuzz(func(t *testing.T, data []byte) {
		n := m.NextPowerOfTwo(len(data))
		if n == 0 || n > 1024 {
			return
		}

		plan, err := NewPlanT[complex64](n)
		if err != nil {
			t.Fatalf("NewPlan(%d) returned error: %v", n, err)
		}

		src := make([]complex64, n)
		for i := range src {
			src[i] = complex(float32(data[i%len(data)]), float32(i))
		}

		out1 := make([]complex64, n)
		out2 := make([]complex64, n)

		err = plan.Forward(out1, src)
		if err != nil {
			t.Fatalf("Forward() returned error: %v", err)
		}

		err = plan.Forward(out2, src)
		if err != nil {
			t.Fatalf("Forward() returned error: %v", err)
		}

		for i := range out1 {
			if absComplex64(out1[i]-out2[i]) > 1e-6 {
				t.Fatalf("out1[%d] != out2[%d] (%v vs %v)", i, i, out1[i], out2[i])
			}
		}
	})
}

func FuzzNoPanicValidInput(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8})

	f.Fuzz(func(t *testing.T, data []byte) {
		n := m.NextPowerOfTwo(len(data))
		if n == 0 || n > 1024 {
			return
		}

		plan, err := NewPlanT[complex64](n)
		if err != nil {
			t.Fatalf("NewPlan(%d) returned error: %v", n, err)
		}

		src := make([]complex64, n)
		for i := range src {
			src[i] = complex(float32(data[i%len(data)]), float32(-i))
		}

		dst := make([]complex64, n)
		_ = plan.Forward(dst, src)
		_ = plan.Inverse(dst, src)
	})
}

func FuzzRoundTripReal(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8})
	f.Add([]byte{0, 1})
	f.Add([]byte{9, 10, 11, 12, 13, 14, 15, 16})

	f.Fuzz(func(t *testing.T, data []byte) {
		n := m.NextPowerOfTwo(len(data))
		if n < 2 || n > 1024 || n%2 != 0 {
			return
		}

		plan, err := NewPlanReal(n)
		if err != nil {
			t.Fatalf("NewPlanReal(%d) returned error: %v", n, err)
		}

		src := make([]float32, n)
		for i := range src {
			src[i] = float32(data[i%len(data)])
		}

		freq := make([]complex64, plan.SpectrumLen())

		err = plan.Forward(freq, src)
		if err != nil {
			t.Fatalf("Forward() returned error: %v", err)
		}

		out := make([]float32, n)

		err = plan.Inverse(out, freq)
		if err != nil {
			t.Fatalf("Inverse() returned error: %v", err)
		}

		for i := range src {
			if abs := out[i] - src[i]; abs < -1e-3 || abs > 1e-3 {
				t.Fatalf("out[%d] = %v, want %v", i, out[i], src[i])
			}
		}
	})
}
