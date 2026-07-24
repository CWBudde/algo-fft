package transform

import (
	"testing"
	"unsafe"
)

// Benchmarks for the packed mixed-radix Stockham engine. They call
// stockhamPackedRun directly (like the correctness tests) so the numbers are
// comparable on every build, including SIMD builds where the dispatch toggle
// is off. RoundTrip alternates forward and inverse to expose the cache cost
// of the twiddle tables both directions touch.

func BenchmarkStockhamPackedForward(b *testing.B) {
	for _, n := range []int{4096, 65536, 1 << 20} {
		b.Run(sizeLabel(n)+"/complex64", func(b *testing.B) {
			benchmarkStockhamPacked[complex64](b, n, false)
		})
		b.Run(sizeLabel(n)+"/complex128", func(b *testing.B) {
			benchmarkStockhamPacked[complex128](b, n, false)
		})
	}
}

func BenchmarkStockhamPackedInverse(b *testing.B) {
	for _, n := range []int{4096, 65536, 1 << 20} {
		b.Run(sizeLabel(n)+"/complex64", func(b *testing.B) {
			benchmarkStockhamPacked[complex64](b, n, true)
		})
		b.Run(sizeLabel(n)+"/complex128", func(b *testing.B) {
			benchmarkStockhamPacked[complex128](b, n, true)
		})
	}
}

func BenchmarkStockhamPackedRoundTrip(b *testing.B) {
	for _, n := range []int{4096, 65536, 1 << 20} {
		b.Run(sizeLabel(n)+"/complex64", func(b *testing.B) {
			benchmarkStockhamPackedRoundTrip[complex64](b, n)
		})
		b.Run(sizeLabel(n)+"/complex128", func(b *testing.B) {
			benchmarkStockhamPackedRoundTrip[complex128](b, n)
		})
	}
}

func benchmarkStockhamPacked[T Complex](b *testing.B, n int, inverse bool) {
	b.Helper()

	src := make([]T, n)
	for i := range src {
		src[i] = complexFromFloats[T](float64(i%7)-3, float64(i%5)-2)
	}

	twiddle := ComputeTwiddleFactors[T](n)

	packed := ComputePackedTwiddles(n, 4, twiddle)
	if packed == nil {
		b.Fatalf("packed table for n=%d is nil", n)
	}

	dst := make([]T, n)
	scratch := make([]T, n)

	var zero T

	b.ReportAllocs()
	b.SetBytes(int64(n) * int64(unsafe.Sizeof(zero)))
	b.ResetTimer()

	for b.Loop() {
		if !stockhamPackedRun(dst, src, twiddle, scratch, packed, inverse) {
			b.Fatalf("stockhamPackedRun(%d) returned false", n)
		}
	}
}

func benchmarkStockhamPackedRoundTrip[T Complex](b *testing.B, n int) {
	b.Helper()

	src := make([]T, n)
	for i := range src {
		src[i] = complexFromFloats[T](float64(i%7)-3, float64(i%5)-2)
	}

	twiddle := ComputeTwiddleFactors[T](n)

	packed := ComputePackedTwiddles(n, 4, twiddle)
	if packed == nil {
		b.Fatalf("packed table for n=%d is nil", n)
	}

	spectrum := make([]T, n)
	dst := make([]T, n)
	scratch := make([]T, n)

	var zero T

	b.ReportAllocs()
	b.SetBytes(2 * int64(n) * int64(unsafe.Sizeof(zero)))
	b.ResetTimer()

	for b.Loop() {
		if !stockhamPackedRun(spectrum, src, twiddle, scratch, packed, false) {
			b.Fatalf("forward stockhamPackedRun(%d) returned false", n)
		}

		if !stockhamPackedRun(dst, spectrum, twiddle, scratch, packed, true) {
			b.Fatalf("inverse stockhamPackedRun(%d) returned false", n)
		}
	}
}

func complexFromFloats[T Complex](re, im float64) T {
	var zero T
	if _, ok := any(zero).(complex64); ok {
		v, _ := any(complex(float32(re), float32(im))).(T)
		return v
	}

	v, _ := any(complex(re, im)).(T)

	return v
}

func sizeLabel(n int) string {
	switch {
	case n >= 1<<20 && n%(1<<20) == 0:
		return itoa(n>>20) + "M"
	case n >= 1024 && n%1024 == 0:
		return itoa(n>>10) + "K"
	default:
		return itoa(n)
	}
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}

	var buf [20]byte
	i := len(buf)

	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}

	return string(buf[i:])
}
