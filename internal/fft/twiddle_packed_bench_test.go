package fft

import "testing"

func BenchmarkPackedTwiddleLookup_Radix4_4096(b *testing.B) {
	benchmarkPackedTwiddleLookup[complex64](b, 4096, 4)
}

func BenchmarkPackedTwiddleLookup_Radix8_4096(b *testing.B) {
	benchmarkPackedTwiddleLookup[complex64](b, 4096, 8)
}

func BenchmarkPackedTwiddleLookup_Radix16_4096(b *testing.B) {
	benchmarkPackedTwiddleLookup[complex64](b, 4096, 16)
}

func BenchmarkPackedTwiddleLookup_Radix4_65536(b *testing.B) {
	benchmarkPackedTwiddleLookup[complex64](b, 65536, 4)
}

func BenchmarkPackedTwiddleLookup_Radix8_65536(b *testing.B) {
	benchmarkPackedTwiddleLookup[complex64](b, 65536, 8)
}

func BenchmarkPackedTwiddleLookup_Radix16_65536(b *testing.B) {
	benchmarkPackedTwiddleLookup[complex64](b, 65536, 16)
}

func BenchmarkBaseTwiddleLookup_Radix4_4096(b *testing.B) {
	benchmarkBaseTwiddleLookup[complex64](b, 4096, 4)
}

func BenchmarkBaseTwiddleLookup_Radix8_4096(b *testing.B) {
	benchmarkBaseTwiddleLookup[complex64](b, 4096, 8)
}

func BenchmarkBaseTwiddleLookup_Radix16_4096(b *testing.B) {
	benchmarkBaseTwiddleLookup[complex64](b, 4096, 16)
}

func BenchmarkBaseTwiddleLookup_Radix4_65536(b *testing.B) {
	benchmarkBaseTwiddleLookup[complex64](b, 65536, 4)
}

func BenchmarkBaseTwiddleLookup_Radix8_65536(b *testing.B) {
	benchmarkBaseTwiddleLookup[complex64](b, 65536, 8)
}

func BenchmarkBaseTwiddleLookup_Radix16_65536(b *testing.B) {
	benchmarkBaseTwiddleLookup[complex64](b, 65536, 16)
}

func benchmarkPackedTwiddleLookup[T Complex](b *testing.B, n, radix int) {
	twiddle := ComputeTwiddleFactors[T](n)
	packed := ComputePackedTwiddles[T](n, radix, twiddle)
	if packed == nil {
		b.Fatalf("ComputePackedTwiddles(%d, %d) returned nil", n, radix)
	}

	total := 0
	values := packed.Values
	b.ResetTimer()
	for b.Loop() {
		for i := range values {
			total += int(real(complex128(values[i])))
		}
	}
	_ = total
}

func benchmarkBaseTwiddleLookup[T Complex](b *testing.B, n, radix int) {
	twiddle := ComputeTwiddleFactors[T](n)

	total := 0
	b.ResetTimer()
	for b.Loop() {
		for size := radix; size <= n; size *= radix {
			step := n / size
			span := size / radix
			for j := 0; j < span; j++ {
				base := j * step
				for k := 1; k < radix; k++ {
					total += int(real(complex128(twiddle[(k*base)%n])))
				}
			}
		}
	}
	_ = total
}
