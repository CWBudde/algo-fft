package algofft

import "testing"

func BenchmarkConvolve_64x64(b *testing.B)     { benchmarkConvolve(b, 64, 64) }
func BenchmarkConvolve_256x256(b *testing.B)   { benchmarkConvolve(b, 256, 256) }
func BenchmarkConvolve_1024x1024(b *testing.B) { benchmarkConvolve(b, 1024, 1024) }

// Awkward output lengths: convLen = aLen+bLen-1 is prime or otherwise not
// executable exactly, so an unpadded plan routes to Rader or Bluestein.
func BenchmarkConvolve_66x66(b *testing.B)     { benchmarkConvolve(b, 66, 66) }     // convLen 131 (prime)
func BenchmarkConvolve_129x129(b *testing.B)   { benchmarkConvolve(b, 129, 129) }   // convLen 257 (Rader)
func BenchmarkConvolve_500x510(b *testing.B)   { benchmarkConvolve(b, 500, 510) }   // convLen 1009 (prime)
func BenchmarkConvolve_2001x2001(b *testing.B) { benchmarkConvolve(b, 2001, 2001) } // convLen 4001 (Rader)

func BenchmarkConvolve128_500x510(b *testing.B) { benchmarkConvolve128(b, 500, 510) }

func BenchmarkConvolver_500x510(b *testing.B) { benchmarkConvolver(b, 500, 510) }

func benchmarkConvolve(b *testing.B, aLen, bLen int) {
	b.Helper()

	a := make([]complex64, aLen)
	bData := make([]complex64, bLen)

	for i := range a {
		a[i] = complex(float32(i%13)-6, float32(i%7)-3)
	}

	for i := range bData {
		bData[i] = complex(float32(i%11)-5, float32(i%5)-2)
	}

	outLen := aLen + bLen - 1
	dst := make([]complex64, outLen)

	b.ReportAllocs()
	b.SetBytes(int64(outLen * 8))
	b.ResetTimer()

	for b.Loop() {
		err := Convolve(dst, a, bData)
		if err != nil {
			b.Fatalf("Convolve() returned error: %v", err)
		}
	}
}

func benchmarkConvolve128(b *testing.B, aLen, bLen int) {
	b.Helper()

	a := make([]complex128, aLen)
	bData := make([]complex128, bLen)

	for i := range a {
		a[i] = complex(float64(i%13)-6, float64(i%7)-3)
	}

	for i := range bData {
		bData[i] = complex(float64(i%11)-5, float64(i%5)-2)
	}

	outLen := aLen + bLen - 1
	dst := make([]complex128, outLen)

	b.ReportAllocs()
	b.SetBytes(int64(outLen * 16))
	b.ResetTimer()

	for b.Loop() {
		err := Convolve128(dst, a, bData)
		if err != nil {
			b.Fatalf("Convolve128() returned error: %v", err)
		}
	}
}

func benchmarkConvolver(b *testing.B, aLen, bLen int) {
	b.Helper()

	a := make([]complex64, aLen)
	bData := make([]complex64, bLen)

	for i := range a {
		a[i] = complex(float32(i%13)-6, float32(i%7)-3)
	}

	for i := range bData {
		bData[i] = complex(float32(i%11)-5, float32(i%5)-2)
	}

	conv, err := NewConvolver[complex64](aLen, bLen)
	if err != nil {
		b.Fatalf("NewConvolver() returned error: %v", err)
	}

	outLen := aLen + bLen - 1
	dst := make([]complex64, outLen)

	b.ReportAllocs()
	b.SetBytes(int64(outLen * 8))
	b.ResetTimer()

	for b.Loop() {
		err := conv.Convolve(dst, a, bData)
		if err != nil {
			b.Fatalf("Convolve() returned error: %v", err)
		}
	}
}
