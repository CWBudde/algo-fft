package algoforge

import "testing"

func BenchmarkConvolve_64x64(b *testing.B)     { benchmarkConvolve(b, 64, 64) }
func BenchmarkConvolve_256x256(b *testing.B)   { benchmarkConvolve(b, 256, 256) }
func BenchmarkConvolve_1024x1024(b *testing.B) { benchmarkConvolve(b, 1024, 1024) }

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
