package algofft

import "testing"

func BenchmarkConvolveReal_64x64(b *testing.B)     { benchmarkConvolveReal(b, 64, 64) }
func BenchmarkConvolveReal_256x256(b *testing.B)   { benchmarkConvolveReal(b, 256, 256) }
func BenchmarkConvolveReal_1024x1024(b *testing.B) { benchmarkConvolveReal(b, 1024, 1024) }

func benchmarkConvolveReal(b *testing.B, aLen, bLen int) {
	b.Helper()

	a := make([]float32, aLen)
	bData := make([]float32, bLen)

	for i := range a {
		a[i] = float32(i%17) - 8
	}

	for i := range bData {
		bData[i] = float32(i%9) - 4
	}

	outLen := aLen + bLen - 1
	dst := make([]float32, outLen)

	b.ReportAllocs()
	b.SetBytes(int64(outLen * 4))
	b.ResetTimer()

	for b.Loop() {
		err := ConvolveReal(dst, a, bData)
		if err != nil {
			b.Fatalf("ConvolveReal() returned error: %v", err)
		}
	}
}
