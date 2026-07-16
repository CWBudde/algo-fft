package algofft

import (
	"math/cmplx"
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/reference"
)

// ForceGeneric benchmarks measure the pure-Go fallback dispatch path in a
// default (SIMD-enabled) binary. Each case validates one output against the
// reference DFT or the round-trip input before timing, so the fallback path
// that gets benchmarked is also known to be correct.

func BenchmarkPlanForward_1024_ForceGeneric(b *testing.B) {
	benchmarkPlanForwardForceGeneric(b, 1024)
}

func BenchmarkPlanInverse_1024_ForceGeneric(b *testing.B) {
	benchmarkPlanInverseForceGeneric(b, 1024)
}

func BenchmarkPlanForward_512_Complex128_ForceGeneric(b *testing.B) {
	benchmarkPlanForwardComplex128ForceGeneric(b, 512)
}

func BenchmarkPlanInverse_512_Complex128_ForceGeneric(b *testing.B) {
	benchmarkPlanInverseComplex128ForceGeneric(b, 512)
}

func benchmarkPlanForwardForceGeneric(b *testing.B, fftSize int) {
	b.Helper()
	forceGenericForBenchmark(b)

	plan, err := NewPlanT[complex64](fftSize)
	if err != nil {
		b.Fatalf("NewPlan(%d) returned error: %v", fftSize, err)
	}

	src := make([]complex64, fftSize)
	for i := range src {
		src[i] = complex(float32(i+1), float32(-i))
	}

	dst := make([]complex64, fftSize)
	if err := plan.Forward(dst, src); err != nil {
		b.Fatalf("Forward() returned error: %v", err)
	}
	assertComplex64CloseRel(b, dst, reference.NaiveDFT(src), 1e-5)

	b.ReportAllocs()
	b.SetBytes(int64(fftSize * 8)) // 8 bytes per complex64 for throughput calculation
	b.ResetTimer()

	for b.Loop() {
		fwdErr := plan.Forward(dst, src)
		if fwdErr != nil {
			b.Fatalf("Forward() returned error: %v", fwdErr)
		}
	}
}

func benchmarkPlanInverseForceGeneric(b *testing.B, fftSize int) {
	b.Helper()
	forceGenericForBenchmark(b)

	plan, err := NewPlanT[complex64](fftSize)
	if err != nil {
		b.Fatalf("NewPlan(%d) returned error: %v", fftSize, err)
	}

	src := make([]complex64, fftSize)
	for i := range src {
		src[i] = complex(float32(i+1), float32(-i))
	}

	freq := make([]complex64, fftSize)
	if err := plan.Forward(freq, src); err != nil {
		b.Fatalf("Forward() returned error: %v", err)
	}

	dst := make([]complex64, fftSize)
	if err := plan.Inverse(dst, freq); err != nil {
		b.Fatalf("Inverse() returned error: %v", err)
	}
	assertComplex64CloseRel(b, dst, src, 1e-5)

	b.ReportAllocs()
	b.SetBytes(int64(fftSize * 8)) // 8 bytes per complex64 for throughput calculation
	b.ResetTimer()

	for b.Loop() {
		invErr := plan.Inverse(dst, freq)
		if invErr != nil {
			b.Fatalf("Inverse() returned error: %v", invErr)
		}
	}
}

func benchmarkPlanForwardComplex128ForceGeneric(b *testing.B, fftSize int) {
	b.Helper()
	forceGenericForBenchmark(b)

	plan, err := NewPlanT[complex128](fftSize)
	if err != nil {
		b.Fatalf("NewPlan(%d) returned error: %v", fftSize, err)
	}

	src := make([]complex128, fftSize)
	for i := range src {
		src[i] = complex(float64(i+1), float64(-i))
	}

	dst := make([]complex128, fftSize)
	if err := plan.Forward(dst, src); err != nil {
		b.Fatalf("Forward() returned error: %v", err)
	}
	assertComplex128CloseRel(b, dst, reference.NaiveDFT128(src), 1e-12)

	b.ReportAllocs()
	b.SetBytes(int64(fftSize * 16)) // 16 bytes per complex128 for throughput calculation
	b.ResetTimer()

	for b.Loop() {
		fwdErr := plan.Forward(dst, src)
		if fwdErr != nil {
			b.Fatalf("Forward() returned error: %v", fwdErr)
		}
	}
}

func benchmarkPlanInverseComplex128ForceGeneric(b *testing.B, fftSize int) {
	b.Helper()
	forceGenericForBenchmark(b)

	plan, err := NewPlanT[complex128](fftSize)
	if err != nil {
		b.Fatalf("NewPlan(%d) returned error: %v", fftSize, err)
	}

	src := make([]complex128, fftSize)
	for i := range src {
		src[i] = complex(float64(i+1), float64(-i))
	}

	freq := make([]complex128, fftSize)
	if err := plan.Forward(freq, src); err != nil {
		b.Fatalf("Forward() returned error: %v", err)
	}

	dst := make([]complex128, fftSize)
	if err := plan.Inverse(dst, freq); err != nil {
		b.Fatalf("Inverse() returned error: %v", err)
	}
	assertComplex128CloseRel(b, dst, src, 1e-12)

	b.ReportAllocs()
	b.SetBytes(int64(fftSize * 16)) // 16 bytes per complex128 for throughput calculation
	b.ResetTimer()

	for b.Loop() {
		invErr := plan.Inverse(dst, freq)
		if invErr != nil {
			b.Fatalf("Inverse() returned error: %v", invErr)
		}
	}
}

// forceGenericForBenchmark forces the pure-Go fallback dispatch for plans
// created afterwards. Kernels are selected at plan creation, so this must be
// called before NewPlanT. Real detection is restored when the benchmark ends.
func forceGenericForBenchmark(b *testing.B) {
	b.Helper()

	features := cpu.DetectFeatures()
	features.ForceGeneric = true
	cpu.SetForcedFeatures(features)
	b.Cleanup(cpu.ResetDetection)
}

// assertComplex64CloseRel fails the benchmark if any element of got deviates
// from want by more than tol relative to the largest magnitude in want.
func assertComplex64CloseRel(b *testing.B, got, want []complex64, tol float64) {
	b.Helper()

	if len(got) != len(want) {
		b.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
	}

	var maxMag float64
	for i := range want {
		if mag := cmplx.Abs(complex128(want[i])); mag > maxMag {
			maxMag = mag
		}
	}

	limit := tol * maxMag
	for i := range got {
		diff := cmplx.Abs(complex128(got[i]) - complex128(want[i]))
		if diff > limit {
			b.Fatalf("index %d: got %v, want %v, diff %g > %g", i, got[i], want[i], diff, limit)
		}
	}
}

// assertComplex128CloseRel fails the benchmark if any element of got deviates
// from want by more than tol relative to the largest magnitude in want.
func assertComplex128CloseRel(b *testing.B, got, want []complex128, tol float64) {
	b.Helper()

	if len(got) != len(want) {
		b.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
	}

	var maxMag float64
	for i := range want {
		if mag := cmplx.Abs(want[i]); mag > maxMag {
			maxMag = mag
		}
	}

	limit := tol * maxMag
	for i := range got {
		diff := cmplx.Abs(got[i] - want[i])
		if diff > limit {
			b.Fatalf("index %d: got %v, want %v, diff %g > %g", i, got[i], want[i], diff, limit)
		}
	}
}
