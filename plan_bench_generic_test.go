package algofft

import (
	"math/cmplx"
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/reference"
)

// ForceGeneric benchmarks measure the pure-Go fallback kernel dispatch path
// in a default (SIMD-enabled) binary. Each case validates one output against
// the reference DFT or the round-trip input before timing, so the fallback
// path that gets benchmarked is also known to be correct.
//
// ForceGeneric alone is not enough to reach the fallback kernels: generic
// (SIMDNone) codelets are registered for every power-of-two size from 4
// through 16384, so an auto-strategy plan would bind a zero-dispatch codelet
// and never touch the fallback dispatch. Plans are therefore created via
// newForceGenericFallbackPlan, which forces KernelStockham to bypass codelet
// binding (all registered codelets are DIT) and asserts the plan landed on
// the fallback path.

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

	plan := newForceGenericFallbackPlan[complex64](b, fftSize)

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

	plan := newForceGenericFallbackPlan[complex64](b, fftSize)

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

	plan := newForceGenericFallbackPlan[complex128](b, fftSize)

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

	plan := newForceGenericFallbackPlan[complex128](b, fftSize)

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

// newForceGenericFallbackPlan creates a plan that exercises the pure-Go
// fallback kernel dispatch path. Generic codelets cover all power-of-two
// sizes up to 16384 even under ForceGeneric, so the plan forces
// KernelStockham to skip codelet binding (every registered codelet is DIT).
// The bound algorithm is asserted so a future Stockham codelet cannot
// silently move the benchmark back onto the zero-dispatch codelet path.
func newForceGenericFallbackPlan[T Complex](b *testing.B, fftSize int) *Plan[T] {
	b.Helper()

	plan, err := NewPlanWithOptions[T](fftSize, PlanOptions{Strategy: KernelStockham})
	if err != nil {
		b.Fatalf("NewPlanWithOptions(%d) returned error: %v", fftSize, err)
	}

	if algo := plan.Algorithm(); algo != "stockham" {
		b.Fatalf("plan bound algorithm %q, want fallback kernel dispatch (stockham)", algo)
	}

	return plan
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
