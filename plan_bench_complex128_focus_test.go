package algofft

import (
	"fmt"
	"strings"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/fft"
)

func BenchmarkPlanForward_128_Complex128_Focus(b *testing.B) {
	benchmarkPlanForwardComplex128Focus(b, 128)
}

func BenchmarkPlanForward_512_Complex128_Focus(b *testing.B) {
	benchmarkPlanForwardComplex128Focus(b, 512)
}

func BenchmarkPlanForward_8192_Complex128_Focus(b *testing.B) {
	benchmarkPlanForwardComplex128Focus(b, 8192)
}

func BenchmarkPlanInverse_128_Complex128_Focus(b *testing.B) {
	benchmarkPlanInverseComplex128Focus(b, 128)
}

func BenchmarkPlanInverse_512_Complex128_Focus(b *testing.B) {
	benchmarkPlanInverseComplex128Focus(b, 512)
}

func BenchmarkPlanInverse_8192_Complex128_Focus(b *testing.B) {
	benchmarkPlanInverseComplex128Focus(b, 8192)
}

func benchmarkPlanForwardComplex128Focus(b *testing.B, fftSize int) {
	b.Helper()

	plan, err := NewPlanT[complex128](fftSize)
	if err != nil {
		b.Fatalf("NewPlan(%d) returned error: %v", fftSize, err)
	}

	logPlanDetails(b, plan)

	src := make([]complex128, fftSize)
	for i := range src {
		src[i] = complex(float64(i+1), float64(-i))
	}

	dst := make([]complex128, fftSize)

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

func benchmarkPlanInverseComplex128Focus(b *testing.B, fftSize int) {
	b.Helper()

	plan, err := NewPlanT[complex128](fftSize)
	if err != nil {
		b.Fatalf("NewPlan(%d) returned error: %v", fftSize, err)
	}

	logPlanDetails(b, plan)

	src := make([]complex128, fftSize)
	for i := range src {
		src[i] = complex(float64(i+1), float64(-i))
	}

	dst := make([]complex128, fftSize)

	b.ReportAllocs()
	b.SetBytes(int64(fftSize * 16)) // 16 bytes per complex128 for throughput calculation
	b.ResetTimer()

	for b.Loop() {
		invErr := plan.Inverse(dst, src)
		if invErr != nil {
			b.Fatalf("Inverse() returned error: %v", invErr)
		}
	}
}

func logPlanDetails[T Complex](b *testing.B, plan *Plan[T]) {
	b.Helper()
	b.Logf(
		"plan: size=%d strategy=%s algorithm=%s twiddle=%s",
		plan.n,
		kernelStrategyName(plan.kernelStrategy),
		plan.algorithm,
		twiddleLayoutSummary(plan),
	)
}

func kernelStrategyName(strategy KernelStrategy) string {
	switch strategy {
	case KernelDIT:
		return "DIT"
	case KernelStockham:
		return "Stockham"
	case KernelSixStep:
		return "SixStep"
	case KernelEightStep:
		return "EightStep"
	case KernelBluestein:
		return "Bluestein"
	case KernelRecursive:
		return "Recursive"
	default:
		return "Auto"
	}
}

func twiddleLayoutSummary[T Complex](plan *Plan[T]) string {
	parts := make([]string, 0, 6)
	parts = append(parts, fmt.Sprintf("twiddle=%d", len(plan.twiddle)))

	if len(plan.codeletTwiddleForward) > 0 {
		parts = append(parts, fmt.Sprintf("codelet_fwd=%d", len(plan.codeletTwiddleForward)))
	}

	if len(plan.codeletTwiddleInverse) > 0 {
		parts = append(parts, fmt.Sprintf("codelet_inv=%d", len(plan.codeletTwiddleInverse)))
	}

	if plan.packedTwiddle4 != nil {
		parts = append(parts, packedSummary("packed4", plan.packedTwiddle4))
	}

	if plan.packedTwiddle8 != nil {
		parts = append(parts, packedSummary("packed8", plan.packedTwiddle8))
	}

	if plan.packedTwiddle16 != nil {
		parts = append(parts, packedSummary("packed16", plan.packedTwiddle16))
	}

	return strings.Join(parts, ",")
}

func packedSummary[T Complex](label string, packed *fft.PackedTwiddles[T]) string {
	return fmt.Sprintf("%s(radix=%d,stages=%d,values=%d)", label, packed.Radix, len(packed.StageOffsets), len(packed.Values))
}
