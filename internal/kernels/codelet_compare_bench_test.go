package kernels

import (
	"fmt"
	"math/rand"
	"runtime"
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/registry"
)

// codeletBenchInputSeed fixes the input across runs so a candidate ordering
// measured today is comparable with one measured tomorrow.
const codeletBenchInputSeed = 42

// codeletBenchInput returns 2*n pseudo-random float64 values in [-1, 1).
//
// The values are generated as float64 for both precisions and narrowed to
// float32 by the complex64 caller, so the two precision arms of a size see
// numerically identical input and their ratio stays like-for-like.
//
// Random input matters here: the previous fill was a period-35 pattern whose
// spectrum is almost entirely zero, which made the benchmark partly time
// cancellation and denormal behaviour that differs per candidate.
func codeletBenchInput(n int) []float64 {
	rng := rand.New(rand.NewSource(codeletBenchInputSeed)) //nolint:gosec // deterministic benchmark input, not cryptography

	values := make([]float64, 2*n)
	for i := range values {
		values[i] = rng.Float64()*2 - 1
	}

	return values
}

// BenchmarkCodeletCandidates64 benchmarks every registered complex64 codelet
// that is runnable on this CPU, grouped by size, so per-size registry
// priorities can be validated against measured performance.
func BenchmarkCodeletCandidates64(b *testing.B) {
	features := cpu.DetectFeatures()

	for _, size := range registry.Registry64.GetAvailableSizes(features) {
		for _, entry := range registry.Registry64.GetAllForSize(size) {
			if entry.Priority < 0 || !registry.CPUSupports(features, entry.SIMDLevel) {
				continue
			}

			b.Run(fmt.Sprintf("size%d/%s/forward", size, entry.Signature), func(b *testing.B) {
				benchmarkCodelet64(b, &entry, false)
			})
			b.Run(fmt.Sprintf("size%d/%s/inverse", size, entry.Signature), func(b *testing.B) {
				benchmarkCodelet64(b, &entry, true)
			})
		}
	}
}

// BenchmarkCodeletCandidates128 is the complex128 twin of
// BenchmarkCodeletCandidates64.
func BenchmarkCodeletCandidates128(b *testing.B) {
	features := cpu.DetectFeatures()

	for _, size := range registry.Registry128.GetAvailableSizes(features) {
		for _, entry := range registry.Registry128.GetAllForSize(size) {
			if entry.Priority < 0 || !registry.CPUSupports(features, entry.SIMDLevel) {
				continue
			}

			b.Run(fmt.Sprintf("size%d/%s/forward", size, entry.Signature), func(b *testing.B) {
				benchmarkCodelet128(b, &entry, false)
			})
			b.Run(fmt.Sprintf("size%d/%s/inverse", size, entry.Signature), func(b *testing.B) {
				benchmarkCodelet128(b, &entry, true)
			})
		}
	}
}

func benchmarkCodelet64(b *testing.B, entry *registry.CodeletEntry[complex64], inverse bool) {
	b.Helper()

	size := entry.Size
	src := make([]complex64, size)
	dst := make([]complex64, size)
	scratch := make([]complex64, size)

	values := codeletBenchInput(size)
	for i := range src {
		src[i] = complex(float32(values[2*i]), float32(values[2*i+1]))
	}

	twiddle := ComputeTwiddleFactors[complex64](size)
	twiddleForward, twiddleInverse, fwdBacking, invBacking := prepareCodeletTwiddles64(size, twiddle, entry)

	fn, tw := entry.Forward, twiddleForward
	if inverse {
		fn, tw = entry.Inverse, twiddleInverse
	}

	if !fn(dst, src, tw, scratch) {
		b.Skipf("codelet %s rejected size %d", entry.Signature, size)
	}

	b.ReportAllocs()
	b.SetBytes(int64(size) * 8)
	b.ResetTimer()

	for range b.N {
		fn(dst, src, tw, scratch)
	}

	b.StopTimer()
	runtime.KeepAlive(fwdBacking)
	runtime.KeepAlive(invBacking)
}

func benchmarkCodelet128(b *testing.B, entry *registry.CodeletEntry[complex128], inverse bool) {
	b.Helper()

	size := entry.Size
	src := make([]complex128, size)
	dst := make([]complex128, size)
	scratch := make([]complex128, size)

	values := codeletBenchInput(size)
	for i := range src {
		src[i] = complex(values[2*i], values[2*i+1])
	}

	twiddle := ComputeTwiddleFactors[complex128](size)
	twiddleForward, twiddleInverse, fwdBacking, invBacking := prepareCodeletTwiddles128(size, twiddle, entry)

	fn, tw := entry.Forward, twiddleForward
	if inverse {
		fn, tw = entry.Inverse, twiddleInverse
	}

	if !fn(dst, src, tw, scratch) {
		b.Skipf("codelet %s rejected size %d", entry.Signature, size)
	}

	b.ReportAllocs()
	b.SetBytes(int64(size) * 16)
	b.ResetTimer()

	for range b.N {
		fn(dst, src, tw, scratch)
	}

	b.StopTimer()
	runtime.KeepAlive(fwdBacking)
	runtime.KeepAlive(invBacking)
}
