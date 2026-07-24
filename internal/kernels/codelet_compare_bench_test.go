package kernels

import (
	"fmt"
	"runtime"
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/registry"
)

// BenchmarkCodeletCandidates64 benchmarks every registered complex64 codelet
// that is runnable on this CPU, grouped by size, so per-size registry
// priorities can be validated against measured performance.
func BenchmarkCodeletCandidates64(b *testing.B) {
	features := cpu.DetectFeatures()

	for _, size := range registry.Registry64.Sizes() {
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

	for _, size := range registry.Registry128.Sizes() {
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

	for i := range src {
		src[i] = complex(float32(i%7)-3, float32(i%5)-2)
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

	for i := range src {
		src[i] = complex(float64(i%7)-3, float64(i%5)-2)
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
