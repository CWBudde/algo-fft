//go:build amd64 && !purego

package fft

import (
	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/kernels"
	m "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/planner"
)

// AVX-512 dispatch: the generic AVX-512 radix-2 DIT kernel slots between the
// tuned AVX2 size-specific codelets (which still win at the sizes they cover)
// and the generic AVX2 kernels (which it beats at every measured size).
//
// This tier serves plans without a registry codelet binding; at sizes
// 1024/4096/8192/16384 the same kernel is additionally registered as a
// complex64 codelet, so those plans bind it directly
// (internal/kernels/dit_avx512_amd64.go).
//
// Measured on Xeon 2.8 GHz (Skylake-SP class), forward complex64:
//
//	size    AVX-512   AVX2 generic  AVX2 Stockham  best AVX2 codelet
//	1024    7.8 µs    10.4 µs       -              (none in DIT switch)
//	2048    16.2 µs   -             21.7 µs        13.7 µs (radix-4+2)
//	4096    33.6 µs   46.4 µs       48.3 µs        36.1 µs (radix-4)
//	16384   180 µs    219 µs        227 µs         (none)
//	65536   0.91 ms   -             1.11 ms        (none)
//	2^21    120 ms    -             166 ms         (none)
//
// complex128 (no AVX2 Stockham asm exists; Go Stockham shown):
//
//	size    AVX-512   AVX2 generic  Go Stockham
//	1024    9.4 µs    12.4 µs       -
//	4096    46.7 µs   59.4 µs       115 µs
//	16384   233 µs    269 µs        562 µs
//	2^21    232 ms    -             331 ms
//
// Sizes the auto heuristic resolves to fftypes.KernelStockham also run the AVX-512
// DIT kernel: it computes the identical transform faster at every measured
// size but one (complex64 n=2^19, ~5% slower — outweighed by 15-40% wins
// elsewhere). An explicitly forced fftypes.KernelStockham stays on the Stockham
// path: PlanOptions.Strategy documents force semantics, and the measurement
// planner relies on forced strategies timing the kernels they name. Plans
// keep the auto/forced distinction visible here by passing fftypes.KernelAuto for
// heuristic choices (see kernelSelectionStrategy in plan.go).

// avx2SizeSpecificDITComplex64Covers reports whether the complex64 DIT switch
// in avx2SizeSpecificOrGenericDITComplex64 has a size-specific codelet for n.
// Keep in sync with that switch (kernels_amd64_size_specific.go).
func avx2SizeSpecificDITComplex64Covers(n int) bool {
	switch n {
	case 8, 16, 32, 64, 128, 256, 512, 2048, 8192:
		return true
	default:
		return false
	}
}

// avx2SizeSpecificDITComplex128Covers reports whether the complex128 DIT
// switch in avx2SizeSpecificOrGenericDITComplex128 has a size-specific codelet
// for n. Keep in sync with that switch (kernels_amd64_size_specific.go).
func avx2SizeSpecificDITComplex128Covers(n int) bool {
	switch n {
	case 4, 8, 16, 32, 64, 512:
		return true
	default:
		return false
	}
}

// avx512FirstKernel chains the AVX-512 generic kernel in front of the AVX2
// dispatch chain, except where the AVX2 chain is known to win or is
// explicitly requested:
//   - DIT-resolved sizes covered by a tuned AVX2 codelet (see table above)
//   - an explicitly forced fftypes.KernelStockham (algorithm choice is honored)
//
// The AVX-512 kernel declines n < 16, so those sizes fall through to the
// AVX2 chain as before.
func avx512FirstKernel[T Complex](
	strategy fftypes.KernelStrategy, avx512, avx2 kernels.Kernel[T], coveredByAVX2 func(int) bool,
) kernels.Kernel[T] {
	return func(dst, src, twiddle, scratch []T) bool {
		n := len(src)
		if !m.IsPowerOf2(n) {
			return false
		}

		switch planner.ResolveKernelStrategyWithDefault(n, strategy) {
		case fftypes.KernelDIT:
			if coveredByAVX2(n) {
				return avx2(dst, src, twiddle, scratch)
			}
		case fftypes.KernelStockham:
			if strategy == fftypes.KernelStockham {
				// Explicitly forced Stockham: honor the algorithm choice.
				// Auto plans reach here with strategy == fftypes.KernelAuto (see
				// kernelSelectionStrategy in plan.go) and get the faster
				// AVX-512 DIT substitution below.
				return avx2(dst, src, twiddle, scratch)
			}
		default:
			return false
		}

		if avx512(dst, src, twiddle, scratch) {
			return true
		}

		return avx2(dst, src, twiddle, scratch)
	}
}

// avx512SizeSpecificOrGenericComplex64 returns the complex64 kernel pair for
// AVX-512 hosts: AVX2 codelets where they win, AVX-512 generic elsewhere,
// AVX2 generic as the safety net.
func avx512SizeSpecificOrGenericComplex64(strategy fftypes.KernelStrategy) kernels.Kernels[complex64] {
	avx2 := avx2SizeSpecificOrGenericComplex64(strategy)

	return kernels.Kernels[complex64]{
		Forward: avx512FirstKernel(strategy, forwardAVX512Complex64, avx2.Forward, avx2SizeSpecificDITComplex64Covers),
		Inverse: avx512FirstKernel(strategy, inverseAVX512Complex64, avx2.Inverse, avx2SizeSpecificDITComplex64Covers),
	}
}

// avx512SizeSpecificOrGenericComplex128 is the complex128 analogue of
// avx512SizeSpecificOrGenericComplex64.
func avx512SizeSpecificOrGenericComplex128(strategy fftypes.KernelStrategy) kernels.Kernels[complex128] {
	avx2 := avx2SizeSpecificOrGenericComplex128(strategy)

	return kernels.Kernels[complex128]{
		Forward: avx512FirstKernel(strategy, forwardAVX512Complex128, avx2.Forward, avx2SizeSpecificDITComplex128Covers),
		Inverse: avx512FirstKernel(strategy, inverseAVX512Complex128, avx2.Inverse, avx2SizeSpecificDITComplex128Covers),
	}
}
