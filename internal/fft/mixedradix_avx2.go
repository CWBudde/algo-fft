//go:build amd64 && asm && !purego

package fft

import (
	"github.com/MeKo-Christian/algo-fft/internal/cpu"
	"github.com/MeKo-Christian/algo-fft/internal/kernels"
	"github.com/MeKo-Christian/algo-fft/internal/memory"
	"runtime"
)

func init() {
	// Override the recursion hooks with AVX2-aware versions.
	recursiveStep64 = mixedRadixRecursivePingPongComplex64AVX2
	recursiveStep128 = mixedRadixRecursivePingPongComplex128AVX2
}

// mixedRadixRecursivePingPongComplex64AVX2 checks for AVX2 codelets before recursing.
func mixedRadixRecursivePingPongComplex64AVX2(dst, src, work []complex64, n, stride, step int, radices []int, twiddle []complex64, inverse bool) {
	// Optimization: check if we have an AVX2 codelet for this sub-transform size.
	// We only do this for n > 1 (base cases are handled by the pure Go recursion anyway).
	if n > 1 {
		features := cpu.DetectFeatures()
		if entry := kernels.Registry64.Lookup(n, features); entry != nil && entry.SIMDLevel >= kernels.SIMDAVX2 {
			// Found an AVX2 kernel for this sub-transform!

			// 1. Prepare Input
			var inputBuf []complex64
			if stride == 1 {
				inputBuf = src[:n]
			} else {
				// Gather strided input into 'work' buffer (scratch space)
				inputBuf = work[:n]
				for i := range n {
					inputBuf[i] = src[i*stride]
				}
			}

			// 2. Prepare Twiddles
			twiddleBuf := make([]complex64, n)
			for i := range n {
				twiddleBuf[i] = twiddle[i*step]
			}

			// 3. Prepare Scratch for Kernel
			kernelScratch := make([]complex64, n)

			// 4. Call Kernel
			codeletTwiddle := twiddleBuf
			var codeletTwiddleBacking []byte
			if entry.TwiddleSize != nil && entry.PrepareTwiddle != nil {
				twiddleLen := entry.TwiddleSize(n)
				if twiddleLen > 0 {
					prepared, preparedBacking := memory.AllocAlignedComplex64(twiddleLen)
					entry.PrepareTwiddle(n, inverse, prepared)
					codeletTwiddle = prepared
					codeletTwiddleBacking = preparedBacking
				}
			}

			success := false
			if inverse {
				if entry.Inverse != nil {
					entry.Inverse(dst[:n], inputBuf, codeletTwiddle, kernelScratch)
					// Undo built-in scaling of the Inverse codelet (1/n)
					scale := complex64(complex(float32(n), 0))
					for i := range n {
						dst[i] *= scale
					}
					success = true
				}
			} else {
				if entry.Forward != nil {
					entry.Forward(dst[:n], inputBuf, codeletTwiddle, kernelScratch)
					success = true
				}
			}

			if success {
				runtime.KeepAlive(codeletTwiddleBacking)
				return
			}
		}
	}

	// Fallback to pure Go implementation.
	mixedRadixRecursivePingPongComplex64(dst, src, work, n, stride, step, radices, twiddle, inverse)
}

// mixedRadixRecursivePingPongComplex128AVX2 is the complex128 version.
func mixedRadixRecursivePingPongComplex128AVX2(dst, src, work []complex128, n, stride, step int, radices []int, twiddle []complex128, inverse bool) {
	if n > 1 {
		features := cpu.DetectFeatures()
		if entry := kernels.Registry128.Lookup(n, features); entry != nil && entry.SIMDLevel >= kernels.SIMDAVX2 {
			var inputBuf []complex128
			if stride == 1 {
				inputBuf = src[:n]
			} else {
				inputBuf = work[:n]
				for i := range n {
					inputBuf[i] = src[i*stride]
				}
			}

			twiddleBuf := make([]complex128, n)
			for i := range n {
				twiddleBuf[i] = twiddle[i*step]
			}

			kernelScratch := make([]complex128, n)

			success := false
			codeletTwiddle := twiddleBuf
			var codeletTwiddleBacking []byte
			if entry.TwiddleSize != nil && entry.PrepareTwiddle != nil {
				twiddleLen := entry.TwiddleSize(n)
				if twiddleLen > 0 {
					prepared, preparedBacking := memory.AllocAlignedComplex128(twiddleLen)
					entry.PrepareTwiddle(n, inverse, prepared)
					codeletTwiddle = prepared
					codeletTwiddleBacking = preparedBacking
				}
			}

			if inverse {
				if entry.Inverse != nil {
					entry.Inverse(dst[:n], inputBuf, codeletTwiddle, kernelScratch)
					// Undo built-in scaling of the Inverse codelet (1/n)
					scale := complex128(complex(float64(n), 0))
					for i := range n {
						dst[i] *= scale
					}
					success = true
				}
			} else {
				if entry.Forward != nil {
					entry.Forward(dst[:n], inputBuf, codeletTwiddle, kernelScratch)
					success = true
				}
			}

			if success {
				runtime.KeepAlive(codeletTwiddleBacking)
				return
			}
		}
	}

	mixedRadixRecursivePingPongComplex128(dst, src, work, n, stride, step, radices, twiddle, inverse)
}
