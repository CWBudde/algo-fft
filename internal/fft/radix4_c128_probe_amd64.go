//go:build amd64 && !purego && fftprobe

package fft

// Measurement harness for the complex128 generic AVX2 radix-4 pair. Built only
// under `-tags fftprobe`, so no ordinary build, test or benchmark sees any of
// it, and no production dispatch reaches the kernels.
//
// Why it is kept rather than deleted. On 2026-08-01 the pair was wired into
// forwardAVX2Complex128Asm/inverseAVX2Complex128Asm with the same radix-4 ->
// radix-4-mixed -> radix-2 preamble the complex64 twins use, verified against
// reference.NaiveDFT128, confirmed by an instrumented run to actually fire
// (pure radix-4 at n = 64/256/1024, mixed at 128/512/32768) -- and then lost
// every size on the i7-1255U:
//
//	         64    128    256    512     1K     2K     4K     8K
//	forward  1.08  1.12   1.56   1.54   1.30   1.24   1.32   1.16
//	inverse  1.15  1.10   1.19   0.90   1.70   2.71   2.76   2.61
//
// That is a decisive loss *on one machine*, which is not the same as a dead
// kernel. This project has already caught a complex128 result failing to
// transfer between exactly these two hosts: the Skylake-SP sweep has the
// radix-8 ladder winning complex128 at every size by 7-28% while the i7-1255U
// has it losing from 2048 up, and docs/CODELET_BENCHMARKS.md records that as
// refuting the i7-1255U byte-stride rule outright. complex128 on AVX2 is
// precisely where microarchitecture has been observed to dominate.
//
// The mechanism argued for deleting it -- a YMM holds four complex64 but only
// two complex128, so radix-4's 4-way butterfly has no width left to exploit --
// predicts a loss on any AVX2 host. It is also the same species of argument as
// the pass-count and Y-operand-census predictions that both turned out wrong
// about this very kernel, so it does not get to close the question on its own.
//
// What would close it: a sweep on the Xeon (AVX-512 host, but these are AVX2
// kernels and run there). Until then the pair stays in-tree behind this tag,
// out of every production build, so the question stays re-measurable rather
// than becoming folklore. Note the sanctioned route to that box is a commit +
// push followed by `git pull` there -- direct file transfer is blocked -- so
// deleting the files is what would actually prevent the measurement.
//
// Take the number with:
//
//	PATH=/usr/local/go/bin:$PATH taskset -c 0 go test -tags fftprobe \
//	  -run '^$' -bench 'BenchmarkC128Radix[24]' -benchtime=0.5s -count=5 ./internal/fft/
//
// Ratios are radix-4 over radix-2 within one process. A win on the Xeon earns
// a host-gated dispatch decision; another loss closes the question for good and
// the files can go.

import (
	kasm "github.com/cwbudde/algo-fft/internal/asm/amd64"
)

// forwardAVX2Complex128Radix4Probe runs the pure radix-4 kernel (power-of-four
// lengths), then the radix-4-then-2 "mixed" kernel (other powers of two), and
// reports whether either accepted. It mirrors the preamble the complex64
// dispatch uses, minus the radix-2 fallback, so a benchmark measures radix-4
// alone rather than radix-4-or-radix-2.
func forwardAVX2Complex128Radix4Probe(dst, src, twiddle, scratch []complex128) bool {
	if kasm.ForwardAVX2Complex128Radix4Asm(dst, src, twiddle, scratch, nil) {
		return true
	}

	return kasm.ForwardAVX2Complex128Radix4MixedAsm(dst, src, twiddle, scratch, nil)
}

// inverseAVX2Complex128Radix4Probe is the inverse twin.
//
// The complex128 radix-4 asm, unlike its complex64 counterpart's inv_r4_scale
// loop, does not apply the 1/n normalisation itself, so this applies it. That
// trailing pass is part of what the ratio above measures, and it is not the
// whole of the inverse penalty -- the loss survives independently of it.
func inverseAVX2Complex128Radix4Probe(dst, src, twiddle, scratch []complex128) bool {
	n := len(src)

	if kasm.InverseAVX2Complex128Radix4Asm(dst, src, twiddle, scratch, nil) {
		ScaleComplex128InPlace(dst, 1.0/float64(n))

		return true
	}

	if kasm.InverseAVX2Complex128Radix4MixedAsm(dst, src, twiddle, scratch, nil) {
		ScaleComplex128InPlace(dst, 1.0/float64(n))

		return true
	}

	return false
}

// forwardAVX2Complex128Radix2Probe and inverseAVX2Complex128Radix2Probe are the
// incumbent this is measured against: the radix-2 kernel the production
// dispatch actually uses, with the same cached permutation it gets there.
func forwardAVX2Complex128Radix2Probe(dst, src, twiddle, scratch []complex128) bool {
	return kasm.ForwardAVX2Complex128Asm(dst, src, twiddle, scratch, cachedBitReversalIndices(len(src)))
}

func inverseAVX2Complex128Radix2Probe(dst, src, twiddle, scratch []complex128) bool {
	return kasm.InverseAVX2Complex128Asm(dst, src, twiddle, scratch, cachedBitReversalIndices(len(src)))
}
