//go:build amd64 && !purego

package amd64

// AVX-512 size-4 and size-8 complex128 codelets. All require only AVX512F;
// callers gate on cpu.Features.HasAVX512, which comes from
// golang.org/x/sys/cpu.X86.HasAVX512 = CPUID leaf 7 EBX bit 16 (AVX512F) plus
// OS support, so DQ/BW/VL are NOT implied. The sign flips therefore use the
// F-only VPXORQ rather than VXORPD, which is DQ. Each function returns false
// for any length other than its own size so callers can fall back.
//
// Neither the twiddle nor the scratch slice is read: at n = 4 and n = 8 all
// twiddles are exact constants (1, +-i, +-(1+-i)/sqrt(2)) that are baked into
// the kernels, and the whole transform is register-resident (one ZMM at n = 4,
// two at n = 8), so in-place calls need no working buffer.

//go:noescape
func ForwardAVX512Size4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseAVX512Size4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardAVX512Size8Radix8Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseAVX512Size8Radix8Complex128Asm(dst, src, twiddle, scratch []complex128) bool
