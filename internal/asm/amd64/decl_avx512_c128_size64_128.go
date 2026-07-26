//go:build amd64 && !purego

package amd64

// AVX-512 size-specific complex128 codelets. All require only AVX512F;
// callers must gate on cpu.Features.HasAVX512. Each pair returns false for any
// length other than the size it is written for, so callers can fall back.
//
// A ZMM register holds 4 complex128, so these transforms are fully
// register-resident (16 of 32 ZMM for n=64) and touch memory only once on load
// and once on store. The twiddle argument is unused -- the twiddle factors for
// a fixed n are compile-time constants and are embedded in the assembly -- but
// its length is still validated so the codelets behave like their siblings.

//go:noescape
func ForwardAVX512Size64Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseAVX512Size64Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

// The size-128 pair splits into two 64-point radix-4 sub-transforms plus a
// radix-2 combine. 128 complex128 would need all 32 ZMM, so the idle half is
// spilled to dst (or to scratch when the transform is in place).

//go:noescape
func ForwardAVX512Size128Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseAVX512Size128Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool
