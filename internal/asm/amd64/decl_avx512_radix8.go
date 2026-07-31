//go:build amd64 && !purego

package amd64

// ============================================================================
// Size-generic 512-bit radix-8 DIT kernel
// ============================================================================
//
// The AVX-512 sibling of Radix8Complex64Asm/Radix8Complex128Asm. Both widths
// consume exactly the same packed twiddle table and stage-1 group index table,
// because everything that is a property of n alone lives in
// internal/kernels/radix8_generic.go and is shared by all three ladders (Go,
// AVX2, AVX-512).
//
// Why a second radix-8 kernel rather than a wider radix-4 one: the 256-bit
// radix-8 stage measured 1.24-1.56x a radix-4 stage per pass at n = 512..2048,
// all of which fits in L1, so the penalty is not a memory effect. It is the
// register budget -- eight live streams plus two rotation masks and the
// sqrt(2)/2 broadcast leave five scratch YMM of sixteen, which forces the
// twiddle planes to be re-broadcast from memory every iteration and leaves no
// room to keep a second butterfly in flight across the radix-8 dependency
// chain. Thirty-two ZMM leave twenty-one scratch instead of five, which is the
// thing being tested here.

// Radix8AVX512Complex64Asm runs a radix-8 decimation-in-time FFT of length
// n = 8^k, 2*8^k or 4*8^k (n >= 64) entirely in 512-bit registers, eight
// butterflies per instruction.
//
// twiddle must hold the n+8 packed twiddle-plane elements produced by
// prepareTwiddleRadix8Complex64, and idx the n/8 stage-1 group indices. limit
// is the largest span the radix-8 stages may cover (n, n/2 or n/4); the
// quotient n/limit selects the single radix-2 or radix-4 tail stage that
// finishes the shapes that need one. inverse selects the conjugated butterfly,
// and scale (1/n for the inverse, 1 for the forward) is folded into stage 1.
//
// The n >= 64 floor is stage 1's: it retires eight groups per iteration, so it
// needs n/8 >= 8. Below that the per-size AVX-512 codelets own the range.
//
//go:noescape
func Radix8AVX512Complex64Asm(
	dst, src, twiddle, scratch []complex64, idx []int32, limit int, inverse bool, scale float32,
) bool

// Radix8AVX512Complex128Asm is the complex128 counterpart of
// Radix8AVX512Complex64Asm: the same size-generic radix-8 DIT in 512-bit
// registers, four butterflies per instruction instead of eight.
//
// twiddle must hold the packed twiddle-plane elements produced by
// prepareTwiddleRadix8Complex128, and idx the n/8 stage-1 group indices (the
// permutation is precision-independent, so every radix-8 kernel in the tree
// shares one table). The floor is n >= 32, because a ZMM holds four complex128
// and stage 1 needs n/8 >= 4.
//
//go:noescape
func Radix8AVX512Complex128Asm(
	dst, src, twiddle, scratch []complex128, idx []int32, limit int, inverse bool, scale float64,
) bool
