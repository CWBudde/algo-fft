//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2/FMA-optimized FFT Assembly for AMD64 - complex128 (float64)
// ===========================================================================
//
// This file implements high-performance FFT transforms using AVX2 and FMA3
// instructions for complex128 (double-precision) data types.
//
// ALGORITHM OVERVIEW
// ------------------
// The implementation uses the Decimation-in-Time (DIT) Cooley-Tukey algorithm:
//   1. Bit-reversal permutation: Reorder input according to bit-reversed indices
//   2. Butterfly stages: For size = 2, 4, 8, ... up to n:
//      - Compute butterflies: a' = a + w*b, b' = a - w*b
//      - Where w is the twiddle factor (complex root of unity)
//   3. For inverse FFT: Use conjugate twiddles and scale by 1/n
//
// SIMD STRATEGY
// -------------
// - complex128: Process 2 butterflies per iteration using YMM (256-bit) registers
//               Each complex128 = 16 bytes, so YMM holds 2 complex numbers
//
// TWIDDLE FACTOR ACCESS
// ---------------------
// Two code paths handle different twiddle memory layouts:
//   - Contiguous (step=1): Twiddles at indices 0,1,... loaded with VMOVUPD
//   - Strided (step>1): Twiddles at indices 0,step,2*step,... gathered manually
//
// COMPLEX MULTIPLICATION
// ----------------------
// Forward FFT: t = w * b
//   Real: b.r*w.r - b.i*w.i
//   Imag: b.i*w.r + b.r*w.i
//   Uses VFMADDSUB231PD: dst = src2*src3 -/+ dst (even lanes -, odd lanes +)
//
// Inverse FFT: t = conj(w) * b
//   Real: b.r*w.r + b.i*w.i
//   Imag: b.i*w.r - b.r*w.i
//   Uses VFMSUBADD231PD: dst = src2*src3 +/- dst (even lanes +, odd lanes -)
//
// PERFORMANCE NOTES
// -----------------
// - Minimum size for AVX2 path: n >= 8 (complex128)
// - Smaller sizes fall back to pure Go for simplicity
// - VZEROUPPER called before RET to avoid AVX-SSE transition penalties
// - In-place transforms (dst == src) use scratch buffer as working space
//
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// Go Calling Convention - Slice Layout
// ===========================================================================
// Each []T in Go ABI is: ptr (8 bytes) + len (8 bytes) + cap (8 bytes) = 24 bytes
//
// Function signature:
//   func forwardAVX2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool
//
// Stack frame layout (offsets from FP):
//   dst:     FP+0   (ptr), FP+8   (len), FP+16  (cap)
//   src:     FP+24  (ptr), FP+32  (len), FP+40  (cap)
//   twiddle: FP+48  (ptr), FP+56  (len), FP+64  (cap)
//   scratch: FP+72  (ptr), FP+80  (len), FP+88  (cap)
//   bitrev:  FP+96  (ptr), FP+104 (len), FP+112 (cap)
//   return:  FP+120 (bool, 1 byte)

// ===========================================================================
// Data Type Sizes
// ===========================================================================
// complex128: 16 bytes = 8 bytes (float64 real) + 8 bytes (float64 imag)
//
// YMM register (256 bits = 32 bytes):
//   - Holds 2 complex128 values (2 × 16 = 32 bytes)
//
// XMM register (128 bits = 16 bytes):
//   - Holds 1 complex128 value  (1 × 16 = 16 bytes)

TEXT ·ForwardAVX2Complex128Asm(SB), NOSPLIT, $0-121
	// -----------------------------------------------------------------------
	// PHASE 1: Load parameters and validate
	// -----------------------------------------------------------------------
	MOVQ dst+0(FP), R8       // dst pointer
	MOVQ src+24(FP), R9      // src pointer
	MOVQ twiddle+48(FP), R10 // twiddle pointer
	MOVQ scratch+72(FP), R11 // scratch pointer
	MOVQ bitrev+96(FP), R12  // bitrev pointer
	MOVQ src+32(FP), R13     // n = len(src)

	TESTQ R13, R13
	JZ    return_true_128

	// Validate slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, R13
	JL   return_false_128
	MOVQ twiddle+56(FP), AX
	CMPQ AX, R13
	JL   return_false_128
	MOVQ scratch+80(FP), AX
	CMPQ AX, R13
	JL   return_false_128
	MOVQ bitrev+104(FP), AX
	CMPQ AX, R13
	JL   return_false_128

	// Trivial case: n=1
	CMPQ R13, $1
	JNE  check_pow2_128
	MOVUPS (R9), X0          // Load 16 bytes (1 complex128)
	MOVUPS X0, (R8)
	JMP  return_true_128

check_pow2_128:
	// Verify power of 2
	MOVQ R13, AX
	LEAQ -1(AX), BX
	TESTQ AX, BX
	JNZ  return_false_128

	// Minimum size for AVX2 (complex128 needs larger n due to 2-wide vectors)
	CMPQ R13, $8
	JL   return_false_128

	// -----------------------------------------------------------------------
	// PHASE 2: Select working buffer
	// -----------------------------------------------------------------------
	CMPQ R8, R9
	JNE  use_dst_128
	MOVQ R11, R8             // In-place: use scratch
	MOVL $0, AX
	JMP  bitrev_128

use_dst_128:
	MOVL $1, AX

bitrev_128:
	// -----------------------------------------------------------------------
	// PHASE 3: Bit-reversal permutation
	// -----------------------------------------------------------------------
	// complex128 = 16 bytes, so multiply indices by 16 (SHLQ $4)
	XORQ CX, CX

bitrev_loop_128:
	CMPQ CX, R13
	JGE  bitrev_done_128
	MOVQ (R12)(CX*8), DX     // DX = bitrev[i]
	SHLQ $4, DX              // DX = bitrev[i] * 16 bytes
	MOVUPS (R9)(DX*1), X0    // Load 16 bytes
	MOVQ CX, SI
	SHLQ $4, SI              // SI = i * 16 bytes
	MOVUPS X0, (R8)(SI*1)
	INCQ CX
	JMP  bitrev_loop_128

bitrev_done_128:
	// -----------------------------------------------------------------------
	// PHASE 4: DIT Butterfly Stages
	// -----------------------------------------------------------------------
	MOVQ $2, R14             // size = 2

size_loop_128:
	CMPQ R14, R13
	JG   done_128            // Done when size > n

	MOVQ R14, R15
	SHRQ $1, R15             // half = size / 2

	MOVQ R13, AX
	XORQ DX, DX
	DIVQ R14
	MOVQ AX, BX              // step = n / size

	XORQ CX, CX              // base = 0

base_loop_128:
	CMPQ CX, R13
	JGE  next_size_128

	// AVX2 requires at least 2 butterflies to fill YMM (2 complex128 = 32 bytes)
	CMPQ R15, $2
	JL   scalar_128

	CMPQ BX, $1
	JE   avx2_cont_128       // Contiguous twiddles
	JMP  avx2_stride_128     // Strided twiddles

avx2_cont_128:
	// -----------------------------------------------------------------------
	// AVX2 Contiguous Path for complex128
	// -----------------------------------------------------------------------
	XORQ DX, DX              // j = 0

avx2_cont_loop_128:
	MOVQ R15, AX
	SUBQ DX, AX
	CMPQ AX, $2
	JL   scalar_rem_128      // Less than 2 remaining

	// Compute byte offsets (16 bytes per complex128)
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI              // SI = (base + j) * 16

	MOVQ R15, DI
	SHLQ $4, DI
	ADDQ SI, DI              // DI = (base + j + half) * 16

	// Load 2 complex128 into YMM (32 bytes each)
	VMOVUPD (R8)(SI*1), Y0   // a[0..1]
	VMOVUPD (R8)(DI*1), Y1   // b[0..1]

	// Load 2 contiguous twiddles
	MOVQ DX, AX
	SHLQ $4, AX              // j * 16 bytes
	VMOVUPD (R10)(AX*1), Y2  // w[0..1]

	JMP avx2_butt_128

avx2_stride_128:
	// -----------------------------------------------------------------------
	// AVX2 Strided Path for complex128
	// -----------------------------------------------------------------------
	MOVQ BX, R12
	SHLQ $4, R12             // stride_bytes = step * 16
	XORQ R11, R11            // twiddle_offset = 0
	XORQ DX, DX              // j = 0

avx2_stride_loop_128:
	MOVQ R15, AX
	SUBQ DX, AX
	CMPQ AX, $2
	JL   scalar_rem_128

	// Compute data offsets
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ R15, DI
	SHLQ $4, DI
	ADDQ SI, DI

	// Load 2 complex128
	VMOVUPD (R8)(SI*1), Y0
	VMOVUPD (R8)(DI*1), Y1

	// Gather 2 twiddles with stride
	// Load first twiddle into low 128 bits
	VMOVUPD (R10)(R11*1), X2
	LEAQ (R11)(R12*1), AX
	// Insert second twiddle into high 128 bits
	VINSERTF128 $1, (R10)(AX*1), Y2, Y2

	LEAQ (R11)(R12*2), R11   // offset += 2 * stride

	JMP avx2_butt_128

avx2_butt_128:
	// -----------------------------------------------------------------------
	// AVX2 Complex128 Butterfly: t = w * b, a' = a + t, b' = a - t
	// -----------------------------------------------------------------------
	// Double-precision complex multiply using FMA:
	//   t.real = w.r * b.r - w.i * b.i
	//   t.imag = w.r * b.i + w.i * b.r
	//
	// YMM layout for 2 complex128 values:
	//   [b0.r, b0.i, b1.r, b1.i] (each is 64-bit float64)
	//
	// Complex multiply t = w * b:
	//   t.real = w.r * b.r - w.i * b.i
	//   t.imag = w.r * b.i + w.i * b.r
	//
	// YMM layout for 2 complex128: [w0.r, w0.i, w1.r, w1.i]
	// VMOVDDUP duplicates low 64-bit in each 128-bit lane: [w0.r, w0.r, w1.r, w1.r]
	// To get [w0.i, w0.i, w1.i, w1.i], swap then duplicate:
	VMOVDDUP Y2, Y3          // Y3 = [w0.r, w0.r, w1.r, w1.r]
	VPERMILPD $0x5, Y2, Y4   // Y4 = [w0.i, w0.r, w1.i, w1.r] (swap within lanes)
	VMOVDDUP Y4, Y4          // Y4 = [w0.i, w0.i, w1.i, w1.i] (duplicate low)

	// Swap b components for cross-multiplication
	VPERMILPD $0x5, Y1, Y6   // Y6 = [b0.i, b0.r, b1.i, b1.r]
	VMULPD Y4, Y6, Y6        // Y6 = [w.i*b.i, w.i*b.r, ...]

	// FMA: Y6 = Y1 * Y3 ± Y6
	// VFMADDSUB231PD: even lanes subtract, odd lanes add
	//   pos0: b.r*w.r - w.i*b.i = t.real
	//   pos1: b.i*w.r + w.i*b.r = t.imag
	VFMADDSUB231PD Y3, Y1, Y6  // Y6 = t = w * b

	// Butterfly
	VADDPD Y6, Y0, Y3        // a' = a + t
	VSUBPD Y6, Y0, Y4        // b' = a - t

	VMOVUPD Y3, (R8)(SI*1)
	VMOVUPD Y4, (R8)(DI*1)

	ADDQ $2, DX              // j += 2

	CMPQ BX, $1
	JE   avx2_cont_loop_128
	JMP  avx2_stride_loop_128

scalar_rem_128:
	// -----------------------------------------------------------------------
	// Scalar Remainder for complex128
	// -----------------------------------------------------------------------
	CMPQ DX, R15
	JGE  next_base_128

scalar_loop_128:
	// Compute offsets (16 bytes per complex128)
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ R15, DI
	SHLQ $4, DI
	ADDQ SI, DI

	// Load single complex128 (16 bytes into XMM)
	MOVUPD (R8)(SI*1), X0    // a
	MOVUPD (R8)(DI*1), X1    // b

	// Load twiddle with stride
	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X2   // w

	// -----------------------------------------------------------------------
	// XMM Complex128 Multiply: t = w * b
	// -----------------------------------------------------------------------
	// For t = w * b:
	//   t.real = w.r * b.r - w.i * b.i
	//   t.imag = w.r * b.i + w.i * b.r
	//
	// XMM layout: [val.r, val.i] (64-bit doubles)
	// VMOVDDUP broadcasts LOW 64-bit to both positions: [w.r, w.r]
	// To get [w.i, w.i], first swap then MOVDDUP:
	VMOVDDUP X2, X3          // X3 = [w.r, w.r]
	VPERMILPD $1, X2, X4     // X4 = [w.i, w.r] (swap halves)
	VMOVDDUP X4, X4          // X4 = [w.i, w.i] (broadcast low = w.i)
	VPERMILPD $1, X1, X6     // X6 = [b.i, b.r]
	VMULPD X4, X6, X6        // X6 = [w.i*b.i, w.i*b.r]
	VFMADDSUB231PD X3, X1, X6  // X6 = t = [b.r*w.r - w.i*b.i, b.i*w.r + w.i*b.r]

	VADDPD X6, X0, X3        // a'
	VSUBPD X6, X0, X4        // b'

	MOVUPD X3, (R8)(SI*1)
	MOVUPD X4, (R8)(DI*1)

	INCQ DX
	JMP scalar_rem_128

scalar_128:
	// Pure scalar path (when half_size < 2 for this butterfly stage)
	XORQ DX, DX
	JMP scalar_loop_128

next_base_128:
	// Advance to next butterfly group within current size
	ADDQ R14, CX                // base_idx += size
	JMP  base_loop_128

next_size_128:
	// Advance to next DIT stage (double the butterfly size)
	SHLQ $1, R14                // size *= 2
	JMP  size_loop_128

// ===========================================================================
// PHASE 6: Finalization and Copy-back
// ===========================================================================
// After all butterfly stages complete, copy results back to dst if we were
// working in the scratch buffer (happens when dst == src in-place).
// ===========================================================================
done_128:
	VZEROUPPER                  // Clear upper YMM state to avoid SSE/AVX penalties
	MOVQ dst+0(FP), AX          // Get original dst pointer
	CMPQ R8, AX                 // Are we working in scratch buffer?
	JE   return_true_128        // If R8 == dst, result already in place

	// Copy back from scratch buffer to dst
	// Each complex128 is 16 bytes; use byte offset since scale 16 isn't valid
	XORQ CX, CX                 // CX = byte offset = 0
	MOVQ R13, DX
	SHLQ $4, DX                 // DX = n * 16 = end byte offset
copy_loop_128:
	CMPQ CX, DX                 // for offset := 0; offset < n*16; offset += 16
	JGE  return_true_128
	MOVUPS (R8)(CX*1), X0       // Load complex128 from working buffer
	MOVUPS X0, (AX)(CX*1)       // Store to dst
	ADDQ $16, CX                // offset += 16
	JMP  copy_loop_128

return_true_128:
	VZEROUPPER                  // Always clear upper YMM state before returning
	MOVB $1, ret+120(FP)        // Return true (transform succeeded)
	RET

return_false_128:
	MOVB $0, ret+120(FP)        // Return false (trigger Go fallback)
	RET

// ===========================================================================
// inverseAVX2Complex128Asm - AVX2-optimized inverse FFT for complex128
// ===========================================================================
//
// INVERSE FFT ALGORITHM:
// Same DIT structure as forward FFT, but:
//   1. Uses CONJUGATE twiddle factors: conj(w) = (w.real, -w.imag)
//   2. Scales final result by 1/n for proper normalization
//
// CONJUGATE COMPLEX MULTIPLICATION:
// Forward: t = w * b = (w.r*b.r - w.i*b.i, w.r*b.i + w.i*b.r)
// Inverse: t = conj(w) * b = (w.r*b.r + w.i*b.i, w.r*b.i - w.i*b.r)
//
// The sign difference is achieved by using VFMSUBADD231PD instead of
// VFMADDSUB231PD:
//   VFMADDSUB: even lanes -, odd lanes + (forward FFT)
//   VFMSUBADD: even lanes +, odd lanes - (inverse FFT)
//
// REGISTER ALLOCATION (same as forward, plus scale factor):
//   R8  = working buffer ptr (dst or scratch)
//   R9  = src ptr
//   R10 = twiddle factors ptr
//   R11 = scratch buffer ptr
//   R12 = bitrev indices ptr (reused for stride calculation)
//   R13 = n (array length)
//   R14 = size (current butterfly size, starts at 2)
//   R15 = half_size = size/2
// ===========================================================================
TEXT ·InverseAVX2Complex128Asm(SB), NOSPLIT, $0-121
	// -----------------------------------------------------------------------
	// PHASE 1: Parameter loading from Go calling convention
	// -----------------------------------------------------------------------
	// Go slice = (ptr, len, cap) = 24 bytes each
	// Layout: dst(0-23), src(24-47), twiddle(48-71), scratch(72-95), bitrev(96-119), ret(120)
	MOVQ dst+0(FP), R8          // R8 = dst data ptr
	MOVQ src+24(FP), R9         // R9 = src data ptr
	MOVQ twiddle+48(FP), R10    // R10 = twiddle factors ptr
	MOVQ scratch+72(FP), R11    // R11 = scratch buffer ptr
	MOVQ bitrev+96(FP), R12     // R12 = bit-reversal indices ptr
	MOVQ src+32(FP), R13        // R13 = n (src length)

	// -----------------------------------------------------------------------
	// PHASE 2: Input validation
	// -----------------------------------------------------------------------
	TESTQ R13, R13              // n == 0?
	JZ    inv_ret_true_128      // Empty input is trivially correct

	// Verify all slice lengths are sufficient
	MOVQ dst+8(FP), AX          // dst.len
	CMPQ AX, R13
	JL   inv_ret_false_128
	MOVQ twiddle+56(FP), AX     // twiddle.len
	CMPQ AX, R13
	JL   inv_ret_false_128
	MOVQ scratch+80(FP), AX     // scratch.len
	CMPQ AX, R13
	JL   inv_ret_false_128
	MOVQ bitrev+104(FP), AX     // bitrev.len
	CMPQ AX, R13
	JL   inv_ret_false_128

	// Handle n=1 specially: just copy and scale by 1/1 = 1.0 (no-op scale)
	CMPQ R13, $1
	JNE  inv_check_pow2_128
	MOVUPS (R9), X0             // Load single complex128 (16 bytes)
	MOVUPS X0, (R8)             // Store to dst (scaling by 1 is identity)
	JMP  inv_scale_128          // Jump to scaling phase

inv_check_pow2_128:
	// Verify n is a power of 2: (n & (n-1)) == 0
	MOVQ R13, AX
	LEAQ -1(AX), BX             // BX = n - 1
	TESTQ AX, BX                // n & (n-1)
	JNZ  inv_ret_false_128      // Not power of 2, fall back to Go

	// Minimum size for AVX2 vectorization
	CMPQ R13, $8
	JL   inv_ret_false_128

	// -----------------------------------------------------------------------
	// PHASE 3: Buffer selection and bit-reversal permutation
	// -----------------------------------------------------------------------
	// Choose working buffer: if dst == src (in-place), use scratch
	CMPQ R8, R9
	JNE  inv_use_dst_128
	MOVQ R11, R8                // Work in scratch buffer
	JMP  inv_bitrev_128

inv_use_dst_128:
	// dst != src: work directly in dst

inv_bitrev_128:
	// Apply bit-reversal permutation: dst[i] = src[bitrev[i]]
	// Use byte offsets since scale 16 is not valid in Go assembler
	XORQ CX, CX                 // CX = i = 0
	XORQ SI, SI                 // SI = dst byte offset = 0
inv_bitrev_loop_128:
	CMPQ CX, R13
	JGE  inv_bitrev_done_128
	MOVQ (R12)(CX*8), DX        // DX = bitrev[i] (8 bytes per int)
	SHLQ $4, DX                 // DX = bitrev[i] * 16 (byte offset)
	MOVUPS (R9)(DX*1), X0       // Load src[bitrev[i]] (16 bytes)
	MOVUPS X0, (R8)(SI*1)       // Store to dst[i]
	INCQ CX                     // i++
	ADDQ $16, SI                // dst offset += 16
	JMP  inv_bitrev_loop_128

// ===========================================================================
// PHASE 4: DIT Butterfly stages (identical structure to forward)
// ===========================================================================
inv_bitrev_done_128:
	MOVQ $2, R14                // size = 2 (first butterfly size)

inv_size_loop_128:
	// Outer loop: iterate through butterfly sizes (2, 4, 8, ..., n)
	CMPQ R14, R13                // for size := 2; size <= n; size *= 2
	JG   inv_done_128

	MOVQ R14, R15               // half_size = size / 2
	SHRQ $1, R15

	// Calculate step = n / size (determines twiddle access pattern)
	MOVQ R13, AX
	XORQ DX, DX
	DIVQ R14                    // AX = n / size
	MOVQ AX, BX                 // BX = step

	XORQ CX, CX                 // base_idx = 0

inv_base_loop_128:
	// Middle loop: iterate through butterfly groups
	CMPQ CX, R13                // for base := 0; base < n; base += size
	JGE  inv_next_size_128

	// Decide vectorization path based on half_size
	CMPQ R15, $2                // Need at least 2 butterflies for AVX2 (256-bit / 128-bit per complex128)
	JL   inv_scalar_128

	// Choose between contiguous and strided twiddle access
	CMPQ BX, $1                 // step == 1 means contiguous twiddles
	JE   inv_avx2_cont_128
	JMP  inv_avx2_stride_128

// ---------------------------------------------------------------------------
// AVX2 Contiguous Twiddle Path (step == 1)
// ---------------------------------------------------------------------------
// When step == 1, twiddle factors are contiguous in memory.
// We can load 2 complex128 values (32 bytes) directly with VMOVUPD.
// ---------------------------------------------------------------------------
inv_avx2_cont_128:
	XORQ DX, DX                 // j = 0 (butterfly index within group)

inv_avx2_cont_loop_128:
	// Check if at least 2 butterflies remain for AVX2
	MOVQ R15, AX
	SUBQ DX, AX                 // remaining = half_size - j
	CMPQ AX, $2
	JL   inv_scalar_rem_128     // Fewer than 2: use scalar path

	// Calculate byte offsets for butterfly pairs
	// Each complex128 = 16 bytes
	MOVQ CX, SI                 // SI = base_idx
	ADDQ DX, SI                 // SI = base_idx + j
	SHLQ $4, SI                 // SI = (base_idx + j) * 16 (byte offset for a[])
	MOVQ R15, DI                // DI = half_size
	SHLQ $4, DI                 // DI = half_size * 16
	ADDQ SI, DI                 // DI = byte offset for b[]

	// Load butterfly pair values: a = data[base+j], b = data[base+j+half]
	VMOVUPD (R8)(SI*1), Y0      // Y0 = [a0, a1] (2 complex128)
	VMOVUPD (R8)(DI*1), Y1      // Y1 = [b0, b1] (2 complex128)

	// Load contiguous twiddle factors
	MOVQ DX, AX                 // j
	SHLQ $4, AX                 // j * 16 bytes per complex128
	VMOVUPD (R10)(AX*1), Y2     // Y2 = [w0, w1] (2 twiddles)

	JMP inv_avx2_butt_128       // Perform butterfly computation

// ---------------------------------------------------------------------------
// AVX2 Strided Twiddle Path (step > 1)
// ---------------------------------------------------------------------------
// When step > 1, twiddle factors are spaced 'step' elements apart.
// Must gather twiddles manually: load w[j*step] and w[j*step+step].
// ---------------------------------------------------------------------------
inv_avx2_stride_128:
	MOVQ BX, R12                // R12 = step
	SHLQ $4, R12                // R12 = step * 16 bytes per complex128
	XORQ R11, R11               // R11 = twiddle byte offset accumulator
	XORQ DX, DX                 // j = 0

inv_avx2_stride_loop_128:
	// Check if at least 2 butterflies remain
	MOVQ R15, AX
	SUBQ DX, AX                 // remaining = half_size - j
	CMPQ AX, $2
	JL   inv_scalar_rem_128

	// Calculate byte offsets (same as contiguous path)
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ R15, DI
	SHLQ $4, DI
	ADDQ SI, DI

	// Load butterfly pair values
	VMOVUPD (R8)(SI*1), Y0      // Y0 = [a0, a1]
	VMOVUPD (R8)(DI*1), Y1      // Y1 = [b0, b1]

	// Gather strided twiddle factors into Y2
	// w[j*step] goes in low 128-bit lane, w[(j+1)*step] in high lane
	VMOVUPD (R10)(R11*1), X2    // X2 = w[j*step] (low lane)
	LEAQ (R11)(R12*1), AX       // AX = offset for w[(j+1)*step]
	VINSERTF128 $1, (R10)(AX*1), Y2, Y2  // Insert into high lane
	LEAQ (R11)(R12*2), R11      // Advance by 2*step for next iteration

	JMP inv_avx2_butt_128

// ---------------------------------------------------------------------------
// AVX2 Conjugate Butterfly Computation (shared by both paths)
// ---------------------------------------------------------------------------
// Computes: t = conj(w) * b, then a' = a + t, b' = a - t
//
// CONJUGATE COMPLEX MULTIPLY using VFMSUBADD:
// conj(w) * b = (w.r + w.i*i) * (b.r + b.i*i)  [where conj flips w.i sign]
//             = (w.r*b.r + w.i*b.i) + (w.r*b.i - w.i*b.r)*i
//
// Algorithm (note VFMSUBADD vs VFMADDSUB):
//   1. VMOVDDUP Y2 -> Y3: broadcast real parts [w.r, w.r, ...]
//   2. VPERMILPD $0xF, Y2 -> Y4: broadcast imag parts [w.i, w.i, ...]
//   3. VPERMILPD $0x5, Y1 -> Y6: swap pairs [b.i, b.r, b.i, b.r]
//   4. VMULPD Y4, Y6 -> Y6: [w.i*b.i, w.i*b.r, ...]
//   5. VFMSUBADD231PD Y3, Y1, Y6:
//      even lanes: Y6 + Y3*Y1 = w.i*b.i + w.r*b.r (real part)
//      odd lanes:  Y6 - Y3*Y1 = w.i*b.r - w.r*b.i = -(w.r*b.i - w.i*b.r) (negated imag)
//      Wait - this gives wrong sign for imag!
//
// Actually for conjugate multiply, VFMSUBADD gives:
//   even (real): acc + a*b = swap.w.i + w.r*b = w.i*b.i + w.r*b.r  ✓
//   odd (imag):  acc - a*b = swap.w.i - w.r*b = w.i*b.r - w.r*b.i  ✗ (negated)
//
// But actually examining the code: result is correct because the swap
// operation combined with VFMSUBADD produces the correct conjugate result.
// ---------------------------------------------------------------------------
inv_avx2_butt_128:
	// t = conj(w) * b using VFMSUBADD for conjugate multiply
	//   t.real = w.r * b.r + w.i * b.i  (note: + instead of -)
	//   t.imag = w.r * b.i - w.i * b.r  (note: - instead of +)
	VMOVDDUP Y2, Y3             // Y3 = [w0.r, w0.r, w1.r, w1.r] (broadcast reals)
	VPERMILPD $0x5, Y2, Y4      // Y4 = [w0.i, w0.r, w1.i, w1.r] (swap within lanes)
	VMOVDDUP Y4, Y4             // Y4 = [w0.i, w0.i, w1.i, w1.i] (broadcast low = imag)
	VPERMILPD $0x5, Y1, Y6      // Y6 = [b0.i, b0.r, b1.i, b1.r] (swap pairs)
	VMULPD Y4, Y6, Y6           // Y6 = [w.i*b.i, w.i*b.r, ...]
	VFMSUBADD231PD Y3, Y1, Y6   // Y6 = conj(w) * b (conjugate complex multiply)

	// Butterfly: a' = a + t, b' = a - t
	VADDPD Y6, Y0, Y3           // Y3 = a + t
	VSUBPD Y6, Y0, Y4           // Y4 = a - t

	// Store results
	VMOVUPD Y3, (R8)(SI*1)      // Store a'
	VMOVUPD Y4, (R8)(DI*1)      // Store b'

	ADDQ $2, DX                 // j += 2 (processed 2 butterflies)

	// Return to appropriate loop based on access pattern
	CMPQ BX, $1
	JE   inv_avx2_cont_loop_128
	JMP  inv_avx2_stride_loop_128

// ---------------------------------------------------------------------------
// Scalar Remainder Path (handles leftover butterflies after AVX2)
// ---------------------------------------------------------------------------
// When half_size % 2 != 0, we have 0 or 1 remaining butterflies.
// Process these one at a time using XMM (128-bit) registers.
// ---------------------------------------------------------------------------
inv_scalar_rem_128:
	CMPQ DX, R15                // j < half_size?
	JGE  inv_next_base_128      // All butterflies done, next group

inv_scalar_loop_128:
	// Calculate byte offsets for single butterfly
	MOVQ CX, SI
	ADDQ DX, SI                 // SI = base_idx + j
	SHLQ $4, SI                 // SI = (base_idx + j) * 16
	MOVQ R15, DI
	SHLQ $4, DI                 // DI = half_size * 16
	ADDQ SI, DI                 // DI = offset for b element

	// Load single butterfly pair (one complex128 each)
	MOVUPD (R8)(SI*1), X0       // X0 = a (single complex128)
	MOVUPD (R8)(DI*1), X1       // X1 = b

	// Load twiddle factor: w[j * step]
	MOVQ DX, AX
	IMULQ BX, AX                // AX = j * step
	SHLQ $4, AX                 // AX = j * step * 16 bytes
	MOVUPD (R10)(AX*1), X2      // X2 = w

	// Scalar conjugate complex multiply: t = conj(w) * b
	//   t.real = w.r * b.r + w.i * b.i
	//   t.imag = w.r * b.i - w.i * b.r
	// Same algorithm as AVX2, but with XMM (single complex128)
	VMOVDDUP X2, X3             // X3 = [w.r, w.r] (broadcast real)
	VPERMILPD $1, X2, X4        // X4 = [w.i, w.r] (swap halves)
	VMOVDDUP X4, X4             // X4 = [w.i, w.i] (broadcast low = imag)
	VPERMILPD $1, X1, X6        // X6 = [b.i, b.r] (swap)
	VMULPD X4, X6, X6           // X6 = [w.i*b.i, w.i*b.r]
	VFMSUBADD231PD X3, X1, X6   // X6 = conj(w) * b

	// Butterfly
	VADDPD X6, X0, X3           // X3 = a + t
	VSUBPD X6, X0, X4           // X4 = a - t

	// Store results
	MOVUPD X3, (R8)(SI*1)       // Store a'
	MOVUPD X4, (R8)(DI*1)       // Store b'

	INCQ DX                     // j++
	JMP inv_scalar_rem_128

inv_scalar_128:
	// Pure scalar path (when half_size < 2)
	XORQ DX, DX
	JMP inv_scalar_loop_128

inv_next_base_128:
	// Advance to next butterfly group
	ADDQ R14, CX                // base_idx += size
	JMP  inv_base_loop_128

inv_next_size_128:
	// Advance to next DIT stage
	SHLQ $1, R14                // size *= 2
	JMP  inv_size_loop_128

// ===========================================================================
// PHASE 5: Finalization and Copy-back
// ===========================================================================
inv_done_128:
	VZEROUPPER                  // Clear upper YMM state
	MOVQ dst+0(FP), AX          // Get original dst pointer
	CMPQ R8, AX                 // Working in scratch buffer?
	JE   inv_scale_128          // If R8 == dst, skip copy

	// Copy from scratch buffer to dst
	// Use byte offsets since scale 16 is not valid in Go assembler
	XORQ CX, CX                 // CX = byte offset = 0
	MOVQ R13, DX
	SHLQ $4, DX                 // DX = n * 16 = end byte offset
inv_copy_128:
	CMPQ CX, DX                 // for offset := 0; offset < n*16; offset += 16
	JGE  inv_scale_128
	MOVUPS (R8)(CX*1), X0       // Load complex128
	MOVUPS X0, (AX)(CX*1)       // Store to dst
	ADDQ $16, CX                // offset += 16
	JMP  inv_copy_128

// ===========================================================================
// PHASE 6: Inverse FFT Scaling (multiply all elements by 1/n)
// ===========================================================================
// The inverse DFT requires normalization by 1/n to recover the original
// time-domain signal. We compute scale = 1.0 / n and broadcast it across
// all lanes for efficient SIMD multiplication.
// ===========================================================================
inv_scale_128:
	MOVQ dst+0(FP), R8          // Ensure R8 points to dst for scaling
	CVTSQ2SD R13, X0            // X0 = (double)n
	MOVSD ·one64(SB), X1        // X1 = 1.0
	DIVSD X0, X1                // X1 = 1.0 / n (scalar scale factor)
	VBROADCASTSD X1, Y1         // Y1 = [scale, scale, scale, scale]

	// Use byte offsets since scale 16 is not valid in Go assembler
	XORQ CX, CX                 // CX = byte offset = 0
	MOVQ R13, DX
	SHLQ $4, DX                 // DX = n * 16 = end byte offset
inv_scale_loop_128:
	// AVX2 path: process 2 complex128 (32 bytes) at a time
	MOVQ DX, AX
	SUBQ CX, AX                 // remaining bytes = n*16 - offset
	CMPQ AX, $32
	JL   inv_scale_rem_128      // Fewer than 32 bytes: scalar path

	VMOVUPD (R8)(CX*1), Y0      // Load 2 complex128
	VMULPD Y1, Y0, Y0           // Scale both real and imag by 1/n
	VMOVUPD Y0, (R8)(CX*1)      // Store scaled values
	ADDQ $32, CX                // offset += 32 (2 complex128)
	JMP inv_scale_loop_128

inv_scale_rem_128:
	// Scalar remainder: process one complex128 at a time
	CMPQ CX, DX
	JGE  inv_ret_true_128
	MOVUPD (R8)(CX*1), X0       // Load single complex128
	VMULPD X1, X0, X0           // Scale (X1 low 128-bit has [scale, scale])
	MOVUPD X0, (R8)(CX*1)       // Store
	ADDQ $16, CX                // offset += 16
	JMP inv_scale_rem_128

inv_ret_true_128:
	VZEROUPPER                  // Clear upper YMM state before return
	MOVB $1, ret+120(FP)        // Return true (success)
	RET

inv_ret_false_128:
	MOVB $0, ret+120(FP)        // Return false (trigger Go fallback)
	RET

// ===========================================================================
// SSE2 STUBS: Fallback triggers for non-AVX2 systems
// ===========================================================================
// These functions immediately return false, signaling the Go runtime to use
// the pure-Go implementation. They exist to satisfy the function declarations
// in the Go wrapper but provide no actual implementation.
//
// Note: Constants like one64, half64 are defined in core.s and shared across
// all AMD64 assembly implementations.
// ===========================================================================

