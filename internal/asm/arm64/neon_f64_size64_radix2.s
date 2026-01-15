//go:build arm64 && asm && !purego

// ===========================================================================
// NEON Size-64 Radix-2 FFT Kernels for ARM64 (complex128)
// ===========================================================================
//
// Size 64 = 2^6, radix-2 DIT algorithm using 6 stages.
// Each complex128 element is 16 bytes (real f64 + imag f64).
//
// Register allocation:
//   R8  = working dst pointer (may point to scratch if src==dst)
//   R9  = src pointer
//   R10 = twiddle pointer
//   R11 = scratch pointer
//   R12 = bit-reversal table pointer
//   R13 = n (size, should be 64)
//   R14 = current butterfly size
//   R15 = half = size/2
//   R16 = step = n/size (twiddle stride)
//   R17 = base index for current group
//   R20 = original dst pointer (preserved)
//
// ===========================================================================

#include "textflag.h"

DATA ·neonInv64F64+0(SB)/8, $0x3f90000000000000 // 1/64 = 0.015625
GLOBL ·neonInv64F64(SB), RODATA, $8

// Forward transform, size 64, complex128, radix-2
// func ForwardNEONSize64Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool
TEXT ·ForwardNEONSize64Radix2Complex128Asm(SB), NOSPLIT, $0-97
	// Load slice headers from stack
	MOVD dst+0(FP), R8          // R8 = dst.data
	MOVD src+24(FP), R9         // R9 = src.data
	MOVD twiddle+48(FP), R10    // R10 = twiddle.data
	MOVD scratch+72(FP), R11    // R11 = scratch.data
	MOVD src+32(FP), R13        // R13 = src.len (n)

	// Validate n == 64
	CMP  $64, R13               // n must be exactly 64
	BNE  neon64r2f64_return_false

	// Validate dst capacity >= 64
	MOVD dst+8(FP), R0          // R0 = dst.len
	CMP  $64, R0
	BLT  neon64r2f64_return_false

	// Validate twiddle capacity >= 64
	MOVD twiddle+56(FP), R0     // R0 = twiddle.len
	CMP  $64, R0
	BLT  neon64r2f64_return_false

	// Validate scratch capacity >= 64
	MOVD scratch+80(FP), R0     // R0 = scratch.len
	CMP  $64, R0
	BLT  neon64r2f64_return_false

	// Load bit-reversal table address
	MOVD $bitrev_size64_radix2_f64<>(SB), R12

	// Save original dst pointer
	MOVD R8, R20                // R20 = original dst

	// If src == dst, use scratch as working buffer to avoid aliasing
	CMP  R8, R9                 // compare dst and src pointers
	BNE  neon64r2f64_use_dst    // if different, use dst directly
	MOVD R11, R8                // src==dst: use scratch as working buffer

neon64r2f64_use_dst:
	// =========================================================================
	// Bit-reversal permutation: work[i] = src[bitrev[i]]
	// =========================================================================
	MOVD $0, R0                 // R0 = loop counter i

neon64r2f64_bitrev_loop:
	CMP  $64, R0                // while i < 64
	BGE  neon64r2f64_stage

	LSL  $3, R0, R1             // R1 = i * 8 (offset into bitrev table, 8 bytes per entry)
	ADD  R12, R1, R1            // R1 = &bitrev[i]
	MOVD (R1), R2               // R2 = bitrev[i] (source index)

	LSL  $4, R2, R3             // R3 = bitrev[i] * 16 (complex128 is 16 bytes)
	ADD  R9, R3, R3             // R3 = &src[bitrev[i]]
	MOVD (R3), R4               // R4 = src[bitrev[i]].real (as bits)
	MOVD 8(R3), R5              // R5 = src[bitrev[i]].imag (as bits)

	LSL  $4, R0, R3             // R3 = i * 16
	ADD  R8, R3, R3             // R3 = &work[i]
	MOVD R4, (R3)               // work[i].real = src[bitrev[i]].real
	MOVD R5, 8(R3)              // work[i].imag = src[bitrev[i]].imag

	ADD  $1, R0, R0             // i++
	B    neon64r2f64_bitrev_loop

neon64r2f64_stage:
	// =========================================================================
	// Cooley-Tukey radix-2 DIT butterfly stages
	// For size 64: 6 stages with sizes 2, 4, 8, 16, 32, 64
	// =========================================================================
	MOVD $2, R14                // R14 = butterfly size, starts at 2

neon64r2f64_size_loop:
	CMP  $64, R14               // while size <= 64
	BGT  neon64r2f64_done

	LSR  $1, R14, R15           // R15 = half = size / 2
	UDIV R14, R13, R16          // R16 = step = n / size (twiddle stride)

	MOVD $0, R17                // R17 = base (group start index)

neon64r2f64_base_loop:
	CMP  R13, R17               // while base < n
	BGE  neon64r2f64_next_size

	MOVD $0, R0                 // R0 = j (butterfly index within group)

neon64r2f64_inner_loop:
	CMP  R15, R0                // while j < half
	BGE  neon64r2f64_next_base

	ADD  R17, R0, R1            // R1 = idx_a = base + j
	ADD  R1, R15, R2            // R2 = idx_b = base + j + half

	// Load twiddle factor w = twiddle[j * step]
	MUL  R0, R16, R3            // R3 = j * step
	LSL  $4, R3, R3             // R3 = (j * step) * 16 bytes
	ADD  R10, R3, R3            // R3 = &twiddle[j * step]
	FMOVD 0(R3), F0             // F0 = w.real
	FMOVD 8(R3), F1             // F1 = w.imag

	// Load a = work[idx_a]
	LSL  $4, R1, R4             // R4 = idx_a * 16
	ADD  R8, R4, R4             // R4 = &work[idx_a]
	FMOVD 0(R4), F2             // F2 = a.real
	FMOVD 8(R4), F3             // F3 = a.imag

	// Load b = work[idx_b]
	LSL  $4, R2, R4             // R4 = idx_b * 16
	ADD  R8, R4, R4             // R4 = &work[idx_b]
	FMOVD 0(R4), F4             // F4 = b.real
	FMOVD 8(R4), F5             // F5 = b.imag

	// Compute wb = w * b = (w.r*b.r - w.i*b.i, w.r*b.i + w.i*b.r)
	FMULD F0, F4, F6            // F6 = w.r * b.r
	FMULD F1, F5, F7            // F7 = w.i * b.i
	FSUBD F7, F6, F6            // F6 = wb.real = w.r*b.r - w.i*b.i
	FMULD F0, F5, F7            // F7 = w.r * b.i
	FMULD F1, F4, F8            // F8 = w.i * b.r
	FADDD F8, F7, F7            // F7 = wb.imag = w.r*b.i + w.i*b.r

	// Butterfly: a' = a + wb, b' = a - wb
	FADDD F6, F2, F9            // F9 = a'.real = a.r + wb.r
	FADDD F7, F3, F10           // F10 = a'.imag = a.i + wb.i
	FSUBD F6, F2, F11           // F11 = b'.real = a.r - wb.r
	FSUBD F7, F3, F12           // F12 = b'.imag = a.i - wb.i

	// Store a' = work[idx_a]
	LSL  $4, R1, R4             // R4 = idx_a * 16
	ADD  R8, R4, R4             // R4 = &work[idx_a]
	FMOVD F9, 0(R4)             // work[idx_a].real = a'.r
	FMOVD F10, 8(R4)            // work[idx_a].imag = a'.i

	// Store b' = work[idx_b]
	LSL  $4, R2, R4             // R4 = idx_b * 16
	ADD  R8, R4, R4             // R4 = &work[idx_b]
	FMOVD F11, 0(R4)            // work[idx_b].real = b'.r
	FMOVD F12, 8(R4)            // work[idx_b].imag = b'.i

	ADD  $1, R0, R0             // j++
	B    neon64r2f64_inner_loop

neon64r2f64_next_base:
	ADD  R14, R17, R17          // base += size (advance to next group)
	B    neon64r2f64_base_loop

neon64r2f64_next_size:
	LSL  $1, R14, R14           // size *= 2 (next stage)
	B    neon64r2f64_size_loop

neon64r2f64_done:
	// =========================================================================
	// Copy back from scratch to dst if we used scratch buffer
	// =========================================================================
	CMP  R8, R20                // did we use scratch? (R8 != R20)
	BEQ  neon64r2f64_return_true // no, result already in dst

	MOVD $0, R0                 // R0 = loop counter
neon64r2f64_copy_loop:
	CMP  $64, R0                // while i < 64
	BGE  neon64r2f64_return_true
	LSL  $4, R0, R1             // R1 = i * 16
	ADD  R8, R1, R2             // R2 = &scratch[i]
	MOVD (R2), R3               // R3 = scratch[i].real (as bits)
	MOVD 8(R2), R4              // R4 = scratch[i].imag (as bits)
	ADD  R20, R1, R5            // R5 = &dst[i]
	MOVD R3, (R5)               // dst[i].real = scratch[i].real
	MOVD R4, 8(R5)              // dst[i].imag = scratch[i].imag
	ADD  $1, R0, R0             // i++
	B    neon64r2f64_copy_loop

neon64r2f64_return_true:
	MOVD $1, R0                 // return true
	MOVB R0, ret+96(FP)
	RET

neon64r2f64_return_false:
	MOVD $0, R0                 // return false
	MOVB R0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform, size 64, complex128, radix-2
// Same as forward but with conjugated twiddles and 1/N scaling
// ===========================================================================
TEXT ·InverseNEONSize64Radix2Complex128Asm(SB), NOSPLIT, $0-97
	// Load slice headers from stack
	MOVD dst+0(FP), R8          // R8 = dst.data
	MOVD src+24(FP), R9         // R9 = src.data
	MOVD twiddle+48(FP), R10    // R10 = twiddle.data
	MOVD scratch+72(FP), R11    // R11 = scratch.data
	MOVD src+32(FP), R13        // R13 = src.len (n)

	// Validate n == 64
	CMP  $64, R13
	BNE  neon64r2f64_inv_return_false

	// Validate dst capacity >= 64
	MOVD dst+8(FP), R0
	CMP  $64, R0
	BLT  neon64r2f64_inv_return_false

	// Validate twiddle capacity >= 64
	MOVD twiddle+56(FP), R0
	CMP  $64, R0
	BLT  neon64r2f64_inv_return_false

	// Validate scratch capacity >= 64
	MOVD scratch+80(FP), R0
	CMP  $64, R0
	BLT  neon64r2f64_inv_return_false

	// Load bit-reversal table address
	MOVD $bitrev_size64_radix2_f64<>(SB), R12

	// Save original dst pointer
	MOVD R8, R20

	// If src == dst, use scratch as working buffer
	CMP  R8, R9
	BNE  neon64r2f64_inv_use_dst
	MOVD R11, R8

neon64r2f64_inv_use_dst:
	// =========================================================================
	// Bit-reversal permutation
	// =========================================================================
	MOVD $0, R0

neon64r2f64_inv_bitrev_loop:
	CMP  $64, R0
	BGE  neon64r2f64_inv_stage

	LSL  $3, R0, R1             // offset into bitrev table
	ADD  R12, R1, R1
	MOVD (R1), R2               // R2 = bitrev[i]

	LSL  $4, R2, R3             // source offset (16 bytes per complex128)
	ADD  R9, R3, R3
	MOVD (R3), R4               // load real part
	MOVD 8(R3), R5              // load imag part

	LSL  $4, R0, R3             // dest offset
	ADD  R8, R3, R3
	MOVD R4, (R3)               // store real part
	MOVD R5, 8(R3)              // store imag part

	ADD  $1, R0, R0
	B    neon64r2f64_inv_bitrev_loop

neon64r2f64_inv_stage:
	// =========================================================================
	// Butterfly stages with conjugated twiddles for inverse
	// =========================================================================
	MOVD $2, R14                // size starts at 2

neon64r2f64_inv_size_loop:
	CMP  $64, R14
	BGT  neon64r2f64_inv_done

	LSR  $1, R14, R15           // half = size / 2
	UDIV R14, R13, R16          // step = n / size

	MOVD $0, R17                // base = 0

neon64r2f64_inv_base_loop:
	CMP  R13, R17
	BGE  neon64r2f64_inv_next_size

	MOVD $0, R0                 // j = 0

neon64r2f64_inv_inner_loop:
	CMP  R15, R0
	BGE  neon64r2f64_inv_next_base

	ADD  R17, R0, R1            // idx_a = base + j
	ADD  R1, R15, R2            // idx_b = base + j + half

	// Load twiddle and conjugate (negate imaginary)
	MUL  R0, R16, R3            // j * step
	LSL  $4, R3, R3             // byte offset
	ADD  R10, R3, R3            // &twiddle[j*step]
	FMOVD 0(R3), F0             // w.real
	FMOVD 8(R3), F1             // w.imag
	FNEGD F1, F1                // conjugate: w* = (w.r, -w.i)

	// Load a
	LSL  $4, R1, R4
	ADD  R8, R4, R4
	FMOVD 0(R4), F2             // a.real
	FMOVD 8(R4), F3             // a.imag

	// Load b
	LSL  $4, R2, R4
	ADD  R8, R4, R4
	FMOVD 0(R4), F4             // b.real
	FMOVD 8(R4), F5             // b.imag

	// wb = w* * b
	FMULD F0, F4, F6            // w.r * b.r
	FMULD F1, F5, F7            // (-w.i) * b.i
	FSUBD F7, F6, F6            // wb.real
	FMULD F0, F5, F7            // w.r * b.i
	FMULD F1, F4, F8            // (-w.i) * b.r
	FADDD F8, F7, F7            // wb.imag

	// Butterfly
	FADDD F6, F2, F9            // a' = a + wb
	FADDD F7, F3, F10
	FSUBD F6, F2, F11           // b' = a - wb
	FSUBD F7, F3, F12

	// Store a'
	LSL  $4, R1, R4
	ADD  R8, R4, R4
	FMOVD F9, 0(R4)
	FMOVD F10, 8(R4)

	// Store b'
	LSL  $4, R2, R4
	ADD  R8, R4, R4
	FMOVD F11, 0(R4)
	FMOVD F12, 8(R4)

	ADD  $1, R0, R0
	B    neon64r2f64_inv_inner_loop

neon64r2f64_inv_next_base:
	ADD  R14, R17, R17          // base += size
	B    neon64r2f64_inv_base_loop

neon64r2f64_inv_next_size:
	LSL  $1, R14, R14           // size *= 2
	B    neon64r2f64_inv_size_loop

neon64r2f64_inv_done:
	// =========================================================================
	// Copy back if we used scratch
	// =========================================================================
	CMP  R8, R20
	BEQ  neon64r2f64_inv_scale

	MOVD $0, R0
neon64r2f64_inv_copy_loop:
	CMP  $64, R0
	BGE  neon64r2f64_inv_scale
	LSL  $4, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R3
	MOVD 8(R2), R4
	ADD  R20, R1, R5
	MOVD R3, (R5)
	MOVD R4, 8(R5)
	ADD  $1, R0, R0
	B    neon64r2f64_inv_copy_loop

neon64r2f64_inv_scale:
	// =========================================================================
	// Apply 1/N scaling for inverse transform
	// =========================================================================
	MOVD $·neonInv64F64(SB), R1 // R1 = &(1/64)
	FMOVD (R1), F0              // F0 = 1/64 = 0.015625
	MOVD $0, R0                 // R0 = loop counter

neon64r2f64_inv_scale_loop:
	CMP  $64, R0
	BGE  neon64r2f64_inv_return_true
	LSL  $4, R0, R1             // R1 = i * 16
	ADD  R20, R1, R1            // R1 = &dst[i]
	FMOVD 0(R1), F2             // F2 = dst[i].real
	FMOVD 8(R1), F3             // F3 = dst[i].imag
	FMULD F0, F2, F2            // F2 = dst[i].real / 64
	FMULD F0, F3, F3            // F3 = dst[i].imag / 64
	FMOVD F2, 0(R1)             // store scaled real
	FMOVD F3, 8(R1)             // store scaled imag
	ADD  $1, R0, R0             // i++
	B    neon64r2f64_inv_scale_loop

neon64r2f64_inv_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon64r2f64_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// ===========================================================================
// Data Section - Bit-reversal permutation table for size 64
// ===========================================================================
// Each entry is 8 bytes (uint64), total 64 entries = 512 bytes
// bitrev[i] = reverse_bits(i, 6) where 6 = log2(64)

DATA bitrev_size64_radix2_f64<>+0x000(SB)/8, $0   // bitrev[0] = 0
DATA bitrev_size64_radix2_f64<>+0x008(SB)/8, $32  // bitrev[1] = 32
DATA bitrev_size64_radix2_f64<>+0x010(SB)/8, $16  // bitrev[2] = 16
DATA bitrev_size64_radix2_f64<>+0x018(SB)/8, $48  // bitrev[3] = 48
DATA bitrev_size64_radix2_f64<>+0x020(SB)/8, $8   // bitrev[4] = 8
DATA bitrev_size64_radix2_f64<>+0x028(SB)/8, $40  // bitrev[5] = 40
DATA bitrev_size64_radix2_f64<>+0x030(SB)/8, $24  // bitrev[6] = 24
DATA bitrev_size64_radix2_f64<>+0x038(SB)/8, $56  // bitrev[7] = 56
DATA bitrev_size64_radix2_f64<>+0x040(SB)/8, $4   // bitrev[8] = 4
DATA bitrev_size64_radix2_f64<>+0x048(SB)/8, $36  // bitrev[9] = 36
DATA bitrev_size64_radix2_f64<>+0x050(SB)/8, $20  // bitrev[10] = 20
DATA bitrev_size64_radix2_f64<>+0x058(SB)/8, $52  // bitrev[11] = 52
DATA bitrev_size64_radix2_f64<>+0x060(SB)/8, $12  // bitrev[12] = 12
DATA bitrev_size64_radix2_f64<>+0x068(SB)/8, $44  // bitrev[13] = 44
DATA bitrev_size64_radix2_f64<>+0x070(SB)/8, $28  // bitrev[14] = 28
DATA bitrev_size64_radix2_f64<>+0x078(SB)/8, $60  // bitrev[15] = 60
DATA bitrev_size64_radix2_f64<>+0x080(SB)/8, $2   // bitrev[16] = 2
DATA bitrev_size64_radix2_f64<>+0x088(SB)/8, $34  // bitrev[17] = 34
DATA bitrev_size64_radix2_f64<>+0x090(SB)/8, $18  // bitrev[18] = 18
DATA bitrev_size64_radix2_f64<>+0x098(SB)/8, $50  // bitrev[19] = 50
DATA bitrev_size64_radix2_f64<>+0x0A0(SB)/8, $10  // bitrev[20] = 10
DATA bitrev_size64_radix2_f64<>+0x0A8(SB)/8, $42  // bitrev[21] = 42
DATA bitrev_size64_radix2_f64<>+0x0B0(SB)/8, $26  // bitrev[22] = 26
DATA bitrev_size64_radix2_f64<>+0x0B8(SB)/8, $58  // bitrev[23] = 58
DATA bitrev_size64_radix2_f64<>+0x0C0(SB)/8, $6   // bitrev[24] = 6
DATA bitrev_size64_radix2_f64<>+0x0C8(SB)/8, $38  // bitrev[25] = 38
DATA bitrev_size64_radix2_f64<>+0x0D0(SB)/8, $22  // bitrev[26] = 22
DATA bitrev_size64_radix2_f64<>+0x0D8(SB)/8, $54  // bitrev[27] = 54
DATA bitrev_size64_radix2_f64<>+0x0E0(SB)/8, $14  // bitrev[28] = 14
DATA bitrev_size64_radix2_f64<>+0x0E8(SB)/8, $46  // bitrev[29] = 46
DATA bitrev_size64_radix2_f64<>+0x0F0(SB)/8, $30  // bitrev[30] = 30
DATA bitrev_size64_radix2_f64<>+0x0F8(SB)/8, $62  // bitrev[31] = 62
DATA bitrev_size64_radix2_f64<>+0x100(SB)/8, $1   // bitrev[32] = 1
DATA bitrev_size64_radix2_f64<>+0x108(SB)/8, $33  // bitrev[33] = 33
DATA bitrev_size64_radix2_f64<>+0x110(SB)/8, $17  // bitrev[34] = 17
DATA bitrev_size64_radix2_f64<>+0x118(SB)/8, $49  // bitrev[35] = 49
DATA bitrev_size64_radix2_f64<>+0x120(SB)/8, $9   // bitrev[36] = 9
DATA bitrev_size64_radix2_f64<>+0x128(SB)/8, $41  // bitrev[37] = 41
DATA bitrev_size64_radix2_f64<>+0x130(SB)/8, $25  // bitrev[38] = 25
DATA bitrev_size64_radix2_f64<>+0x138(SB)/8, $57  // bitrev[39] = 57
DATA bitrev_size64_radix2_f64<>+0x140(SB)/8, $5   // bitrev[40] = 5
DATA bitrev_size64_radix2_f64<>+0x148(SB)/8, $37  // bitrev[41] = 37
DATA bitrev_size64_radix2_f64<>+0x150(SB)/8, $21  // bitrev[42] = 21
DATA bitrev_size64_radix2_f64<>+0x158(SB)/8, $53  // bitrev[43] = 53
DATA bitrev_size64_radix2_f64<>+0x160(SB)/8, $13  // bitrev[44] = 13
DATA bitrev_size64_radix2_f64<>+0x168(SB)/8, $45  // bitrev[45] = 45
DATA bitrev_size64_radix2_f64<>+0x170(SB)/8, $29  // bitrev[46] = 29
DATA bitrev_size64_radix2_f64<>+0x178(SB)/8, $61  // bitrev[47] = 61
DATA bitrev_size64_radix2_f64<>+0x180(SB)/8, $3   // bitrev[48] = 3
DATA bitrev_size64_radix2_f64<>+0x188(SB)/8, $35  // bitrev[49] = 35
DATA bitrev_size64_radix2_f64<>+0x190(SB)/8, $19  // bitrev[50] = 19
DATA bitrev_size64_radix2_f64<>+0x198(SB)/8, $51  // bitrev[51] = 51
DATA bitrev_size64_radix2_f64<>+0x1A0(SB)/8, $11  // bitrev[52] = 11
DATA bitrev_size64_radix2_f64<>+0x1A8(SB)/8, $43  // bitrev[53] = 43
DATA bitrev_size64_radix2_f64<>+0x1B0(SB)/8, $27  // bitrev[54] = 27
DATA bitrev_size64_radix2_f64<>+0x1B8(SB)/8, $59  // bitrev[55] = 59
DATA bitrev_size64_radix2_f64<>+0x1C0(SB)/8, $7   // bitrev[56] = 7
DATA bitrev_size64_radix2_f64<>+0x1C8(SB)/8, $39  // bitrev[57] = 39
DATA bitrev_size64_radix2_f64<>+0x1D0(SB)/8, $23  // bitrev[58] = 23
DATA bitrev_size64_radix2_f64<>+0x1D8(SB)/8, $55  // bitrev[59] = 55
DATA bitrev_size64_radix2_f64<>+0x1E0(SB)/8, $15  // bitrev[60] = 15
DATA bitrev_size64_radix2_f64<>+0x1E8(SB)/8, $47  // bitrev[61] = 47
DATA bitrev_size64_radix2_f64<>+0x1F0(SB)/8, $31  // bitrev[62] = 31
DATA bitrev_size64_radix2_f64<>+0x1F8(SB)/8, $63  // bitrev[63] = 63
GLOBL bitrev_size64_radix2_f64<>(SB), RODATA, $512
