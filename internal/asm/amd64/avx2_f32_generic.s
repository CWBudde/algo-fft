//go:build amd64 && fft_asm && !purego

// ===========================================================================
// AVX2/FMA-optimized FFT Assembly for AMD64 - complex64 (float32)
// ===========================================================================
//
// This file implements high-performance FFT transforms using AVX2 and FMA3
// instructions for complex64 (single-precision) data types.
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
// - complex64:  Process 4 butterflies per iteration using YMM (256-bit) registers
//               Each complex64 = 8 bytes, so YMM holds 4 complex numbers
//
// TWIDDLE FACTOR ACCESS
// ---------------------
// Two code paths handle different twiddle memory layouts:
//   - Contiguous (step=1): Twiddles at indices 0,1,2,3,... loaded with VMOVUPS
//   - Strided (step>1): Twiddles at indices 0,step,2*step,... gathered manually
//
// The strided path uses scalar loads + VPUNPCKLQDQ/VINSERTF128 to build vectors.
//
// COMPLEX MULTIPLICATION
// ----------------------
// Forward FFT: t = w * b
//   Real: b.r*w.r - b.i*w.i
//   Imag: b.i*w.r + b.r*w.i
//   Uses VFMADDSUB231PS: dst = src2*src3 -/+ dst (even lanes -, odd lanes +)
//
// Inverse FFT: t = conj(w) * b
//   Real: b.r*w.r + b.i*w.i
//   Imag: b.i*w.r - b.r*w.i
//   Uses VFMSUBADD231PS: dst = src2*src3 +/- dst (even lanes +, odd lanes -)
//
// PERFORMANCE NOTES
// -----------------
// - Minimum size for AVX2 path: n >= 16 (complex64)
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
//   func forwardAVX2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
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
// complex64:  8 bytes  = 4 bytes (float32 real) + 4 bytes (float32 imag)
//
// YMM register (256 bits = 32 bytes):
//   - Holds 4 complex64  values (4 × 8  = 32 bytes)
//
// XMM register (128 bits = 16 bytes):
//   - Holds 2 complex64  values (2 × 8  = 16 bytes)

// ===========================================================================
// Register Allocation (complex64 functions)
// ===========================================================================
// Preserved across function body:
//   R8:  work buffer pointer (dst or scratch, depending on in-place)
//   R9:  src pointer (input data)
//   R10: twiddle pointer (precomputed roots of unity)
//   R11: scratch pointer / stride_bytes (reused in strided loops)
//   R12: bitrev pointer / stride_bytes (reused in strided loops)
//   R13: n (transform length, power of 2)
//
// Loop control:
//   R14: size (outer loop: 2, 4, 8, ... n)
//   R15: half = size/2 (butterflies per group)
//   BX:  step = n/size (twiddle stride)
//   CX:  base (middle loop: 0, size, 2*size, ...)
//   DX:  j (inner loop: 0 to half-1)
//
// Index computation:
//   SI:  index1 byte offset = (base + j) * sizeof(complex)
//   DI:  index2 byte offset = index1 + half * sizeof(complex)
//   AX:  scratch register for various computations

// ===========================================================================
// forwardAVX2Complex64Asm - Forward FFT for complex64 using AVX2/FMA
// ===========================================================================
// Performs forward DFT: X[k] = Σ x[n] * exp(-2πi*n*k/N)
//
// Parameters:
//   dst     []complex64 - Output buffer (len >= n)
//   src     []complex64 - Input buffer  (len = n, defines transform size)
//   twiddle []complex64 - Precomputed twiddle factors (len >= n)
//   scratch []complex64 - Working buffer for in-place transforms (len >= n)
//   bitrev  []int       - Bit-reversal permutation indices (len >= n)
//
// Returns: true if transform completed, false to fall back to Go
// ===========================================================================
TEXT ·ForwardAVX2Complex64Asm(SB), NOSPLIT, $0-121
	// -----------------------------------------------------------------------
	// PHASE 1: Load parameters and validate inputs
	// -----------------------------------------------------------------------
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer
	MOVQ src+32(FP), R13     // R13 = n = len(src)

	// Empty input is valid (no-op)
	TESTQ R13, R13
	JZ    return_true

	// Validate all slice lengths are >= n
	MOVQ dst+8(FP), AX
	CMPQ AX, R13
	JL   return_false        // dst too short

	MOVQ twiddle+56(FP), AX
	CMPQ AX, R13
	JL   return_false        // twiddle too short

	MOVQ scratch+80(FP), AX
	CMPQ AX, R13
	JL   return_false        // scratch too short

	MOVQ bitrev+104(FP), AX
	CMPQ AX, R13
	JL   return_false        // bitrev too short

	// Trivial case: n=1, just copy
	CMPQ R13, $1
	JNE  check_power_of_2
	MOVQ (R9), AX            // Load single complex64 (8 bytes)
	MOVQ AX, (R8)            // Copy to dst
	JMP  return_true

check_power_of_2:
	// Verify n is power of 2: (n & (n-1)) == 0
	// This is required for radix-2 FFT algorithm
	MOVQ R13, AX
	LEAQ -1(AX), BX          // BX = n - 1
	TESTQ AX, BX             // Sets ZF if n & (n-1) == 0
	JNZ  return_false        // Not power of 2, fall back to Go

	// Minimum size for AVX2 vectorization
	// Smaller sizes have too much overhead vs pure Go
	CMPQ R13, $16
	JL   return_false        // Fall back to Go for n < 16

	// -----------------------------------------------------------------------
	// PHASE 2: Select working buffer
	// -----------------------------------------------------------------------
	// For in-place transforms (dst == src), we need a separate work buffer
	// to avoid overwriting source data during bit-reversal
	CMPQ R8, R9
	JNE  use_dst_as_work

	// In-place: use scratch as working buffer
	MOVQ R11, R8             // R8 = work = scratch
	MOVL $0, AX              // Flag: work != dst (unused, kept for clarity)
	JMP  do_bit_reversal

use_dst_as_work:
	// Out-of-place: use dst directly as working buffer
	MOVL $1, AX              // Flag: work == dst (unused, kept for clarity)

do_bit_reversal:
	// -----------------------------------------------------------------------
	// PHASE 3: Bit-reversal permutation
	// -----------------------------------------------------------------------
	// Reorder input using precomputed bit-reversed indices:
	//   work[i] = src[bitrev[i]]  for i = 0..n-1
	//
	// This puts data in "butterfly order" for the DIT algorithm.
	// After this, elements that need to be combined in the first stage
	// are adjacent in memory.
	XORQ CX, CX              // CX = i = 0

bitrev_loop:
	CMPQ CX, R13
	JGE  bitrev_done

	// Load bitrev[i] (int = 8 bytes on amd64)
	MOVQ (R12)(CX*8), DX     // DX = bitrev[i]

	// Load src[bitrev[i]] (complex64 = 8 bytes)
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[i]]

	// Store to work[i]
	MOVQ AX, (R8)(CX*8)      // work[i] = src[bitrev[i]]

	INCQ CX
	JMP  bitrev_loop

bitrev_done:
	// -----------------------------------------------------------------------
	// PHASE 4: Main DIT Butterfly Stages
	// -----------------------------------------------------------------------
	// Outer loop: for size = 2, 4, 8, ... up to n
	//   Each iteration doubles the butterfly group size
	//   - size=2:  combine pairs        (n/2 groups of 2)
	//   - size=4:  combine quads        (n/4 groups of 4)
	//   - size=n:  final combination    (1 group of n)
	MOVQ $2, R14             // R14 = size = 2

size_loop:
	CMPQ R14, R13
	JG   transform_done      // Done when size > n

	// half = number of butterflies per group = size/2
	MOVQ R14, R15
	SHRQ $1, R15             // R15 = half = size >> 1

	// step = twiddle stride = n/size
	// Twiddles are stored for the largest stage (size=n)
	// Smaller stages skip entries: twiddle[j*step]
	MOVQ R13, AX             // AX = n (dividend)
	XORQ DX, DX              // DX:AX = 64-bit dividend
	DIVQ R14                 // AX = n / size
	MOVQ AX, BX              // BX = step

	// Middle loop: process each group
	// for base = 0, size, 2*size, ... up to n
	XORQ CX, CX              // CX = base = 0

base_loop:
	CMPQ CX, R13
	JGE  next_size           // All groups processed

	// -----------------------------------------------------------------------
	// Select vectorization strategy based on butterfly count and twiddle layout
	// -----------------------------------------------------------------------
	// AVX2 requires at least 4 butterflies (fills one YMM register)
	CMPQ R15, $4
	JL   scalar_butterflies  // Too few butterflies for AVX2

	// Two twiddle access patterns:
	// - Contiguous (step=1): Twiddles at 0,1,2,3... - single VMOVUPS load
	// - Strided (step>1): Twiddles at 0,s,2s,3s... - gather with scalar loads
	CMPQ BX, $1
	JE   avx2_contiguous     // Fast path: contiguous twiddles
	JMP  avx2_strided        // Slow path: strided twiddles

avx2_contiguous:
	// -----------------------------------------------------------------------
	// AVX2 Contiguous Twiddle Path (step == 1)
	// -----------------------------------------------------------------------
	// This is the fast path for the largest stages where step=1.
	// Process 4 butterflies per iteration using single VMOVUPS loads.
	XORQ DX, DX              // DX = j = 0 (butterfly index within group)

avx2_loop:
	// Check if 4+ butterflies remain in this group
	MOVQ R15, AX
	SUBQ DX, AX              // AX = remaining = half - j
	CMPQ AX, $4
	JL   scalar_remainder    // Less than 4 left, finish with scalar

	// Compute byte offsets for data access
	// index1 = base + j (first butterfly input)
	// index2 = base + j + half (second butterfly input, half-group apart)
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI              // SI = (base + j) * 8 bytes

	MOVQ R15, DI
	SHLQ $3, DI              // DI = half * 8 bytes
	ADDQ SI, DI              // DI = (base + j + half) * 8 bytes

	// Load 4 complex64 values from each position (32 bytes = 4 × 8)
	// Y0 = a[0..3] = work[index1 : index1+4]
	// Y1 = b[0..3] = work[index2 : index2+4]
	VMOVUPS (R8)(SI*1), Y0   // Y0 = [a0, a1, a2, a3]
	VMOVUPS (R8)(DI*1), Y1   // Y1 = [b0, b1, b2, b3]

	// Load 4 contiguous twiddle factors
	// Y2 = w[0..3] = twiddle[j : j+4] (step=1, so contiguous)
	MOVQ DX, AX
	SHLQ $3, AX              // AX = j * 8 bytes
	VMOVUPS (R10)(AX*1), Y2  // Y2 = [w0, w1, w2, w3]

	// Jump to shared butterfly computation
	JMP  avx2_butterfly

avx2_strided:
	// -----------------------------------------------------------------------
	// AVX2 Strided Twiddle Path (step > 1)
	// -----------------------------------------------------------------------
	// This path handles early stages where twiddle factors are not contiguous.
	// We manually gather 4 (or 8) twiddles using scalar loads, then combine
	// them into YMM registers using VPUNPCKLQDQ and VINSERTF128.
	//
	// Loop is 2x unrolled to process 8 butterflies when possible, reducing
	// loop overhead and improving instruction-level parallelism.
	//
	// Register usage in this section:
	//   R11: current twiddle byte offset (starts at 0)
	//   R12: stride in bytes = step * 8

	MOVQ BX, R12             // R12 = step
	SHLQ $3, R12             // R12 = step * 8 (stride in bytes)
	XORQ R11, R11            // R11 = twiddle_offset = 0
	XORQ DX, DX              // DX = j = 0

avx2_strided_loop:
	// Try to process 8 butterflies (2 × 4-wide AVX2 operations)
	MOVQ R15, AX
	SUBQ DX, AX              // AX = remaining = half - j
	CMPQ AX, $8
	JL   avx2_strided_single // Less than 8, try single block

	// =======================================================================
	// Unrolled Block: Process 8 butterflies in parallel
	// =======================================================================
	// Block 1: butterflies j+0..j+3 using Y0, Y1, Y2
	// Block 2: butterflies j+4..j+7 using Y7, Y8, Y9

	// Compute data offsets for Block 1
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI              // SI = (base + j) * 8

	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI              // DI = (base + j + half) * 8

	// Load data for both blocks (Block 2 is +32 bytes ahead)
	VMOVUPS (R8)(SI*1), Y0   // Block 1: a[j+0..j+3]
	VMOVUPS (R8)(DI*1), Y1   // Block 1: b[j+0..j+3]
	VMOVUPS 32(R8)(SI*1), Y7 // Block 2: a[j+4..j+7]
	VMOVUPS 32(R8)(DI*1), Y8 // Block 2: b[j+4..j+7]

	// -----------------------------------------------------------------------
	// Gather Block 1 twiddles: w[0], w[step], w[2*step], w[3*step]
	// -----------------------------------------------------------------------
	// Each twiddle is 8 bytes (complex64). Load scalar, then pack into YMM.
	// Twiddle indices: 0*step, 1*step, 2*step, 3*step from current offset
	VMOVSD (R10)(R11*1), X2      // X2 = w0
	LEAQ (R11)(R12*1), AX        // AX = offset + 1*stride
	VMOVSD (R10)(AX*1), X3       // X3 = w1
	ADDQ R12, AX                 // AX = offset + 2*stride
	VMOVSD (R10)(AX*1), X4       // X4 = w2
	ADDQ R12, AX                 // AX = offset + 3*stride
	VMOVSD (R10)(AX*1), X5       // X5 = w3

	// Pack into YMM: Y2 = [w0, w1, w2, w3]
	// VPUNPCKLQDQ interleaves low 64 bits: [X2_lo, X3_lo] -> X2
	VPUNPCKLQDQ X3, X2, X2       // X2 = [w0, w1]
	VPUNPCKLQDQ X5, X4, X4       // X4 = [w2, w3]
	VINSERTF128 $1, X4, Y2, Y2   // Y2 = [w0, w1, w2, w3]

	// -----------------------------------------------------------------------
	// Gather Block 2 twiddles: w[4*step], w[5*step], w[6*step], w[7*step]
	// -----------------------------------------------------------------------
	ADDQ R12, AX                 // AX = offset + 4*stride
	VMOVSD (R10)(AX*1), X9       // X9  = w4
	ADDQ R12, AX
	VMOVSD (R10)(AX*1), X10      // X10 = w5
	ADDQ R12, AX
	VMOVSD (R10)(AX*1), X11      // X11 = w6
	ADDQ R12, AX
	VMOVSD (R10)(AX*1), X12      // X12 = w7

	// Pack into YMM: Y9 = [w4, w5, w6, w7]
	VPUNPCKLQDQ X10, X9, X9      // X9  = [w4, w5]
	VPUNPCKLQDQ X12, X11, X11    // X11 = [w6, w7]
	VINSERTF128 $1, X11, Y9, Y9  // Y9  = [w4, w5, w6, w7]

	// -----------------------------------------------------------------------
	// Block 1 Butterfly: t = w * b, a' = a + t, b' = a - t
	// -----------------------------------------------------------------------
	// Complex multiply using FMA: t = w * b
	//   t.real = b.real*w.real - b.imag*w.imag
	//   t.imag = b.imag*w.real + b.real*w.imag
	VMOVSLDUP Y2, Y3             // Y3 = [w.r, w.r, ...] (broadcast real parts)
	VMOVSHDUP Y2, Y4             // Y4 = [w.i, w.i, ...] (broadcast imag parts)
	VSHUFPS $0xB1, Y1, Y1, Y6    // Y6 = b_swapped = [b.i, b.r, ...]
	VMULPS Y4, Y6, Y6            // Y6 = [b.i*w.i, b.r*w.i, ...]
	VFMADDSUB231PS Y3, Y1, Y6    // Y6 = b*w.r -/+ Y6 = [t.r, t.i, ...]
	VADDPS Y6, Y0, Y3            // Y3 = a + t = a'
	VSUBPS Y6, Y0, Y4            // Y4 = a - t = b'
	VMOVUPS Y3, (R8)(SI*1)       // Store a'
	VMOVUPS Y4, (R8)(DI*1)       // Store b'

	// -----------------------------------------------------------------------
	// Block 2 Butterfly (same algorithm, different registers)
	// -----------------------------------------------------------------------
	VMOVSLDUP Y9, Y10            // Y10 = w.r broadcast
	VMOVSHDUP Y9, Y11            // Y11 = w.i broadcast
	VSHUFPS $0xB1, Y8, Y8, Y6    // Y6 = b_swapped (reuse Y6)
	VMULPS Y11, Y6, Y6           // Y6 = b_swap * w.i
	VFMADDSUB231PS Y10, Y8, Y6   // Y6 = t
	VADDPS Y6, Y7, Y10           // Y10 = a'
	VSUBPS Y6, Y7, Y11           // Y11 = b'
	VMOVUPS Y10, 32(R8)(SI*1)    // Store a' (Block 2, +32 bytes)
	VMOVUPS Y11, 32(R8)(DI*1)    // Store b' (Block 2, +32 bytes)

	// Advance to next 8 butterflies
	LEAQ (R11)(R12*8), R11       // twiddle_offset += 8 * stride
	ADDQ $8, DX                  // j += 8
	JMP avx2_strided_loop

avx2_strided_single:
	// -----------------------------------------------------------------------
	// Single AVX2 block: Process 4 butterflies (when 4-7 remain)
	// -----------------------------------------------------------------------
	CMPQ AX, $4
	JL   scalar_remainder        // Less than 4, use scalar

	// Compute data offsets
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI                  // SI = (base + j) * 8
	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI                  // DI = (base + j + half) * 8

	// Load 4 complex64 values
	VMOVUPS (R8)(SI*1), Y0       // a[j+0..j+3]
	VMOVUPS (R8)(DI*1), Y1       // b[j+0..j+3]

	// Gather 4 twiddles with stride
	VMOVSD (R10)(R11*1), X2      // w0
	LEAQ (R11)(R12*1), AX
	VMOVSD (R10)(AX*1), X3       // w1
	ADDQ R12, AX
	VMOVSD (R10)(AX*1), X4       // w2
	ADDQ R12, AX
	VMOVSD (R10)(AX*1), X5       // w3

	// Pack: Y2 = [w0, w1, w2, w3]
	VPUNPCKLQDQ X3, X2, X2
	VPUNPCKLQDQ X5, X4, X4
	VINSERTF128 $1, X4, Y2, Y2

	// Butterfly using FMA
	VMOVSLDUP Y2, Y3             // w.r broadcast
	VMOVSHDUP Y2, Y4             // w.i broadcast
	VSHUFPS $0xB1, Y1, Y1, Y6    // b_swapped
	VMULPS Y4, Y6, Y6            // b_swap * w.i
	VFMADDSUB231PS Y3, Y1, Y6    // t = w * b
	VADDPS Y6, Y0, Y3            // a' = a + t
	VSUBPS Y6, Y0, Y4            // b' = a - t
	VMOVUPS Y3, (R8)(SI*1)
	VMOVUPS Y4, (R8)(DI*1)

	// Advance
	LEAQ (R11)(R12*4), R11       // twiddle_offset += 4 * stride
	ADDQ $4, DX                  // j += 4

	JMP scalar_remainder         // Handle remaining 0-3

avx2_butterfly:
	// -----------------------------------------------------------------------
	// AVX2 Butterfly Computation (shared by contiguous path)
	// -----------------------------------------------------------------------
	// Entry state:
	//   Y0: a[0..3] - first butterfly inputs
	//   Y1: b[0..3] - second butterfly inputs
	//   Y2: w[0..3] - twiddle factors
	//   SI: byte offset of index1
	//   DI: byte offset of index2
	//   DX: current j value
	//   R8: work buffer pointer
	//   BX: step value (for loop-back decision)
	//
	// Complex multiplication algorithm: t = w * b
	// -----------------------------------------------------------------------
	// For complex numbers w = w.r + i*w.i and b = b.r + i*b.i:
	//   t.real = w.r * b.r - w.i * b.i
	//   t.imag = w.r * b.i + w.i * b.r
	//
	// Memory layout in YMM (4 complex64, 32 bytes):
	//   [b0.r, b0.i, b1.r, b1.i, b2.r, b2.i, b3.r, b3.i]
	//    lane0 lane1 lane2 lane3 lane4 lane5 lane6 lane7
	//
	// Strategy using FMA:
	//   1. VMOVSLDUP: broadcast real parts [w.r, w.r, w.r, w.r, ...]
	//   2. VMOVSHDUP: broadcast imag parts [w.i, w.i, w.i, w.i, ...]
	//   3. VSHUFPS 0xB1: swap adjacent pairs [b.i, b.r, ...]
	//   4. Multiply swapped by w.i: [b.i*w.i, b.r*w.i, ...]
	//   5. VFMADDSUB231PS: b*w.r -/+ (swap*w.i)
	//      Even lanes (real): subtract → b.r*w.r - b.i*w.i ✓
	//      Odd lanes (imag):  add      → b.i*w.r + b.r*w.i ✓

	VMOVSLDUP Y2, Y3             // Y3 = [w.r, w.r, ...] duplicated real parts
	VMOVSHDUP Y2, Y4             // Y4 = [w.i, w.i, ...] duplicated imag parts

	// Prepare swapped b for the imaginary term
	VSHUFPS $0xB1, Y1, Y1, Y6    // Y6 = [b.i, b.r, b.i, b.r, ...] (swap pairs)
	VMULPS Y4, Y6, Y6            // Y6 = [b.i*w.i, b.r*w.i, ...]

	// FMA: Y6 = Y1 * Y3 -/+ Y6
	// VFMADDSUB231PS: dst = src2*src3 -/+ dst (even -, odd +)
	VFMADDSUB231PS Y3, Y1, Y6    // Y6 = t = w * b

	// Butterfly outputs
	VADDPS Y6, Y0, Y3            // Y3 = a + t = a'
	VSUBPS Y6, Y0, Y4            // Y4 = a - t = b'

	// Store results back to memory
	VMOVUPS Y3, (R8)(SI*1)       // work[index1] = a'
	VMOVUPS Y4, (R8)(DI*1)       // work[index2] = b'

	ADDQ $4, DX                  // j += 4 (processed 4 butterflies)

	// Loop back to appropriate path based on twiddle layout
	CMPQ BX, $1
	JE   avx2_loop               // Contiguous: continue in avx2_loop
	JMP  avx2_strided_loop       // Strided: continue in avx2_strided_loop

scalar_remainder:
	// -----------------------------------------------------------------------
	// Scalar Remainder: Process 0-3 leftover butterflies after AVX2
	// -----------------------------------------------------------------------
	CMPQ DX, R15
	JGE  next_base               // No remainder, move to next group

scalar_remainder_loop:
	// Compute byte offsets
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI                  // SI = (base + j) * 8

	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI                  // DI = (base + j + half) * 8

	// Load single butterfly inputs (complex64 = 8 bytes)
	MOVSD (R8)(SI*1), X0         // X0 = a = work[index1]
	MOVSD (R8)(DI*1), X1         // X1 = b = work[index2]

	// Load twiddle with stride: twiddle[j * step]
	MOVQ DX, AX
	IMULQ BX, AX                 // AX = j * step
	SHLQ $3, AX                  // AX = (j * step) * 8 bytes
	MOVSD (R10)(AX*1), X2        // X2 = w = twiddle[j * step]

	// -----------------------------------------------------------------------
	// SSE Complex multiply: t = w * b
	// -----------------------------------------------------------------------
	// Same algorithm as AVX2 but using XMM registers (128-bit)
	// Only the lower 64 bits contain valid data

	MOVSLDUP X2, X3              // X3 = [w.r, w.r, ...] (duplicate low float)
	MOVSHDUP X2, X4              // X4 = [w.i, w.i, ...] (duplicate high float)

	MOVAPS X1, X5
	MULPS  X3, X5                // X5 = [b.r*w.r, b.i*w.r, ...]

	MOVAPS X1, X6
	SHUFPS $0xB1, X6, X6         // X6 = [b.i, b.r, ...] (swap pairs)
	MULPS  X4, X6                // X6 = [b.i*w.i, b.r*w.i, ...]

	// ADDSUBPS: adds odd lanes, subtracts even lanes
	// Result: [b.r*w.r - b.i*w.i, b.i*w.r + b.r*w.i, ...] = t
	ADDSUBPS X6, X5              // X5 = t = w * b

	// Butterfly
	MOVAPS X0, X3
	ADDPS  X5, X3                // X3 = a + t = a'
	MOVAPS X0, X4
	SUBPS  X5, X4                // X4 = a - t = b'

	// Store single complex64 (lower 64 bits)
	MOVSD X3, (R8)(SI*1)         // work[index1] = a'
	MOVSD X4, (R8)(DI*1)         // work[index2] = b'

	INCQ DX                      // j++
	CMPQ DX, R15
	JL   scalar_remainder_loop   // Continue until j >= half

next_base:
	// Advance to next butterfly group
	ADDQ R14, CX                 // base += size
	JMP  base_loop

scalar_butterflies:
	// -----------------------------------------------------------------------
	// Pure Scalar Path: When half < 4 (not enough for AVX2)
	// -----------------------------------------------------------------------
	// Used for early stages (size=2, size=4, size=8 when n=8)
	// Same algorithm as scalar_remainder but starts from j=0
	XORQ DX, DX                  // j = 0

scalar_loop:
	CMPQ DX, R15
	JGE  next_base               // Done with this group

	// Compute offsets
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI

	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI

	// Load inputs
	MOVSD (R8)(SI*1), X0         // a
	MOVSD (R8)(DI*1), X1         // b

	// Load twiddle with stride
	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $3, AX
	MOVSD (R10)(AX*1), X2        // w

	// Complex multiply: t = w * b
	MOVSLDUP X2, X3
	MOVSHDUP X2, X4

	MOVAPS X1, X5
	MULPS  X3, X5

	MOVAPS X1, X6
	SHUFPS $0xB1, X6, X6
	MULPS  X4, X6

	ADDSUBPS X6, X5              // t

	// Butterfly
	MOVAPS X0, X3
	ADDPS  X5, X3                // a'
	MOVAPS X0, X4
	SUBPS  X5, X4                // b'

	MOVSD X3, (R8)(SI*1)
	MOVSD X4, (R8)(DI*1)

	INCQ DX
	JMP  scalar_loop

next_size:
	// Advance to next stage (double the butterfly size)
	SHLQ $1, R14                 // size *= 2
	JMP  size_loop

transform_done:
	// -----------------------------------------------------------------------
	// PHASE 5: Finalization
	// -----------------------------------------------------------------------
	// Clear upper YMM bits to avoid AVX-SSE transition penalties
	// (mixing AVX and legacy SSE code causes performance stalls)
	VZEROUPPER

	// If we used scratch as work buffer (in-place transform),
	// copy results back to dst
	MOVQ dst+0(FP), AX
	CMPQ R8, AX
	JE   return_true             // work == dst, no copy needed

	// Copy work (scratch) → dst
	// n elements × 8 bytes per complex64
	XORQ CX, CX                  // i = 0

copy_loop:
	CMPQ CX, R13
	JGE  return_true
	MOVQ (R8)(CX*8), DX          // Load from work
	MOVQ DX, (AX)(CX*8)          // Store to dst
	INCQ CX
	JMP  copy_loop

return_true:
	// Success: transform completed in assembly
	VZEROUPPER                   // Ensure clean state
	MOVB $1, ret+120(FP)         // Return true
	RET

return_false:
	// Failure: fall back to pure Go implementation
	MOVB $0, ret+120(FP)         // Return false
	RET

// ===========================================================================
// forwardAVX2StockhamComplex64Asm - Forward FFT for complex64 (Stockham path)
// ===========================================================================
TEXT ·ForwardAVX2StockhamComplex64Asm(SB), NOSPLIT, $0-121
	// -----------------------------------------------------------------------
	// PHASE 1: Load parameters and validate inputs
	// -----------------------------------------------------------------------
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer (unused)
	MOVQ src+32(FP), R13     // R13 = n = len(src)

	// Empty input is valid (no-op)
	TESTQ R13, R13
	JZ    stockham_return_true

	// Validate all slice lengths are >= n
	MOVQ dst+8(FP), AX
	CMPQ AX, R13
	JL   stockham_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, R13
	JL   stockham_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, R13
	JL   stockham_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, R13
	JL   stockham_return_false

	// Trivial case: n=1, just copy
	CMPQ R13, $1
	JNE  stockham_check_power
	MOVQ (R9), AX
	MOVQ AX, (R8)
	JMP  stockham_return_true

stockham_check_power:
	// Verify n is power of 2
	MOVQ R13, AX
	LEAQ -1(AX), BX
	TESTQ AX, BX
	JNZ  stockham_return_false

	// Minimum size for AVX2 vectorization
	CMPQ R13, $16
	JL   stockham_return_false

	// -----------------------------------------------------------------------
	// PHASE 2: Select buffers (Stockham uses ping-pong)
	// -----------------------------------------------------------------------
	MOVQ R9, SI               // SI = in (src)
	MOVQ R8, DI               // DI = out (dst)
	CMPQ R8, R9
	JNE  stockham_out_ready
	MOVQ R11, DI              // In-place: first out = scratch

stockham_out_ready:
	// m starts at n and halves each stage
	MOVQ R13, R14             // R14 = m

stockham_stage_loop:
	CMPQ R14, $1
	JLE  stockham_done

	// step = n / m
	MOVQ R13, AX
	XORQ DX, DX
	DIVQ R14
	MOVQ AX, BX               // BX = step (also group count)

	XORQ CX, CX               // k = 0

stockham_k_loop:
	CMPQ CX, BX
	JGE  stockham_stage_done

	// half = m / 2
	MOVQ R14, R15
	SHRQ $1, R15

	// baseElem = k * m
	MOVQ CX, AX
	IMULQ R14, AX

	// ptrA = in + baseElem*8
	SHLQ $3, AX              // AX = baseElem * 8
	LEAQ (SI)(AX*1), BP      // BP = ptrA = in + baseElem*8

	// ptrB = ptrA + half*8
	MOVQ R15, AX
	SHLQ $3, AX              // AX = half * 8
	ADDQ BP, AX              // AX = ptrB = ptrA + half*8

	// outBaseElem = k * half
	MOVQ CX, DX
	IMULQ R15, DX

	// ptrOut0 = out + outBaseElem*8
	SHLQ $3, DX              // DX = outBaseElem * 8
	LEAQ (DI)(DX*1), R9      // R9 = ptrOut0 (DI = current output buffer)

	// ptrOut1 = ptrOut0 + (n/2)*8
	MOVQ R13, R12
	SHLQ $2, R12             // R12 = n * 4 bytes
	LEAQ (R9)(R12*1), R12    // R12 = ptrOut1

	// remaining = half
	MOVQ R15, DX             // DX = half

	// Fast path for contiguous twiddles (step == 1)
	CMPQ BX, $1
	JNE  stockham_scalar_strided

	// twiddle offset for contiguous path
	XORQ R11, R11
	CMPQ DX, $4
	JL   stockham_scalar_contig

stockham_vec_loop:
	CMPQ DX, $4
	JL   stockham_scalar_contig

	VMOVUPS (BP), Y0         // a (ptrA)
	VMOVUPS (AX), Y1         // b (ptrB)
	VMOVUPS (R10)(R11*1), Y2  // twiddle

	VADDPS Y1, Y0, Y3         // sum = a + b
	VSUBPS Y1, Y0, Y4         // diff = a - b

	VMOVSLDUP Y2, Y5          // w.r
	VMOVSHDUP Y2, Y6          // w.i
	VSHUFPS $0xB1, Y4, Y4, Y7 // diff swapped
	VMULPS Y6, Y7, Y7
	VFMADDSUB231PS Y5, Y4, Y7 // t = diff * w

	VMOVUPS Y3, (R9)          // out0 = sum
	VMOVUPS Y7, (R12)         // out1 = diff * w

	ADDQ $32, BP
	ADDQ $32, AX
	ADDQ $32, R9
	ADDQ $32, R12
	ADDQ $32, R11
	SUBQ $4, DX
	JMP  stockham_vec_loop

stockham_scalar_contig:
	MOVQ $8, R15              // stride bytes for step==1
	JMP  stockham_scalar_core

stockham_scalar_strided:
	MOVQ BX, R15
	SHLQ $3, R15              // stride bytes = step * 8
	XORQ R11, R11             // twiddle offset bytes

	CMPQ DX, $4
	JL   stockham_scalar_core

stockham_strided_vec_loop:
	CMPQ DX, $4
	JL   stockham_scalar_core

	VMOVUPS (BP), Y0         // a (ptrA)
	VMOVUPS (AX), Y1         // b (ptrB)

	// Gather 4 strided twiddles using running offset
	VMOVSD (R10)(R11*1), X2
	ADDQ R15, R11
	VMOVSD (R10)(R11*1), X3
	ADDQ R15, R11
	VMOVSD (R10)(R11*1), X4
	ADDQ R15, R11
	VMOVSD (R10)(R11*1), X5
	ADDQ R15, R11             // advance to next block
	VPUNPCKLQDQ X3, X2, X2
	VPUNPCKLQDQ X5, X4, X4
	VINSERTF128 $1, X4, Y2, Y2

	VADDPS Y1, Y0, Y3         // sum = a + b
	VSUBPS Y1, Y0, Y4         // diff = a - b

	VMOVSLDUP Y2, Y5          // w.r
	VMOVSHDUP Y2, Y6          // w.i
	VSHUFPS $0xB1, Y4, Y4, Y7 // diff swapped
	VMULPS Y6, Y7, Y7
	VFMADDSUB231PS Y5, Y4, Y7 // t = diff * w

	VMOVUPS Y3, (R9)          // out0 = sum
	VMOVUPS Y7, (R12)         // out1 = diff * w

	ADDQ $32, BP
	ADDQ $32, AX
	ADDQ $32, R9
	ADDQ $32, R12
	SUBQ $4, DX
	JMP  stockham_strided_vec_loop

stockham_scalar_core:
	CMPQ DX, $0
	JLE  stockham_k_done

stockham_scalar_loop:
	MOVSS (BP), X0           // a.r (ptrA)
	MOVSS 4(BP), X1          // a.i
	MOVSS (AX), X2           // b.r (ptrB)
	MOVSS 4(AX), X3          // b.i

	// sum = a + b
	MOVSS X0, X4
	ADDSS X2, X4
	MOVSS X1, X5
	ADDSS X3, X5
	MOVSS X4, (R9)
	MOVSS X5, 4(R9)

	// diff = a - b
	MOVSS X0, X6
	SUBSS X2, X6
	MOVSS X1, X7
	SUBSS X3, X7

	// twiddle (strided)
	MOVSS (R10)(R11*1), X8
	MOVSS 4(R10)(R11*1), X9

	// t = diff * w
	MOVSS X6, X10
	MULSS X8, X10
	MOVSS X7, X11
	MULSS X9, X11
	SUBSS X11, X10

	MOVSS X7, X12
	MULSS X8, X12
	MOVSS X6, X13
	MULSS X9, X13
	ADDSS X13, X12

	MOVSS X10, (R12)
	MOVSS X12, 4(R12)

	ADDQ $8, BP
	ADDQ $8, AX
	ADDQ $8, R9
	ADDQ $8, R12
	ADDQ R15, R11
	DECQ DX
	JNZ  stockham_scalar_loop

stockham_k_done:
	INCQ CX
	JMP  stockham_k_loop

stockham_stage_done:
	// Swap in/out buffers
	MOVQ DI, SI
	MOVQ dst+0(FP), AX
	CMPQ DI, AX
	JE   stockham_out_to_scratch
	MOVQ AX, DI
	JMP  stockham_stage_next

stockham_out_to_scratch:
	MOVQ scratch+72(FP), DI

stockham_stage_next:
	SHRQ $1, R14
	JMP  stockham_stage_loop

stockham_done:
	VZEROUPPER
	MOVQ dst+0(FP), AX
	CMPQ SI, AX
	JE   stockham_return_true

	XORQ CX, CX

stockham_copy_loop:
	CMPQ CX, R13
	JGE  stockham_return_true
	MOVQ (SI)(CX*8), DX
	MOVQ DX, (AX)(CX*8)
	INCQ CX
	JMP  stockham_copy_loop

stockham_return_true:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

stockham_return_false:
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// inverseAVX2Complex64Asm - Inverse FFT for complex64 using AVX2/FMA
// ===========================================================================
// Performs inverse DFT: x[n] = (1/N) * Σ X[k] * exp(+2πi*n*k/N)
//
// The inverse FFT differs from forward in two ways:
//   1. Uses CONJUGATE twiddle factors: conj(w) instead of w
//   2. Scales output by 1/n to ensure round-trip: IFFT(FFT(x)) = x
//
// Conjugate complex multiply: t = conj(w) * b
//   t.real = w.r * b.r + w.i * b.i  (note: + instead of -)
//   t.imag = w.r * b.i - w.i * b.r  (note: - instead of +)
//
// This is achieved by using VFMSUBADD instead of VFMADDSUB:
//   VFMSUBADD: even lanes +, odd lanes - (opposite of VFMADDSUB)
// ===========================================================================
TEXT ·InverseAVX2Complex64Asm(SB), NOSPLIT, $0-121
	// -----------------------------------------------------------------------
	// PHASE 1: Load parameters and validate inputs (same as forward)
	// -----------------------------------------------------------------------
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer
	MOVQ src+32(FP), R13     // R13 = n = len(src)

	// Empty input is valid
	TESTQ R13, R13
	JZ    inv_return_true

	// Validate slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, R13
	JL   inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, R13
	JL   inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, R13
	JL   inv_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, R13
	JL   inv_return_false

	// Trivial case: n=1
	CMPQ R13, $1
	JNE  inv_check_power_of_2
	MOVQ (R9), AX
	MOVQ AX, (R8)
	JMP  inv_return_true

inv_check_power_of_2:
	// Verify power of 2
	MOVQ R13, AX
	LEAQ -1(AX), BX
	TESTQ AX, BX
	JNZ  inv_return_false

	// Minimum size for AVX2
	CMPQ R13, $16
	JL   inv_return_false

	// -----------------------------------------------------------------------
	// PHASE 2: Select working buffer
	// -----------------------------------------------------------------------
	CMPQ R8, R9
	JNE  inv_use_dst_as_work
	MOVQ R11, R8             // In-place: use scratch
	JMP  inv_do_bit_reversal

inv_use_dst_as_work:
	// Out-of-place: use dst directly

inv_do_bit_reversal:
	// -----------------------------------------------------------------------
	// PHASE 3: Bit-reversal permutation
	// -----------------------------------------------------------------------
	XORQ CX, CX

inv_bitrev_loop:
	CMPQ CX, R13
	JGE  inv_bitrev_done
	MOVQ (R12)(CX*8), DX     // DX = bitrev[i]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[i]]
	MOVQ AX, (R8)(CX*8)      // work[i] = src[bitrev[i]]
	INCQ CX
	JMP  inv_bitrev_loop

inv_bitrev_done:
	// -----------------------------------------------------------------------
	// PHASE 4: DIT butterfly stages with CONJUGATE twiddles
	// -----------------------------------------------------------------------
	MOVQ $2, R14             // size = 2

inv_size_loop:
	CMPQ R14, R13
	JG   inv_transform_done

	MOVQ R14, R15
	SHRQ $1, R15             // half = size / 2

	MOVQ R13, AX
	XORQ DX, DX
	DIVQ R14
	MOVQ AX, BX              // step = n / size

	XORQ CX, CX              // base = 0

inv_base_loop:
	CMPQ CX, R13
	JGE  inv_next_size

	// Select AVX2 or scalar path
	CMPQ R15, $4
	JL   inv_scalar_butterflies

	CMPQ BX, $1
	JE   inv_avx2_contiguous
	JMP  inv_avx2_strided

inv_avx2_contiguous:
	// -----------------------------------------------------------------------
	// Inverse AVX2 Contiguous Path (step == 1)
	// -----------------------------------------------------------------------
	XORQ DX, DX

inv_avx2_loop:
	MOVQ R15, AX
	SUBQ DX, AX
	CMPQ AX, $4
	JL   inv_scalar_remainder

	// Compute byte offsets
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI

	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI

	// Load butterfly inputs
	VMOVUPS (R8)(SI*1), Y0   // a
	VMOVUPS (R8)(DI*1), Y1   // b

	// Load twiddles (contiguous)
	MOVQ DX, AX
	SHLQ $3, AX
	VMOVUPS (R10)(AX*1), Y2  // w

	JMP inv_avx2_butterfly

inv_avx2_strided:
	// -----------------------------------------------------------------------
	// Inverse AVX2 Strided Path (step > 1)
	// -----------------------------------------------------------------------
	MOVQ BX, R12
	SHLQ $3, R12             // stride_bytes = step * 8
	XORQ R11, R11            // twiddle_offset = 0
	XORQ DX, DX

inv_avx2_strided_loop:
	MOVQ R15, AX
	SUBQ DX, AX
	CMPQ AX, $8
	JL   inv_avx2_strided_single

	// -----------------------------------------------------------------------
	// Unrolled: Process 8 butterflies with conjugate twiddles
	// -----------------------------------------------------------------------
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI
	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI

	// Load data for both blocks
	VMOVUPS (R8)(SI*1), Y0        // Block 1: a
	VMOVUPS (R8)(DI*1), Y1        // Block 1: b
	VMOVUPS 32(R8)(SI*1), Y7      // Block 2: a
	VMOVUPS 32(R8)(DI*1), Y8      // Block 2: b

	// Gather Block 1 twiddles
	VMOVSD (R10)(R11*1), X2
	LEAQ (R11)(R12*1), AX
	VMOVSD (R10)(AX*1), X3
	ADDQ R12, AX
	VMOVSD (R10)(AX*1), X4
	ADDQ R12, AX
	VMOVSD (R10)(AX*1), X5
	VPUNPCKLQDQ X3, X2, X2
	VPUNPCKLQDQ X5, X4, X4
	VINSERTF128 $1, X4, Y2, Y2   // Y2 = [w0, w1, w2, w3]

	// Gather Block 2 twiddles
	ADDQ R12, AX
	VMOVSD (R10)(AX*1), X9
	ADDQ R12, AX
	VMOVSD (R10)(AX*1), X10
	ADDQ R12, AX
	VMOVSD (R10)(AX*1), X11
	ADDQ R12, AX
	VMOVSD (R10)(AX*1), X12
	VPUNPCKLQDQ X10, X9, X9
	VPUNPCKLQDQ X12, X11, X11
	VINSERTF128 $1, X11, Y9, Y9  // Y9 = [w4, w5, w6, w7]

	// Block 1 Butterfly with CONJUGATE multiply (VFMSUBADD)
	VMOVSLDUP Y2, Y3             // w.r broadcast
	VMOVSHDUP Y2, Y4             // w.i broadcast
	VSHUFPS $0xB1, Y1, Y1, Y6    // b_swapped
	VMULPS Y4, Y6, Y6            // b_swap * w.i
	VFMSUBADD231PS Y3, Y1, Y6    // t = conj(w) * b (note: SUBADD not ADDSUB)
	VADDPS Y6, Y0, Y3            // a'
	VSUBPS Y6, Y0, Y4            // b'
	VMOVUPS Y3, (R8)(SI*1)
	VMOVUPS Y4, (R8)(DI*1)

	// Block 2 Butterfly with CONJUGATE multiply
	VMOVSLDUP Y9, Y10
	VMOVSHDUP Y9, Y11
	VSHUFPS $0xB1, Y8, Y8, Y6
	VMULPS Y11, Y6, Y6
	VFMSUBADD231PS Y10, Y8, Y6   // t = conj(w) * b
	VADDPS Y6, Y7, Y10
	VSUBPS Y6, Y7, Y11
	VMOVUPS Y10, 32(R8)(SI*1)
	VMOVUPS Y11, 32(R8)(DI*1)

	// Advance
	LEAQ (R11)(R12*8), R11
	ADDQ $8, DX
	JMP inv_avx2_strided_loop

inv_avx2_strided_single:
	// Process 4 butterflies with strided twiddles
	CMPQ AX, $4
	JL   inv_scalar_remainder

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI
	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI

	VMOVUPS (R8)(SI*1), Y0
	VMOVUPS (R8)(DI*1), Y1

	// Gather 4 twiddles
	VMOVSD (R10)(R11*1), X2
	LEAQ (R11)(R12*1), AX
	VMOVSD (R10)(AX*1), X3
	ADDQ R12, AX
	VMOVSD (R10)(AX*1), X4
	ADDQ R12, AX
	VMOVSD (R10)(AX*1), X5
	VPUNPCKLQDQ X3, X2, X2
	VPUNPCKLQDQ X5, X4, X4
	VINSERTF128 $1, X4, Y2, Y2

	// Conjugate multiply using VFMSUBADD
	VMOVSLDUP Y2, Y3
	VMOVSHDUP Y2, Y4
	VSHUFPS $0xB1, Y1, Y1, Y6
	VMULPS Y4, Y6, Y6
	VFMSUBADD231PS Y3, Y1, Y6    // t = conj(w) * b
	VADDPS Y6, Y0, Y3
	VSUBPS Y6, Y0, Y4
	VMOVUPS Y3, (R8)(SI*1)
	VMOVUPS Y4, (R8)(DI*1)

	LEAQ (R11)(R12*4), R11
	ADDQ $4, DX
	JMP inv_scalar_remainder

inv_avx2_butterfly:
	// -----------------------------------------------------------------------
	// AVX2 Conjugate Butterfly (contiguous path)
	// -----------------------------------------------------------------------
	// Conjugate complex multiply: t = conj(w) * b
	//   t.real = b.r*w.r + b.i*w.i  (ADD in even lanes)
	//   t.imag = b.i*w.r - b.r*w.i  (SUB in odd lanes)
	//
	// VFMSUBADD231PS: dst = src2*src3 +/- dst
	//   Even lanes: ADD (+ dst)
	//   Odd lanes:  SUB (- dst)
	// This is opposite of forward FFT's VFMADDSUB!

	VMOVSLDUP Y2, Y3             // Y3 = [w.r, w.r, ...]
	VMOVSHDUP Y2, Y4             // Y4 = [w.i, w.i, ...]

	VSHUFPS $0xB1, Y1, Y1, Y6    // Y6 = [b.i, b.r, ...] swapped
	VMULPS Y4, Y6, Y6            // Y6 = [b.i*w.i, b.r*w.i, ...]

	// Key difference: VFMSUBADD instead of VFMADDSUB
	// Result: [b.r*w.r + b.i*w.i, b.i*w.r - b.r*w.i, ...] = conj(w)*b
	VFMSUBADD231PS Y3, Y1, Y6    // Y6 = t = conj(w) * b

	VADDPS Y6, Y0, Y3            // a' = a + t
	VSUBPS Y6, Y0, Y4            // b' = a - t

	VMOVUPS Y3, (R8)(SI*1)
	VMOVUPS Y4, (R8)(DI*1)

	ADDQ $4, DX

	CMPQ BX, $1
	JE   inv_avx2_loop
	JMP  inv_avx2_strided_loop

inv_scalar_remainder:
	// -----------------------------------------------------------------------
	// Scalar Remainder for Inverse: Handle leftover 0-3 butterflies
	// -----------------------------------------------------------------------
	CMPQ DX, R15
	JGE  inv_next_base

inv_scalar_remainder_loop:
	// Compute byte offsets
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI

	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI

	// Load butterfly inputs
	MOVSD (R8)(SI*1), X0     // a
	MOVSD (R8)(DI*1), X1     // b

	// Load twiddle with stride
	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $3, AX
	MOVSD (R10)(AX*1), X2    // w

	// -----------------------------------------------------------------------
	// SSE Scalar Conjugate Complex Multiply: t = conj(w) * b
	// -----------------------------------------------------------------------
	// Conjugate multiply formula:
	//   conj(w) * b = (w.r - i*w.i) * (b.r + i*b.i)
	//              = (w.r*b.r + w.i*b.i) + i*(w.r*b.i - w.i*b.r)
	//
	// Since SSE3 ADDSUBPS has wrong sign pattern for conjugate multiply,
	// we compute both ADD and SUB, then blend the correct lanes.

	MOVSLDUP X2, X3          // X3 = [w.r, w.r]
	MOVSHDUP X2, X4          // X4 = [w.i, w.i]

	MOVAPS X1, X5
	MULPS  X3, X5            // X5 = [b.r*w.r, b.i*w.r]

	MOVAPS X1, X6
	SHUFPS $0xB1, X6, X6     // X6 = [b.i, b.r]
	MULPS  X4, X6            // X6 = [b.i*w.i, b.r*w.i]

	// We need: real = b.r*w.r + b.i*w.i, imag = b.i*w.r - b.r*w.i
	//   X5 = [b.r*w.r, b.i*w.r], X6 = [b.i*w.i, b.r*w.i]
	//   X5+X6: [correct_real, wrong_imag]
	//   X5-X6: [wrong_real, correct_imag]
	// We need to blend: take lane 0 from ADD, lane 1 from SUB

	MOVAPS X5, X7
	ADDPS  X6, X7            // X7 = [real_OK, imag_WRONG]
	MOVAPS X5, X3
	SUBPS  X6, X3            // X3 = [real_WRONG, imag_OK]

	// Blend X7[0] with X3[1] using UNPCKLPS + SHUFPS
	UNPCKLPS X3, X7          // X7 = [X7[0], X3[0], X7[1], X3[1]]
	                         //    = [real_OK, garbage, garbage, imag_OK]
	SHUFPS $0x0C, X7, X7     // imm = 0b00001100 = 0x0C
	                         // X7 = [X7[0], X7[3], ...] = [real_OK, imag_OK]
	MOVAPS X7, X5            // X5 = t = conj(w) * b

	// Butterfly
	MOVAPS X0, X3
	ADDPS  X5, X3            // a' = a + t
	MOVAPS X0, X4
	SUBPS  X5, X4            // b' = a - t

	MOVSD X3, (R8)(SI*1)
	MOVSD X4, (R8)(DI*1)

	INCQ DX
	CMPQ DX, R15
	JL   inv_scalar_remainder_loop

inv_next_base:
	ADDQ R14, CX             // base += size
	JMP  inv_base_loop

inv_scalar_butterflies:
	// -----------------------------------------------------------------------
	// Pure Scalar Path for Inverse (half < 4)
	// -----------------------------------------------------------------------
	// Same conjugate multiply algorithm as inv_scalar_remainder
	XORQ DX, DX

inv_scalar_loop:
	CMPQ DX, R15
	JGE  inv_next_base

	// Compute offsets
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI

	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI

	// Load inputs
	MOVSD (R8)(SI*1), X0     // a
	MOVSD (R8)(DI*1), X1     // b

	// Load twiddle
	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $3, AX
	MOVSD (R10)(AX*1), X2    // w

	// Conjugate multiply: t = conj(w) * b
	MOVSLDUP X2, X3
	MOVSHDUP X2, X4

	MOVAPS X1, X5
	MULPS  X3, X5            // [b.r*w.r, b.i*w.r]

	MOVAPS X1, X6
	SHUFPS $0xB1, X6, X6
	MULPS  X4, X6            // [b.i*w.i, b.r*w.i]

	// Blend for conjugate: ADD[0] with SUB[1]
	MOVAPS X5, X7
	ADDPS  X6, X7
	MOVAPS X5, X3
	SUBPS  X6, X3
	UNPCKLPS X3, X7
	SHUFPS $0x0C, X7, X7     // [real, imag, ...]
	MOVAPS X7, X5            // t

	// Butterfly
	MOVAPS X0, X3
	ADDPS  X5, X3            // a'
	MOVAPS X0, X4
	SUBPS  X5, X4            // b'

	MOVSD X3, (R8)(SI*1)
	MOVSD X4, (R8)(DI*1)

	INCQ DX
	JMP  inv_scalar_loop

inv_next_size:
	SHLQ $1, R14             // size *= 2
	JMP  inv_size_loop

inv_transform_done:
	// -----------------------------------------------------------------------
	// PHASE 5: Copy back (if needed) and Scale by 1/n
	// -----------------------------------------------------------------------
	VZEROUPPER               // Clean AVX state

	// Copy work buffer to dst if we used scratch
	MOVQ dst+0(FP), AX
	CMPQ R8, AX
	JE   inv_scale           // work == dst, skip copy

	XORQ CX, CX
inv_copy_loop:
	CMPQ CX, R13
	JGE  inv_scale
	MOVQ (R8)(CX*8), DX
	MOVQ DX, (AX)(CX*8)
	INCQ CX
	JMP  inv_copy_loop

inv_scale:
	// -----------------------------------------------------------------------
	// PHASE 6: Scale output by 1/n
	// -----------------------------------------------------------------------
	// Inverse FFT requires normalization: x[n] = (1/N) * IDFT_unscaled
	// This ensures: IFFT(FFT(x)) = x
	//
	// We compute scale = 1.0 / n and multiply each complex value by it.
	// Both real and imaginary parts are scaled by the same factor.

	MOVQ dst+0(FP), R8       // Reload dst (may have been clobbered)

	// Compute 1/n as float32
	CVTSQ2SS R13, X0         // X0 = (float32)n
	MOVSS    ·one32(SB), X1  // X1 = 1.0f
	DIVSS    X0, X1          // X1 = 1.0f / n

	// Broadcast scale factor to all 4 lanes
	SHUFPS   $0x00, X1, X1   // X1 = [scale, scale, scale, scale]

	// Scale each element: dst[i] *= scale (both real and imag)
	XORQ CX, CX

inv_scale_loop:
	CMPQ CX, R13
	JGE  inv_return_true

	// Load one complex64 (8 bytes = 64 bits)
	MOVSD (R8)(CX*8), X0

	// Multiply both real and imag by scale
	// Since we broadcast scale to all lanes, this works correctly
	MULPS X1, X0

	// Store back
	MOVSD X0, (R8)(CX*8)

	INCQ CX
	JMP  inv_scale_loop

inv_return_true:
	// Success: inverse transform completed
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

inv_return_false:
	// Failure: fall back to pure Go
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// inverseAVX2StockhamComplex64Asm - Inverse FFT for complex64 (Stockham path)
// ===========================================================================
TEXT ·InverseAVX2StockhamComplex64Asm(SB), NOSPLIT, $0-121
	// -----------------------------------------------------------------------
	// PHASE 1: Load parameters and validate inputs
	// -----------------------------------------------------------------------
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer (unused)
	MOVQ src+32(FP), R13     // R13 = n = len(src)

	// Empty input is valid (no-op)
	TESTQ R13, R13
	JZ    inv_stockham_return_true

	// Validate all slice lengths are >= n
	MOVQ dst+8(FP), AX
	CMPQ AX, R13
	JL   inv_stockham_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, R13
	JL   inv_stockham_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, R13
	JL   inv_stockham_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, R13
	JL   inv_stockham_return_false

	// Trivial case: n=1, just copy
	CMPQ R13, $1
	JNE  inv_stockham_check_power
	MOVQ (R9), AX
	MOVQ AX, (R8)
	JMP  inv_stockham_return_true

inv_stockham_check_power:
	// Verify n is power of 2
	MOVQ R13, AX
	LEAQ -1(AX), BX
	TESTQ AX, BX
	JNZ  inv_stockham_return_false

	// Minimum size for AVX2 vectorization
	CMPQ R13, $16
	JL   inv_stockham_return_false

	// -----------------------------------------------------------------------
	// PHASE 2: Select buffers (Stockham uses ping-pong)
	// -----------------------------------------------------------------------
	MOVQ R9, SI               // SI = in (src)
	MOVQ R8, DI               // DI = out (dst)
	CMPQ R8, R9
	JNE  inv_stockham_out_ready
	MOVQ R11, DI              // In-place: first out = scratch

inv_stockham_out_ready:
	// m starts at n and halves each stage
	MOVQ R13, R14             // R14 = m

inv_stockham_stage_loop:
	CMPQ R14, $1
	JLE  inv_stockham_done

	// step = n / m
	MOVQ R13, AX
	XORQ DX, DX
	DIVQ R14
	MOVQ AX, BX               // BX = step (also group count)

	XORQ CX, CX               // k = 0

inv_stockham_k_loop:
	CMPQ CX, BX
	JGE  inv_stockham_stage_done

	// half = m / 2
	MOVQ R14, R15
	SHRQ $1, R15

	// baseElem = k * m
	MOVQ CX, AX
	IMULQ R14, AX

	// ptrA = in + baseElem*8
	SHLQ $3, AX              // AX = baseElem * 8
	LEAQ (SI)(AX*1), BP      // BP = ptrA = in + baseElem*8

	// ptrB = ptrA + half*8
	MOVQ R15, AX
	SHLQ $3, AX              // AX = half * 8
	ADDQ BP, AX              // AX = ptrB = ptrA + half*8

	// outBaseElem = k * half
	MOVQ CX, DX
	IMULQ R15, DX

	// ptrOut0 = out + outBaseElem*8
	SHLQ $3, DX              // DX = outBaseElem * 8
	LEAQ (DI)(DX*1), R9      // R9 = ptrOut0 (DI = current output buffer)

	// ptrOut1 = ptrOut0 + (n/2)*8
	MOVQ R13, R12
	SHLQ $2, R12             // R12 = n * 4 bytes
	LEAQ (R9)(R12*1), R12    // R12 = ptrOut1

	// remaining = half
	MOVQ R15, DX             // DX = half

	// Fast path for contiguous twiddles (step == 1)
	CMPQ BX, $1
	JNE  inv_stockham_scalar_strided

	// twiddle offset for contiguous path
	XORQ R11, R11
	CMPQ DX, $4
	JL   inv_stockham_scalar_contig

inv_stockham_vec_loop:
	CMPQ DX, $4
	JL   inv_stockham_scalar_contig

	VMOVUPS (BP), Y0         // a (ptrA)
	VMOVUPS (AX), Y1         // b (ptrB)
	VMOVUPS (R10)(R11*1), Y2  // twiddle

	VADDPS Y1, Y0, Y3         // sum = a + b
	VSUBPS Y1, Y0, Y4         // diff = a - b

	// Conjugate multiply: diff * conj(w)
	VMOVSLDUP Y2, Y5          // w.r
	VMOVSHDUP Y2, Y6          // w.i
	VSHUFPS $0xB1, Y4, Y4, Y7 // diff swapped
	VMULPS Y6, Y7, Y7
	VFMSUBADD231PS Y5, Y4, Y7 // t = diff * conj(w)

	VMOVUPS Y3, (R9)          // out0 = sum
	VMOVUPS Y7, (R12)         // out1 = diff * conj(w)

	ADDQ $32, BP
	ADDQ $32, AX
	ADDQ $32, R9
	ADDQ $32, R12
	ADDQ $32, R11
	SUBQ $4, DX
	JMP  inv_stockham_vec_loop

inv_stockham_scalar_contig:
	MOVQ $8, R15              // stride bytes for step==1
	JMP  inv_stockham_scalar_core

inv_stockham_scalar_strided:
	MOVQ BX, R15
	SHLQ $3, R15              // stride bytes = step * 8
	XORQ R11, R11             // twiddle offset bytes

	CMPQ DX, $4
	JL   inv_stockham_scalar_core

inv_stockham_strided_vec_loop:
	CMPQ DX, $4
	JL   inv_stockham_scalar_core

	VMOVUPS (BP), Y0         // a (ptrA)
	VMOVUPS (AX), Y1         // b (ptrB)

	// Gather 4 strided twiddles using running offset
	VMOVSD (R10)(R11*1), X2
	ADDQ R15, R11
	VMOVSD (R10)(R11*1), X3
	ADDQ R15, R11
	VMOVSD (R10)(R11*1), X4
	ADDQ R15, R11
	VMOVSD (R10)(R11*1), X5
	ADDQ R15, R11             // advance to next block
	VPUNPCKLQDQ X3, X2, X2
	VPUNPCKLQDQ X5, X4, X4
	VINSERTF128 $1, X4, Y2, Y2

	VADDPS Y1, Y0, Y3         // sum = a + b
	VSUBPS Y1, Y0, Y4         // diff = a - b

	// Conjugate multiply: diff * conj(w)
	VMOVSLDUP Y2, Y5          // w.r
	VMOVSHDUP Y2, Y6          // w.i
	VSHUFPS $0xB1, Y4, Y4, Y7 // diff swapped
	VMULPS Y6, Y7, Y7
	VFMSUBADD231PS Y5, Y4, Y7 // t = diff * conj(w)

	VMOVUPS Y3, (R9)          // out0 = sum
	VMOVUPS Y7, (R12)         // out1 = diff * conj(w)

	ADDQ $32, BP
	ADDQ $32, AX
	ADDQ $32, R9
	ADDQ $32, R12
	SUBQ $4, DX
	JMP  inv_stockham_strided_vec_loop

inv_stockham_scalar_core:
	CMPQ DX, $0
	JLE  inv_stockham_k_done

inv_stockham_scalar_loop:
	MOVSS (BP), X0           // a.r (ptrA)
	MOVSS 4(BP), X1          // a.i
	MOVSS (AX), X2           // b.r (ptrB)
	MOVSS 4(AX), X3          // b.i

	// sum = a + b
	MOVSS X0, X4
	ADDSS X2, X4
	MOVSS X1, X5
	ADDSS X3, X5
	MOVSS X4, (R9)
	MOVSS X5, 4(R9)

	// diff = a - b
	MOVSS X0, X6
	SUBSS X2, X6
	MOVSS X1, X7
	SUBSS X3, X7

	// twiddle (conjugate, strided)
	MOVSS (R10)(R11*1), X8
	MOVSS 4(R10)(R11*1), X9

	// t = diff * conj(w)
	MOVSS X6, X10
	MULSS X8, X10
	MOVSS X7, X11
	MULSS X9, X11
	ADDSS X11, X10

	MOVSS X7, X12
	MULSS X8, X12
	MOVSS X6, X13
	MULSS X9, X13
	SUBSS X13, X12

	MOVSS X10, (R12)
	MOVSS X12, 4(R12)

	ADDQ $8, BP
	ADDQ $8, AX
	ADDQ $8, R9
	ADDQ $8, R12
	ADDQ R15, R11
	DECQ DX
	JNZ  inv_stockham_scalar_loop

inv_stockham_k_done:
	INCQ CX
	JMP  inv_stockham_k_loop

inv_stockham_stage_done:
	// Swap in/out buffers
	MOVQ DI, SI
	MOVQ dst+0(FP), AX
	CMPQ DI, AX
	JE   inv_stockham_out_to_scratch
	MOVQ AX, DI
	JMP  inv_stockham_stage_next

inv_stockham_out_to_scratch:
	MOVQ scratch+72(FP), DI

inv_stockham_stage_next:
	SHRQ $1, R14
	JMP  inv_stockham_stage_loop

inv_stockham_done:
	VZEROUPPER
	MOVQ dst+0(FP), AX
	CMPQ SI, AX
	JE   inv_stockham_scale

	XORQ CX, CX

inv_stockham_copy_loop:
	CMPQ CX, R13
	JGE  inv_stockham_scale
	MOVQ (SI)(CX*8), DX
	MOVQ DX, (AX)(CX*8)
	INCQ CX
	JMP  inv_stockham_copy_loop

inv_stockham_scale:
	// scale by 1/n
	CVTSQ2SS R13, X0
	MOVSS    ·one32(SB), X1
	DIVSS    X0, X1
	SHUFPS   $0x00, X1, X1

	XORQ CX, CX

inv_stockham_scale_loop:
	CMPQ CX, R13
	JGE  inv_stockham_return_true
	MOVSD (AX)(CX*8), X0
	MULPS X1, X0
	MOVSD X0, (AX)(CX*8)
	INCQ CX
	JMP  inv_stockham_scale_loop

inv_stockham_return_true:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

inv_stockham_return_false:
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// forwardSSE2Complex64Asm - Forward FFT for complex64 using SSE2
// ===========================================================================
// SSE2-only implementation of the radix-2 DIT FFT.
// Vectorizes contiguous twiddle stages with 2-complex (XMM) butterflies.
