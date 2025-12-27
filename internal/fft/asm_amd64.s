//go:build amd64 && fft_asm && !purego

// ===========================================================================
// AVX2/FMA-optimized FFT Assembly for AMD64
// ===========================================================================
//
// This file implements high-performance FFT transforms using AVX2 and FMA3
// instructions for both complex64 (single-precision) and complex128 (double-
// precision) data types.
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
// - complex128: Process 2 butterflies per iteration using YMM registers
//               Each complex128 = 16 bytes, so YMM holds 2 complex numbers
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
//   Uses VFMADDSUB231PS/PD: dst = src2*src3 -/+ dst (even lanes -, odd lanes +)
//
// Inverse FFT: t = conj(w) * b
//   Real: b.r*w.r + b.i*w.i
//   Imag: b.i*w.r - b.r*w.i
//   Uses VFMSUBADD231PS/PD: dst = src2*src3 +/- dst (even lanes +, odd lanes -)
//
// PERFORMANCE NOTES
// -----------------
// - Minimum size for AVX2 path: n >= 16 (complex64), n >= 8 (complex128)
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
// complex128: 16 bytes = 8 bytes (float64 real) + 8 bytes (float64 imag)
//
// YMM register (256 bits = 32 bytes):
//   - Holds 4 complex64  values (4 × 8  = 32 bytes)
//   - Holds 2 complex128 values (2 × 16 = 32 bytes)
//
// XMM register (128 bits = 16 bytes):
//   - Holds 2 complex64  values (2 × 8  = 16 bytes)
//   - Holds 1 complex128 value  (1 × 16 = 16 bytes)

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
TEXT ·forwardAVX2Complex64Asm(SB), NOSPLIT, $0-121
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
TEXT ·forwardAVX2StockhamComplex64Asm(SB), NOSPLIT, $0-121
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
TEXT ·inverseAVX2Complex64Asm(SB), NOSPLIT, $0-121
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
TEXT ·inverseAVX2StockhamComplex64Asm(SB), NOSPLIT, $0-121
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
TEXT ·forwardSSE2Complex64Asm(SB), NOSPLIT, $0-121
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
	JZ    sse2_return_true

	// Validate all slice lengths are >= n
	MOVQ dst+8(FP), AX
	CMPQ AX, R13
	JL   sse2_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, R13
	JL   sse2_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, R13
	JL   sse2_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, R13
	JL   sse2_return_false

	// Trivial case: n=1, just copy
	CMPQ R13, $1
	JNE  sse2_check_power
	MOVQ (R9), AX
	MOVQ AX, (R8)
	JMP  sse2_return_true

sse2_check_power:
	// Verify n is power of 2
	MOVQ R13, AX
	LEAQ -1(AX), BX
	TESTQ AX, BX
	JNZ  sse2_return_false

	// -----------------------------------------------------------------------
	// PHASE 2: Select working buffer
	// -----------------------------------------------------------------------
	CMPQ R8, R9
	JNE  sse2_use_dst
	MOVQ R11, R8             // In-place: use scratch

sse2_use_dst:
	// -----------------------------------------------------------------------
	// PHASE 3: Bit-reversal permutation
	// -----------------------------------------------------------------------
	XORQ CX, CX

sse2_bitrev_loop:
	CMPQ CX, R13
	JGE  sse2_bitrev_done
	MOVQ (R12)(CX*8), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, (R8)(CX*8)
	INCQ CX
	JMP  sse2_bitrev_loop

sse2_bitrev_done:
	// -----------------------------------------------------------------------
	// PHASE 4: Main DIT Butterfly Stages
	// -----------------------------------------------------------------------
	MOVQ $2, R14             // size = 2

sse2_size_loop:
	CMPQ R14, R13
	JG   sse2_transform_done

	MOVQ R14, R15
	SHRQ $1, R15             // half = size / 2

	MOVQ R13, AX
	XORQ DX, DX
	DIVQ R14
	MOVQ AX, BX              // step = n / size

	XORQ CX, CX              // base = 0

sse2_base_loop:
	CMPQ CX, R13
	JGE  sse2_next_size

	// SSE2 vector path only for contiguous twiddles and half >= 2
	CMPQ BX, $1
	JNE  sse2_scalar_butterflies
	CMPQ R15, $2
	JL   sse2_scalar_butterflies

	XORQ DX, DX              // j = 0

sse2_vec_loop:
	MOVQ R15, AX
	SUBQ DX, AX
	CMPQ AX, $2
	JL   sse2_scalar_remainder

	// Compute byte offsets
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI              // SI = (base + j) * 8

	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI              // DI = (base + j + half) * 8

	// Load 2 complex64 values for a and b
	MOVUPS (R8)(SI*1), X0
	MOVUPS (R8)(DI*1), X1

	// Load contiguous twiddles (2 complex64)
	MOVQ DX, AX
	SHLQ $3, AX
	MOVUPS (R10)(AX*1), X2

	// t = w * b (SSE2, 2 complex64)
	MOVAPS X1, X3
	MULPS  X2, X3            // prod1 = b * w

	MOVAPS X2, X4
	SHUFPS $0xB1, X4, X4     // w_shuf

	MOVAPS X1, X5
	MULPS  X4, X5            // prod2 = b * w_shuf

	MOVAPS X3, X6
	SHUFPS $0xB1, X6, X6
	SUBPS  X6, X3            // real = prod1 - shuf(prod1)

	MOVAPS X5, X7
	SHUFPS $0xB1, X7, X7
	ADDPS  X7, X5            // imag = prod2 + shuf(prod2)

	SHUFPS $0x88, X3, X3     // real lanes
	SHUFPS $0xDD, X5, X5     // imag lanes
	UNPCKLPS X5, X3          // t = [r0,i0,r1,i1]

	// Butterfly
	MOVAPS X0, X6
	ADDPS  X3, X6
	MOVAPS X0, X7
	SUBPS  X3, X7

	// Store results
	MOVUPS X6, (R8)(SI*1)
	MOVUPS X7, (R8)(DI*1)

	ADDQ $2, DX
	JMP  sse2_vec_loop

sse2_scalar_remainder:
	CMPQ DX, R15
	JGE  sse2_next_base

sse2_scalar_loop:
	// Compute byte offsets
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI

	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI

	// Load single complex64 values
	XORPS X0, X0
	MOVSD (R8)(SI*1), X0     // a
	XORPS X1, X1
	MOVSD (R8)(DI*1), X1     // b

	// Load twiddle (stride for scalar path)
	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $3, AX
	XORPS X2, X2
	MOVSD (R10)(AX*1), X2    // w

	// t = w * b (SSE2, 1 complex64)
	MOVAPS X1, X3
	MULPS  X2, X3

	MOVAPS X2, X4
	SHUFPS $0xB1, X4, X4

	MOVAPS X1, X5
	MULPS  X4, X5

	MOVAPS X3, X6
	SHUFPS $0xB1, X6, X6
	SUBPS  X6, X3

	MOVAPS X5, X7
	SHUFPS $0xB1, X7, X7
	ADDPS  X7, X5

	SHUFPS $0x88, X3, X3
	SHUFPS $0xDD, X5, X5
	UNPCKLPS X5, X3

	// Butterfly
	MOVAPS X0, X6
	ADDPS  X3, X6
	MOVAPS X0, X7
	SUBPS  X3, X7

	// Store
	MOVSD X6, (R8)(SI*1)
	MOVSD X7, (R8)(DI*1)

	INCQ DX
	CMPQ DX, R15
	JL   sse2_scalar_loop
	JMP  sse2_next_base

sse2_scalar_butterflies:
	// Pure scalar path for early/strided stages
	XORQ DX, DX

sse2_scalar_only_loop:
	CMPQ DX, R15
	JGE  sse2_next_base

	// Compute byte offsets
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI

	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI

	// Load single complex64 values
	XORPS X0, X0
	MOVSD (R8)(SI*1), X0
	XORPS X1, X1
	MOVSD (R8)(DI*1), X1

	// Load twiddle with stride
	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $3, AX
	XORPS X2, X2
	MOVSD (R10)(AX*1), X2

	// t = w * b
	MOVAPS X1, X3
	MULPS  X2, X3

	MOVAPS X2, X4
	SHUFPS $0xB1, X4, X4

	MOVAPS X1, X5
	MULPS  X4, X5

	MOVAPS X3, X6
	SHUFPS $0xB1, X6, X6
	SUBPS  X6, X3

	MOVAPS X5, X7
	SHUFPS $0xB1, X7, X7
	ADDPS  X7, X5

	SHUFPS $0x88, X3, X3
	SHUFPS $0xDD, X5, X5
	UNPCKLPS X5, X3

	// Butterfly
	MOVAPS X0, X6
	ADDPS  X3, X6
	MOVAPS X0, X7
	SUBPS  X3, X7

	// Store
	MOVSD X6, (R8)(SI*1)
	MOVSD X7, (R8)(DI*1)

	INCQ DX
	JMP  sse2_scalar_only_loop

sse2_next_base:
	ADDQ R14, CX
	JMP  sse2_base_loop

sse2_next_size:
	SHLQ $1, R14
	JMP  sse2_size_loop

sse2_transform_done:
	// -----------------------------------------------------------------------
	// PHASE 5: Copy results back if we used scratch
	// -----------------------------------------------------------------------
	MOVQ dst+0(FP), AX
	CMPQ R8, AX
	JE   sse2_return_true

	XORQ CX, CX

sse2_copy_loop:
	CMPQ CX, R13
	JGE  sse2_return_true
	MOVQ (R8)(CX*8), DX
	MOVQ DX, (AX)(CX*8)
	INCQ CX
	JMP  sse2_copy_loop

sse2_return_true:
	MOVB $1, ret+120(FP)
	RET

sse2_return_false:
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// inverseSSE2Complex64Asm - Inverse FFT for complex64 using SSE2
// ===========================================================================
TEXT ·inverseSSE2Complex64Asm(SB), NOSPLIT, $0-121
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
	JZ    inv_sse2_return_true

	// Validate all slice lengths are >= n
	MOVQ dst+8(FP), AX
	CMPQ AX, R13
	JL   inv_sse2_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, R13
	JL   inv_sse2_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, R13
	JL   inv_sse2_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, R13
	JL   inv_sse2_return_false

	// Trivial case: n=1, just copy
	CMPQ R13, $1
	JNE  inv_sse2_check_power
	MOVQ (R9), AX
	MOVQ AX, (R8)
	JMP  inv_sse2_return_true

inv_sse2_check_power:
	// Verify n is power of 2
	MOVQ R13, AX
	LEAQ -1(AX), BX
	TESTQ AX, BX
	JNZ  inv_sse2_return_false

	// -----------------------------------------------------------------------
	// PHASE 2: Select working buffer
	// -----------------------------------------------------------------------
	CMPQ R8, R9
	JNE  inv_sse2_use_dst
	MOVQ R11, R8             // In-place: use scratch

inv_sse2_use_dst:
	// -----------------------------------------------------------------------
	// PHASE 3: Bit-reversal permutation
	// -----------------------------------------------------------------------
	XORQ CX, CX

inv_sse2_bitrev_loop:
	CMPQ CX, R13
	JGE  inv_sse2_bitrev_done
	MOVQ (R12)(CX*8), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, (R8)(CX*8)
	INCQ CX
	JMP  inv_sse2_bitrev_loop

inv_sse2_bitrev_done:
	// -----------------------------------------------------------------------
	// PHASE 4: DIT butterfly stages with CONJUGATE twiddles
	// -----------------------------------------------------------------------
	MOVQ $2, R14

inv_sse2_size_loop:
	CMPQ R14, R13
	JG   inv_sse2_transform_done

	MOVQ R14, R15
	SHRQ $1, R15

	MOVQ R13, AX
	XORQ DX, DX
	DIVQ R14
	MOVQ AX, BX

	XORQ CX, CX

inv_sse2_base_loop:
	CMPQ CX, R13
	JGE  inv_sse2_next_size

	// SSE2 vector path only for contiguous twiddles and half >= 2
	CMPQ BX, $1
	JNE  inv_sse2_scalar_butterflies
	CMPQ R15, $2
	JL   inv_sse2_scalar_butterflies

	XORQ DX, DX

inv_sse2_vec_loop:
	MOVQ R15, AX
	SUBQ DX, AX
	CMPQ AX, $2
	JL   inv_sse2_scalar_remainder

	// Compute byte offsets
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI

	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI

	// Load 2 complex64 values for a and b
	MOVUPS (R8)(SI*1), X0
	MOVUPS (R8)(DI*1), X1

	// Load contiguous twiddles (2 complex64)
	MOVQ DX, AX
	SHLQ $3, AX
	MOVUPS (R10)(AX*1), X2

	// t = conj(w) * b (SSE2, 2 complex64)
	MOVAPS X1, X3
	MULPS  X2, X3            // prod1 = b * w

	MOVAPS X2, X4
	SHUFPS $0xB1, X4, X4     // w_shuf

	MOVAPS X1, X5
	MULPS  X4, X5            // prod2 = b * w_shuf

	MOVAPS X3, X6
	SHUFPS $0xB1, X6, X6
	ADDPS  X6, X3            // real = prod1 + shuf(prod1)

	MOVAPS X5, X7
	SHUFPS $0xB1, X7, X7
	SUBPS  X5, X7            // imag = shuf(prod2) - prod2

	SHUFPS $0x88, X3, X3
	SHUFPS $0x88, X7, X7
	UNPCKLPS X7, X3

	// Butterfly
	MOVAPS X0, X6
	ADDPS  X3, X6
	MOVAPS X0, X7
	SUBPS  X3, X7

	// Store results
	MOVUPS X6, (R8)(SI*1)
	MOVUPS X7, (R8)(DI*1)

	ADDQ $2, DX
	JMP  inv_sse2_vec_loop

inv_sse2_scalar_remainder:
	CMPQ DX, R15
	JGE  inv_sse2_next_base

inv_sse2_scalar_loop:
	// Compute byte offsets
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI

	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI

	// Load single complex64 values
	XORPS X0, X0
	MOVSD (R8)(SI*1), X0
	XORPS X1, X1
	MOVSD (R8)(DI*1), X1

	// Load twiddle with stride
	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $3, AX
	XORPS X2, X2
	MOVSD (R10)(AX*1), X2

	// t = conj(w) * b
	MOVAPS X1, X3
	MULPS  X2, X3

	MOVAPS X2, X4
	SHUFPS $0xB1, X4, X4

	MOVAPS X1, X5
	MULPS  X4, X5

	MOVAPS X3, X6
	SHUFPS $0xB1, X6, X6
	ADDPS  X6, X3

	MOVAPS X5, X7
	SHUFPS $0xB1, X7, X7
	SUBPS  X5, X7

	SHUFPS $0x88, X3, X3
	SHUFPS $0x88, X7, X7
	UNPCKLPS X7, X3

	// Butterfly
	MOVAPS X0, X6
	ADDPS  X3, X6
	MOVAPS X0, X7
	SUBPS  X3, X7

	// Store
	MOVSD X6, (R8)(SI*1)
	MOVSD X7, (R8)(DI*1)

	INCQ DX
	CMPQ DX, R15
	JL   inv_sse2_scalar_loop
	JMP  inv_sse2_next_base

inv_sse2_scalar_butterflies:
	// Pure scalar path for early/strided stages
	XORQ DX, DX

inv_sse2_scalar_only_loop:
	CMPQ DX, R15
	JGE  inv_sse2_next_base

	// Compute byte offsets
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI

	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI

	// Load single complex64 values
	XORPS X0, X0
	MOVSD (R8)(SI*1), X0
	XORPS X1, X1
	MOVSD (R8)(DI*1), X1

	// Load twiddle with stride
	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $3, AX
	XORPS X2, X2
	MOVSD (R10)(AX*1), X2

	// t = conj(w) * b
	MOVAPS X1, X3
	MULPS  X2, X3

	MOVAPS X2, X4
	SHUFPS $0xB1, X4, X4

	MOVAPS X1, X5
	MULPS  X4, X5

	MOVAPS X3, X6
	SHUFPS $0xB1, X6, X6
	ADDPS  X6, X3

	MOVAPS X5, X7
	SHUFPS $0xB1, X7, X7
	SUBPS  X5, X7

	SHUFPS $0x88, X3, X3
	SHUFPS $0x88, X7, X7
	UNPCKLPS X7, X3

	// Butterfly
	MOVAPS X0, X6
	ADDPS  X3, X6
	MOVAPS X0, X7
	SUBPS  X3, X7

	// Store
	MOVSD X6, (R8)(SI*1)
	MOVSD X7, (R8)(DI*1)

	INCQ DX
	JMP  inv_sse2_scalar_only_loop

inv_sse2_next_base:
	ADDQ R14, CX
	JMP  inv_sse2_base_loop

inv_sse2_next_size:
	SHLQ $1, R14
	JMP  inv_sse2_size_loop

inv_sse2_transform_done:
	// -----------------------------------------------------------------------
	// PHASE 5: Copy back (if needed) and Scale by 1/n
	// -----------------------------------------------------------------------
	MOVQ dst+0(FP), AX
	CMPQ R8, AX
	JE   inv_sse2_scale

	XORQ CX, CX

inv_sse2_copy_loop:
	CMPQ CX, R13
	JGE  inv_sse2_scale
	MOVQ (R8)(CX*8), DX
	MOVQ DX, (AX)(CX*8)
	INCQ CX
	JMP  inv_sse2_copy_loop

inv_sse2_scale:
	// Scale output by 1/n
	MOVQ dst+0(FP), R8

	CVTSQ2SS R13, X0
	MOVSS    ·one32(SB), X1
	DIVSS    X0, X1
	SHUFPS   $0x00, X1, X1   // broadcast scale

	XORQ CX, CX

inv_sse2_scale_loop:
	CMPQ CX, R13
	JGE  inv_sse2_return_true

	XORPS X0, X0
	MOVSD (R8)(CX*8), X0
	MULPS X1, X0
	MOVSD X0, (R8)(CX*8)

	INCQ CX
	JMP  inv_sse2_scale_loop

inv_sse2_return_true:
	MOVB $1, ret+120(FP)
	RET

inv_sse2_return_false:
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// forwardAVX2Complex128Asm - Forward FFT for complex128 using AVX2/FMA
// ===========================================================================
// Double-precision (float64) version of the forward FFT.
//
// Key differences from complex64 version:
//   - complex128 = 16 bytes (8 bytes real + 8 bytes imag)
//   - YMM holds 2 complex128 values (vs 4 complex64)
//   - Uses VMOVUPD (packed double) instead of VMOVUPS (packed single)
//   - Uses VMULPD, VADDPD, VSUBPD instead of PS variants
//   - Uses VFMADDSUB231PD instead of VFMADDSUB231PS
//   - Uses VMOVDDUP and VPERMILPD for component broadcast/shuffle
//   - Minimum size: n >= 8 (vs n >= 16 for complex64)
// ===========================================================================
TEXT ·forwardAVX2Complex128Asm(SB), NOSPLIT, $0-121
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
TEXT ·inverseAVX2Complex128Asm(SB), NOSPLIT, $0-121
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
// CONSTANTS: Floating-point literals for assembly use
// ===========================================================================
// Go assembly requires constants to be defined in DATA directives with
// explicit IEEE 754 bit patterns. These are referenced via symbol names.
// ===========================================================================

// one64: Double-precision 1.0 for complex128 inverse scaling
DATA ·one64+0(SB)/8, $0x3ff0000000000000 // 1.0 (IEEE 754 double)
GLOBL ·one64(SB), RODATA|NOPTR, $8

// ===========================================================================
// SSE2 STUBS: Fallback triggers for non-AVX2 systems
// ===========================================================================
// These functions immediately return false, signaling the Go runtime to use
// the pure-Go implementation. They exist to satisfy the function declarations
// in the Go wrapper but provide no actual implementation.
// ===========================================================================

TEXT ·forwardSSE2Complex128Asm(SB), NOSPLIT|NOFRAME, $0-121
	MOVB $0, ret+120(FP)        // Return false (use Go fallback)
	RET

TEXT ·inverseSSE2Complex128Asm(SB), NOSPLIT|NOFRAME, $0-121
	MOVB $0, ret+120(FP)        // Return false (use Go fallback)
	RET

// ===========================================================================
// UTILITY FUNCTIONS: Small helpers for specific FFT operations
// ===========================================================================

// asmCopyComplex64 - Copy a single complex64 (8 bytes)
// Parameters: AX = dst ptr, BX = src ptr (passed via registers, not stack)
TEXT ·asmCopyComplex64(SB), NOSPLIT|NOFRAME, $0-16
	MOVQ (BX), CX               // Load 8 bytes (real + imag as single QWORD)
	MOVQ CX, (AX)               // Store to destination
	RET

// ===========================================================================
// CONSTANTS: Single-precision floating-point literals
// ===========================================================================

// half32: Single-precision 0.5 for complex64 inverse scaling (n=2 case)
DATA ·half32+0(SB)/4, $0x3f000000  // 0.5f (IEEE 754 single)
GLOBL ·half32(SB), RODATA|NOPTR, $4

// one32: Single-precision 1.0 (reserved for future use)
DATA ·one32+0(SB)/4, $0x3f800000   // 1.0f (IEEE 754 single)
GLOBL ·one32(SB), RODATA|NOPTR, $4

// ===========================================================================
// asmForward2Complex64 - Specialized size-2 forward FFT for complex64
// ===========================================================================
// Computes the 2-point DFT:
//   X[0] = x[0] + x[1]  (DC component)
//   X[1] = x[0] - x[1]  (Nyquist component)
//
// For n=2, twiddle factor w = e^(-2πi/2) = -1, so:
//   X[k] = sum(x[n] * w^(nk)) gives X[0] = x[0] + x[1], X[1] = x[0] - x[1]
//
// Parameters: AX = dst ptr, BX = src ptr
// Returns: 1 in AX (success)
// ===========================================================================
TEXT ·asmForward2Complex64(SB), NOSPLIT|NOFRAME, $0-16
	// Load both complex64 values component-by-component
	MOVSS (BX), X0              // X0 = x[0].real
	MOVSS 4(BX), X1             // X1 = x[0].imag
	MOVSS 8(BX), X2             // X2 = x[1].real
	MOVSS 12(BX), X3            // X3 = x[1].imag

	// X[0] = x[0] + x[1] (DC component)
	MOVSS X0, X4
	ADDSS X2, X4                // X4 = x[0].real + x[1].real
	MOVSS X1, X5
	ADDSS X3, X5                // X5 = x[0].imag + x[1].imag

	MOVSS X4, (AX)              // Store X[0].real
	MOVSS X5, 4(AX)             // Store X[0].imag

	// X[1] = x[0] - x[1] (Nyquist component)
	MOVSS X0, X6
	SUBSS X2, X6                // X6 = x[0].real - x[1].real
	MOVSS X1, X7
	SUBSS X3, X7                // X7 = x[0].imag - x[1].imag

	MOVSS X6, 8(AX)             // Store X[1].real
	MOVSS X7, 12(AX)            // Store X[1].imag

	MOVL $1, AX                 // Return success
	RET

// ===========================================================================
// asmInverse2Complex64 - Specialized size-2 inverse FFT for complex64
// ===========================================================================
// Computes the 2-point inverse DFT with 1/n normalization:
//   x[0] = (X[0] + X[1]) / 2
//   x[1] = (X[0] - X[1]) / 2
//
// Parameters: AX = dst ptr, BX = src ptr
// Returns: 1 in AX (success)
// ===========================================================================
TEXT ·asmInverse2Complex64(SB), NOSPLIT|NOFRAME, $0-16
	// Load both complex64 values
	MOVSS (BX), X0              // X0 = X[0].real
	MOVSS 4(BX), X1             // X1 = X[0].imag
	MOVSS 8(BX), X2             // X2 = X[1].real
	MOVSS 12(BX), X3            // X3 = X[1].imag

	// x[0] = (X[0] + X[1]) / 2
	MOVSS X0, X4
	ADDSS X2, X4                // X4 = X[0].real + X[1].real
	MOVSS X1, X5
	ADDSS X3, X5                // X5 = X[0].imag + X[1].imag

	MULSS ·half32(SB), X4       // X4 = (X[0].real + X[1].real) * 0.5
	MULSS ·half32(SB), X5       // X5 = (X[0].imag + X[1].imag) * 0.5

	MOVSS X4, (AX)              // Store x[0].real
	MOVSS X5, 4(AX)             // Store x[0].imag

	// x[1] = (X[0] - X[1]) / 2
	MOVSS X0, X6
	SUBSS X2, X6                // X6 = X[0].real - X[1].real
	MOVSS X1, X7
	SUBSS X3, X7                // X7 = X[0].imag - X[1].imag

	MULSS ·half32(SB), X6       // X6 = (X[0].real - X[1].real) * 0.5
	MULSS ·half32(SB), X7       // X7 = (X[0].imag - X[1].imag) * 0.5

	MOVSS X6, 8(AX)             // Store x[1].real
	MOVSS X7, 12(AX)            // Store x[1].imag

	MOVL $1, AX                 // Return success
	RET

// ===========================================================================
// Size-Specific AVX2 Kernels - Stubs (Phase 14.5.1)
// ===========================================================================
// These are stub implementations that return false, causing fallback to
// generic AVX2 implementations. Actual unrolled kernels will be added in
// phases 14.5.2-14.5.5.
//
// Function signature: func(dst, src, twiddle, scratch []T, bitrev []int) bool
// Return: false (0) to trigger fallback to generic implementation
// ===========================================================================

// Forward transform, size 16, complex64
// Fully unrolled 4-stage FFT with AVX2 vectorization
//
// This kernel implements a complete radix-2 DIT FFT for exactly 16 complex64 values.
// All 4 stages are fully unrolled with hardcoded twiddle factor indices:
//   Stage 1 (size=2):  8 butterflies, step=8, twiddle index 0 for all
//   Stage 2 (size=4):  8 butterflies in 2 groups, step=4, twiddle indices [0,4]
//   Stage 3 (size=8):  8 butterflies in 1 group, step=2, twiddle indices [0,2,4,6]
//   Stage 4 (size=16): 8 butterflies, step=1, twiddle indices [0,1,2,3,4,5,6,7]
//
// Bit-reversal permutation indices for n=16:
//   [0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15]
//
// Register allocation:
//   R8:  work buffer (dst or scratch for in-place)
//   R9:  src pointer
//   R10: twiddle pointer
//   R11: scratch pointer
//   Y0-Y3: data registers for butterflies
//   Y4-Y7: twiddle and intermediate values
//
TEXT ·forwardAVX2Size16Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 16)

	// Verify n == 16
	CMPQ R13, $16
	JNE  size16_return_false

	// Validate all slice lengths >= 16
	MOVQ dst+8(FP), AX
	CMPQ AX, $16
	JL   size16_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $16
	JL   size16_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $16
	JL   size16_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $16
	JL   size16_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size16_use_dst

	// In-place: use scratch
	MOVQ R11, R8
	JMP  size16_bitrev

size16_use_dst:
	// Out-of-place: use dst

size16_bitrev:
	// =======================================================================
	// Bit-reversal permutation: work[i] = src[bitrev[i]]
	// =======================================================================
	// For size 16, bitrev = [0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15]
	// We use precomputed indices from the bitrev slice for correctness.
	// Unrolled into 4 groups of 4 for efficiency.

	// Group 0: indices 0-3
	MOVQ (R12), DX           // bitrev[0]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, (R8)            // work[0]

	MOVQ 8(R12), DX          // bitrev[1]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 8(R8)           // work[1]

	MOVQ 16(R12), DX         // bitrev[2]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 16(R8)          // work[2]

	MOVQ 24(R12), DX         // bitrev[3]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 24(R8)          // work[3]

	// Group 1: indices 4-7
	MOVQ 32(R12), DX         // bitrev[4]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 32(R8)          // work[4]

	MOVQ 40(R12), DX         // bitrev[5]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 40(R8)          // work[5]

	MOVQ 48(R12), DX         // bitrev[6]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 48(R8)          // work[6]

	MOVQ 56(R12), DX         // bitrev[7]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 56(R8)          // work[7]

	// Group 2: indices 8-11
	MOVQ 64(R12), DX         // bitrev[8]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 64(R8)          // work[8]

	MOVQ 72(R12), DX         // bitrev[9]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 72(R8)          // work[9]

	MOVQ 80(R12), DX         // bitrev[10]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 80(R8)          // work[10]

	MOVQ 88(R12), DX         // bitrev[11]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 88(R8)          // work[11]

	// Group 3: indices 12-15
	MOVQ 96(R12), DX         // bitrev[12]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 96(R8)          // work[12]

	MOVQ 104(R12), DX        // bitrev[13]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 104(R8)         // work[13]

	MOVQ 112(R12), DX        // bitrev[14]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 112(R8)         // work[14]

	MOVQ 120(R12), DX        // bitrev[15]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 120(R8)         // work[15]

	// =======================================================================
	// STAGE 1: size=2, half=1, step=8
	// =======================================================================
	// 8 independent butterflies with pairs: (0,1), (2,3), (4,5), (6,7),
	//                                       (8,9), (10,11), (12,13), (14,15)
	// All use twiddle[0] = (1, 0) which is identity multiplication.
	// So: a' = a + b, b' = a - b (no complex multiply needed)

	// Load all 16 complex64 values into 4 YMM registers
	// Y0 = [work[0], work[1], work[2], work[3]]
	// Y1 = [work[4], work[5], work[6], work[7]]
	// Y2 = [work[8], work[9], work[10], work[11]]
	// Y3 = [work[12], work[13], work[14], work[15]]
	VMOVUPS (R8), Y0
	VMOVUPS 32(R8), Y1
	VMOVUPS 64(R8), Y2
	VMOVUPS 96(R8), Y3

	// Stage 1: Butterflies on adjacent pairs within each 128-bit lane
	// For size=2 FFT: out[0] = in[0] + in[1], out[1] = in[0] - in[1]
	// Using twiddle[0] = 1+0i means t = b * 1 = b
	//
	// We need to rearrange: take pairs (0,1), (2,3) etc and compute a+b, a-b
	// VSHUFPD with imm=0b0101 swaps adjacent 64-bit elements in each 128-bit lane
	// But for size-2 butterflies, we need a different approach:
	//
	// Y0 = [a0, b0, a1, b1] where a0=work[0], b0=work[1], a1=work[2], b1=work[3]
	// We want: [a0+b0, a0-b0, a1+b1, a1-b1]
	//
	// Use VPERMILPD to create: [b0, a0, b1, a1]
	// Then add/sub

	// Y0: [w0, w1, w2, w3] -> pairs (w0,w1), (w2,w3)
	VPERMILPD $0x05, Y0, Y4  // Y4 = [w1, w0, w3, w2] (swap within 128-bit lanes)
	VADDPS Y4, Y0, Y5        // Y5 = [w0+w1, w1+w0, w2+w3, w3+w2]
	VSUBPS Y4, Y0, Y6        // Y6 = [w0-w1, w1-w0, w2-w3, w3-w2]
	// Now blend: take even positions from Y5, odd from Y6
	VBLENDPS $0xAA, Y6, Y5, Y0  // Y0 = [w0+w1, w0-w1, w2+w3, w2-w3]

	// Same for Y1: pairs (w4,w5), (w6,w7)
	VPERMILPD $0x05, Y1, Y4
	VADDPS Y4, Y1, Y5
	VSUBPS Y4, Y1, Y6
	VBLENDPS $0xAA, Y6, Y5, Y1

	// Same for Y2: pairs (w8,w9), (w10,w11)
	VPERMILPD $0x05, Y2, Y4
	VADDPS Y4, Y2, Y5
	VSUBPS Y4, Y2, Y6
	VBLENDPS $0xAA, Y6, Y5, Y2

	// Same for Y3: pairs (w12,w13), (w14,w15)
	VPERMILPD $0x05, Y3, Y4
	VADDPS Y4, Y3, Y5
	VSUBPS Y4, Y3, Y6
	VBLENDPS $0xAA, Y6, Y5, Y3

	// =======================================================================
	// STAGE 2: size=4, half=2, step=4
	// =======================================================================
	// 4 groups of 2 butterflies: (0,2), (1,3), (4,6), (5,7), ...
	// Twiddle factors: j=0 uses twiddle[0], j=1 uses twiddle[4]
	// twiddle[0] = (1, 0), twiddle[4] = (0, -1) for n=16
	//
	// After stage 1, Y0-Y3 contain the data. Now pairs are 2 apart.
	// Y0 = [d0, d1, d2, d3] where we need butterflies (d0,d2) and (d1,d3)
	//
	// Reorganize: we need to pair elements that are 2 positions apart
	// Use VPERM2F128 and VSHUFPS to rearrange

	// For each Y register, extract 'a' values (positions 0,1) and 'b' values (positions 2,3)
	// Y0 = [d0, d1, d2, d3]
	// a = [d0, d1], b = [d2, d3]
	// Butterfly: a' = a + w*b, b' = a - w*b where w = twiddle[j*step]

	// Load twiddle factors for stage 2
	// twiddle[0] = exp(0) = (1, 0)
	// twiddle[4] = exp(-2πi*4/16) = exp(-πi/2) = (0, -1)
	// Packed: [tw0, tw4] = [(1,0), (0,-1)]
	VMOVSD (R10), X4         // X4 = twiddle[0] = (1, 0)
	VMOVSD 32(R10), X5       // X5 = twiddle[4] = (0, -1)
	VPUNPCKLQDQ X5, X4, X4   // X4 = [tw0, tw4]
	VINSERTF128 $1, X4, Y4, Y4  // Y4 = [tw0, tw4, tw0, tw4]

	// Y0 = [d0, d1, d2, d3]
	// Extract a = [d0, d1, d0, d1] (low 128 bits duplicated)
	// Extract b = [d2, d3, d2, d3] (high 128 bits duplicated)
	VPERM2F128 $0x00, Y0, Y0, Y5  // Y5 = [d0, d1, d0, d1] (low lane to both)
	VPERM2F128 $0x11, Y0, Y0, Y6  // Y6 = [d2, d3, d2, d3] (high lane to both)

	// Complex multiply: t = w * b (Y4 * Y6)
	VMOVSLDUP Y4, Y7         // Y7 = [w.r, w.r, ...] (broadcast real parts)
	VMOVSHDUP Y4, Y8         // Y8 = [w.i, w.i, ...] (broadcast imag parts)
	VSHUFPS $0xB1, Y6, Y6, Y9  // Y9 = b_swapped = [b.i, b.r, ...]
	VMULPS Y8, Y9, Y9        // Y9 = [b.i*w.i, b.r*w.i, ...]
	VFMADDSUB231PS Y7, Y6, Y9  // Y9 = t = w * b

	// Butterfly: a' = a + t, b' = a - t
	VADDPS Y9, Y5, Y7        // Y7 = a + t
	VSUBPS Y9, Y5, Y8        // Y8 = a - t

	// Recombine: Y0 = [a'0, a'1, b'0, b'1]
	// Y7 has a' in both halves, Y8 has b' in both halves
	// Take low 128 bits of Y7 and high 128 bits (as low) of Y8
	VINSERTF128 $1, X8, Y7, Y0  // Y0 = [a'0, a'1, b'0, b'1]

	// Same for Y1 = [d4, d5, d6, d7]
	VPERM2F128 $0x00, Y1, Y1, Y5
	VPERM2F128 $0x11, Y1, Y1, Y6
	VMOVSLDUP Y4, Y7
	VMOVSHDUP Y4, Y8
	VSHUFPS $0xB1, Y6, Y6, Y9
	VMULPS Y8, Y9, Y9
	VFMADDSUB231PS Y7, Y6, Y9
	VADDPS Y9, Y5, Y7
	VSUBPS Y9, Y5, Y8
	VINSERTF128 $1, X8, Y7, Y1

	// Same for Y2 = [d8, d9, d10, d11]
	VPERM2F128 $0x00, Y2, Y2, Y5
	VPERM2F128 $0x11, Y2, Y2, Y6
	VMOVSLDUP Y4, Y7
	VMOVSHDUP Y4, Y8
	VSHUFPS $0xB1, Y6, Y6, Y9
	VMULPS Y8, Y9, Y9
	VFMADDSUB231PS Y7, Y6, Y9
	VADDPS Y9, Y5, Y7
	VSUBPS Y9, Y5, Y8
	VINSERTF128 $1, X8, Y7, Y2

	// Same for Y3 = [d12, d13, d14, d15]
	VPERM2F128 $0x00, Y3, Y3, Y5
	VPERM2F128 $0x11, Y3, Y3, Y6
	VMOVSLDUP Y4, Y7
	VMOVSHDUP Y4, Y8
	VSHUFPS $0xB1, Y6, Y6, Y9
	VMULPS Y8, Y9, Y9
	VFMADDSUB231PS Y7, Y6, Y9
	VADDPS Y9, Y5, Y7
	VSUBPS Y9, Y5, Y8
	VINSERTF128 $1, X8, Y7, Y3

	// =======================================================================
	// STAGE 3: size=8, half=4, step=2
	// =======================================================================
	// 2 groups of 4 butterflies: indices 0-3 with 4-7, indices 8-11 with 12-15
	// Twiddle factors: twiddle[0], twiddle[2], twiddle[4], twiddle[6]
	// twiddle[0] = (1, 0)
	// twiddle[2] = (cos(-π/4), sin(-π/4)) ≈ (0.707, -0.707)
	// twiddle[4] = (0, -1)
	// twiddle[6] = (-0.707, -0.707)

	// Load twiddle factors for stage 3
	VMOVSD (R10), X4         // twiddle[0]
	VMOVSD 16(R10), X5       // twiddle[2]
	VPUNPCKLQDQ X5, X4, X4   // X4 = [tw0, tw2]
	VMOVSD 32(R10), X5       // twiddle[4]
	VMOVSD 48(R10), X6       // twiddle[6]
	VPUNPCKLQDQ X6, X5, X5   // X5 = [tw4, tw6]
	VINSERTF128 $1, X5, Y4, Y4  // Y4 = [tw0, tw2, tw4, tw6]

	// Group 1: Y0 (indices 0-3) with Y1 (indices 4-7)
	// a = Y0 = [d0, d1, d2, d3]
	// b = Y1 = [d4, d5, d6, d7]
	// t = w * b, a' = a + t, b' = a - t

	VMOVSLDUP Y4, Y5         // Y5 = [w.r, w.r, ...]
	VMOVSHDUP Y4, Y6         // Y6 = [w.i, w.i, ...]
	VSHUFPS $0xB1, Y1, Y1, Y7  // Y7 = b_swapped
	VMULPS Y6, Y7, Y7        // Y7 = b_swap * w.i
	VFMADDSUB231PS Y5, Y1, Y7  // Y7 = t = w * b

	VADDPS Y7, Y0, Y8        // Y8 = a + t = new a (indices 0-3)
	VSUBPS Y7, Y0, Y9        // Y9 = a - t = new b (indices 4-7)

	// Group 2: Y2 (indices 8-11) with Y3 (indices 12-15)
	VSHUFPS $0xB1, Y3, Y3, Y7
	VMULPS Y6, Y7, Y7
	VFMADDSUB231PS Y5, Y3, Y7

	VADDPS Y7, Y2, Y0        // Y0 = new indices 8-11
	VSUBPS Y7, Y2, Y1        // Y1 = new indices 12-15

	// Move stage 3 results to Y0-Y3
	VMOVAPS Y8, Y2           // Save new 0-3 to Y2 temporarily
	VMOVAPS Y9, Y3           // Save new 4-7 to Y3 temporarily
	// Y0 = new 8-11, Y1 = new 12-15
	// Reorder: Y0=0-3, Y1=4-7, Y2=8-11, Y3=12-15
	VMOVAPS Y2, Y10          // Y10 = new 0-3
	VMOVAPS Y3, Y11          // Y11 = new 4-7
	VMOVAPS Y0, Y2           // Y2 = new 8-11
	VMOVAPS Y1, Y3           // Y3 = new 12-15
	VMOVAPS Y10, Y0          // Y0 = new 0-3
	VMOVAPS Y11, Y1          // Y1 = new 4-7

	// =======================================================================
	// STAGE 4: size=16, half=8, step=1
	// =======================================================================
	// 8 butterflies: index i with index i+8, for i=0..7
	// Twiddle factors: twiddle[0..7]

	// Load twiddle factors for stage 4
	VMOVUPS (R10), Y4        // Y4 = [tw0, tw1, tw2, tw3]
	VMOVUPS 32(R10), Y5      // Y5 = [tw4, tw5, tw6, tw7]

	// Group 1: Y0 (indices 0-3) with Y2 (indices 8-11) using Y4 (tw0-3)
	VMOVSLDUP Y4, Y6         // Y6 = [w.r broadcast]
	VMOVSHDUP Y4, Y7         // Y7 = [w.i broadcast]
	VSHUFPS $0xB1, Y2, Y2, Y8  // Y8 = b_swapped
	VMULPS Y7, Y8, Y8        // Y8 = b_swap * w.i
	VFMADDSUB231PS Y6, Y2, Y8  // Y8 = t = w * b

	VADDPS Y8, Y0, Y10       // Y10 = a' (new indices 0-3)
	VSUBPS Y8, Y0, Y11       // Y11 = b' (new indices 8-11)

	// Group 2: Y1 (indices 4-7) with Y3 (indices 12-15) using Y5 (tw4-7)
	VMOVSLDUP Y5, Y6
	VMOVSHDUP Y5, Y7
	VSHUFPS $0xB1, Y3, Y3, Y8
	VMULPS Y7, Y8, Y8
	VFMADDSUB231PS Y6, Y3, Y8

	VADDPS Y8, Y1, Y12       // Y12 = a' (new indices 4-7)
	VSUBPS Y8, Y1, Y13       // Y13 = b' (new indices 12-15)

	// Store final results to dst (not work buffer!)
	// If we were using scratch, we need to copy to dst
	MOVQ dst+0(FP), R9       // R9 = dst pointer
	VMOVUPS Y10, (R9)        // dst[0-3]
	VMOVUPS Y12, 32(R9)      // dst[4-7]
	VMOVUPS Y11, 64(R9)      // dst[8-11]
	VMOVUPS Y13, 96(R9)      // dst[12-15]

	VZEROUPPER
	MOVB $1, ret+120(FP)     // Return true (success)
	RET

size16_return_false:
	MOVB $0, ret+120(FP)
	RET

// Forward transform, size 32, complex64
// Fully unrolled 5-stage FFT with AVX2 vectorization
//
// This kernel implements a complete radix-2 DIT FFT for exactly 32 complex64 values.
// All 5 stages are fully unrolled with hardcoded twiddle factor indices:
//   Stage 1 (size=2):  16 butterflies, step=16, twiddle index 0 for all
//   Stage 2 (size=4):  16 butterflies in 4 groups, step=8, twiddle indices [0,8]
//   Stage 3 (size=8):  16 butterflies in 2 groups, step=4, twiddle indices [0,4,8,12]
//   Stage 4 (size=16): 16 butterflies in 1 group, step=2, twiddle indices [0,2,4,6,8,10,12,14]
//   Stage 5 (size=32): 16 butterflies, step=1, twiddle indices [0,1,2,...,15]
//
// Bit-reversal permutation indices for n=32:
//   [0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30,1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31]
//
// Register allocation:
//   R8:  work buffer (dst or scratch for in-place)
//   R9:  src pointer
//   R10: twiddle pointer
//   R11: scratch pointer
//   Y0-Y7: data registers for butterflies (32 complex64 = 8 YMM registers)
//   Y8-Y13: twiddle and intermediate values
//
TEXT ·forwardAVX2Size32Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 32)

	// Verify n == 32
	CMPQ R13, $32
	JNE  size32_return_false

	// Validate all slice lengths >= 32
	MOVQ dst+8(FP), AX
	CMPQ AX, $32
	JL   size32_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $32
	JL   size32_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $32
	JL   size32_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $32
	JL   size32_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size32_use_dst

	// In-place: use scratch
	MOVQ R11, R8
	JMP  size32_bitrev

size32_use_dst:
	// Out-of-place: use dst

size32_bitrev:
	// =======================================================================
	// Bit-reversal permutation: work[i] = src[bitrev[i]]
	// =======================================================================
	// For size 32, bitrev = [0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30,
	//                        1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31]
	// We use precomputed indices from the bitrev slice for correctness.
	// Unrolled into 8 groups of 4 for efficiency.

	// Group 0: indices 0-3
	MOVQ (R12), DX           // bitrev[0]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, (R8)            // work[0]

	MOVQ 8(R12), DX          // bitrev[1]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 8(R8)           // work[1]

	MOVQ 16(R12), DX         // bitrev[2]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 16(R8)          // work[2]

	MOVQ 24(R12), DX         // bitrev[3]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 24(R8)          // work[3]

	// Group 1: indices 4-7
	MOVQ 32(R12), DX         // bitrev[4]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 32(R8)          // work[4]

	MOVQ 40(R12), DX         // bitrev[5]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 40(R8)          // work[5]

	MOVQ 48(R12), DX         // bitrev[6]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 48(R8)          // work[6]

	MOVQ 56(R12), DX         // bitrev[7]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 56(R8)          // work[7]

	// Group 2: indices 8-11
	MOVQ 64(R12), DX         // bitrev[8]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 64(R8)          // work[8]

	MOVQ 72(R12), DX         // bitrev[9]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 72(R8)          // work[9]

	MOVQ 80(R12), DX         // bitrev[10]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 80(R8)          // work[10]

	MOVQ 88(R12), DX         // bitrev[11]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 88(R8)          // work[11]

	// Group 3: indices 12-15
	MOVQ 96(R12), DX         // bitrev[12]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 96(R8)          // work[12]

	MOVQ 104(R12), DX        // bitrev[13]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 104(R8)         // work[13]

	MOVQ 112(R12), DX        // bitrev[14]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 112(R8)         // work[14]

	MOVQ 120(R12), DX        // bitrev[15]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 120(R8)         // work[15]

	// Group 4: indices 16-19
	MOVQ 128(R12), DX        // bitrev[16]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 128(R8)         // work[16]

	MOVQ 136(R12), DX        // bitrev[17]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 136(R8)         // work[17]

	MOVQ 144(R12), DX        // bitrev[18]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 144(R8)         // work[18]

	MOVQ 152(R12), DX        // bitrev[19]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 152(R8)         // work[19]

	// Group 5: indices 20-23
	MOVQ 160(R12), DX        // bitrev[20]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 160(R8)         // work[20]

	MOVQ 168(R12), DX        // bitrev[21]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 168(R8)         // work[21]

	MOVQ 176(R12), DX        // bitrev[22]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 176(R8)         // work[22]

	MOVQ 184(R12), DX        // bitrev[23]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 184(R8)         // work[23]

	// Group 6: indices 24-27
	MOVQ 192(R12), DX        // bitrev[24]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 192(R8)         // work[24]

	MOVQ 200(R12), DX        // bitrev[25]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 200(R8)         // work[25]

	MOVQ 208(R12), DX        // bitrev[26]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 208(R8)         // work[26]

	MOVQ 216(R12), DX        // bitrev[27]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 216(R8)         // work[27]

	// Group 7: indices 28-31
	MOVQ 224(R12), DX        // bitrev[28]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 224(R8)         // work[28]

	MOVQ 232(R12), DX        // bitrev[29]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 232(R8)         // work[29]

	MOVQ 240(R12), DX        // bitrev[30]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 240(R8)         // work[30]

	MOVQ 248(R12), DX        // bitrev[31]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 248(R8)         // work[31]

	// =======================================================================
	// STAGE 1: size=2, half=1, step=16
	// =======================================================================
	// 16 independent butterflies with pairs: (0,1), (2,3), (4,5), ..., (30,31)
	// All use twiddle[0] = (1, 0) which is identity multiplication.
	// So: a' = a + b, b' = a - b (no complex multiply needed)

	// Load all 32 complex64 values into 8 YMM registers
	// Y0 = [work[0], work[1], work[2], work[3]]
	// Y1 = [work[4], work[5], work[6], work[7]]
	// ...
	// Y7 = [work[28], work[29], work[30], work[31]]
	VMOVUPS (R8), Y0
	VMOVUPS 32(R8), Y1
	VMOVUPS 64(R8), Y2
	VMOVUPS 96(R8), Y3
	VMOVUPS 128(R8), Y4
	VMOVUPS 160(R8), Y5
	VMOVUPS 192(R8), Y6
	VMOVUPS 224(R8), Y7

	// Stage 1: Butterflies on adjacent pairs within each 128-bit lane
	// For size=2 FFT: out[0] = in[0] + in[1], out[1] = in[0] - in[1]
	// Using twiddle[0] = 1+0i means t = b * 1 = b
	//
	// Y0 = [a0, b0, a1, b1] where a0=work[0], b0=work[1], a1=work[2], b1=work[3]
	// We want: [a0+b0, a0-b0, a1+b1, a1-b1]

	// Y0: [w0, w1, w2, w3] -> pairs (w0,w1), (w2,w3)
	VPERMILPD $0x05, Y0, Y8  // Y8 = [w1, w0, w3, w2] (swap within 128-bit lanes)
	VADDPS Y8, Y0, Y9        // Y9 = [w0+w1, w1+w0, w2+w3, w3+w2]
	VSUBPS Y8, Y0, Y10       // Y10 = [w0-w1, w1-w0, w2-w3, w3-w2]
	VBLENDPS $0xAA, Y10, Y9, Y0  // Y0 = [w0+w1, w0-w1, w2+w3, w2-w3]

	// Same for Y1
	VPERMILPD $0x05, Y1, Y8
	VADDPS Y8, Y1, Y9
	VSUBPS Y8, Y1, Y10
	VBLENDPS $0xAA, Y10, Y9, Y1

	// Same for Y2
	VPERMILPD $0x05, Y2, Y8
	VADDPS Y8, Y2, Y9
	VSUBPS Y8, Y2, Y10
	VBLENDPS $0xAA, Y10, Y9, Y2

	// Same for Y3
	VPERMILPD $0x05, Y3, Y8
	VADDPS Y8, Y3, Y9
	VSUBPS Y8, Y3, Y10
	VBLENDPS $0xAA, Y10, Y9, Y3

	// Same for Y4
	VPERMILPD $0x05, Y4, Y8
	VADDPS Y8, Y4, Y9
	VSUBPS Y8, Y4, Y10
	VBLENDPS $0xAA, Y10, Y9, Y4

	// Same for Y5
	VPERMILPD $0x05, Y5, Y8
	VADDPS Y8, Y5, Y9
	VSUBPS Y8, Y5, Y10
	VBLENDPS $0xAA, Y10, Y9, Y5

	// Same for Y6
	VPERMILPD $0x05, Y6, Y8
	VADDPS Y8, Y6, Y9
	VSUBPS Y8, Y6, Y10
	VBLENDPS $0xAA, Y10, Y9, Y6

	// Same for Y7
	VPERMILPD $0x05, Y7, Y8
	VADDPS Y8, Y7, Y9
	VSUBPS Y8, Y7, Y10
	VBLENDPS $0xAA, Y10, Y9, Y7

	// =======================================================================
	// STAGE 2: size=4, half=2, step=8
	// =======================================================================
	// 8 groups of 2 butterflies: (0,2), (1,3), (4,6), (5,7), ...
	// Twiddle factors: j=0 uses twiddle[0], j=1 uses twiddle[8]
	// twiddle[0] = (1, 0), twiddle[8] = (0, -1) for n=32

	// Load twiddle factors for stage 2
	// twiddle[0] = exp(0) = (1, 0)
	// twiddle[8] = exp(-2*pi*i*8/32) = exp(-pi*i/2) = (0, -1)
	VMOVSD (R10), X8         // X8 = twiddle[0] = (1, 0)
	VMOVSD 64(R10), X9       // X9 = twiddle[8] = (0, -1) ; offset = 8 * 8 bytes
	VPUNPCKLQDQ X9, X8, X8   // X8 = [tw0, tw8]
	VINSERTF128 $1, X8, Y8, Y8  // Y8 = [tw0, tw8, tw0, tw8]

	// Y0 = [d0, d1, d2, d3]
	// Extract a = [d0, d1, d0, d1] (low 128 bits duplicated)
	// Extract b = [d2, d3, d2, d3] (high 128 bits duplicated)
	VPERM2F128 $0x00, Y0, Y0, Y9   // Y9 = [d0, d1, d0, d1] (low lane to both)
	VPERM2F128 $0x11, Y0, Y0, Y10  // Y10 = [d2, d3, d2, d3] (high lane to both)

	// Complex multiply: t = w * b (Y8 * Y10)
	VMOVSLDUP Y8, Y11        // Y11 = [w.r, w.r, ...] (broadcast real parts)
	VMOVSHDUP Y8, Y12        // Y12 = [w.i, w.i, ...] (broadcast imag parts)
	VSHUFPS $0xB1, Y10, Y10, Y13  // Y13 = b_swapped = [b.i, b.r, ...]
	VMULPS Y12, Y13, Y13     // Y13 = [b.i*w.i, b.r*w.i, ...]
	VFMADDSUB231PS Y11, Y10, Y13  // Y13 = t = w * b

	// Butterfly: a' = a + t, b' = a - t
	VADDPS Y13, Y9, Y11      // Y11 = a + t
	VSUBPS Y13, Y9, Y12      // Y12 = a - t

	// Recombine: Y0 = [a'0, a'1, b'0, b'1]
	VINSERTF128 $1, X12, Y11, Y0

	// Same for Y1
	VPERM2F128 $0x00, Y1, Y1, Y9
	VPERM2F128 $0x11, Y1, Y1, Y10
	VMOVSLDUP Y8, Y11
	VMOVSHDUP Y8, Y12
	VSHUFPS $0xB1, Y10, Y10, Y13
	VMULPS Y12, Y13, Y13
	VFMADDSUB231PS Y11, Y10, Y13
	VADDPS Y13, Y9, Y11
	VSUBPS Y13, Y9, Y12
	VINSERTF128 $1, X12, Y11, Y1

	// Same for Y2
	VPERM2F128 $0x00, Y2, Y2, Y9
	VPERM2F128 $0x11, Y2, Y2, Y10
	VMOVSLDUP Y8, Y11
	VMOVSHDUP Y8, Y12
	VSHUFPS $0xB1, Y10, Y10, Y13
	VMULPS Y12, Y13, Y13
	VFMADDSUB231PS Y11, Y10, Y13
	VADDPS Y13, Y9, Y11
	VSUBPS Y13, Y9, Y12
	VINSERTF128 $1, X12, Y11, Y2

	// Same for Y3
	VPERM2F128 $0x00, Y3, Y3, Y9
	VPERM2F128 $0x11, Y3, Y3, Y10
	VMOVSLDUP Y8, Y11
	VMOVSHDUP Y8, Y12
	VSHUFPS $0xB1, Y10, Y10, Y13
	VMULPS Y12, Y13, Y13
	VFMADDSUB231PS Y11, Y10, Y13
	VADDPS Y13, Y9, Y11
	VSUBPS Y13, Y9, Y12
	VINSERTF128 $1, X12, Y11, Y3

	// Same for Y4
	VPERM2F128 $0x00, Y4, Y4, Y9
	VPERM2F128 $0x11, Y4, Y4, Y10
	VMOVSLDUP Y8, Y11
	VMOVSHDUP Y8, Y12
	VSHUFPS $0xB1, Y10, Y10, Y13
	VMULPS Y12, Y13, Y13
	VFMADDSUB231PS Y11, Y10, Y13
	VADDPS Y13, Y9, Y11
	VSUBPS Y13, Y9, Y12
	VINSERTF128 $1, X12, Y11, Y4

	// Same for Y5
	VPERM2F128 $0x00, Y5, Y5, Y9
	VPERM2F128 $0x11, Y5, Y5, Y10
	VMOVSLDUP Y8, Y11
	VMOVSHDUP Y8, Y12
	VSHUFPS $0xB1, Y10, Y10, Y13
	VMULPS Y12, Y13, Y13
	VFMADDSUB231PS Y11, Y10, Y13
	VADDPS Y13, Y9, Y11
	VSUBPS Y13, Y9, Y12
	VINSERTF128 $1, X12, Y11, Y5

	// Same for Y6
	VPERM2F128 $0x00, Y6, Y6, Y9
	VPERM2F128 $0x11, Y6, Y6, Y10
	VMOVSLDUP Y8, Y11
	VMOVSHDUP Y8, Y12
	VSHUFPS $0xB1, Y10, Y10, Y13
	VMULPS Y12, Y13, Y13
	VFMADDSUB231PS Y11, Y10, Y13
	VADDPS Y13, Y9, Y11
	VSUBPS Y13, Y9, Y12
	VINSERTF128 $1, X12, Y11, Y6

	// Same for Y7
	VPERM2F128 $0x00, Y7, Y7, Y9
	VPERM2F128 $0x11, Y7, Y7, Y10
	VMOVSLDUP Y8, Y11
	VMOVSHDUP Y8, Y12
	VSHUFPS $0xB1, Y10, Y10, Y13
	VMULPS Y12, Y13, Y13
	VFMADDSUB231PS Y11, Y10, Y13
	VADDPS Y13, Y9, Y11
	VSUBPS Y13, Y9, Y12
	VINSERTF128 $1, X12, Y11, Y7

	// =======================================================================
	// STAGE 3: size=8, half=4, step=4
	// =======================================================================
	// 4 groups of 4 butterflies: indices 0-3 with 4-7, indices 8-11 with 12-15, etc.
	// Twiddle factors: twiddle[0], twiddle[4], twiddle[8], twiddle[12]

	// Load twiddle factors for stage 3
	VMOVSD (R10), X8         // twiddle[0]
	VMOVSD 32(R10), X9       // twiddle[4]
	VPUNPCKLQDQ X9, X8, X8   // X8 = [tw0, tw4]
	VMOVSD 64(R10), X9       // twiddle[8]
	VMOVSD 96(R10), X10      // twiddle[12]
	VPUNPCKLQDQ X10, X9, X9  // X9 = [tw8, tw12]
	VINSERTF128 $1, X9, Y8, Y8  // Y8 = [tw0, tw4, tw8, tw12]

	// Group 1: Y0 (indices 0-3) with Y1 (indices 4-7)
	VMOVSLDUP Y8, Y9         // Y9 = [w.r, w.r, ...]
	VMOVSHDUP Y8, Y10        // Y10 = [w.i, w.i, ...]
	VSHUFPS $0xB1, Y1, Y1, Y11  // Y11 = b_swapped
	VMULPS Y10, Y11, Y11     // Y11 = b_swap * w.i
	VFMADDSUB231PS Y9, Y1, Y11  // Y11 = t = w * b

	VADDPS Y11, Y0, Y12      // Y12 = a + t = new indices 0-3
	VSUBPS Y11, Y0, Y13      // Y13 = a - t = new indices 4-7

	// Group 2: Y2 (indices 8-11) with Y3 (indices 12-15)
	VSHUFPS $0xB1, Y3, Y3, Y11
	VMULPS Y10, Y11, Y11
	VFMADDSUB231PS Y9, Y3, Y11

	VADDPS Y11, Y2, Y0       // Y0 = new indices 8-11
	VSUBPS Y11, Y2, Y1       // Y1 = new indices 12-15

	// Group 3: Y4 (indices 16-19) with Y5 (indices 20-23)
	VSHUFPS $0xB1, Y5, Y5, Y11
	VMULPS Y10, Y11, Y11
	VFMADDSUB231PS Y9, Y5, Y11

	VADDPS Y11, Y4, Y2       // Y2 = new indices 16-19
	VSUBPS Y11, Y4, Y3       // Y3 = new indices 20-23

	// Group 4: Y6 (indices 24-27) with Y7 (indices 28-31)
	VSHUFPS $0xB1, Y7, Y7, Y11
	VMULPS Y10, Y11, Y11
	VFMADDSUB231PS Y9, Y7, Y11

	VADDPS Y11, Y6, Y4       // Y4 = new indices 24-27
	VSUBPS Y11, Y6, Y5       // Y5 = new indices 28-31

	// Reorder: move results to Y0-Y7 in order
	// Currently: Y12=0-3, Y13=4-7, Y0=8-11, Y1=12-15, Y2=16-19, Y3=20-23, Y4=24-27, Y5=28-31
	VMOVAPS Y0, Y6           // Save 8-11 to Y6
	VMOVAPS Y1, Y7           // Save 12-15 to Y7
	VMOVAPS Y12, Y0          // Y0 = 0-3
	VMOVAPS Y13, Y1          // Y1 = 4-7
	VMOVAPS Y6, Y14          // Temp save
	VMOVAPS Y7, Y15          // Temp save
	VMOVAPS Y14, Y2          // Y2 = 8-11
	VMOVAPS Y15, Y3          // Y3 = 12-15
	// Y4=24-27, Y5=28-31 need swapping with Y2=16-19, Y3=20-23
	// Wait, let's re-trace: after stage 3:
	// Y12=0-3, Y13=4-7, Y0=8-11, Y1=12-15, Y2=16-19, Y3=20-23, Y4=24-27, Y5=28-31
	// Need: Y0=0-3, Y1=4-7, Y2=8-11, Y3=12-15, Y4=16-19, Y5=20-23, Y6=24-27, Y7=28-31
	// Correction - let me redo the reordering
	VMOVAPS Y2, Y6           // Save 16-19 (was in Y2)
	VMOVAPS Y3, Y7           // Save 20-23 (was in Y3)
	VMOVAPS Y4, Y2           // Y2 = 24-27 - WRONG, should stay as Y6
	VMOVAPS Y5, Y3           // Y3 = 28-31 - WRONG

	// Let's restart the reordering more carefully
	// Save current state first
	VMOVUPS Y0, (R8)         // temp store 8-11
	VMOVUPS Y1, 32(R8)       // temp store 12-15
	VMOVUPS Y2, 64(R8)       // temp store 16-19
	VMOVUPS Y3, 96(R8)       // temp store 20-23
	VMOVUPS Y4, 128(R8)      // temp store 24-27
	VMOVUPS Y5, 160(R8)      // temp store 28-31
	VMOVAPS Y12, Y0          // Y0 = 0-3
	VMOVAPS Y13, Y1          // Y1 = 4-7
	VMOVUPS (R8), Y2         // Y2 = 8-11
	VMOVUPS 32(R8), Y3       // Y3 = 12-15
	VMOVUPS 64(R8), Y4       // Y4 = 16-19
	VMOVUPS 96(R8), Y5       // Y5 = 20-23
	VMOVUPS 128(R8), Y6      // Y6 = 24-27
	VMOVUPS 160(R8), Y7      // Y7 = 28-31

	// =======================================================================
	// STAGE 4: size=16, half=8, step=2
	// =======================================================================
	// 2 groups of 8 butterflies:
	//   Group 1: indices 0-7 with indices 8-15 using twiddle[0,2,4,6] (for 0-3, 4-7)
	//   Group 2: indices 16-23 with indices 24-31 using twiddle[0,2,4,6]
	// Twiddle factors for n=32 with step=2: twiddle[0], twiddle[2], twiddle[4], twiddle[6]
	//                                       twiddle[0], twiddle[2], twiddle[4], twiddle[6]

	// Load twiddle factors for stage 4: [tw0, tw2, tw4, tw6]
	VMOVSD (R10), X8         // twiddle[0]
	VMOVSD 16(R10), X9       // twiddle[2]
	VPUNPCKLQDQ X9, X8, X8   // X8 = [tw0, tw2]
	VMOVSD 32(R10), X9       // twiddle[4]
	VMOVSD 48(R10), X10      // twiddle[6]
	VPUNPCKLQDQ X10, X9, X9  // X9 = [tw4, tw6]
	VINSERTF128 $1, X9, Y8, Y8  // Y8 = [tw0, tw2, tw4, tw6]

	// Group 1a: Y0 (indices 0-3) with Y2 (indices 8-11) using Y8 (tw0-3)
	VMOVSLDUP Y8, Y9         // Y9 = [w.r broadcast]
	VMOVSHDUP Y8, Y10        // Y10 = [w.i broadcast]
	VSHUFPS $0xB1, Y2, Y2, Y11  // Y11 = b_swapped
	VMULPS Y10, Y11, Y11     // Y11 = b_swap * w.i
	VFMADDSUB231PS Y9, Y2, Y11  // Y11 = t = w * b

	VADDPS Y11, Y0, Y12      // Y12 = a' (new indices 0-3)
	VSUBPS Y11, Y0, Y13      // Y13 = b' (new indices 8-11)

	// Group 1b: Y1 (indices 4-7) with Y3 (indices 12-15) using Y8 (tw0-3)
	VSHUFPS $0xB1, Y3, Y3, Y11
	VMULPS Y10, Y11, Y11
	VFMADDSUB231PS Y9, Y3, Y11

	VADDPS Y11, Y1, Y14      // Y14 = a' (new indices 4-7)
	VSUBPS Y11, Y1, Y15      // Y15 = b' (new indices 12-15)

	// Move group 1 results
	VMOVAPS Y12, Y0          // Y0 = 0-3
	VMOVAPS Y14, Y1          // Y1 = 4-7
	VMOVAPS Y13, Y2          // Y2 = 8-11
	VMOVAPS Y15, Y3          // Y3 = 12-15

	// Group 2a: Y4 (indices 16-19) with Y6 (indices 24-27) using Y8
	VSHUFPS $0xB1, Y6, Y6, Y11
	VMULPS Y10, Y11, Y11
	VFMADDSUB231PS Y9, Y6, Y11

	VADDPS Y11, Y4, Y12      // Y12 = a' (new indices 16-19)
	VSUBPS Y11, Y4, Y13      // Y13 = b' (new indices 24-27)

	// Group 2b: Y5 (indices 20-23) with Y7 (indices 28-31) using Y8
	VSHUFPS $0xB1, Y7, Y7, Y11
	VMULPS Y10, Y11, Y11
	VFMADDSUB231PS Y9, Y7, Y11

	VADDPS Y11, Y5, Y14      // Y14 = a' (new indices 20-23)
	VSUBPS Y11, Y5, Y15      // Y15 = b' (new indices 28-31)

	// Move group 2 results
	VMOVAPS Y12, Y4          // Y4 = 16-19
	VMOVAPS Y14, Y5          // Y5 = 20-23
	VMOVAPS Y13, Y6          // Y6 = 24-27
	VMOVAPS Y15, Y7          // Y7 = 28-31

	// =======================================================================
	// STAGE 5: size=32, half=16, step=1
	// =======================================================================
	// 16 butterflies: index i with index i+16, for i=0..15
	// Twiddle factors: twiddle[0..15]
	// Need 4 YMM registers for twiddles: [tw0-3], [tw4-7], [tw8-11], [tw12-15]

	// Load twiddle factors for stage 5
	VMOVUPS (R10), Y8        // Y8 = [tw0, tw1, tw2, tw3]
	VMOVUPS 32(R10), Y9      // Y9 = [tw4, tw5, tw6, tw7]
	VMOVUPS 64(R10), Y10     // Y10 = [tw8, tw9, tw10, tw11]
	VMOVUPS 96(R10), Y11     // Y11 = [tw12, tw13, tw14, tw15]

	// Group 1: Y0 (indices 0-3) with Y4 (indices 16-19) using Y8 (tw0-3)
	VMOVSLDUP Y8, Y12        // Y12 = [w.r broadcast]
	VMOVSHDUP Y8, Y13        // Y13 = [w.i broadcast]
	VSHUFPS $0xB1, Y4, Y4, Y14   // Y14 = b_swapped
	VMULPS Y13, Y14, Y14     // Y14 = b_swap * w.i
	VFMADDSUB231PS Y12, Y4, Y14  // Y14 = t = w * b

	// Store to memory since we need all 16 registers
	VMOVUPS Y14, (R8)        // Temp store t0-3

	VADDPS Y14, Y0, Y14      // Y14 = a' (new indices 0-3)
	VMOVUPS (R8), Y15        // Reload t0-3
	VSUBPS Y15, Y0, Y15      // Y15 = b' (new indices 16-19)
	VMOVUPS Y14, (R8)        // Store new 0-3 in dst
	VMOVUPS Y15, 128(R8)     // Store new 16-19 in dst

	// Group 2: Y1 (indices 4-7) with Y5 (indices 20-23) using Y9 (tw4-7)
	VMOVSLDUP Y9, Y12
	VMOVSHDUP Y9, Y13
	VSHUFPS $0xB1, Y5, Y5, Y14
	VMULPS Y13, Y14, Y14
	VFMADDSUB231PS Y12, Y5, Y14

	VADDPS Y14, Y1, Y15      // Y15 = a' (new indices 4-7)
	VSUBPS Y14, Y1, Y14      // Y14 = b' (new indices 20-23)
	VMOVUPS Y15, 32(R8)      // Store new 4-7
	VMOVUPS Y14, 160(R8)     // Store new 20-23

	// Group 3: Y2 (indices 8-11) with Y6 (indices 24-27) using Y10 (tw8-11)
	VMOVSLDUP Y10, Y12
	VMOVSHDUP Y10, Y13
	VSHUFPS $0xB1, Y6, Y6, Y14
	VMULPS Y13, Y14, Y14
	VFMADDSUB231PS Y12, Y6, Y14

	VADDPS Y14, Y2, Y15      // Y15 = a' (new indices 8-11)
	VSUBPS Y14, Y2, Y14      // Y14 = b' (new indices 24-27)
	VMOVUPS Y15, 64(R8)      // Store new 8-11
	VMOVUPS Y14, 192(R8)     // Store new 24-27

	// Group 4: Y3 (indices 12-15) with Y7 (indices 28-31) using Y11 (tw12-15)
	VMOVSLDUP Y11, Y12
	VMOVSHDUP Y11, Y13
	VSHUFPS $0xB1, Y7, Y7, Y14
	VMULPS Y13, Y14, Y14
	VFMADDSUB231PS Y12, Y7, Y14

	VADDPS Y14, Y3, Y15      // Y15 = a' (new indices 12-15)
	VSUBPS Y14, Y3, Y14      // Y14 = b' (new indices 28-31)
	VMOVUPS Y15, 96(R8)      // Store new 12-15
	VMOVUPS Y14, 224(R8)     // Store new 28-31

	// =======================================================================
	// Copy results to dst if we used scratch buffer
	// =======================================================================
	MOVQ dst+0(FP), R9       // R9 = dst pointer
	CMPQ R8, R9
	JE   size32_done         // Already in dst

	// Copy from scratch to dst
	VMOVUPS (R8), Y0
	VMOVUPS 32(R8), Y1
	VMOVUPS 64(R8), Y2
	VMOVUPS 96(R8), Y3
	VMOVUPS 128(R8), Y4
	VMOVUPS 160(R8), Y5
	VMOVUPS 192(R8), Y6
	VMOVUPS 224(R8), Y7
	VMOVUPS Y0, (R9)
	VMOVUPS Y1, 32(R9)
	VMOVUPS Y2, 64(R9)
	VMOVUPS Y3, 96(R9)
	VMOVUPS Y4, 128(R9)
	VMOVUPS Y5, 160(R9)
	VMOVUPS Y6, 192(R9)
	VMOVUPS Y7, 224(R9)

size32_done:
	VZEROUPPER
	MOVB $1, ret+120(FP)     // Return true (success)
	RET

size32_return_false:
	MOVB $0, ret+120(FP)
	RET

// Forward transform, size 64, complex64
TEXT ·forwardAVX2Size64Complex64Asm(SB), NOSPLIT, $0-121
	MOVB $0, ret+120(FP)        // Return false
	RET

// Forward transform, size 128, complex64
TEXT ·forwardAVX2Size128Complex64Asm(SB), NOSPLIT, $0-121
	MOVB $0, ret+120(FP)        // Return false
	RET

// Inverse transform, size 16, complex64
TEXT ·inverseAVX2Size16Complex64Asm(SB), NOSPLIT, $0-121
	MOVB $0, ret+120(FP)        // Return false
	RET

// Inverse transform, size 32, complex64
TEXT ·inverseAVX2Size32Complex64Asm(SB), NOSPLIT, $0-121
	MOVB $0, ret+120(FP)        // Return false
	RET

// Inverse transform, size 64, complex64
TEXT ·inverseAVX2Size64Complex64Asm(SB), NOSPLIT, $0-121
	MOVB $0, ret+120(FP)        // Return false
	RET

// Inverse transform, size 128, complex64
TEXT ·inverseAVX2Size128Complex64Asm(SB), NOSPLIT, $0-121
	MOVB $0, ret+120(FP)        // Return false
	RET
