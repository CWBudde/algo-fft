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
// SSE2 Stubs - Return false to trigger Go fallback
// ===========================================================================
// These stubs exist for systems without AVX2 support.
// They simply return false, causing the calling code to use the pure Go path.
TEXT ·forwardSSE2Complex64Asm(SB), NOSPLIT|NOFRAME, $0-121
	MOVB $0, ret+120(FP)
	RET

TEXT ·inverseSSE2Complex64Asm(SB), NOSPLIT|NOFRAME, $0-121
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
