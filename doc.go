// Package algoforge provides high-performance Fast Fourier Transform (FFT) implementations.
//
// # Overview
//
// algoforge is a production-ready FFT library for Go with focus on:
//   - Correctness through comprehensive testing
//   - Performance via SIMD optimization across architectures
//   - Flexibility supporting various transform types and data layouts
//   - Clean, ergonomic Go API
//
// # Plan Constructors
//
// The library provides multiple ways to create FFT plans:
//
// Generic constructor (recommended for type-safe code):
//
//	plan, err := algoforge.NewPlan[complex64](1024)
//	plan128, err := algoforge.NewPlan[complex128](1024)
//
// Explicit precision constructors:
//
//	plan32, err := algoforge.NewPlan32(1024)   // complex64 (single precision)
//	plan64, err := algoforge.NewPlan64(1024)   // complex128 (double precision)
//
// Convenience alias (defaults to complex64 for best performance):
//
//	plan, err := algoforge.NewPlan(1024)  // equivalent to NewPlan32
//
// # Basic Usage (1D FFT)
//
// Create a plan for a specific FFT size, then reuse it for multiple transforms:
//
//	plan, err := algoforge.NewPlan(1024)
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	// Prepare input and output slices
//	input := make([]complex64, 1024)
//	output := make([]complex64, 1024)
//
//	// Perform forward FFT
//	if err := plan.Forward(output, input); err != nil {
//		log.Fatal(err)
//	}
//
//	// Perform inverse FFT
//	if err := plan.Inverse(output, input); err != nil {
//		log.Fatal(err)
//	}
//
// # Double Precision (complex128)
//
// For applications requiring higher precision:
//
//	plan64, err := algoforge.NewPlan64(1024)
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	input := make([]complex128, 1024)
//	output := make([]complex128, 1024)
//
//	if err := plan64.Forward(output, input); err != nil {
//		log.Fatal(err)
//	}
//
// Or using the generic constructor:
//
//	plan, err := algoforge.NewPlan[complex128](1024)
//
// Use complex128 when:
//   - Accumulating many transforms (error compounds less)
//   - Working with very large FFT sizes (>65536 points)
//   - Requiring high dynamic range in frequency domain
//
// # Real FFT
//
// For real-valued input signals, use PlanReal for ~2x performance improvement:
//
//	planReal, err := algoforge.NewPlanReal(1024)
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	// Real input: 1024 float32 samples
//	input := make([]float32, 1024)
//
//	// Complex output: N/2+1 = 513 complex64 values (conjugate-symmetric)
//	output := make([]complex64, 513)
//
//	if err := planReal.Forward(output, input); err != nil {
//		log.Fatal(err)
//	}
//
//	// Inverse: reconstruct real signal from half-spectrum
//	reconstructed := make([]float32, 1024)
//	if err := planReal.Inverse(reconstructed, output); err != nil {
//		log.Fatal(err)
//	}
//
// The output contains only the non-redundant half of the spectrum due to
// conjugate symmetry of real signals. Index 0 is DC, index N/2 is Nyquist.
//
// # 2D FFT
//
// For image processing and 2D signal analysis:
//
//	plan2D, err := algoforge.NewPlan2D(256, 256) // rows, cols
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	// Data stored in row-major order: data[row*cols + col]
//	input := make([]complex64, 256*256)
//	output := make([]complex64, 256*256)
//
//	if err := plan2D.Forward(output, input); err != nil {
//		log.Fatal(err)
//	}
//
// Non-square matrices are fully supported:
//
//	planRect, err := algoforge.NewPlan2D(128, 512) // 128 rows, 512 cols
//
// # 3D FFT
//
// For volumetric data (medical imaging, fluid dynamics):
//
//	plan3D, err := algoforge.NewPlan3D(64, 64, 64) // depth, rows, cols
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	// Data stored as: data[z*rows*cols + y*cols + x]
//	volume := make([]complex64, 64*64*64)
//	spectrum := make([]complex64, 64*64*64)
//
//	if err := plan3D.Forward(spectrum, volume); err != nil {
//		log.Fatal(err)
//	}
//
// # N-Dimensional FFT
//
// For arbitrary dimensions (4D, 5D, etc.):
//
//	dims := []int{8, 16, 32, 64} // 4D: 8x16x32x64
//	planND, err := algoforge.NewPlanND(dims)
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	size := 8 * 16 * 32 * 64
//	input := make([]complex64, size)
//	output := make([]complex64, size)
//
//	if err := planND.Forward(output, input); err != nil {
//		log.Fatal(err)
//	}
//
// # Batch Processing
//
// Process multiple signals of the same length efficiently:
//
//	plan, _ := algoforge.NewPlan(256)
//
//	// 100 signals, each 256 samples, stored sequentially
//	count := 100
//	signals := make([]complex64, count*256)
//	spectra := make([]complex64, count*256)
//
//	// Transform all signals in one call
//	if err := plan.ForwardBatch(spectra, signals, count, 256); err != nil {
//		log.Fatal(err)
//	}
//
// For interleaved data layouts, adjust the stride parameter accordingly.
//
// # Strided Data
//
// Transform non-contiguous data (e.g., matrix columns):
//
//	plan, _ := algoforge.NewPlan(128)
//
//	// 128x256 matrix in row-major order
//	matrix := make([]complex64, 128*256)
//
//	// Transform column 0 (stride = 256, the number of columns)
//	colOutput := make([]complex64, 128)
//	if err := plan.ForwardStrided(colOutput, matrix, 256); err != nil {
//		log.Fatal(err)
//	}
//
// # Convolution via FFT
//
// Efficient O(N log N) convolution for filtering and correlation:
//
//	// Convolve two complex signals
//	a := make([]complex64, 1000)
//	b := make([]complex64, 500)
//	result := make([]complex64, len(a)+len(b)-1) // 1499 samples
//
//	if err := algoforge.Convolve(result, a, b); err != nil {
//		log.Fatal(err)
//	}
//
// For real-valued signals (audio filtering, etc.):
//
//	signal := make([]float32, 44100)  // 1 second of audio
//	kernel := make([]float32, 441)    // 10ms filter kernel
//	filtered := make([]float32, len(signal)+len(kernel)-1)
//
//	if err := algoforge.ConvolveReal(filtered, signal, kernel); err != nil {
//		log.Fatal(err)
//	}
//
// # Correlation
//
// Cross-correlation and auto-correlation:
//
//	// Cross-correlate two signals (find where b occurs in a)
//	if err := algoforge.Correlate(result, a, b); err != nil {
//		log.Fatal(err)
//	}
//
//	// Auto-correlation (find periodicity in a signal)
//	if err := algoforge.AutoCorrelate(result, signal); err != nil {
//		log.Fatal(err)
//	}
//
// # Transform Types Summary
//
// The library supports several transform types:
//   - Complex FFT: forward and inverse transforms of complex-valued signals
//   - Real FFT: optimized transforms for real-valued input signals (PlanReal)
//   - Multi-dimensional: 2D, 3D, and arbitrary N-dimensional FFTs
//   - Batch: efficient processing of multiple transforms with same Plan
//   - Strided: transform non-contiguous data without copying
//
// # Size Support
//
// Plans support:
//   - Power-of-2 sizes: optimized Radix-2 and Radix-4 algorithms
//   - Composite sizes: mixed-radix Radix-2/3/4/5 algorithms
//   - Arbitrary sizes: Bluestein's algorithm (Chirp-Z transform)
//
// # Performance
//
// The library achieves high performance through:
//   - Zero-allocation transforms with pre-allocated Plans
//   - SIMD optimization (AVX2 on amd64, NEON on ARM64)
//   - CPU feature detection and runtime dispatch
//   - Cache-efficient memory access patterns
//
// Plans are safe for concurrent use (read-only during transforms).
//
// # Precision
//
// Two precision levels are available:
//   - complex64 (Plan32 / NewPlan / NewPlan32): 32-bit float components, faster, lower memory
//   - complex128 (Plan64 / NewPlan64): 64-bit float components, higher precision
//
// The generic Plan[T] type unifies both precisions with compile-time type safety.
//
// Numerical accuracy is maintained through careful algorithm implementation.
// See the documentation for precision characteristics of different transform types.
//
// # Thread Safety
//
// Plan objects are safe for concurrent use during transform operations.
// Multiple goroutines can safely call transform methods with different input/output buffers.
//
// # Error Handling
//
// All transform methods return an error if inputs are invalid:
//   - ErrInvalidLength: FFT size is not supported
//   - ErrNilSlice: input or output slice is nil
//   - ErrLengthMismatch: slice sizes don't match Plan dimensions
//   - ErrInvalidStride: stride parameter is invalid for the data layout
//
// # Examples
//
// See the examples/ directory for more detailed usage patterns:
//   - examples/basic/: Simple 1D FFT usage
//   - examples/audio/: Audio spectrum analysis
//   - examples/image/: 2D FFT for image processing
//   - examples/benchmark/: Performance comparison tool
package algoforge
