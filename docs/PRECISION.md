# Numerical Precision in algofft

This document describes the numerical precision characteristics of the algofft library, including error bounds, precision recommendations, and guidelines for choosing between `complex64` and `complex128`.

## Precision Overview

algofft supports two precision levels:

- **complex64** (float32): ~6-7 decimal digits of precision
- **complex128** (float64): ~15-16 decimal digits of precision

The choice between them involves trade-offs between performance, memory usage, and numerical accuracy.

## Error Sources in FFT

Numerical errors in FFT implementations come from several sources:

1. **Twiddle factor quantization**: Precomputed roots of unity have finite precision
2. **Floating-point arithmetic**: Rounding errors in butterfly operations
3. **Error accumulation**: Errors compound through O(log N) stages
4. **Bit-reversal permutation**: Indexing doesn't introduce error, but affects memory access patterns

### Expected Error Bounds

For a size-N FFT with well-conditioned input:

| Precision  | Single Transform | Round-trip (Forward + Inverse) |
| ---------- | ---------------- | ------------------------------ |
| complex64  | ~10⁻⁶ relative   | ~10⁻⁵ relative                 |
| complex128 | ~10⁻¹⁴ relative  | ~10⁻¹³ relative                |

These bounds scale roughly as `ε × log₂(N)` where `ε` is machine epsilon.

## Parseval's Theorem

Energy conservation is verified to within:

- **complex64**: relative error < 10⁻⁵
- **complex128**: relative error < 10⁻¹³

This is tested in `precision_test.go::TestPrecisionParseval`.

## Error Accumulation Over Repeated Transforms

Repeated forward/inverse cycles accumulate error linearly with the number of cycles:

| Precision  | 10 cycles | 100 cycles | 1000 cycles |
| ---------- | --------- | ---------- | ----------- |
| complex64  | ~10⁻⁵     | ~10⁻⁴      | ~10⁻³       |
| complex128 | ~10⁻¹³    | ~10⁻¹²     | ~10⁻¹¹      |

**Recommendation**: For applications requiring repeated transforms (e.g., iterative solvers), use complex128 or periodically reinitialize from high-precision reference data.

## Precision Profiling Sweep (Round-Trip)

The test suite includes a size sweep that logs round-trip errors for both precisions
along with `log2(size)` and the error ratio (complex64 / complex128). Use the output
to plot error vs `log2(size)` if needed. Run:

```bash
go test -v -run TestPrecisionRoundTripSweep ./...
```

Example log format:

```
size=4096 log2=12 err64=... err128=... ratio=...
```

## FFT Size Considerations

Precision degrades slightly with FFT size due to increased accumulation depth:

### complex64 Recommended Maximum Sizes

| Application                     | Max Size | Rationale                         |
| ------------------------------- | -------- | --------------------------------- |
| Signal processing (audio/video) | 65536    | Perceptual tolerance masks errors |
| Real-time applications          | 16384    | Low latency + acceptable accuracy |
| Scientific computing            | 4096     | Higher precision requirements     |

### complex128 Maximum Sizes

- Practically unlimited for sizes up to 2²⁴ (16,777,216)
- Tested up to 262,144 with excellent precision
- Errors remain < 10⁻¹¹ even for very large transforms

## Precision Comparison: complex64 vs complex128

### When to Use complex64

✅ **Good for:**

- Real-time audio/video processing
- Embedded systems with limited memory
- Applications where performance is critical
- When human perception is the final judge (audio, images)
- Prototyping and development

### When to Use complex128

✅ **Good for:**

- Scientific computing requiring high accuracy
- Iterative algorithms (solvers, optimization)
- Very large FFT sizes (> 65536)
- Financial calculations
- Applications where errors compound across multiple operations
- Reference implementations for validation

### Performance vs Precision Trade-off

| Metric                 | complex64  | complex128 | Ratio        |
| ---------------------- | ---------- | ---------- | ------------ |
| Memory per sample      | 8 bytes    | 16 bytes   | 2×           |
| Cache efficiency       | Better     | Worse      | ~1.5× faster |
| SIMD throughput (AVX2) | 4 elements | 2 elements | 2×           |
| Precision              | 10⁻⁶       | 10⁻¹⁴      | 10⁸×         |

**Typical performance**: complex64 is 1.5-2× faster than complex128 on modern CPUs with SIMD.

## Algorithm-Specific Precision

### Stockham Autosort

- **Advantage**: No explicit bit-reversal pass reduces indexing complexity
- **Precision**: Identical to DIT for same precision type
- **Cache behavior**: Better locality may slightly improve numerical stability in practice

### DIT (Decimation-in-Time)

- **Advantage**: Classic algorithm, well-studied error characteristics
- **Precision**: Meets theoretical bounds
- **Bit-reversal**: Performed in-place without precision loss

### Bluestein (Arbitrary Lengths)

- **Error**: Slightly higher due to convolution via zero-padded FFT
- **Recommendation**: Use complex128 for non-power-of-2 sizes when precision matters
- **Tested**: Meets same error bounds as radix-2 algorithms

## Known Signals and Analytical Results

The library has been validated against analytical FFT results:

### Impulse (δ[0])

- **Expected**: FFT = [1, 1, 1, ..., 1]
- **Observed**: Max error < 10⁻¹² (complex128)

### Sine Wave (sin(2πft))

- **Expected**: Peaks at ±frequency bins
- **Observed**: Peak magnitude error < 0.1% of expected value

### Cosine Wave (cos(2πft))

- **Expected**: Real-valued peaks at ±frequency bins
- **Observed**: Imaginary component < 10⁻¹⁰ × real component

## Testing and Validation

Precision is continuously validated through:

1. **Cross-validation**: Comparison with O(N²) naive DFT (`internal/reference`)
2. **Property tests**: Parseval's theorem, linearity, shift theorem
3. **Round-trip tests**: `Inverse(Forward(x)) ≈ x`
4. **Known signals**: Analytical results for standard test signals
5. **Cross-precision**: complex64 results compared with complex128 baseline

See `precision_test.go` for complete test suite.

## Precision Gotchas

### 1. Denormalized Numbers

Very small values (< 10⁻³⁸ for float32) may flush to zero on some architectures with FTZ (flush-to-zero) enabled. This is rare in practice but can affect scientific applications.

**Mitigation**: Use complex128 or scale inputs to normal range.

### 2. Catastrophic Cancellation

Subtracting nearly-equal values loses precision. This is inherent to floating-point arithmetic, not FFT-specific.

**Example**: `x - y` where `x ≈ y` can lose significant digits.

### 3. Large Dynamic Range

If input contains both very large and very small values (e.g., 10¹⁰ and 10⁻¹⁰), precision degrades.

**Mitigation**: Normalize inputs to similar magnitude or use complex128.

## Recommendations Summary

| Use Case                     | Recommended Precision | Max Size | Notes                   |
| ---------------------------- | --------------------- | -------- | ----------------------- |
| Audio processing             | complex64             | 65536    | Perceptual tolerance    |
| Video/Image processing       | complex64             | 16384    | Real-time performance   |
| Radar/Sonar                  | complex128            | 262144   | High dynamic range      |
| Scientific simulations       | complex128            | 524288   | Accuracy critical       |
| Machine learning (inference) | complex64             | 4096     | Speed matters           |
| Machine learning (training)  | complex128            | 16384    | Stability important     |
| Embedded systems             | complex64             | 4096     | Memory constrained      |
| Financial calculations       | complex128            | Any      | Regulatory requirements |

## Benchmarking Precision

To measure precision for your specific application:

```go
// Example: measure round-trip error
plan, _ := algofft.NewPlan64(n)
original := /* your data */
freq := make([]complex128, n)
reconstructed := make([]complex128, n)

plan.Forward(freq, original)
plan.Inverse(reconstructed, freq)

// Measure error
var maxError float64
for i := range original {
    err := cmplx.Abs(reconstructed[i] - original[i])
    if err > maxError {
        maxError = err
    }
}
```

See `precision_test.go` for more comprehensive examples.

## Two accuracy findings from the 2026-07 rounds

**Component-wise complex64 multiplication costs 3–9% relative-L2 error, i.e.
sub-ulp.** Go's compiler does not implement scalar `complex64 * complex64` in
single precision — it widens all four components to float64, multiplies in
double precision and rounds back. Replacing those products with
`math.MulComplex64` (component-wise: `MULSS` ×3, `VFMADD231SS`, `SUBSS`) is a
large speed win but gives up the accidental double-precision accumulation.
Everything stays at ~10⁻⁷ (float32 ε is 1.19e-07) and the peak-normalized error
is unchanged or lower. complex128 is bit-identical.

**FMA fusion improves accuracy where the twiddle work is densest**, as
one-rounding-instead-of-two predicts: fusing the AVX2 codelets the registry
actually selects moved max relative error by −28.5% at complex64 size 16 and
−38.2% at size 32.

### Reading `cmd/measure_correctness` correctly

The tool reports **relative L2** against `reference.NaiveDFTWide` (float64
output for float32 input), mean and max over trials, plus a peak-normalized
column. It also prints the build configuration, which is what makes its numbers
comparable at all, and seeds with a fixed `rand.NewSource(42)` so an A/B over a
code change compares identical input vectors.

Two things to know before quoting it:

- It used to max a **per-bin** relative error against a **complex64** reference
  — an extreme-value statistic over an unstable quantity — and that misleading
  number nearly caused good work to be reverted. Do not reintroduce it.
- Relative L2 attenuates a single wrong bin by ~1/√n, which is exactly the
  failure a broken codelet produces. That is why the peak-normalized column
  exists; read both.
- **Its complex128 column measures the reference, not the FFT.**
  `NaiveDFT128` builds twiddles from un-reduced angles, so its own error grows
  as O(n). Recorded, not fixed.

## References

- IEEE 754 Standard for Floating-Point Arithmetic
- Numerical Recipes in C (Chapter 12: Fast Fourier Transform)
- FFTW documentation on precision and accuracy
- _What Every Computer Scientist Should Know About Floating-Point Arithmetic_ (Goldberg, 1991)

## Updates

This document is based on empirical testing as of 2025-01-27. Precision characteristics may improve with:

- SIMD optimizations (AVX-512, ARM SVE)
- Improved twiddle factor generation
- Alternative algorithms (split-radix, prime-factor)

Run the precision test suite (`go test -v -run=Precision ./...`) to verify current behavior.
