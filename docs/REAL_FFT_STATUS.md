# Real FFT Implementation Status

## TL;DR

✅ **Real FFT IS ALREADY IMPLEMENTED!** You can use it today for 4096-sample FFTs.

❌ **Missing:** `float64` precision support (currently only `float32`)

---

## What You Have Now (✅ Fully Working)

### 1D Real FFT (float32 only)

```go
// Create a plan for 4096 real samples
planReal, err := algofft.NewPlanReal(4096)
if err != nil {
    panic(err)
}

// Input: 4096 real float32 samples
input := make([]float32, 4096)
// ... fill with your audio/signal data

// Output: 2049 complex values (N/2+1 due to conjugate symmetry)
output := make([]complex64, 4096/2+1)

// Forward transform (time → frequency)
err = planReal.Forward(output, input)

// Inverse transform (frequency → time)
recovered := make([]float32, 4096)
err = planReal.Inverse(recovered, output)
```

### Key Features Already Working

- ✅ **Optimized Pack Method**: Treats N reals as N/2 complex, performs N/2 complex FFT
- ✅ **Conjugate Symmetry**: Output only contains N/2+1 bins (not redundant N bins)
- ✅ **Precomputed Weights**: U[k] weights cached for fast recombination
- ✅ **Zero Allocations**: After plan creation, transforms allocate no memory
- ✅ **Normalized Variants**: `ForwardNormalized()` and `ForwardUnitary()`
- ✅ **Batch Support**: Process multiple real FFTs efficiently
- ✅ **Strided Access**: Transform non-contiguous data
- ✅ **2D/3D Real FFT**: Multi-dimensional real transforms
- ✅ **Real Convolution**: Efficient convolution via real FFT

### Performance on AVX2 (4096 samples)

Based on your AVX2 system, for size=4096:

- **Algorithm Selected**: Stockham autosort (auto-selected for sizes > 1024)
- **SIMD Path**: AVX2 vectorized complex64 operations
- **Expected Throughput**: ~300-500 MB/s (depending on CPU model)
- **Allocations**: 0 during steady-state transforms
- **Memory**: Plan allocates ~40 KB (twiddles + scratch buffers)

---

## What's Missing (❌ Not Yet Implemented)

### float64 Precision Support

**Current limitation:** `PlanReal` only works with `float32` input/output.

If you have `float64` data (common in scientific computing, high-precision audio), you must currently:

```go
// ❌ Current workaround (loses precision)
float64Data := []float64{1.0, 2.0, 3.0, ...}

// Downcast to float32
float32Data := make([]float32, len(float64Data))
for i, v := range float64Data {
    float32Data[i] = float32(v)  // Precision loss!
}

planReal, _ := algofft.NewPlanReal(len(float32Data))
// ... use planReal
```

**What we need to add:** Generic `PlanReal[F Float]` type

```go
// ✅ Future API (after Phase 11.4)
planReal32, _ := algofft.NewPlanReal32(4096)  // float32 → complex64
planReal64, _ := algofft.NewPlanReal64(4096)  // float64 → complex128

// Or generic
planReal, _ := algofft.NewPlanRealT[float64](4096)

input := make([]float64, 4096)    // Full precision
output := make([]complex128, 2049)
planReal64.Forward(output, input)
```

---

## What I Added to PLAN.md

I've added **Phase 11.4: Generic Real FFT API** with the following tasks:

1. **Design generic type** `PlanReal[F Float, C Complex]`

   - Type constraint: `Float = float32 | float64`
   - Auto-pair: `float32` with `complex64`, `float64` with `complex128`

2. **Implement generic constructor** `NewPlanRealT[F Float](n int)`

   - Type dispatch to correct backend
   - Share pack/unpack logic via generics

3. **Add convenience constructors**

   - `NewPlanReal32()` - current behavior (backward compatible)
   - `NewPlanReal64()` - new for float64 support
   - `NewPlanReal()` remains alias to `NewPlanReal32()`

4. **Implement float64 kernels**

   - Pack/unpack for `float64` → `complex128`
   - Weight computation at full `float64` precision
   - Test: <1e-12 round-trip error (vs <1e-6 for float32)

5. **Testing & docs**
   - Correctness tests for both precisions
   - Precision comparison benchmarks
   - Updated examples and README

**Estimated effort:** 1-2 days of focused work

---

## How the Dispatch Works for Real FFT

When you call `planReal.Forward(dst, src)` for 4096 samples:

### Plan Creation (One-Time)

```
NewPlanReal(4096)
  ├─> Validates n=4096 is even
  ├─> Creates Plan[complex64] for n/2=2048
  │     └─> CPU detection: AVX2 available
  │     └─> Strategy selection: 2048 > 1024 → Stockham
  │     └─> Kernel binding: AVX2 Stockham complex64
  │     └─> Precomputes: twiddles, bit-reversal, packed twiddles
  └─> Precomputes U[k] weights for recombination
```

### Transform Execution (Every Call)

```
planReal.Forward(dst, src)
  ├─> Pack: treat 4096 reals as 2048 complex values
  │     └─> z[k] = src[2k] + i*src[2k+1] for k=0..2047
  ├─> Complex FFT on 2048 samples (AVX2 Stockham kernel)
  │     └─> Zero-dispatch to AVX2 vectorized butterfly
  │     └─> Processes 4 complex64 per YMM register
  │     └─> No bit-reversal (Stockham autosort)
  └─> Unpack using U[k] weights
        └─> dst[k] = U[k]*Z[k] + conj(U[k])*conj(Z[N/2-k])
        └─> Outputs 2049 bins (0 to N/2 inclusive)
```

**Performance:** The real FFT is ~2x faster than an equivalent 4096-point complex FFT because it only does a 2048-point complex FFT internally.

---

## Recommendations for Your Use Case

### If you're using float32 data (typical for audio)

✅ **You're all set!** Use the existing `NewPlanReal()` API:

```go
plan, err := algofft.NewPlanReal(4096)
// ... use plan.Forward() and plan.Inverse()
```

### If you need float64 precision

You have two options:

1. **Short-term workaround**: Downcast to float32

   - ❌ Loses precision (~7 decimal digits)
   - ✅ Works today

2. **Wait for Phase 11.4** or contribute the implementation
   - ✅ Full precision (~15 decimal digits)
   - ⏳ Estimated 1-2 days to implement
   - See [PLAN.md:419-469](PLAN.md#L419-L469) for detailed tasks

### For production audio applications

The current `float32` real FFT is **production-ready** with:

- Zero allocations after plan creation
- SIMD acceleration (AVX2 on your system)
- Extensive testing (correctness, round-trip, properties)
- Benchmarking infrastructure

---

## Next Steps

1. **Try the existing API** - it's fully functional!
2. **If you need float64** - let me know if you want help implementing Phase 11.4
3. **Benchmark your use case** - use `just bench` to measure performance

Would you like me to:

- Show you example code for your specific use case?
- Help implement the generic float64 support?
- Explain any other part of the dispatch system?
