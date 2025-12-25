# Benchmarks

This file tracks baseline benchmark results for `algoforge`. Results are
hardware- and Go-version-specific.

## How to Update

1. Run benchmarks and update the baseline output file:

```bash
scripts/run_benchmarks.sh benchmarks/baseline.txt
```

2. Regenerate the table below:

```bash
scripts/bench_md.sh benchmarks/baseline.txt > /tmp/benchmarks.md
```

3. Replace the "Baseline Results" table with the regenerated output.

## Baseline Results

**Date**: 2025-12-24  
**Go**: go1.25.0  
**OS/Arch**: linux/amd64  
**CPU**: 12th Gen Intel(R) Core(TM) i7-1255U

| Benchmark                                                |   ns/op |    MB/s |    B/op | allocs/op |
| -------------------------------------------------------- | ------: | ------: | ------: | --------: |
| BenchmarkPlanForward_16-12                               |   117.4 | 1090.55 |       0 |         0 |
| BenchmarkPlanForward_64-12                               |   503.5 | 1016.91 |       0 |         0 |
| BenchmarkPlanForward_256-12                              |    2617 |  782.46 |       0 |         0 |
| BenchmarkPlanForward_1024-12                             |   12850 |  637.50 |       0 |         0 |
| BenchmarkPlanForward_4096-12                             |   70165 |  467.01 |       0 |         0 |
| BenchmarkPlanForward_16384-12                            |  401326 |  326.60 |       0 |         0 |
| BenchmarkPlanForward_65536-12                            | 1992006 |  263.20 |       0 |         0 |
| BenchmarkPlanInverse_16-12                               |   213.9 |  598.35 |       0 |         0 |
| BenchmarkPlanInverse_64-12                               |   833.7 |  614.13 |       0 |         0 |
| BenchmarkPlanInverse_256-12                              |    4409 |  464.45 |       0 |         0 |
| BenchmarkPlanInverse_1024-12                             |   23178 |  353.43 |       0 |         0 |
| BenchmarkPlanInverse_4096-12                             |   96256 |  340.43 |       0 |         0 |
| BenchmarkPlanInverse_16384-12                            |  502509 |  260.84 |       0 |         0 |
| BenchmarkPlanInverse_65536-12                            | 2325128 |  225.49 |       0 |         0 |
| BenchmarkNewPlan_16-12                                   |    1497 |       - |    1776 |        30 |
| BenchmarkNewPlan_256-12                                  |   13380 |       - |   18480 |        44 |
| BenchmarkNewPlan_4096-12                                 |  351846 |       - |  532916 |        70 |
| BenchmarkNewPlan_16384-12                                |  827891 |       - | 1286583 |        74 |
| BenchmarkReferenceDFT_16-12                              |    4015 |   31.88 |     640 |         3 |
| BenchmarkReferenceDFT_64-12                              |   75484 |    6.78 |    2560 |         3 |
| BenchmarkReferenceDFT_256-12                             | 1096580 |    1.87 |   10240 |         3 |
| BenchmarkPlanForward_MemStats_1024-12                    |   13234 |  619.01 |       0 |         0 |
| BenchmarkPlanReusePatterns_1024/ReusePlanReuseBuffers-12 |   13291 |  616.37 |       0 |         0 |
| BenchmarkPlanReusePatterns_1024/ReusePlanAllocBuffers-12 |   18533 |  442.02 |   16384 |         2 |
| BenchmarkPlanReusePatterns_1024/NewPlanEachIter-12       |   64360 |  127.28 |   89872 |        54 |
| BenchmarkPooledVsRegular/Pooled/64-12                    |    6463 |       - |    3997 |        38 |
| BenchmarkPooledVsRegular/Regular/64-12                   |    4755 |       - |    5024 |        37 |
| BenchmarkPooledVsRegular/Pooled/256-12                   |   14831 |       - |   14013 |        45 |
| BenchmarkPooledVsRegular/Regular/256-12                  |   11819 |       - |   18480 |        44 |
| BenchmarkPooledVsRegular/Pooled/1024-12                  |   43724 |       - |   54797 |        53 |
| BenchmarkPooledVsRegular/Regular/1024-12                 |   49471 |       - |   73488 |        52 |
| BenchmarkPooledVsRegular/Pooled/4096-12                  |  266400 |       - |  453772 |        72 |
| BenchmarkPooledVsRegular/Regular/4096-12                 |  265161 |       - |  532915 |        70 |
| BenchmarkNewPlan_64-12                                   |    3523 |       - |    5024 |        37 |
| BenchmarkNewPlan_1024-12                                 |   48279 |       - |   73488 |        52 |
| BenchmarkNewPlan_65536-12                                | 4480630 |       - | 8299029 |        98 |
| BenchmarkComputeTwiddleFactors64_16-12                   |   276.9 |       - |     128 |         1 |
| BenchmarkComputeTwiddleFactors64_256-12                  |    4329 |       - |    2048 |         1 |
| BenchmarkComputeTwiddleFactors64_1024-12                 |   18623 |       - |    8192 |         1 |
| BenchmarkComputeTwiddleFactors64_4096-12                 |  120621 |       - |   32768 |         1 |
| BenchmarkComputeTwiddleFactors64_65536-12                | 2126632 |       - |  524289 |         1 |
| BenchmarkComputeTwiddleFactors128_1024-12                |   30968 |       - |   16384 |         1 |
| BenchmarkComputeBitReversalIndices_16-12                 |   88.96 |       - |     128 |         1 |
| BenchmarkComputeBitReversalIndices_256-12                |    3011 |       - |    2048 |         1 |
| BenchmarkComputeBitReversalIndices_1024-12               |    9904 |       - |    8192 |         1 |
| BenchmarkComputeBitReversalIndices_4096-12               |   36147 |       - |   32768 |         1 |
| BenchmarkComputeBitReversalIndices_65536-12              |  651953 |       - |  524288 |         1 |
| BenchmarkPackedTwiddleLookup_Radix4_4096-12              |   908.9 |       - |       0 |         0 |
| BenchmarkPackedTwiddleLookup_Radix8_4096-12              |   907.7 |       - |       0 |         0 |
| BenchmarkPackedTwiddleLookup_Radix16_4096-12             |   767.4 |       - |       0 |         0 |
| BenchmarkPackedTwiddleLookup_Radix4_65536-12             |    9771 |       - |       0 |         0 |
| BenchmarkPackedTwiddleLookup_Radix8_65536-12             |    5103 |       - |       0 |         0 |
| BenchmarkPackedTwiddleLookup_Radix16_65536-12            |    9325 |       - |       0 |         0 |
| BenchmarkBaseTwiddleLookup_Radix4_4096-12                |    9362 |       - |       0 |         0 |
| BenchmarkBaseTwiddleLookup_Radix8_4096-12                |    9535 |       - |       0 |         0 |
| BenchmarkBaseTwiddleLookup_Radix16_4096-12               |    9718 |       - |       0 |         0 |
| BenchmarkBaseTwiddleLookup_Radix4_65536-12               |  157334 |       - |       0 |         0 |
| BenchmarkBaseTwiddleLookup_Radix8_65536-12               |   72736 |       - |       0 |         0 |
| BenchmarkBaseTwiddleLookup_Radix16_65536-12              |  177567 |       - |       0 |         0 |

## AVX2 Performance (Phase 14.2)

Build with `-tags fft_asm` to enable AVX2 optimizations on amd64.
These results compare the baseline Pure Go implementation with the AVX2 optimized version.

**Date**: 2025-12-25
**CPU**: 12th Gen Intel(R) Core(TM) i7-1255U

### complex64 Forward FFT

| Size  | Pure Go (ns) | Pure Go (MB/s) | AVX2 (ns) | AVX2 (MB/s) | Speedup  |
| ----: | -----------: | -------------: | --------: | ----------: | -------: |
| 64    | 602          | 850            | 185       | 2763        | **3.3x** |
| 256   | 2988         | 685            | 770       | 2660        | **3.9x** |
| 1024  | 14550        | 563            | 3228      | 2537        | **4.5x** |
| 4096  | 69113        | 474            | 14505     | 2259        | **4.8x** |
| 16384 | 336675       | 389            | 75015     | 1747        | **4.5x** |

### complex128 Forward FFT (AVX2)

| Size | AVX2 (ns) | AVX2 (MB/s) |
| ---: | --------: | ----------: |
| 64   | 229       | 4477        |
| 256  | 1020      | 4018        |
| 1024 | 4511      | 3632        |
| 4096 | 22795     | 2875        |

### Key Optimizations Applied

1. **SIMD Vectorization**: Process 4 complex64 or 2 complex128 per iteration using YMM registers
2. **FMA Instructions**: `VFMADDSUB231PS/PD` for fused multiply-add in butterfly computation
3. **2x Loop Unrolling**: Process 8 butterflies in strided path to reduce loop overhead
4. **Manual Twiddle Gathering**: Use scalar loads + pack for strided twiddle access
5. **Zero Allocations**: All transforms are allocation-free after plan creation

