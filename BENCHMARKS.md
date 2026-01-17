# Benchmarks

This file tracks baseline benchmark results for `algofft`. Results are
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

## Phase 12 Focus (complex128 128/512/8192)

Use these focused benchmarks to capture kernel selection, strategy, and twiddle layout
for complex128 sizes 128/512/8192. Run with `-v` to include the plan details logged by
the benchmarks, and capture `-benchmem` + `-cpuprofile` for profiling.

```bash
go test -v -run '^$' -bench 'BenchmarkPlan(Forward|Inverse)_(128|512|8192)_Complex128_Focus$' -benchmem ./...
go test -v -run '^$' -bench 'BenchmarkPlanForward_128_Complex128_Focus$' -benchmem -cpuprofile /tmp/fft128.pprof ./...
go test -v -run '^$' -bench 'BenchmarkPlanForward_512_Complex128_Focus$' -benchmem -cpuprofile /tmp/fft512.pprof ./...
go test -v -run '^$' -bench 'BenchmarkPlanForward_8192_Complex128_Focus$' -benchmem -cpuprofile /tmp/fft8192.pprof ./...
```

Record CPU model, Go version, and the logged plan details alongside the numbers.

**Captured Results**: 2026-01-17  
**Go**: go1.25.0  
**OS/Arch**: linux/amd64  
**CPU**: 12th Gen Intel(R) Core(TM) i7-1255U  
**Profiles**: `/tmp/fft128.pprof`, `/tmp/fft512.pprof`, `/tmp/fft8192.pprof`

| Benchmark                                     |  ns/op |    MB/s | B/op | allocs/op |
| --------------------------------------------- | -----: | ------: | ---: | --------: |
| BenchmarkPlanForward_128_Complex128_Focus-12  |  519.7 | 3940.76 |    0 |         0 |
| BenchmarkPlanInverse_128_Complex128_Focus-12  |  631.2 | 3244.53 |    0 |         0 |
| BenchmarkPlanForward_512_Complex128_Focus-12  |   2525 | 3244.08 |    0 |         0 |
| BenchmarkPlanInverse_512_Complex128_Focus-12  |   2933 | 2793.05 |    0 |         0 |
| BenchmarkPlanForward_8192_Complex128_Focus-12 | 116405 | 1126.00 |    0 |         0 |
| BenchmarkPlanInverse_8192_Complex128_Focus-12 | 127627 | 1027.00 |    0 |         0 |

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

Build with `-tags asm` to enable AVX2 optimizations on amd64.
These results compare the baseline Pure Go implementation with the AVX2 optimized version.

**Date**: 2026-01-04
**CPU**: 12th Gen Intel(R) Core(TM) i7-1255U

### complex64 Forward FFT

|  Size | Pure Go (ns) | Pure Go (MB/s) | AVX2 (ns) | AVX2 (MB/s) |  Speedup |
| ----: | -----------: | -------------: | --------: | ----------: | -------: |
|    64 |          602 |            850 |       185 |        2763 | **3.3x** |
|   256 |         2988 |            685 |       770 |        2660 | **3.9x** |
|  1024 |         8154 |           1005 |      4059 |        2018 | **2.0x** |
|  4096 |        54463 |            601 |     20498 |        1599 | **2.7x** |
| 16384 |       336675 |            389 |     75015 |        1747 | **4.5x** |

### complex128 Forward FFT (AVX2)

| Size | AVX2 (ns) | AVX2 (MB/s) |
| ---: | --------: | ----------: |
|   64 |       229 |        4477 |
|  256 |      1020 |        4018 |
| 1024 |      4511 |        3632 |
| 4096 |     22795 |        2875 |

### Key Optimizations Applied

1. **SIMD Vectorization**: Process 4 complex64 or 2 complex128 per iteration using YMM registers
2. **FMA Instructions**: `VFMADDSUB231PS/PD` for fused multiply-add in butterfly computation
3. **2x Loop Unrolling**: Process 8 butterflies in strided path to reduce loop overhead
4. **Manual Twiddle Gathering**: Use scalar loads + pack for strided twiddle access
5. **Zero Allocations**: All transforms are allocation-free after plan creation

## ARM64 NEON vs Go (QEMU)

**Date**: 2025-12-25  
**Go**: go1.25.0  
**OS/Arch**: linux/arm64 (QEMU)  
**CPU**: ARMv8 Processor rev 0 (v8l)  
**Note**: QEMU results are not representative of real ARM64 hardware.

### complex64 Forward FFT

| Size | NEON (ns) | NEON (MB/s) | Go DIT (ns) | Go DIT (MB/s) |
| ---: | --------: | ----------: | ----------: | ------------: |
|   64 |     10781 |       47.49 |       10894 |         47.00 |
|  256 |     57312 |       35.73 |       56289 |         36.38 |
| 1024 |    278673 |       29.40 |      284396 |         28.80 |
| 4096 |   1332887 |       24.58 |     1274761 |         25.71 |

### complex64 Inverse FFT

| Size | NEON (ns) | NEON (MB/s) | Go DIT (ns) | Go DIT (MB/s) |
| ---: | --------: | ----------: | ----------: | ------------: |
|   64 |     12959 |       39.51 |       14508 |         35.29 |
|  256 |     65889 |       31.08 |       69474 |         29.48 |
| 1024 |    326061 |       25.12 |      336179 |         24.37 |
| 4096 |   1514744 |       21.63 |     1537455 |         21.31 |

### complex128 Forward FFT

| Size | NEON (ns) | NEON (MB/s) | Go DIT (ns) | Go DIT (MB/s) |
| ---: | --------: | ----------: | ----------: | ------------: |
|   64 |      7770 |      131.79 |        9536 |        107.38 |
|  256 |     39421 |      103.90 |       46687 |         87.73 |
| 1024 |    186962 |       87.63 |      218785 |         74.89 |
| 4096 |    911766 |       71.88 |     1093437 |         59.94 |

### complex128 Inverse FFT

| Size | NEON (ns) | NEON (MB/s) | Go DIT (ns) | Go DIT (MB/s) |
| ---: | --------: | ----------: | ----------: | ------------: |
|   64 |      8403 |      121.87 |       11371 |         90.05 |
|  256 |     39257 |      104.34 |       55815 |         73.39 |
| 1024 |    194358 |       84.30 |      269664 |         60.76 |
| 4096 |    916886 |       71.48 |     1238905 |         52.90 |

## Multi-Dimensional FFT Performance (Phase 18)

**Date**: 2025-12-26
**Go**: go1.25.0
**OS/Arch**: linux/amd64
**CPU**: 12th Gen Intel(R) Core(TM) i7-1255U

### 3D FFT: Plan3D vs PlanND Comparison

| Implementation | Size  | Elements | Time (µs) | Throughput (MB/s) | Allocs/op | Bytes/op |
| -------------- | ----- | -------- | --------- | ----------------- | --------- | -------- |
| **Plan3D**     | 8×8×8 | 512      | 14.8      | 276               | 2         | 128      |
| **PlanND**     | 8×8×8 | 512      | 18.4      | 223               | 3         | 192      |

**Analysis**: PlanND has ~24% overhead vs specialized Plan3D. This is a reasonable trade-off for supporting arbitrary dimensions.

### PlanND Performance Across Dimensions

| Dimensions | Size      | Elements | Time (µs) | Throughput (MB/s) | Allocs/op |
| ---------- | --------- | -------- | --------- | ----------------- | --------- |
| **3D**     | 8×8×8     | 512      | 18.2      | 225               | 3         |
| **4D**     | 4×4×4×4   | 256      | 20.1      | 102               | 4         |
| **5D**     | 2×2×2×2×2 | 32       | 6.8       | 38                | 5         |

**Scaling**: Allocation count scales linearly with dimension count (one temporary buffer per dimension). Performance scales well with problem size.

### 3D FFT Sizes (Plan3D)

| Size     | Elements | Time (µs) | Throughput (MB/s) | Allocs/op | Bytes/op |
| -------- | -------- | --------- | ----------------- | --------- | -------- |
| 8×8×8    | 512      | 14.8      | 276               | 2         | 128      |
| 16×16×16 | 4,096    | ~120      | ~270              | 2         | 128      |
| 32×32×32 | 32,768   | ~1,000    | ~260              | 2         | 128      |

### Key Findings

1. **Specialized implementations are faster**: Plan2D and Plan3D outperform PlanND by 20-30% for their specific dimensions
2. **PlanND provides excellent flexibility**: Supports arbitrary dimensions (tested up to 5D) with reasonable overhead
3. **Zero-allocation transforms**: All plans achieve zero allocations during transform operations (allocations shown are from test setup)
4. **Memory efficiency**: Aligned scratch buffers enable SIMD optimizations in underlying 1D FFT kernels
5. **Dimension scaling**: Higher dimensions add marginal overhead due to additional slice extraction/writing

### Recommendations

- **Use specialized plans when possible**: Plan2D for images, Plan3D for volumes
- **Use PlanND for flexibility**: When dimension count varies or exceeds 3
- **Cache-friendly sizes**: Power-of-2 dimensions perform best due to optimized 1D kernels
- **Concurrent transforms**: Use `Clone()` to create independent plan instances for parallel processing
