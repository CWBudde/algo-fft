# Codelet benchmarks

The measurement record behind every `Priority` and `RankLevel` in
`cmd/gencodelets/specs.go`. When a spec row changes, the evidence for it belongs
here.

Two harnesses produce everything below, and both run **all candidates for a size
inside one process**, because only same-process ratios survive a shared or
thermally-limited host:

- `BenchmarkCodeletCandidates64/128` (`internal/kernels/codelet_compare_bench_test.go`)
  walks the runtime registry, skipping `Priority < 0` and CPU-unsupported rows,
  and times both directions from a seeded RNG so runs are comparable across days.
- `scripts/bench_gated.sh` + `scripts/bench_gated_analyze.sh` (`just bench-gated
<sizes...>`) wrap that benchmark in a canary-gated sweep: each (precision,
  size) group is bracketed by a canary cell of known quiet-machine cost, group
  and cell order rotate per pass, and the analyser rejects any group whose two
  canaries disagree. Ratios are taken **within** a group and then medianed —
  never a ratio of medians. See `PLAN.md` §2.2 for why.

Two standing traps:

- `bench_gated.sh` reads `OUTDIR` from the environment; `bench_gated_analyze.sh`
  takes the directory as a **positional argument** and ignores `OUTDIR`. Passing
  `OUTDIR=` to the analyser silently analyses a stale directory. Check that
  accepted + rejected equals groups × passes before reading a single ratio.
- Recalibrate `GOOD` from the observed canary floor each round rather than
  reusing the previous value. A stale floor does not bias the ratios, but it
  lets in windows that should have been rejected.

Do not compare absolute figures across runs or hosts. Only the ratios travel.

## AVX2 tier (i7-1255U) — incumbent audit

Canary-gated sweeps, i7-1255U (Alder Lake, AVX2, no AVX-512), pinned to core 0.
Every registered power-of-two size has now been ranked. Each cell is the
**median within-group ratio to that size's incumbent**, forward; `< 1.00` means
the candidate beats the row the registry currently selects.

| sizes                       | date       | groups accepted | conditions                              |
| --------------------------- | ---------- | --------------- | --------------------------------------- |
| 8, 16, 32, 64, 16384        | 2026-07-30 | 159 / 160       | canary floor 1565 ns, 48-49 C           |
| 256, 512, 1024              | 2026-07-30 | 93 / 96         | canary floor 1593 ns, 50-51 C           |
| 128, 32768                  | 2026-07-29 | 30 / 32         |                                         |
| 512, 1024, 2048, 4096, 8192 | 2026-07-29 | 92 / 100        | superseded at 512/1024 by the row above |

Size 4 registers exactly one candidate per tier, so it has nothing to rank.

### What the audit changed

- **complex128 at n = 8 was mis-tuned.** `dit8_radix4_avx2` was the registered
  choice; `dit8_radix8_avx2` beats it at 0.970 forward / 0.859 inverse over 16
  groups and took the row. Reading it honestly: the forward gap is 0.2 ns and
  would not justify a change on its own, and the ~100 ns per-call plan dispatch
  swamps the whole difference; the inverse gap, 8.2 -> 7.0 ns, is the
  substantive part. `dit8_radix2_avx2` ties on inverse but loses forward, so
  radix-8 wins both directions rather than trading them. Two SSE2 rows also beat
  the old incumbent on forward, but registry ordering is SIMD-level major, so
  they could never be selected on an AVX2 host.
- **The no-tail radix-4 variant has been folded into the default kernel.** The
  2026-07-29 sweeps ranked `dit{512,2048,8192,32768}_radix4_notail_avx2` at
  0.86-0.93 of the then-incumbent, and `dit128_radix4fused_avx2` at 0.93-0.96.
  Those `notail` signatures no longer exist: the behaviour is now what
  `dit<N>_radix4_avx2` does, which is visible in the re-sweep — complex64 at
  n = 512 measures 415 ns where the old incumbent was 433 and the old `notail`
  candidate 409. `radix4fused` remains a separate row at 128 (both precisions)
  and 2048 complex64, where the fusion is a different trade.
- **Nothing at 256, 512 or 1024 needed a priority change.** All six incumbents
  reconfirmed, by 2.1x or more over every other candidate.

### Shadowed AVX2 candidates above 1.5x the winner

These are the deletion candidates in `PLAN.md` §4. None of them can be selected
on any amd64 CPU today.

|     n | prec | candidate                    |  fwd |  inv | note                                              |
| ----: | ---- | ---------------------------- | ---: | ---: | ------------------------------------------------- |
|    16 | c64  | `dit16_radix4_avx2`          | 1.74 | 1.76 |                                                   |
|    32 | c64  | `dit32_radix4_then2_avx2`    | 2.56 | 2.74 | loses to `..._sse3` (2.44)                        |
|    64 | c64  | `dit64_radix4_avx2`          | 2.29 | 2.36 |                                                   |
|   128 | c64  | `dit128_radix2_avx2`         | 3.99 | 4.25 |                                                   |
|   256 | c64  | `dit256_radix2_avx2`         | 2.10 | 2.21 |                                                   |
|   256 | c64  | `dit256_radix16_avx2`        | 3.13 | 3.29 |                                                   |
|   512 | c64  | `dit512_radix2_avx2`         | 2.13 | 2.21 |                                                   |
|   512 | c64  | `dit512_radix8_avx2`         | 3.01 | 3.71 |                                                   |
|   512 | c64  | `dit512_radix16x32_avx2`     | 3.78 | 4.18 |                                                   |
|  1024 | c64  | `dit1024_radix32x32_avx2`    | 8.06 | 9.81 | slower than its own pure-Go twin (7.40)           |
|  4096 | c64  | `dit4096_sixstep_avx2`       | 4.61 | 4.57 | stale crossover, not a bad kernel                 |
|  8192 | c64  | `dit8192_sixstep64x128_avx2` | 5.24 | 5.26 | stale crossover                                   |
| 16384 | c64  | `dit16384_sixstep_avx2`      | 2.37 | 2.38 | stale crossover                                   |
|    16 | c128 | `dit16_radix2_avx2`          | 2.38 | 2.79 | **loses to pure Go (1.97)** and to both SSE2 rows |
|    32 | c128 | `dit32_radix2_avx2`          | 3.02 | 3.55 | loses to both SSE2 rows                           |
|    64 | c128 | `dit64_radix2_avx2`          | 3.40 | 3.89 | loses to both SSE2 rows                           |
|   128 | c128 | `dit128_radix2_avx2`         | 3.41 | 3.92 | loses to `dit128_radix2_sse2` (3.14)              |
|   256 | c128 | `dit256_radix16_avx2`        | 2.61 | 2.40 |                                                   |
|   256 | c128 | `dit256_radix2_avx2`         | 3.95 | 4.47 | loses to `dit256_radix2_sse2` (3.70)              |
|   512 | c128 | `dit512_radix8_avx2`         | 2.49 | 2.66 |                                                   |
|   512 | c128 | `dit512_radix2_avx2`         | 3.99 | 4.32 |                                                   |

The `*_radix2_avx2` group is **structurally dominated, not badly written**:
radix-2 makes `log2 n` full passes where radix-4 makes half as many, and at
complex128 a 256-bit register holds only two elements, so there is no width left
to recover the difference. That several of them lose to their own SSE2 twins is
the symptom, not a second defect.

The higher-radix group is **implementation-limited**: `dit256_radix16_avx2`
loses at 2.6-3.1x despite radix-16 needing a quarter of radix-4's passes, and
only one of
`dit1024_radix32x32_avx2`'s two stages is vectorised, and the two size-512
kernels are the same shape. That result does not test whether a properly
vectorised radix-8 would win — see the `PLAN.md` §4 item that proposes one.

## AVX-512 tier (Xeon Gold 5218)

Evidence behind the `SIMDAVX512` rows, including the one deliberately disabled
entry. This is the only AVX-512 hardware reachable; Cascade Lake downclocks
under AVX-512, so it is a pessimistic machine for the tier.

### Measurement setup

|            |                                                                                                                            |
| ---------- | -------------------------------------------------------------------------------------------------------------------------- |
| CPU        | Intel Xeon Gold 5218 @ 2.30 GHz (Cascade Lake-SP), 2 vCPU                                                                  |
| Features   | `avx512f avx512dq avx512cd avx512bw avx512vl avx512_vnni`                                                                  |
| Go         | 1.25.5, linux/amd64                                                                                                        |
| Command    | `go test -run '^$' -bench '^BenchmarkCodeletCandidates{64,128}$/^size<N>$/' -benchtime=300ms -count=7 ./internal/kernels/` |
| Statistic  | median of 7                                                                                                                |
| Host state | idle (1-minute load average 0.22 at start, 1.03 at end)                                                                    |

`BenchmarkCodeletCandidates64/128` runs **every** registered candidate for a
size in a single process, so the AVX-512 codelet and the alternative it is
compared against are measured under identical conditions. That matters: on a
2-vCPU host, concurrent load inflates every row by up to 2-4x, and only
same-process ratios survive it. Runs were accepted only when the pre-existing
AVX2/SSE2/pure-Go rows reproduced the pre-merge baseline to within ~3%.

Do not compare absolute figures here against numbers measured in a different
run or on a different host.

### complex64

"Best other" is the fastest non-AVX-512 candidate registered at that size.

| Size | Codelet                      |   Forward |     Best other | Speedup |   Inverse |     Best other | Speedup |
| ---: | ---------------------------- | --------: | -------------: | ------: | --------: | -------------: | ------: |
|    8 | `dit8_radix8_avx512`         |  14.58 ns |   15.38 (avx2) |   1.05x |  15.44 ns |   16.68 (avx2) |   1.08x |
|   16 | `dit16_radix16_avx512`       |  22.44 ns |   25.43 (avx2) |   1.13x |  23.56 ns |   28.35 (avx2) |   1.20x |
|   32 | `dit32_radix4_then2_avx512`  |  31.19 ns |   67.13 (avx2) |   2.15x |  34.19 ns |   73.35 (avx2) |   2.15x |
|   64 | `dit64_radix2_avx512`        |  49.65 ns |  145.40 (avx2) |   2.93x |  51.95 ns |  167.60 (avx2) |   3.23x |
|   64 | `dit64_radix4_avx512`        |  49.60 ns |              — |       — |  52.08 ns |              — |       — |
|  128 | `dit128_radix8_then2_avx512` | 187.00 ns |  785.00 (sse3) |   4.20x | 194.00 ns |  859.90 (avx2) |   4.43x |
|  256 | `dit256_radix8_then2_avx512` | 386.30 ns | 1061.00 (avx2) |   2.75x | 413.10 ns | 1251.00 (avx2) |   3.03x |

The two size-64 kernels are the same 8x8 four-step transform under different
sub-FFT decompositions; they emit the same 148 instructions and measure within
0.1%, so `radix2` holds the higher priority arbitrarily. A single benchmark run
cannot separate them: the candidate benchmarked **first** in a process is
consistently ~3% slower, and registry order follows priority.

### complex128

Before this set, complex128 had no AVX-512 codelets at all.

| Size | Codelet                      |   Forward |    Best other | Speedup |   Inverse |     Best other | Speedup |
| ---: | ---------------------------- | --------: | ------------: | ------: | --------: | -------------: | ------: |
|    8 | `dit8_radix8_avx512`         |  13.81 ns |  15.87 (sse2) |   1.15x |  14.29 ns |   19.74 (avx2) |   1.38x |
|   16 | `dit16_radix4_avx512`        |  20.96 ns |  53.23 (avx2) |   2.54x |  24.08 ns |   61.43 (sse2) |   2.55x |
|   32 | `dit32_radix4_then2_avx512`  |  52.82 ns | 134.60 (avx2) |   2.55x |  54.89 ns |  165.90 (avx2) |   3.02x |
|   64 | `dit64_radix4_avx512`        |  92.28 ns | 335.60 (sse2) |   3.64x |  93.47 ns |  376.40 (sse2) |   4.03x |
|  128 | `dit128_radix4_then2_avx512` | 230.30 ns | 846.80 (avx2) |   3.68x | 225.60 ns | 1010.00 (sse2) |   4.48x |

Note that at sizes 8, 16, 64 and 128 the fastest pre-existing codelet is an
**SSE2** one, not the AVX2 one registered at higher priority — the complex128
AVX2 codelets are weaker than their priorities imply.

### The disabled entry: complex128 size 4

`dit4_radix4_avx512` is registered with `Priority: -1`.

| Direction |  AVX-512 |  Pure Go |     SSE2 |
| --------- | -------: | -------: | -------: |
| forward   | 11.54 ns |  8.27 ns | 10.86 ns |
| inverse   | 12.07 ns | 11.18 ns |        — |

It loses, so it must not be selected: codelet selection prefers the higher SIMD
level over priority, so registering it at a positive priority would make
AVX-512 hosts _slower_.

The cause is not instruction count — the kernel is 11 vector ops with no
multiply. A 4-point butterfly network across the four lanes of a single ZMM
needs two levels of lane-crossing `VSHUFF64X2` (3 cycles each), about 7 cycles
of serial shuffle latency, whereas the SSE2 kernel keeps each complex128 in its
own XMM and needs no shuffle at all for stage 1. Packing n = 4 into one register
trades free register-level parallelism for shuffle latency, and at 64 bytes
there is no data-movement win to pay for it.

The row is kept so the kernel is not lost. Note that negative-priority entries
are **skipped by the behavioural test sweeps**
(`TestForwardInverseAllCodeletsVsReference*`, `TestRoundTripAllCodelets*`,
`TestCodeletsZeroAlloc*` all `continue` on `Priority < 0`), so this kernel was
verified at a positive priority before being disabled.

### Why these kernels are fast

A ZMM holds 8 complex64 or 4 complex128, so at every size above the transform
is register-resident: 2 to 32 ZMM out of 32. The kernels **load once, run every
stage in registers, and store once**. Three consequences:

1. **No memory traffic between stages.** The AVX2 codelets at the same sizes
   need twice their register file (e.g. 64 complex128 needs 32 YMM) and spill.
   This is why AVX2 loses to SSE2 at complex128 sizes 64 and 128.
2. **No bit-reversal pass and no permutation table.** The input permutation is
   absorbed into _which load lands in which register_: lane _m_ of the
   contiguous ZMM loads already is stage-1 radix-4 butterfly _m_'s input
   quadruple. The residual movement is a single 4x4 or 8x8 lane transpose.
3. **`scratch` is never touched**, so the kernels need no in-place
   (`dst == src`) branch at all.

The remaining win is **shuffle elimination rather than wider arithmetic**. The
AVX2 size-16 complex64 kernel is shuffle-port bound, not FP bound (~53 port-5
uops against ~43 FP ops that retire two per cycle); ~15 of those shuffles exist
only to assemble strided twiddle vectors and 8 more form a final transpose.
Embedding the twiddles as rodata constants and folding the digit reversal into
the last layer's operand permutes removes all three cost centres.

Twiddle constants are computed in float64 from the exact angle and rounded
**once** to the target type. Deriving them by repeated multiplication or
squaring in the target precision is measurably worse: an earlier size-256
kernel that squared in float32 passed every codelet-registry test and still
failed `TestForwardMatchesReferenceSmall`, whose absolute 5e-3 tolerance is
~1.3e-6 relative at n=256. Exact rodata constants were both faster (~35 fewer
instructions per body) and 7x more accurate.

Cascade Lake's 512-bit downclock did not erase these wins: on this Gold 5218 the
AVX-512 turbo is only about one bin below AVX2.

### Instruction set constraint

All of these kernels use **AVX512F only**. `internal/cpu` derives
`Features.HasAVX512` from `golang.org/x/sys/cpu.X86.HasAVX512`, which is
CPUID leaf 7 EBX bit 16 — AVX512 **Foundation** alone. A kernel using AVX512DQ
or AVX512BW instructions behind that gate would fault with #UD on an
AVX512F-only part (Knights Landing / Knights Mill).

Avoid these; the F-only equivalents are bit-identical when no mask register is
used:

| AVX512DQ (do not use)            | AVX512F (use instead)            |
| -------------------------------- | -------------------------------- |
| `VXORPS` on ZMM                  | `VPXORD`                         |
| `VXORPD` on ZMM                  | `VPXORQ`                         |
| `VANDPS` / `VANDPD` on ZMM       | `VPANDD` / `VPANDQ`              |
| `VINSERTF64X2` / `VEXTRACTF64X2` | `VINSERTF32X4` / `VEXTRACTF32X4` |
| `VINSERTF32X8` / `VEXTRACTF32X8` | `VINSERTF64X4` / `VEXTRACTF64X4` |

`VINSERTF32X4` selects the same 128-bit lane via `imm[1:0]`. The 512-bit
`VSHUFF32X4` / `VSHUFF64X2` forms are AVX512F; only the 256-bit VL forms
require AVX512VL.

Self-check:

```bash
grep -nE '^\s*(VXORPS|VXORPD|VANDPS|VANDPD|VORPS|VORPD|VINSERTF64X2|VEXTRACTF64X2|VINSERTF32X8|VEXTRACTF32X8)\b.*Z[0-9]+' internal/asm/amd64/avx512_*.s
```

Anything it prints violates the contract in `internal/asm/amd64/decl_avx512.go`.

Two further notes for anyone extending these kernels:

- `go tool objdump` **cannot decode EVEX** — it prints `?` for the 0x62 prefix.
  Use GNU `objdump -d -M intel` on `go test -c` output instead.
- `MOVD AX, X26` does not assemble; XMM16-31 require EVEX. Use
  `VBROADCASTSS ·sym(SB), Zn`. Embedded broadcast is spelled
  `VMULPS.BCST ·sym(SB), Zsrc, Zdst`.

### Sizes without a size-specific codelet

Power-of-two sizes with no registered AVX-512 codelet are still served on
AVX-512 hosts by the generic AVX-512 radix-2 DIT kernel
(`internal/asm/amd64/avx512_f32_generic.s`, `avx512_f64_generic.s`) through the
dispatch tier in `internal/fft/kernels_amd64_avx512.go`. That kernel is
additionally bound as a complex64 codelet at sizes 1024, 4096, 8192 and 16384;
see `internal/kernels/dit_avx512_amd64.go`.

### AVX-512 against the best AVX2 codelet (complex64)

Pinned, idle host. **These numbers predate the 256-bit AVX2 radix-4 kernels**,
which moved the AVX2 column substantially; re-measure before acting on them.
The AVX-512 codelet is radix-2 while every AVX2 winner here is radix-4, so this
is an algorithm comparison, not a vector-width one.

| size  | AVX-512 fwd | best AVX2 fwd | fwd Δ  | AVX-512 inv | best AVX2 inv | inv Δ     |
| ----- | ----------- | ------------- | ------ | ----------- | ------------- | --------- |
| 1024  | 9151 ns     | 8210 ns       | +11.5% | 10662 ns    | 10141 ns      | +5.1%     |
| 4096  | 40786 ns    | 39726 ns      | +2.7%  | 45995 ns    | 50651 ns      | **−9.2%** |
| 8192  | 89129 ns    | 83269 ns      | +7.0%  | 96315 ns    | 102567 ns     | **−6.1%** |
| 16384 | 199838 ns   | 188084 ns     | +6.2%  | 221941 ns   | 233577 ns     | **−5.0%** |

Consequence: the AVX-512 rows sit at `Priority 10` against 24–28 for AVX2, so
the registry never selects them even on an AVX-512 CPU. For forward that is
currently right; it discards a real 5–9% on inverse at n >= 4096. See `PLAN.md`
§6 for the open item.
