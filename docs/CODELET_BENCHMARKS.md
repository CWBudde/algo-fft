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
- `GOOD` is per (host, toolchain, canary) and is no longer a literal in the
  script. Derive it on an **idle** machine by running the sweep script with
  `CALIBRATE=1`, which takes the minimum over `CALREPS` samples and records it
  in `benchmarks/canary-calibration.tsv`; the sweep refuses to run rather than
  inherit a floor that does not belong to the machine. A stale floor
  does not bias the ratios, but it lets in windows that should have been
  rejected — and it fails silently in that direction, which is why calibrating
  under load is worse than not calibrating at all.

  **`GOOD` values quoted in the sections below are not comparable to today's.**
  Everything before 2026-07-31 was calibrated against the old canary,
  `BenchmarkDITComplex128/Size256/Radix16/Forward` (~1.6 µs); the canary is now
  the frozen `BenchmarkGateCanary` (~13 µs), so only the gate _ratios_ carry
  across the change.

Do not compare absolute figures across runs or hosts. Only the ratios travel.

## Generic tier (i7-1255U, `-tags purego`) — the radix-8 ladder

Evidence behind the nine `dit<N>_radix8ladder_generic` rows and behind deleting
`dit256_radix16_generic`. Sweep of 2026-07-30: `GOOD=1660` from the observed
floor, `PASSES=8`, `-tags fftprobe,purego`, 51–52 °C, **110 of 112 groups
accepted** (a second sweep covering 64/128/256 accepted 48 of 48). Each cell is
the median within-group ratio to that size's pure-Go incumbent; `< 1.00` means
the ladder beat the row the registry then selected.

|     n | passes 8 : 4 |   c64 fwd | c64 inv† |  c128 fwd | c128 inv† |
| ----: | ------------ | --------: | -------: | --------: | --------: |
|    64 | 1:2          |     1.112 |    0.894 |     1.110 |     0.875 |
|   128 | 2:3          |     1.054 |    0.881 |     1.112 |     0.896 |
|   256 | 2:4          |     1.004 |    0.724 |     1.005 |     0.852 |
|   512 | 3:5          | **0.807** |    0.823 |     0.986 |     0.867 |
|  1024 | 4:5          | **0.900** |    0.933 |     1.097 |     0.913 |
|  2048 | 4:6          |     0.988 |    0.737 | **0.928** |     0.829 |
|  4096 | 4:6          | **0.859** |    0.766 | **0.792** |     0.738 |
|  8192 | 5:7          | **0.889** |    0.882 | **0.934** |     0.835 |
| 16384 | 5:7          | **0.790** |    0.773 | **0.834** |     0.807 |
| 32768 | 5:8          |     1.062 |    0.961 |     1.084 |     0.931 |

† **The inverse column is stale and must not be used to justify a row.** It was
taken before the `1/n` scaling sweep, when the incumbents still finished the
inverse with a separate pass doing `MulComplex64(dst[i], complex(1/n, 0))` while
the ladder already folded `1/n` into stage 1. Removing that pass from the
incumbents erased most of the gap: a partial re-sweep (contended, 18 of 48
groups rejected — not usable for a decision) put
`dit512_radix4_then2_generic` back in front at n = 512 complex128 at 0.921
forward / 1.001 inverse. The **forward** column is unaffected, because the
scaling sweep touched no forward path.

So nine rows were promoted immediately, being exactly those the ladder wins on
forward alone: 512/1024/4096/8192/16384 complex64 and 2048/4096/8192/16384
complex128. The four cells where forward is a tie and the case rested on the
inverse — 256 and 2048 complex64, 256 and 512 complex128 — were held back for a
clean re-measurement, which follows.

### The four tie rows, re-measured after the scaling sweep

Same day, once the `1/n` sweep had landed, so **both sides of every comparison
are current** and the inverse column is usable again. `GOOD=1700` from a fresh
canary floor of 1693, `GATE=1.15`, `PASSES=8`, `-tags fftprobe,purego`, 53 °C,
core 0, **46 of 48 groups accepted** (2 over gate, 0 drift, 0 incomplete).

|    n | prec |   fwd rel |   inv rel | incumbent                      |
| ---: | ---- | --------: | --------: | ------------------------------ |
|  256 | c64  |     1.014 | **0.784** | `dit256_radix4_generic`        |
| 2048 | c64  | **0.968** | **0.764** | `dit2048_radix4_then2_generic` |
|  256 | c128 |     1.003 | **0.888** | `dit256_radix4_generic`        |
|  512 | c128 |     1.002 | **0.886** | `dit512_radix4_then2_generic`  |

All four are promoted. Forward is a tie within noise at three of them and a win
at 2048 complex64; the inverse win of 11–22% is what carries them, and this time
it is measured against an incumbent that no longer pays the trailing scaling
pass. That kills the earlier partial re-sweep's 0.921 forward for
`dit512_radix4_then2_generic` at n = 512 complex128 — that run was contended and
should not have been read at all.

The same sweep re-confirms two of the nine earlier promotions from the other
direction, now that the ladder is the incumbent: at n = 512 complex64
`dit512_radix4_then2_generic` measures 1.224/1.100 against it, and at n = 2048
complex128 1.074/1.187.

Thirteen of the ladder's twenty registered cells are therefore production rows.
What remains under `-tags fftprobe` is 64 and 128 (both precisions), 1024
complex128, and 32768 (both precisions).

Three findings the sweep produced beyond the ranking:

- **The butterfly was the whole problem, not the radix.** At n = 512 complex64
  the old `dit512_radix8_generic` measures 2714/3220 ns against the ladder's
  2371/2469 — same radix, same pass count, 13%/23% apart. The old kernel spends
  a full complex multiply on `W_8^2 = −i`, which is a free swap-and-negate, and
  on `W_8^{1,3}`, which cost half that, and computes each of those three
  products twice per butterfly.
- **n = 32768 loses forward while having the best pass ratio (5:8).** Its last
  radix-8 stage holds eight live streams 4096 elements apart — 32 KiB at
  complex64, 64 KiB at complex128 — so all eight map to the same L1 sets. Same
  collision the radix-4 fused tail hits at n = 2048 complex128 with a 4 KiB
  stride (see `forwardRadix4AVX2FusedComplex64`). Blocking the stage would test
  it.
- **`dit256_radix16_generic` was mis-tuned in both precisions.** At complex64 it
  was the _selected_ row while losing to plain `dit256_radix4_generic` at 0.682
  forward; at complex128 it sat at 1.677. Both spec rows are deleted per §2.1
  rule 6. The kernel functions stayed at the time, because
  `BenchmarkDITComplex128/Size256/Radix16/Forward` was then the canary every
  gated sweep was calibrated against and removing the code would have
  invalidated `GOOD`.

  That reason expired on 2026-07-31. Using a codelet from the package under test
  as the canary is precisely what let `GOOD` go stale twice — every successful
  optimisation moved the reference — so the canary is now the frozen,
  dependency-free `BenchmarkGateCanary`. The size-16 radix kernels are no longer
  load-bearing for measurement; whether they still earn their place is an
  ordinary dead-code question, not a measurement constraint.

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
- **The default radix-4 kernel absorbed most of the tail gap — but the no-tail
  probe was not folded in, and cannot be.** The 2026-07-29 sweeps ranked
  `dit{512,2048,8192,32768}_radix4_notail_avx2` at 0.86-0.93 of the
  then-incumbent, and `dit128_radix4fused_avx2` at 0.93-0.96. The default kernel
  then got materially faster: complex64 at n = 512 measures 415 ns in the
  re-sweep where the old incumbent was 433 and the old `notail` candidate 409.
  `radix4fused` remains a separate row at 128 (both precisions) and 2048
  complex64, where the fusion is a different trade.

  What did **not** happen is the `notail` behaviour becoming the default. That
  probe skips the tail combine and therefore computes the wrong answer on
  purpose — it is a bound on what any fusion could recover, never a shippable
  candidate. Its signatures still exist
  (`internal/kernels/radix4_avx2_tail_probe_amd64.go`, registered only under
  `-tags fftprobe` at priority 85, below the incumbent's 90, so `registry.Lookup`
  never returns it). The stage speedup lifted the probe and the default kernel
  together, which is why the _ratio_ barely moved even though every absolute
  number dropped: the later re-check below still finds `notail` fastest at 512,
  2048, 8192 and 32768 in both precisions, at 0.84-0.93.

  Practical consequence, because it costs a debugging detour every time: a
  `-tags fftprobe` correctness run **fails by design** on
  `dit<N>_radix4_notail_avx2` at 128/512/2048/8192/32768 in both precisions.
  That is the probe working as intended, not a regression in whatever you just
  changed.

- **Nothing at 256, 512 or 1024 needed a priority change.** All six incumbents
  reconfirmed, by 2.1x or more over every other candidate.

### The size-generic AVX2 radix-8 ladder

`internal/asm/amd64/avx2_f{32,64}_radix8.s`, swept 2026-07-30 against each
size's registered incumbent: 16 groups (8 sizes x 2 precisions) x 8 passes,
**118 accepted, 10 rejected (5 over gate, 5 drift), 0 incomplete** — which is
the full 128, so nothing went unaccounted for. Canary floor 1600 ns at gate
1.12, load 1.3 at start, 47-49 C. Ratios are within-group, forward / inverse.

|     n | complex64         | complex128        |
| ----: | ----------------- | ----------------- |
|   256 | **0.953 / 0.940** | 1.000 / 0.998     |
|   512 | **0.903 / 0.934** | **0.933 / 0.979** |
|  1024 | **0.983 / 0.940** | 0.982 / 1.025     |
|  2048 | **0.953 / 0.946** | **0.942 / 0.922** |
|  4096 | 1.067 / 1.075     | 0.973 / 1.007     |
|  8192 | 1.078 / 1.034     | 1.038 / 1.017     |
| 16384 | 1.016 / 1.042     | 1.012 / 0.996     |
| 32768 | 1.011 / 1.030     | **0.931 / 0.985** |

Bold cells took a spec row at priority 95. Seven of sixteen: the pure-Go
result did **not** transfer wholesale, and it was never safe to assume it
would — the pure-Go radix-4 the prototype beat is a much weaker opponent than
the 256-bit radix-4 kernel here.

**The complex64 column is decided entirely by the last stage's stride.**
Writing out `m * 8` bytes between the eight streams of the final radix-8 stage:

|     n | last m | stride | result |
| ----: | -----: | -----: | -----: |
|   256 |      8 |   64 B |    win |
|   512 |     64 |  512 B |    win |
|  1024 |     64 |  512 B |    win |
|  2048 |     64 |  512 B |    win |
|  4096 |    512 |  4 KiB |   loss |
|  8192 |    512 |  4 KiB |   loss |
| 16384 |    512 |  4 KiB |   loss |
| 32768 |   4096 | 32 KiB |   loss |

Every cell at or below 512 B wins and every cell at or above 4 KiB loses, with
no exceptions in either direction. Eight streams a multiple of 4 KiB apart all
map to one L1 set — the same collision `forwardRadix4AVX2FusedComplex64`
documents at n = 2048 complex128, and radix-8 is twice as exposed because it
doubles the live streams from four to eight. This is a prediction the probe's
header made before the sweep ran, not a story fitted afterwards.

complex128 does **not** follow the rule cleanly: n = 32768 wins at a 64 KiB
stride. At 512 KiB the working set is far past L2, so the cell is decided by
memory traffic, where the ladder's third fewer passes wins outright. Treat the
stride rule as established for complex64 and as a hypothesis for complex128.

#### Blocking the wide stages does not work — and the stride rule is not about conflicts

The obvious reading of that table is that the far-apart streams collide in one
L1 set, so the fix is to stop interleaving them. Written and measured
2026-07-30, **it is not.**

The tiled version walks j in chunks of 64, copies each stream's chunk into a
contiguous tile — applying that stream's twiddle during the gather, so a copy
run touches two streams instead of the fifteen a stage really interleaves
(eight data plus seven twiddle planes, all m apart) — runs the butterflies
inside the tile and copies back. Arithmetic identical, bit for bit. Swept
against the unblocked ladder in the same group, `-tags fftprobe,purego`, 8
groups x 12 passes, **96 accepted, 0 rejected**, canary floor 1580 ns at gate
1.12, 48-50 C. Ratios are blocked / unblocked, so above 1.00 means blocking
cost time:

|     n | complex64 fwd / inv | complex128 fwd / inv |
| ----: | ------------------- | -------------------- |
|  4096 | 1.078 / 1.083       | 1.115 / 1.105        |
|  8192 | 1.065 / 1.073       | 1.118 / 1.087        |
| 16384 | 1.074 / 1.065       | 1.102 / 1.088        |
| 32768 | 1.103 / 1.132       | 1.138 / 1.141        |

Blocking costs 6.5-14%, every cell, both precisions, both directions. It is
worst at n = 32768, which is exactly the cell the collision story predicted it
would rescue.

The reason is simple once seen: **a single FFT stage has no reuse to capture.**
Every element is read once and written once. Blocking is a technique for
turning repeated traversals of a large working set into repeated traversals of
a small one; applied to a loop that already touches each datum once, it cannot
remove any traffic and can only add the tile's extra read and write. The
measured 6.5-14% is about the size of one extra pass through L1, which is what
it is.

So the stride rule stands as an empirical correlation and its _explanation_ does
not. The 4 KiB threshold is more likely plain capacity — at m\*sizeof(T) = 4 KiB
the stage's eight streams span 32 KiB, which is the whole of L1, while at 512 B
they span 4 KiB and stay resident — than conflict misses or 4K store-to-load
aliasing. A capacity limit is not something a tile fixes; it wants a
decomposition that shrinks the span, which is what four-step and six-step
already do. The blocked code was reverted; only this note remains.

#### n = 32768 re-checked

That cell was worth a second run: it is the only complex128 win at a large
stride, its inverse margin was thin, and a single-shot non-interleaved ranking
pass (`TestRadix4AVX2Complex128Ranking`, taken at load 11.5) disagreed with it,
putting `dit32768_radix4_avx2` ahead at 104550 ns against 108709. Re-swept
alone at 12 passes, load 1.15, **24 of 24 groups accepted, none rejected** —
and with radix-8 now the incumbent, so the ratio is read in the other
direction:

|     n | prec | `dit32768_radix4_avx2` vs the ladder |
| ----: | ---- | ------------------------------------ |
| 32768 | c128 | 1.045 fwd / 1.013 inv — ladder wins  |
| 32768 | c64  | ladder 0.999 fwd / 1.012 inv — a tie |

So the complex128 promotion holds, at 4.5% forward rather than the first
sweep's 6.9%; two gated runs agree on sign and roughly on size, and the
single-shot ranking pass was the outlier. The complex64 cell at this size
firms up from a 1.011/1.030 loss to a tie, which is still not a win — it stays
on the probe list.

The lesson is the one the ranking test's own comment already carries: a
sequential in-process ranking pass and a canary-gated interleaved sweep are not
the same instrument, and where they disagree by single-digit percent the gated
sweep is the one to believe.

Incidental, and unchanged by this round: `dit<N>_radix4_notail_avx2` — the
probe that skips the tail combine and therefore computes the wrong answer, and
exists only to bound the question — is still the fastest cell at 512, 2048,
8192 and 32768 in both precisions, at 0.84-0.93. The tail is still 7-16% of
those kernels and nothing has recovered it.

### n = 128 closed, and the odd-exponent question settled (2026-08-01)

The radix-8 ladder's remaining unmeasured cells were 128 in both precisions.
Motivation: an odd-exponent length n = 2*4^k is also 8*4^(k-1), so radix-8
_removes_ the radix-2 tail where the radix-4 kernel can at best fuse it. That
is the principled "specialise the odd exponent" kernel, and it needs no new
assembly.

`GOOD=5216`, `GATE=1.25`, 16 passes, `-tags fftprobe`, core 0, 42 C throughout,
**95 accepted + 1 drift = 96**, full accounting. Ratios to each group's
incumbent:

| cell      |   fwd |   inv | incumbent                 | outcome                |
| --------- | ----: | ----: | ------------------------- | ---------------------- |
| 128 c64   | 0.984 | 0.989 | `dit128_radix4fused_avx2` | 1.1-1.6%, not promoted |
| 128 c128  | 1.026 | 1.037 | `dit128_radix4fused_avx2` | lost                   |
| 8192 c64  | 1.068 | 1.078 | `dit8192_radix4_avx2`     | lost (re-derived)      |
| 8192 c128 | 0.979 | 1.012 | `dit8192_radix4_avx2`     | fwd win, inv loss      |
| 32768 c64 | 1.017 | 1.004 | `dit32768_radix4_avx2`    | tie/lost (re-derived)  |

No cell is promoted. The 128 complex64 margin is the only arguable one and it
is 1.1-1.6% in the single group that lost a pass to drift — against a bar that
has been 11-22% for every radix-8 promotion so far.

Two independent checks came free. The complex64 8192/32768 losses re-derive to
1.068 and 1.017, inside the 1.011-1.078 range recorded on 2026-07-30, so the
harness agrees with itself across five weeks. And the complex128 32768 group,
where radix-8 is already the incumbent, confirms that row from the other
direction: radix-4 measures 1.052/1.058 against it.

**The tail is the whole remaining prize at these sizes.**
`dit<N>_radix4_notail_avx2` measures 0.867-0.933 across all six groups — a
6.7-13.3% cost that neither fusion nor radix-8 recovers. At n = 128 the fused
variant is _already_ the incumbent and notail still shows 9-13% left on the
table. Anything further at odd-exponent sizes should attack the combine, not
the radix.

A process note worth keeping: this sweep registered a second probe file for
sizes `radix8_avx2_probe_amd64.go` already covered, so every affected group
ranked the same signature twice. It cost a full sweep to notice and changed no
ratio (a duplicated candidate does not bias a within-group comparison), but the
sync test written to prevent exactly this checked `cmd/gencodelets/specs.go`
and not the sibling probe files. Check both.

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

**Update (2026-08-01): "implementation-limited" was half right.** The radix-8
half of that claim held — a correct ladder won 0.87 geomean in pure Go and
seven of sixteen AVX2 cells. The radix-16 half did not. A correct pure-Go
radix-16 ladder loses every cell (see below), so `dit256_radix16_avx2` was a bad
implementation _of an algorithm that also does not pay_. The two defects were
independent, and fixing the first would not have rescued it.

## Ruled out: kernels deleted in 1f7977b, and why

Commit `1f7977b` ("feat: Add radix-16 FFT implementation and associated
tests") deleted ~15,200 lines out of the AVX2 tier alongside adding the
pure-Go radix-16 ladder below. Two of those files were restored afterward for
reuse elsewhere and are **not** part of this record:
`avx2_f32_transpose{64x64,128x128}.s` (six symbols — plain transpose plus
fused transpose+twiddle and transpose+conj-twiddle), now reachable through
`internal/math`'s out-of-place transpose API.

`avx2_f64_generic_radix4_{even,odd}.s` was restored too, measured, and **kept
in-tree behind `-tags fftprobe`** rather than deleted — it lost on the
i7-1255U, which is a one-host result in the one precision this project has
caught failing to transfer. Keeping it was vindicated on 2026-08-01: the same
kernel wins forward by 8-17% on the Xeon at every odd-exponent size. See
"complex128 generic AVX2: radix-2 wins on the i7-1255U, loses on the Xeon"
below. The rest were audited on 2026-08-01 and stay dead; nothing below should
be revived without re-running the same census.

| file                                                                                                                                     | evidence                                                                                                                                                                                                                                                                                                                        | verdict  |
| ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `avx2_f32_size512_radix16x32.s`                                                                                                          | 0 `Y` operands, 3,862 `X` operands. Header promised a 256-bit design that was never written. Also four-step-shaped, not high-radix.                                                                                                                                                                                             | dead     |
| `avx2_f32_size512_radix8.s`                                                                                                              | 4 `Y` vs 1,905 `X`, and **all four `Y` hits are the header comment itself** ("Y0-Y7: 8 complex64 vectors … 4 parallel 8-pt butterflies") — the body has zero. The same unwritten-design defect, in its purest form. Superseded by the size-generic `avx2_f32_radix8.s` (see "The size-generic AVX2 radix-8 ladder" above).      | dead     |
| `avx2_f32_size256_radix16.s`                                                                                                             | Genuinely 256-bit (2,023 `Y` operands), but it is a 16×16 matrix factorisation with two full transposes through scratch plus a per-call `W_16` table rebuilt on the stack — it tests four-step, not high-radix. Radix-16 is independently ruled out for every instruction set (see "Generic tier — the radix-16 ladder" below). | dead     |
| `avx2_f64_size128_radix2.s`                                                                                                              | 4 `Y` vs 951 `X`, and the four `Y` are a `VMOVUPS` load/store pair in a copy loop — no 256-bit compute at all.                                                                                                                                                                                                                  | dead     |
| `avx2_f64_size256_radix2.s`                                                                                                              | 4 `Y` vs 1,641 `X`. Radix-2 at complex128 is structurally dominated regardless: a 256-bit register holds only two complex128 elements, so there is no vector width left to recover the log2(n) passes.                                                                                                                          | dead     |
| the six-step AVX2 drivers (`dit_4096_sixstep_amd64_avx2.go`, `dit_8192_sixstep_64x128_amd64_avx2.go`, `dit_16384_sixstep_amd64_avx2.go`) | not fully dead — see `PLAN.md` §4, "The six-step AVX2 drivers are worth a second look, but not now"                                                                                                                                                                                                                             | deferred |

**A file named `avx2_*.s` is not necessarily a 256-bit kernel.** Several of the
files above carried headers describing a `Y`-register design that was never
implemented; the body was XMM-width throughout. Run a one-line census before
ever trusting such a header or benchmarking such a kernel:

```
grep -o 'Y[0-9]\+' file.s | wc -l   # 256-bit (YMM) operand count
grep -o 'X[0-9]\+' file.s | wc -l   # 128-bit (XMM) operand count
```

A real 256-bit implementation has the bulk of its operands in `Y`; an
aspirational header leaves them all in `X`.

Two refinements, both of which this audit hit. The raw count above scans
comments as well as code, and in `avx2_f32_size512_radix8.s` **every** `Y` it
finds is the header comment promising the design — so strip comments before
believing a small non-zero count:

```
grep -o 'Y[0-9]\+' <(sed 's://.*::' file.s) | wc -l
```

And a small non-zero count that survives comment-stripping still is not
evidence of vector compute: in `avx2_f64_size128_radix2.s` the four remaining
`Y` are one `VMOVUPS` load/store pair in a copy loop. Read the hits when there
are few enough to read; the ratio only means something at scale.

All five dead `.s` files are recoverable at `git show bd87b0e:<path>` if
anyone ever needs to re-examine them — for example
`git show bd87b0e:internal/asm/amd64/avx2_f32_size512_radix8.s`. `bd87b0e` is
the last commit before the `1f7977b` deletion, so its content is
byte-identical to what `1f7977b` removed.

### complex128 generic AVX2: radix-2 wins on the i7-1255U, loses on the Xeon

A sixth file, `avx2_f64_generic_radix4_{even,odd}.s`, was restored on the
strength of two arguments that both turned out to be wrong, and is recorded
here rather than in the table above because it was **measured**, not censused.

The reasoning for restoring it: it is a genuine 256-bit kernel (586 `Y` vs 445
`X`, with FMA) and radix-4 makes `log2(n)/2` passes against radix-2's
`log2(n)`. Its complex64 twins are live in production, so complex128 was left
running radix-2 alone — an apparent oversight.

It was wired into `forwardAVX2Complex128Asm`/`inverseAVX2Complex128Asm` with
the same radix-4 → radix-4-mixed → radix-2 preamble the complex64 path uses,
validated against `reference.NaiveDFT128`/`NaiveIDFT128`, and confirmed by an
instrumented run to actually fire rather than fall through — pure radix-4 at
n = 64/256/1024, mixed at 128/512/32768. Then it lost every size. Ratios to the
radix-2 kernel, median of 3, same process, `taskset -c 0`:

|         |   64 |  128 |  256 |  512 |   1K |   2K |   4K |   8K |
| ------- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| forward | 1.08 | 1.12 | 1.56 | 1.54 | 1.30 | 1.24 | 1.32 | 1.16 |
| inverse | 1.15 | 1.10 | 1.19 | 0.90 | 1.70 | 2.71 | 2.76 | 2.61 |

The same harness against the **complex64** pair, which shares the dispatch
shape, gives 0.87 / 0.72 / 0.71 / 0.59 / 0.58 / 0.54 at 256…8192 — radix-4
winning by up to 1.9x. So the protocol is not insensitive, and the existing
complex64 dispatch is confirmed correct as a side effect.

The cause was thought to be register width, not the algorithm: a YMM holds four
complex64 but only **two** complex128, so radix-4's 4-way butterfly has no width
left to exploit at double precision while radix-2 keeps its lanes full.
**The Xeon sweep below refutes this** — the same kernel wins forward by 8-17% on
Skylake-SP at every odd-exponent size. Register width is identical on both
hosts, so whatever drives the i7-1255U loss, it is not width.

Two confounders were ruled out before accepting the result: both precisions
pass `nil` for `bitrev`, so the radix-2 path's `cachedBitReversalIndices` is
not the source of the gap; and the inverse penalty survives independently of
the trailing `1/n` scaling pass that the complex128 radix-4 asm — unlike its
complex64 twin's `inv_r4_scale` loop — omits and the Go wrapper had to supply.
(The second of those is stated too strongly: radix-2 runs its own scaling pass
in asm, so both sides pay one. See the Xeon subsection below.)

Caveat on precision: the host was thermally noisy (package 76–100 °C at load
~2.0), so these ratios are not trustworthy at the few-percent level. They do
not need to be — the effects are 1.2x–2.8x. A 5% claim from this run would be
worthless; a 2x one is not.

The complex128 dispatch is back to radix-2 only, and no production build
reaches these kernels. **The `.s` files stay in the tree** behind
`internal/fft/radix4_c128_probe_amd64.go` (`-tags fftprobe`), with their own
correctness tests and a same-process comparison benchmark.

That is deliberate, and it is the general rule for this repo: a **structural**
loss — XMM-width against a 256-bit peer, a constant table rebuilt per call —
justifies deleting a kernel, because it cannot win anywhere. A **measured** loss
on one host does not. These two are the second kind, and this document already
records complex128 on AVX2 as the precision where microarchitecture has been
observed to dominate: see "The radix-8 ladder on Skylake-SP — and the stride
rule failing to transfer", where complex128 wins at every size on the Xeon while
losing from 2048 up on the i7-1255U.

The mechanistic argument for deleting anyway — a YMM holds four complex64 but
only two complex128, so radix-4 has no width left to exploit — predicts a loss
on any AVX2 host. It is also the same species of argument as the pass-count and
`Y`-operand-census predictions that were both wrong about this very kernel, so
it does not get to close the question alone.

What closes it: a sweep on the Xeon. Deleting the files is precisely what would
prevent that, since the sanctioned route to that host is commit + push then
`git pull` there.

#### The Xeon answers it: the mixed kernel wins forward by 8-17% (2026-08-01)

The width argument was wrong, and the deletion would have destroyed a kernel
that is the fastest thing available on Skylake-SP at half its sizes.

Both hosts, same compiled `-tags fftprobe` binary, `taskset -c 0`,
`-benchtime=200ms`, 7 reps. Ratios are radix-4 over radix-2, **median of the
per-rep ratios**, `s` is the spread (max - min) across the 7 reps. Under 1.00
means radix-4 wins.

| size | fwd Xeon          | fwd i7-1255U  | inv Xeon      | inv i7-1255U  |
| ---: | :---------------- | :------------ | :------------ | :------------ |
|   64 | 0.989 (s0.08)     | 1.084 (s0.21) | 1.129 (s0.03) | 1.249 (s0.22) |
|  128 | **0.834** (s0.09) | 1.138 (s0.17) | 0.991 (s0.11) | 1.224 (s0.08) |
|  256 | 1.045 (s0.14)     | 1.543 (s0.33) | 1.233 (s0.01) | 1.538 (s0.27) |
|  512 | **0.892** (s0.13) | 1.331 (s0.31) | 1.111 (s0.03) | 1.280 (s0.12) |
|   1K | 1.042 (s0.03)     | 1.335 (s0.29) | 1.264 (s0.09) | 1.368 (s0.31) |
|   2K | **0.904** (s0.02) | 1.203 (s0.45) | 1.143 (s0.02) | 1.303 (s0.30) |
|   4K | 1.018 (s0.04)     | 1.267 (s0.30) | 1.264 (s0.03) | 1.286 (s0.14) |
|   8K | **0.918** (s0.06) | 1.209 (s0.28) | 1.164 (s0.02) | 1.254 (s0.22) |

The forward column splits **exactly by shape**, which is why this is a result
and not noise. 128 / 512 / 2K / 8K are the odd exponents — the `_mixed`
radix-4-then-2 entry point — and all four win by 8-17% with spreads of 0.02-0.13.
64 / 256 / 1K / 4K are the powers of four taking the pure radix-4 entry point,
and all four tie or lose slightly (0.989-1.045). One kernel of the pair wins on
this host; the other does not. On the i7-1255U the same mixed kernel loses those
same four cells by 14-33%.

So the two entry points want different verdicts on different hosts, and no
static registry ordering expresses that. The Wisdom cache is the mechanism that
does — it is keyed by size + precision + CPU features and is already how
per-host tuning is persisted.

Inverse loses on both hosts at every size but n=128. That is not the scaling
pass: `InverseAVX2Complex128Asm` runs its own `inv_scale_loop_128` over the
buffer, so both sides of the ratio pay one, and the earlier note above (that the
Go wrapper "had to supply" the pass radix-2 omits) is wrong about radix-2. The
inverse gap is unexplained and is the open part of this question.

Two method notes, both of which changed the numbers:

- `go test -bench ... -count=7` repeats each sub-benchmark **consecutively**, so
  it puts radix-4/8K and radix-2/8K ~30 s apart — the same back-to-back shape
  that reversed the sign of the pure-Go n=512 result. The table above comes from
  one process invocation per (direction, size) cell, which runs the pair ~0.4 s
  apart, repeated 7 times.
- The i7-1255U column was re-measured with that harness rather than reused, and
  it reproduced the older back-to-back medians to within a few percent
  (1.084/1.138/1.543/1.331 against 1.08/1.12/1.56/1.54). Its spreads are
  0.17-0.45 against the Xeon's 0.02-0.14 — the package hit 100 °C during the run
  — so the laptop column supports a "loses by 20%+" claim and nothing finer.

## Generic tier — the radix-16 ladder, and where the radix ladder stops

Sweep of 2026-08-01, `-tags fftprobe,purego`, `GOOD=5216` recalibrated on an
idle i7-1255U at 46 °C, 18 groups x 16 passes, 282 accepted + 6 over gate = 288
(full accounting). Ratios are radix-16 against the pure-Go radix-8 ladder,
taken **within** each group.

|     n | passes 16:8 | c64 fwd | c64 inv | c128 fwd | c128 inv |
| ----: | ----------: | ------: | ------: | -------: | -------: |
|   256 |         2:3 |   1.158 |   1.221 |    1.163 |    1.225 |
|   512 |         3:3 |   1.138 |   1.166 |    1.163 |    1.158 |
|  1024 |         3:4 |   1.024 |   1.101 |    1.018 |    1.029 |
|  2048 |         3:4 |   1.114 |   1.115 |    1.107 |    1.110 |
|  4096 |         3:4 |   1.139 |   1.166 |    1.305 |    1.356 |
|  8192 |         4:5 |   1.128 |   1.197 |    1.298 |    1.332 |
| 16384 |         4:5 |   1.122 |   1.138 |    1.294 |    1.303 |
| 32768 |         4:5 |   1.126 |   1.128 |    1.253 |    1.278 |

Not one cell wins, in either precision or either direction, while making 25-33%
fewer passes at every size except 512. The pass advantage is real and is
delivered; the butterfly consumes all of it. Radix-8 won 0.87 geomean on this
same harness, so the protocol is not insensitive — the radix is.

Three costs grow as the pass count shrinks, which is what makes this the end of
the ladder rather than one bad rung:

- **Diminishing passes.** `log2(n)/4` against radix-8's `log2(n)/3` is 25%
  fewer, where radix-8 bought 33% over radix-4.
- **Growing twiddle cost.** 15 planes per stage against 7, so the table
  streamed per stage roughly doubles while the passes saved shrink.
- **Growing gather cost.** Stage 1 is a 16x16 digit-reversed transpose against
  8x8 — quadratic in the radix.

This closes radix-16 for every instruction set, not only the generic tier. AVX2
has 16 YMM against radix-16's 16 live streams; AVX-512's 32 ZMM leave ~12
scratch, which is structurally the position AVX2 radix-8 was measured in at
1.24-1.56x per pass. A radix that cannot win where registers are free will not
win where they are scarce.

n = 65536 is the one uncompared cell: no radix-8 or radix-4 row is registered
there, so the probe is its own incumbent and its 1.000 means nothing. The
corroborating signal from the AVX-512 sweep is at n = 8192 complex64, where the
_simplified_ `dit8192_radix4_notail_avx2` scored 0.906 against the AVX-512
radix-8 ladder's 0.900 — a pure simplification within half a percent of an
entire new ISA tier.

## AVX-512 tier (Xeon Gold 5218)

Evidence behind the `SIMDAVX512` rows, including the one deliberately disabled
entry. This is the only AVX-512 hardware reachable; Cascade Lake downclocks
under AVX-512, so it is a pessimistic machine for the tier.

### The generic radix-2 AVX-512 rows were selecting a 3-4x loss (2026-07-31)

`dit{1024,4096,8192,16384}_radix2_avx512` were registered as "the sizes where it
beats the best AVX2 codelet". That was true when written and is false now: the
size-generic AVX2 radix-4 kernel and the radix-8 ladder both landed afterwards,
and neither could be compared against these on the development laptop, which has
no AVX-512. Because registry ordering is SIMD-level major, these outranked every
AVX2 row **on level alone**, so an AVX-512 host ran a codelet 3-4x slower than
one sitting in the same registry.

Canary-gated sweep on the idle Xeon (load 0.05), `-tags fftprobe`, 8 passes,
`GOOD=11298` from `CALIBRATE=1`, `GATE=1.20`, **112 of 112 groups accepted, 0
rejected** — complex64, forward:

|     n | `radix2_avx512` | best AVX2 at that size |    gap |
| ----: | --------------: | ---------------------- | -----: |
|  1024 |         8903 ns | `radix4_avx2` 2193 ns  | 4.06 x |
|  4096 |        38654 ns | `radix4_avx2` 11990 ns | 3.22 x |
|  8192 |        87682 ns | `radix4_avx2` 28734 ns | 3.05 x |
| 16384 |       198990 ns | `radix4_avx2` 68608 ns | 2.90 x |

Two causes compound: radix-2 runs ~2x the passes of radix-4 (13 vs 6.5 at
n = 8192), and Skylake-SP drops core frequency under 512-bit work.

Fixed with `RankLevel: SIMDAVX2` rather than deletion. `SIMDLevel` stays
`SIMDAVX512`, so the rows still execute only where legal; ranking them at AVX2
with `Priority 10` puts them below every AVX2 row (priority 90) and still above
SSE3 — which is what the measurement says, since they do beat SSE3 (8903 vs
10127 ns at n = 1024). Deleting them would over-generalise from one
microarchitecture: a part without Skylake-SP's AVX-512 downclocking may rank
differently. Re-measure before removing or re-promoting.

Only those four rows are demoted, and a second sweep over 8/16/32/64/128/256
(**96 of 96 groups accepted**) is what bounds it: at every one of those sizes the
_per-size assembly_ AVX-512 codelet is genuinely the fastest candidate in its
group, so the tier is healthy where it is hand-written.

| n (c64) | AVX-512 incumbent          | best AVX2 behind it |    gap |
| ------: | -------------------------- | ------------------- | -----: |
|      64 | `dit64_radix2_avx512` 46.2 | `radix8ladder` 115  | 2.50 x |
|     128 | `radix8_then2_avx512` 176  | `radix4_notail` 205 | 1.16 x |
|     256 | `radix8_then2_avx512` 371  | `radix8ladder` 454  | 1.22 x |

So the split is not AVX-512 vs AVX2 but hand-written-per-size vs size-generic
radix-2: `dit64_radix2_avx512` is itself radix-2 and wins comfortably. Radix-2
is not the problem at small n, where ZMM width dominates; it becomes the problem
at large n, where its pass count and Skylake-SP's downclocking compound.

**The general trap:** a codelet at a higher SIMD level is selected on level
before priority, so it silently keeps its slot as the tier below it improves.
Any row justified by "it beats the level below" needs re-measuring whenever that
level gains a kernel — and on a machine that can actually run it.

### The radix-8 ladder on Skylake-SP — and the stride rule failing to transfer

Same sweep, ladder ÷ `radix4_avx2` within group, forward:

|     n |       c64 |      c128 |
| ----: | --------: | --------: |
|   512 | **0.934** | **0.749** |
|  1024 |     0.992 | **0.843** |
|  2048 |     1.013 | **0.722** |
|  4096 |     1.143 | **0.928** |
|  8192 |     1.197 | **0.912** |
| 16384 |     1.082 | **0.851** |
| 32768 |     1.151 | **0.840** |

complex128 wins at every size by 7-28%; complex64 wins only at 512, ties at
1024, and loses from 2048 up.

This **refutes the byte-stride rule** recorded for the i7-1255U (radix-8 wins
where its widest stage strides ≤ 512 B, loses at ≥ 4 KiB). complex128 has twice
the byte stride of complex64 at equal n, so under that rule it should fail
earlier and harder — and n = 32768 complex128, whose widest stage strides 64 KiB,
should be the worst cell on the board. It is a 16% win. The complex64 crossover
also moved, to 2048 here from 4096 on the laptop.

So the stride correlation is real on one machine and is not the mechanism. The
better-supported reading is the compute-to-memory ratio: complex128 does twice
the work per byte through the same 256-bit path, so the ladder's fewer passes
pay while its access pattern costs less. Together with the failed blocking
experiment (below), that closes the conflict-miss story for good.

#### Why the radix-8 stage costs more per pass — it is registers, not cache

Normalising the same sweep by pass count (ladder shapes: `8^k` / `2·8^k` /
`4·8^k`, so n = 8192 is 4 radix-8 stages plus a radix-2 tail against radix-4's
6 + 1), complex64 forward:

|     n |  KB | ladder ns/pass | radix-4 ns/pass | ratio |
| ----: | --: | -------------: | --------------: | ----: |
|   512 |   4 |            317 |             204 |  1.56 |
|  1024 |   8 |            544 |             439 |  1.24 |
|  2048 |  16 |           1318 |             868 |  1.52 |
|  4096 |  32 |           3425 |            1998 |  1.71 |
|  8192 |  64 |           6877 |            4105 |  1.67 |
| 32768 | 256 |          37549 |           20389 |  1.84 |

L1d is 32 KiB here, so n ≤ 2048 is fully resident — and the penalty is already
1.24–1.56x there. It is not a memory effect. Most of the rest is expected
(radix-8 does proportionally more work per pass: 7/5 at n = 8192); normalised
per operation, radix-8 is only ~1.2x less efficient, while total op counts
slightly favour it (4.08 vs 4.25 N·log2 N).

That residual is a register-budget effect, and the kernel's own header
(`internal/asm/amd64/avx2_f32_radix8.s`) names it: eight live streams plus two
rotation masks and the sqrt(2)/2 broadcast leave **five** scratch YMM of 16,
"exactly enough" for one butterfly. It never spills — frame `$0`, no `(SP)`
references anywhere — but it has no slack either, so it re-broadcasts the
twiddle planes from memory every iteration and cannot keep a second butterfly in
flight to cover a radix-8's 3-level dependency chain (radix-4 is 2 levels and
has registers to spare). Extra load uops plus exposed latency, with no spill to
make it visible.

This is the thing to fix, and it is what makes an AVX-512 radix-8 worth writing:
32 ZMM leaves 21 scratch rather than 5, and embedded broadcast (`{1to16}`)
folds the re-broadcast into the instruction encoding — the exact workaround the
header describes. Predicted 1.4–1.7x over `radix4_avx2` at n = 8192, the low end
being the pass-count bound (5/7) if the result turns out memory-bound, the high
end assuming width and register slack recover the per-op gap net of Skylake-SP's
AVX-512 downclocking.

Do not fold these figures into the i7-1255U tables below — different
microarchitecture, different cache geometry, and a different incumbent per size.

#### The AVX-512 radix-8 ladder: prediction half right, 16 rows promoted

Sweep of 2026-07-31, Xeon Gold 5218, 8 passes × 20 groups, **160 accepted / 0
rejected**, `benchmarks/gated-avx512r8`. Ratios are the ladder against the
incumbent each size then selected, median within group. Under 1.00 wins.

|     n |   c64 fwd |   c64 inv |  c128 fwd |  c128 inv |
| ----: | --------: | --------: | --------: | --------: |
|    64 |     1.968 |     1.861 |     1.464 |     1.455 |
|   128 |     1.039 |     0.997 |     1.256 |     1.283 |
|   256 | **0.947** | **0.921** | **0.708** | **0.743** |
|   512 | **0.777** | **0.774** | **0.814** | **0.766** |
|  1024 | **0.740** | **0.748** | **0.697** | **0.753** |
|  2048 | **0.703** | **0.751** | **0.884** | **0.882** |
|  4096 | **0.790** | **0.807** | **0.702** | **0.695** |
|  8192 | **0.900** | **0.890** | **0.700** | **0.696** |
| 16384 | **0.808** | **0.813** | **0.708** | **0.711** |
| 32768 | **0.883** | **0.887** | **0.869** | **0.865** |

**The register-budget diagnosis above is confirmed for complex128 and only
complex128.** The stated bar was 1.4–1.7× over `dit8192_radix4_avx2` at
n = 8192: complex128 lands 1.43× (0.700), inside the predicted band; complex64
lands 1.11× (0.900), well short. The likely reason for the split is stage 1 —
complex64 needs a `VPGATHERDQ` 8×8 transpose that complex128 gets from plain XMM
loads — so the c64 ladder pays a gather the c128 ladder does not, on top of the
same win.

Sizes 256–32768 are promoted to production rows in both precisions
(`cmd/gencodelets/specs_avx512.go`, Priority 50, ranked at `SIMDAVX512`).
n = 64 and n = 128 are **not**: 64 loses outright in both precisions, and 128 is
a c128 loss with c64 at parity (1.039/0.997 — a tie is not a promotion). Both
stay in `radix8_avx512_probe_amd64.go`, whose size lists were trimmed to exactly
the unpromoted cells so no signature is registered twice.

Two things the sweep does not establish, and which the ratios above must not be
read as covering:

- The rows were promoted from a sweep taken while the ladder was
  `RankLevel`-demoted to `SIMDSSE2`. Promotion makes it the incumbent, so the
  comparison it won is no longer the comparison the registry makes. Nothing in
  the numbers changes, but a re-sweep now reports the ladder against _itself_ at
  these sizes unless the old incumbent is temporarily re-promoted.
- Only `n = 8192` had a stated bar. The other nine sizes were measured, not
  predicted, and 2048 c128 (0.884) and 32768 (0.869/0.883) are visibly weaker
  than the ~0.70 the rest of the c128 column holds — unexplained.

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

## NEON tier (Apple M5, darwin/arm64) — the codelets contain no NEON

First measurement of the NEON ladder on real arm64 hardware (2026-08-01). Every
prior NEON priority was ladder-mirrored from x86 and QEMU-verified for
correctness only, so nothing in this file covered them. Host: Apple M5, 4P+6E,
darwin 25.6, cross-compiled test binary (`go test -c`), `BenchmarkCodeletCandidates64/128`,
`benchtime=500ms`, `count=6`. Ratios are same-process, within-size, against the
best non-NEON candidate at that size — the only figure that travels.

**Every registered NEON codelet lost**: 55 of 56 (precision × size × direction)
cells, ratios **1.09–3.76**, median ~1.6. The lone win was c128/size32/inverse
(0.92). Worst cells: c128 size 8 forward 3.76×, c64 size 4 inverse 3.23×, c64
size 64 forward 2.54×.

The cause is not microarchitectural. A mnemonic census of all 42 size-specific
`neon_f*_size*.s` files finds **zero vector instructions** — they are scalar
`FMOVS/FADDS/FMULS/FSUBS/FNEGS` throughout, with no `FMLA`/`FMLS` either, plus
dead weight like 281 register-to-register `FMOVS` in `neon_f32_size1024_radix4.s`
and a scalar bit-reversal loop. Only `neon_f32_generic.s` is genuinely
vectorized; `neon_f64_generic.s` is scalar too.

The reason the files came out that way is a real toolchain constraint worth
recording: **Go's arm64 assembler has no vector FP add, subtract or multiply.**
`VFADD`/`VFSUB`/`VFMUL` are "unrecognized instruction"; `VFMLA` and `VFMLS` are
the only vector FP arithmetic mnemonics it accepts, and `VADD`/`VSUB` are
integer-only. The workaround is the one `neon_f32_generic.s` already uses:
synthesize multiply as `VEOR`-zero + `VFMLA`, and add/sub against a vector of
1.0 (`·neonOnes`) with `VFMLA`/`VFMLS`. On Apple cores FMA has the same
throughput as FADD, so this costs nothing. Two further traps: `VLD2`/`VST2`
register lists must be **contiguous** (`[V0,V1]`, never `[V0,V4]`), and
`RankLevel` cannot demote below the generic tier because the registry reads
`RankLevel == SIMDNone` as unset.

### Size 16, complex64 — first vectorized rewrite

`neon_f32_size16_radix4_v2.s` (`dit16_radix4_neon_v2`) replaces the scalar v1
with a 4×4 Cooley-Tukey decomposition: `VLD2` deinterleaves four complex per
register, two rounds of vertical DFT4 sandwich a `VTRN1/VTRN2` + `VZIP1/VZIP2`
4×4 transpose, and each store is a single `VST2` of four consecutive outputs.
It needs no bit-reversal pass and no scratch buffer — every input is in
registers before the first store, so `dst` may alias `src`.

`count=8`, same process, mean ns/op:

| candidate                |  forward |   inverse |
| ------------------------ | -------: | --------: |
| **dit16_radix4_neon_v2** | **9.50** | **10.44** |
| dit16_radix16_generic    |    16.23 |     17.74 |
| dit16_radix4_generic     |    16.25 |     20.49 |
| dit16_radix4_neon (v1)   |    35.68 |     38.53 |
| dit16_radix2_neon        |    61.78 |     68.84 |

v2 is **3.76× / 3.69×** faster than v1 and **1.71× / 1.70×** faster than the
best pure-Go candidate — the same size v1 lost by 1.83× / 2.19×. v2 is now
Priority 28; v1 drops to Priority 1, keeping it under the reference tests and
re-measurable on other arm64 hosts rather than deleted on one machine's number.

Correctness: naive-DFT cross-check, round-trip and in-place all pass under both
QEMU and the M5. The remaining 41 files are unconverted and still lose.
