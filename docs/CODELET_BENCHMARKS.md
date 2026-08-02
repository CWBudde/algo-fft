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

| file                                                                                                                                     | evidence                                                                                                                                                                                                                                                                                                                                                                                       | verdict  |
| ---------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `avx2_f32_size512_radix16x32.s`                                                                                                          | 0 `Y` operands, 3,862 `X` operands. Header promised a 256-bit design that was never written. Also four-step-shaped, not high-radix.                                                                                                                                                                                                                                                            | dead     |
| `avx2_f32_size512_radix8.s`                                                                                                              | 4 `Y` vs 1,905 `X`, and **all four `Y` hits are the header comment itself** ("Y0-Y7: 8 complex64 vectors … 4 parallel 8-pt butterflies") — the body has zero. The same unwritten-design defect, in its purest form. Superseded by the size-generic `avx2_f32_radix8.s` (see "The size-generic AVX2 radix-8 ladder" above).                                                                     | dead     |
| `avx2_f32_size256_radix16.s`                                                                                                             | Genuinely 256-bit (2,023 `Y` operands), but it is a 16×16 matrix factorisation with two full transposes through scratch plus a per-call `W_16` table rebuilt on the stack — it tests four-step, not high-radix. Radix-16 is independently ruled out for every instruction set (see "Generic tier — the radix-16 ladder" below).                                                                | dead     |
| `avx2_f64_size128_radix2.s`                                                                                                              | 4 `Y` vs 951 `X`, and the four `Y` are a `VMOVUPS` load/store pair in a copy loop — no 256-bit compute at all.                                                                                                                                                                                                                                                                                 | dead     |
| `avx2_f64_size256_radix2.s`                                                                                                              | 4 `Y` vs 1,641 `X`. Radix-2 at complex128 is structurally dominated regardless: a 256-bit register holds only two complex128 elements, so there is no vector width left to recover the log2(n) passes.                                                                                                                                                                                         | dead     |
| the six-step AVX2 drivers (`dit_4096_sixstep_amd64_avx2.go`, `dit_8192_sixstep_64x128_amd64_avx2.go`, `dit_16384_sixstep_amd64_avx2.go`) | not fully dead — deferred to Phase 3's "Use the AVX2 transposes that already exist". They were the **only** callers of `Transpose{64x64,128x128}Complex64AVX2Asm`, which is why the §1.1 census now reports those six symbols orphaned; that item is a restore from `1f7977b^`, not the wiring change its text assumes. Their spec rows had already been unregistered separately in `08c8e7b`. | deferred |

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

## The six-step / four-step crossovers (2026-08-02)

Evidence behind the `Six-step`, `Six-step 64×128`, `Eight-step` and `Four-step`
family verdicts, and behind whatever `dit<N>_sixstep*_generic` priority
`cmd/gencodelets/specs.go` now carries. The question PLAN.md §1.2 asked was
where each decomposition overtakes the flat radix ladder, after radix-4 got
2–4× faster and moved the crossing point up.

Three separate things had to be established, and only one of them was a
crossover:

### Eight-step is six-step

`internal/kernels/eightstep.go` does not implement an eight-step FFT. Its
transform bodies are `sixstep.go`'s with the names changed: same
`intSqrt(n); if m*m != n { return false }` perfect-square rejection, same two
`math.TransposeSquare`-bracketed Stockham row passes, same twiddle stage. A
diff of the two files with the family names normalised away leaves exactly two
differences, both of them structural rather than algorithmic — the `stdmath`
import, and the `fillRowTwiddle`/`intSqrt` helpers that `sixstep.go` hosts for
both callers. There is no eighth step anywhere in the file.

This is a source fact, not a measurement, and it settles the family without a
benchmark: `KernelEightStep` is a second name for `KernelSixStep`, so any
sweep pitting one against the other measures noise. The family's matrix verdict
is therefore `untested` rather than a loss — PLAN.md §2.2, a poor
implementation disqualifies the file and not the algorithm, and here there is
not even an implementation to disqualify. `BenchmarkStepCrossover` carries no
eight-step arm for the same reason.

It also retires one inherited number. The eight-step loss recorded in
`internal/planner/selection.go` ("at 2^22 complex64 Stockham runs 157/171 ms
against eight-step's 201/269 ms") is a measurement of six-step at a
non-square-friendly size, not of an eight-step algorithm. The rule it justified
— removing the ≥ 2^22 eight-step branch — stands, because six-step lost there
too; the label on it was wrong.

### The ≥ 2^18 half was already answered, in a code comment

`BenchmarkSquareAutoRule` (`plan_autosquare_bench_test.go`) measured every
strategy the auto rule could pick at 2^18, 2^20 and 2^22 — the only power-of-two
squares its branches could reach — on both the SIMD and purego builds, at both
precisions and in both directions. Stockham won or tied against six-step,
eight-step, four-step and split-radix in every arm but one. That result has been
load-bearing since 2026-07-28: it is why `resolveKernelStrategy` has no square
branch at all today. It had never been written down anywhere but
`internal/planner/selection.go:61-90`, so it is recorded here:

| arm                               | result                                                          |
| --------------------------------- | --------------------------------------------------------------- |
| split-radix, 2^18–2^22            | Stockham wins every arm except purego 2^18 c64 fwd (−3%, noise) |
| split-radix, 2^20 c128 fwd        | 80.3 ms vs six-step 39.3 ms vs Stockham 49.7 ms — costs 2×      |
| eight-step, 2^22 c64, SIMD        | 201/269 ms (fwd/inv) vs Stockham 157/171 ms                     |
| eight-step, 2^22 c64, purego      | 203/247 ms vs Stockham 102/113 ms                               |
| **six-step, 2^20 c128 fwd, SIMD** | **39.3 ms vs Stockham 49.7 ms — the one dissenting arm**        |

The dissent is accepted knowingly rather than encoded: a precision- and
direction-blind rule cannot express it, the same size's other three arms all
favour Stockham, and wisdom/measure mode picks it per machine where it matters.

**What this leaves open is the band below it.** `BenchmarkSquareAutoRule`'s
floor is 2^18 and the six-step codelet sizes stop at 16384, so 32768–131072 had
never been measured against anything — which is exactly where §1.2's own
estimate ("it is above 16384 on this host") put the crossing point.
`BenchmarkStepCrossover` covers it.

### The six-step codelet rows lose to the radix-8 ladder in the generic tier

Canary-gated sweep of 2026-08-02, i7-1255U pinned to core 0, `-tags purego`,
`GOOD=5216`, `GATE=1.25`, 12 passes × 6 groups = 72, **60 accepted + 12 over
gate + 0 drift + 0 incomplete = 72**, full accounting; 50–54 °C throughout;
`benchmarks/gated-sixstep-purego`. Ratios are the six-step row against its
group's incumbent, taken within each group and then medianed. The incumbent at
all three sizes is `dit<N>_radix8ladder_generic` — the flat ladder §1.2 asked
these decompositions to overtake.

Purego is the decisive build rather than a fallback check: registry ordering is
SIMD-level major, so on any AVX2 host the AVX2 row takes the cell whatever
priority a `SIMDNone` row carries. The generic-tier ranking _is_ the question
these rows' priorities answer.

|     n | row                             | c64 fwd | c64 inv | c128 fwd | c128 inv |
| ----: | ------------------------------- | ------: | ------: | -------: | -------: |
|  4096 | `dit4096_sixstep_generic`       |   1.428 |   1.489 |    1.488 |    1.896 |
|  8192 | `dit8192_sixstep64x128_generic` |   1.706 |   1.914 |    2.058 |    2.091 |
| 16384 | `dit16384_sixstep_generic`      |   1.892 |   2.111 |    2.202 |    2.063 |

For scale, the middle row of each group — `dit<N>_radix4[_then2]_generic`, the
ladder's own predecessor — sits at 1.09–1.36. Six-step is not merely behind the
best row; it is behind the row the best row replaced, in all twelve cells.

Re-run on the Xeon Gold 5218, `PASSES=8`, **48 accepted + 0 rejected = 48**,
load average 0.66, `taskset -c 0`. Same incumbent, same direction, slightly
harder:

|     n | row                             | c64 fwd | c64 inv | c128 fwd | c128 inv |
| ----: | ------------------------------- | ------: | ------: | -------: | -------: |
|  4096 | `dit4096_sixstep_generic`       |   1.594 |   1.719 |    1.736 |    1.833 |
|  8192 | `dit8192_sixstep64x128_generic` |   2.003 |   2.173 |    2.162 |    2.351 |
| 16384 | `dit16384_sixstep_generic`      |   2.069 |   1.809 |    2.141 |    2.339 |

**This pair of sweeps is a fair fight, and it is the one result here that is a
loss on merit.** The six-step _codelets_ do not share the scalar-row defect
described below: `dit_4096_sixstep.go` and its siblings run their rows through
`forwardDIT64Radix4Complex64` / `forwardDIT128Radix2Complex64` — the tuned
pure-Go leaves — and the incumbent they are measured against is the pure-Go
radix-8 ladder. Both arms are scalar, at the same sizes, in the same build. So
the 64×64, 64×128 and 128×128 decompositions genuinely lose to the flat ladder
at 4096/8192/16384, on two hosts, in both precisions and both directions.

**The disposition that follows, and which this round did not carry out.**
Under `PLAN.md` §2.2 a measured, non-structural loss of ≥ 1.5× belongs behind
`-tags fftprobe`: every Xeon cell and eight of twelve laptop cells clear that
bar, and the second host that would justify keeping them registered has now
been measured. The six rows are left registered at their existing
**non-selectable** priorities (25/30 against the ladder's 50) because that part
of the rule — never leave a beaten codelet at a _selectable_ priority — is
already satisfied, and because the migration is a code change rather than the
priority adjustment this round was scoped to. It is carried as an explicit
action on the row-binding item in `PLAN.md`, not as a silent omission.

### And no decomposition overtakes the incumbent route anywhere in 16384–131072

`BenchmarkStepCrossover`, i7-1255U pinned to core 0, `-count=5`, medians of 5,
both builds. Ungated — these are plan-level arms, not registry candidates, so
the canary harness does not apply; the arms for one size run adjacent in one
process, which is what makes the within-size ordering trustworthy even though
the absolute numbers are not comparable across the two builds.

Read against what a default plan actually binds, which had to be checked rather
than inferred: codelets cover 4096–65536 on the SIMD build and 4096–32768 on
purego, and `Stockham` is the auto route only above those.

SIMD build, forward, µs (the DIT arm is the bound codelet, i.e. the default):

|      n |    DIT |   Stockham | SixStep | FourStep |
| -----: | -----: | ---------: | ------: | -------: |
|  16384 |   24.6 |      174.0 |   846.7 |    862.7 |
|  32768 |   86.3 |      217.7 |       — |   1006.0 |
|  65536 |  125.8 |      748.3 |  3620.4 |   3646.9 |
| 131072 | 2300.6 | **1373.4** |       — |   6070.1 |

purego, forward, µs:

|      n |    DIT |   Stockham |    SixStep | FourStep |
| -----: | -----: | ---------: | ---------: | -------: |
|  16384 |  224.3 |  **193.5** |     1006.4 |    462.6 |
|  32768 |  496.8 |      726.4 |          — |   1726.4 |
|  65536 | 2823.4 |     2094.2 | **1846.6** |   1973.1 |
| 131072 | 2624.7 | **3700.3** |          — |   5332.6 |

**One cell in the whole band goes to a decomposition, and it is not a result.**
purego / 65536 / complex64 / forward: six-step 1846.6 µs against the auto
route's 2094.2, a 12% edge. This host does not support a 12% cross-arm claim —
`docs/BENCHMARKING.md` puts the believable floor near 15% even within one
binary — and the same cell's inverse goes the other way by 3.3× (2710.5 vs
820.9). Nothing is promoted on it.

Every other cell loses, most of them enormously: 17–35× against the bound
codelet on the SIMD build, 2–10× on purego.

### Why — the row passes are scalar, and that disqualifies the file, not the algorithm

The 17–35× figures are far too large to be a decomposition losing on merit, so
they were attributed rather than recorded. Splitting `sixStepForward` at
n = 65536 complex64 (SIMD build, core 0, `-count=3` medians):

| stage                                   |    cost |
| --------------------------------------- | ------: |
| 3 × `math.TransposeSquare`              |  162 µs |
| 2 × row passes (256 rows of 256)        | 3240 µs |
| twiddle stage, as written (`(i*j)%n`)   |  286 µs |
| twiddle stage, subtract-wrap instead    |  245 µs |
| whole six-step                          | 3730 µs |
| one flat pure-Go Stockham over all of n | 2870 µs |

The row passes are 87% of it, and the reason is that `sixStepForward` calls the
package-internal, pure-Go `stockhamForward` for them unconditionally. On a SIMD
build it is therefore a scalar kernel racing AVX2 codelets, and the 17–35×
measures that and not the decomposition. `fourStepForward` has the same shape
via `rowStockham`; PLAN.md Phase 3 already names it — "the row passes still use
the scalar Stockham butterflies, the main handicap against the monolithic
kernels".

Even inside pure Go the implementation does not pay: two row passes cost
3240 µs where one flat Stockham over the whole array costs 2870 µs, before
the transposes and twiddles are added. A six-step whose rows are the same
scalar Stockham it is competing against cannot win — it can only add stages.

**This applies to the strategy kernel, not to the codelets.** The two are
different implementations of the same decomposition and only the strategy form
has scalar rows: `ForwardSixStepComplex64` calls the generic `stockhamForward`,
while `forwardDIT4096SixStepComplex64` calls the tuned `forwardDIT64Radix4…`
leaf. So the 17–35× figures here are confounded and prove nothing about the
decomposition, whereas the 1.43–2.35× codelet figures above are a fair
pure-Go comparison and do. Keeping the two apart is the whole difference
between "six-step loses" and "this six-step file loses".

So under PLAN.md §2.2 this is an implementation loss, not an algorithm loss,
and none of the three families may be recorded as `closed`. What the AVX2
six-step drivers deleted in `1f7977b` did — call `ForwardAVX2Size64Radix4…` for
the rows and the AVX2 transposes for the shuffles — is exactly the missing
piece, which is why that restore and this verdict are the same question.

**A second, smaller defect, recorded so it is not re-found:** the twiddle stage
indexes with `twiddle[(i*j)%n]`, an integer division per element, where
`fourstep.go` already uses a subtract-wrap for the identical stage. It is worth
14% of that stage and ~1% of the transform — real, but not the reason six-step
loses, and specifically _not_ what the numbers above should be attributed to.

### The four-step split model picks the worst split it is offered

`BenchmarkFourStepSplitSweep`, same host and pinning, `-count=1`. `fourStepSplit`
derives n1×n2 from `cpu.DetectCaches()`; the sweep times every split so the
derived one can be checked against the measured optimum. It chose the balanced
√n×√n split at every size, and the balanced split measured worst or near-worst:

|       n | derived split | its cost | best measured |     cost | gap  |
| ------: | ------------- | -------: | ------------- | -------: | ---- |
|  262144 | 512×512       |  9290 µs | 256×1024      |  8769 µs | 5.9% |
| 1048576 | 1024×1024     | 42962 µs | 32768×32      | 40749 µs | 5.4% |

At 2^20 the derived split is the slowest of all eleven. The gaps are ~5%, so
the individual numbers are at this host's noise floor, but the _pattern_ — the
model degenerating to the balanced split, and the balanced split being the one
to avoid — is consistent across both sizes and has a mechanism: √n×√n is
six-step's shape, so the model is steering four-step onto the one decomposition
whose only distinguishing feature it thereby discards.

This is the parameter a second host moves, and it is now known to be
mis-derived on the first one.

### The Xeon: the one favourable cell does not reproduce, and the DIT/Stockham crossover moves

`BenchmarkStepCrossover` re-run on the Xeon Gold 5218 (Skylake-SP, AVX-512,
22 MiB L3 against the laptop's 12), `taskset -c 0`, `-count=5`, both builds,
load average 0.66 at start. Per the standing rule these numbers are **not**
folded into the i7-1255U tables above — only the orderings travel.

This host was worth the trip twice over.

**First, it kills the one cell that went to a decomposition.** The laptop's
purego / 65536 / complex64 / forward result — six-step 1846.6 µs against
Stockham's 2094.2, the 12% edge already discounted as below the noise floor —
inverts completely:

| purego, 65536, c64, fwd   |   i7-1255U | Xeon Gold 5218 |
| ------------------------- | ---------: | -------------: |
| six-step                  | **1846.6** |         5404.3 |
| Stockham (the auto route) |     2094.2 |     **1977.5** |

2.7× the other way. So across two hosts, four builds, two precisions, two
directions and four sizes there is **no cell anywhere** in which six-step or
four-step beats the route the planner already takes. The families lose on both
machines and the verdicts do not rest on one.

**Second, and not a six-step result at all: the DIT/Stockham crossover sits on
opposite sides of n = 131072 on the two hosts.** No codelet is registered at
that size in any tier, so both machines fall through to the `ditAutoThreshold`
heuristic, which answers Stockham:

| n = 131072, SIMD build, forward |   i7-1255U | Xeon Gold 5218 |
| ------------------------------- | ---------: | -------------: |
| DIT, complex64                  |     2300.6 |     **2422.2** |
| Stockham, complex64 (auto)      | **1373.4** |         2785.3 |
| DIT, complex128                 |     2903.6 |     **3683.8** |
| Stockham, complex128 (auto)     | **2452.8** |         4979.7 |

Auto is right on the laptop and costs the Xeon 15% at complex64 and **26%** at
complex128. That belongs to §1.4's threshold item rather than this one, and it
is recorded here because it is precisely the class of result a single host
cannot produce — the same reason `PLAN.md` §2.2 keeps losing kernels
re-measurable instead of deleting them.

## Split-radix, and the 32×32 / 16×32 decompositions (2026-08-02)

Evidence behind the `Split-radix`, `Radix-32×32` and `Radix-16×32` family
verdicts. Split-radix was the matrix's largest untested cell: full strategy
plumbing, a pure-Go kernel, no codelet row at any ISA, and auto-selected
nowhere.

**It had to be registered before it could be measured.** The canary-gated
harness only ranks candidates the registry knows about — `bench_gated.sh` drives
`BenchmarkCodeletCandidates<prec>` and the analyzer parses that name shape — so
a family with no rows is invisible to the protocol `PLAN.md` §1.2 asks for. Rows
were added at priority 1 (last in the pure-Go tier, ranked by nothing, but
visible to the sweep and to the registry-driven reference tests) at 256…32768.
The ladder stops there deliberately: **n = 65536 has no generic row at all**, so
a row there would have made an unmeasured kernel the selected purego route. That
band is measured at plan level instead.

### The gated sweep: split-radix loses every codelet-sized cell

Xeon Gold 5218, `-tags purego`, `GOOD=11298`, 16 groups × 8 passes,
**128 accepted + 0 rejected = 128** (full accounting), `benchmarks/gated-sr-purego`.
Ratios against the group incumbent, taken **within** each group.

|     n | incumbent           | c64 fwd | c64 inv | c128 fwd | c128 inv |
| ----: | ------------------- | ------: | ------: | -------: | -------: |
|   256 | radix8ladder        |   1.338 |   1.401 |    1.347 |    1.513 |
|   512 | radix8ladder        |   1.205 |   1.292 |    1.259 |    1.328 |
|  1024 | radix8ladder/radix4 |   1.176 |   1.251 |    1.254 |    1.251 |
|  2048 | radix8ladder        |   1.190 |   1.274 |    1.206 |    1.322 |
|  4096 | radix8ladder        |   1.155 |   1.180 |    1.143 |    1.246 |
|  8192 | radix8ladder        |   1.113 |   1.153 |    1.149 |    1.221 |
| 16384 | radix8ladder        |   1.173 |   1.173 |    1.186 |    1.300 |
| 32768 | radix4_then2        |   1.235 |   1.197 |    1.274 |    1.234 |

Thirty-two cells, no win. But the loss is **shallow and bounded** — only one cell
(256 complex128 inverse, 1.513) reaches §2.2's 1.5× bar, and the curve has a
clear minimum at n = 8192 (1.11) before widening again. Split-radix also _beats_
`dit16384_radix4_generic` in both precisions (1.173 vs 1.246 complex64) and ties
`dit4096_radix4_generic`, so it is mid-pack rather than dominated: it loses to
the tuned radix-8 ladder, not to everything.

That is §2.2's "registered, low priority" case exactly, and the rows are already
there — priority 1, never selected, timed by the wisdom tuner, correctness-tested
for the first time.

### The band above the codelets: split-radix wins where the ladder stops

`BenchmarkStepCrossover`, Xeon, `taskset -c 0`, `-benchtime=0.5s -count=5`,
medians of five. Two builds. Ratios are split-radix against each named arm.

| n (purego)  | split-radix | vs Stockham |    vs DIT |
| ----------- | ----------: | ----------: | --------: |
| 16384 c64   |      316 µs |   **0.748** |     1.159 |
| 32768 c64   |      704 µs |   **0.750** |     1.215 |
| 65536 c64   |     1662 µs |   **0.840** | **0.686** |
| 131072 c64  |     3744 µs |   **0.899** | **0.630** |
| 16384 c128  |      338 µs |   **0.819** |     1.221 |
| 32768 c128  |      813 µs |   **0.889** |     1.282 |
| 65536 c128  |     1833 µs |   **0.897** | **0.526** |
| 131072 c128 |     4201 µs |       1.037 | **0.596** |

**The two harnesses cross-validate.** At 16384 and 32768 the plan-level DIT arm
binds the same codelet the gated sweep ranks, and the two independent
measurements agree to within 2%: 1.159/1.215 here against 1.173/1.235 there.
That is worth more than either number alone — it says the plan-level arms are
measuring the kernel and not the plan.

**And it locates the crossover precisely.** Split-radix loses to the bound
codelet at every size that has one, and beats everything at every size that does
not. The flip is at n = 65536, and it is a large flip: 0.686 against the DIT
route and 0.840 against Stockham, which is what auto actually picks there.

### Why that is a coverage gap, not an algorithmic win

The honest reading of the flip is not "split-radix is the best large-n pure-Go
kernel". It is that **the generic codelet tier stops at 32768**. At 65536 the
DIT arm falls off the registry onto `internal/kernels/dit.go`'s size switch and
costs 2422 µs where the registered 32768 row costs 580 µs for half the work —
the route gets worse per point, and split-radix wins by comparison rather than
on merit. Extrapolating the `radix4_then2` row's cost per point puts a
registered 65536 ladder codelet near 1250–1400 µs, which would beat split-radix
outright.

So there are two separable actions, and only the first is this item's:

- Split-radix at 65536 is a **measured 16% win over the route auto takes today**
  on purego, available for one spec row. §2.1 gate 5 is satisfied on this host.
- Extending the generic ladder past 32768 is probably the larger win, and it is
  a Phase 3 sizing question rather than a §1.2 family verdict.

### The 32×32 / 16×32 decompositions

`PLAN.md` §1.2 blamed this family's loss on "only one of two stages vectorised".
**That defect is in files that no longer exist**: `avx2_f32_size1024_radix32x32.s`
and `avx2_f64_size1024_radix32x32.s` were deleted in `08c8e7b`,
`avx2_f32_size512_radix16x32.s` in `1f7977b`. What survives is four pure-Go rows,
and the item's other figure — the pure-Go 32×32 "loses 7.2×/5.2× to
`dit1024_radix4_generic`" — does not reproduce either. Same sweep, same
accounting:

| row                          |    n | c64 fwd | c64 inv | c128 fwd | c128 inv |
| ---------------------------- | ---: | ------: | ------: | -------: | -------: |
| `dit1024_radix32x32_generic` | 1024 |   1.264 |   1.794 |    1.522 |    1.979 |
| `dit512_radix16x32_generic`  |  512 |   1.230 |   1.339 |    1.470 |    1.527 |

Against `dit1024_radix4_generic` specifically, 32×32 measures **1.255×**, not
7.2×. Whatever produced that figure was measuring something else.

The shape that survives is a **forward/inverse asymmetry**, and it is the whole
story for 32×32: forward loses 1.26–1.52 while inverse loses 1.79–1.98. A gap
that large between two directions of the same decomposition is an inverse-path
defect, not a decomposition verdict — the same class of finding as the scaling
pass that AGENTS.md records sitting in 28 kernel files.

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

#### The radix-4 tail on the Xeon: headroom confirmed, fusion closed (2026-08-01)

Sweep at the five odd-exponent sizes, `-tags fftprobe`, `PASSES=10`,
`GOOD=11298`, **100 accepted / 0 rejected**, `benchmarks/gated-tail`. Ratios are
within group, against `dit<N>_radix4_avx2` (the unfused kernel) rather than
against the incumbent — on this host the incumbent at 512 and up is now the
AVX-512 radix-8 ladder, so an incumbent-relative number would not be about the
tail at all.

| prec |     n | notail/plain | fused/plain | fused/plain (i7-1255U) |
| ---- | ----: | -----------: | ----------: | ---------------------: |
| c64  |   128 |        0.876 |       0.976 |                  0.955 |
| c64  |   512 |        0.884 |       0.980 |                  0.971 |
| c64  |  2048 |        0.869 |       1.040 |                  0.943 |
| c64  |  8192 |        0.899 |       1.117 |                  1.034 |
| c64  | 32768 |        0.923 |       1.072 |                  1.004 |
| c128 |   128 |        0.849 |       0.966 |                  0.935 |
| c128 |   512 |    **0.766** |       0.870 |                  1.002 |
| c128 |  2048 |        0.829 |       1.035 |                  1.110 |
| c128 |  8192 |        0.875 |       1.026 |                  1.006 |
| c128 | 32768 |        0.879 |       1.012 |                  1.020 |

**The headroom transfers and is larger here**: the tail costs 7.7-23.4% on the
Xeon against 6.7-13.3% on the laptop, in every cell of both precisions. That is
now a two-microarchitecture result and the strongest reason to keep attacking it.

**Fusion is closed as the lever.** It recovers at most 2.4% (c64 128/512, c128
128), and from n = 2048 up it is a net loss in both precisions on both hosts.
Ten cells, two hosts, no cell where it captures even a third of what `notail`
says is available.

**An L1-associativity explanation was predicted and is refuted.** Both hosts
alias at 4 KB (64 sets × 64 B), but the Xeon has 8 ways against the laptop's 12,
and the fused loop holds ~11 mutually-aliasing streams — 8 data plus 3 twiddle.
That predicts fusion degrading distinctly on the Xeon at the large sizes. It
does for c64 (+0.068 to +0.097 at 2048/8192/32768) and **fails for c128**, which
has twice the byte stride and identical set-aliasing yet gets _better_: -0.132 at
512, -0.075 at 2048, -0.008 at 32768. A way-count mechanism cannot produce that
sign split. Whatever fusion costs, it is not L1 conflict misses.

The surviving explanation is the register budget, the same one that explains the
AVX2 radix-8 ladder: fusion pins Y0-Y7 across group 1's whole computation and
leaves only Y8-Y13 scratch, so group 1 re-loads its twiddle broadcasts every
iteration (the loop comment in `avx2_f32_radix4.s` says so outright). That is a
prediction about ZMM, not about cache, and it is untested.

One figure worth carrying: at n = 8192 complex64 the tail-free probe scores
**0.998 against the AVX-512 radix-8 ladder** while the real AVX2 kernel scores
1.110. A radix-4 AVX2 kernel that did not pay the tail would match an AVX-512
kernel at that size.

For complex128 the Xeon has already routed around the problem — the AVX2
radix-8 ladder, which removes the tail structurally rather than fusing it, beats
`radix4_avx2` here by 25% at 512 (1.225 vs 1.629), 28% at 2048 (1.135 vs 1.584)
and 16% at 32768 (1.156 vs 1.374). The open cell is complex64, where the ladder
did not win on either host.

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

The reason the files came out that way is a toolchain constraint worth recording:
**Go's arm64 assembler has no mnemonic for vector FP add, subtract or multiply.**
`VFADD`/`VFSUB`/`VFMUL`, and bare `FADD`, are all "unrecognized instruction";
`VFMLA` and `VFMLS` are the only vector FP arithmetic mnemonics it accepts, and
`VADD`/`VSUB` are integer-only.

**That is a gap in the assembler's instruction table, not in the hardware, and
treating it as a hard limit was a mistake that cost a factor of two.** The
encodings can be emitted directly with `WORD`; `internal/asm/arm64/neon_fp.h`
now provides verified macros for `FADD`/`FSUB`/`FMUL`/`FMLA`/`FMLS` plus
register move and xor, in both `.4S` and `.2D`. The workaround this file
originally recommended — the one `neon_f32_generic.s` uses, synthesizing add and
subtract against a vector of 1.0 (`·neonOnes`) with `VFMLA`/`VFMLS` — costs
**two instructions per add or subtract**, a register for the constant, and two
prologue instructions to load it. The claim that "on Apple cores FMA has the
same throughput as FADD, so this costs nothing" was wrong: the throughput claim
is true, but it is not the issue — the instruction _count_ is. A radix-4
butterfly is sixteen adds and subtracts.

The macros take register **numbers**, not names, because the encoding embeds the
number and Go's asm preprocessor has no token pasting (`##` fails with "'#' must
be first item on line"). So a converted file carries two macro families: numeric
ones for arithmetic and name-taking ones for the shuffles and loads that do have
mnemonics. Nothing checks the `V<n>` ↔ `n` correspondence; the registry-driven
reference tests are what catch a wrong number.

One further trap: `VLD2`/`VST2` register lists must be **contiguous**
(`[V0,V1]`, never `[V0,V4]`).

`RankLevel` could not demote below the generic tier — the registry reads
`RankLevel == SIMDNone` as unset, and `SIMDLevel` is unsigned, so there was no
level under `SIMDNone` and no way to tie-break beneath a generic sibling at
priority 0. `CodeletEntry.RankBelowGeneric` now expresses exactly that; see
"The selection problem this exposes" below.

### Size 16, complex64 — first vectorized rewrite

`neon_f32_size16_radix4.s` (`dit16_radix4_neon`) was rewritten as a 4x4
Cooley-Tukey decomposition: `VLD2` deinterleaves four complex per register, two
rounds of vertical DFT4 sandwich a `VTRN1/VTRN2` + `VZIP1/VZIP2` 4x4 transpose,
and each store is a single `VST2` of four consecutive outputs. It needs no
bit-reversal pass and no scratch buffer — every input is in registers before the
first store, so `dst` may alias `src`. The scalar implementation it replaced was
deleted rather than shadowed; note that `·neonInv16` had to move into this file,
because `neon_f32_size16_radix2.s` references it.

A/B against the scalar version while both were registered (`count=8`, same
process, mean ns/op):

| candidate                |  forward |   inverse |
| ------------------------ | -------: | --------: |
| **vectorized (kept)**    | **9.50** | **10.44** |
| dit16_radix16_generic    |    16.23 |     17.74 |
| dit16_radix4_generic     |    16.25 |     20.49 |
| scalar dit16_radix4_neon |    35.68 |     38.53 |
| dit16_radix2_neon        |    61.78 |     68.84 |

**3.76x / 3.69x** faster than the scalar version and **1.71x / 1.70x** faster
than the best pure-Go candidate — a size the scalar version had lost by
1.83x / 2.19x. After deleting the scalar file, re-measured on a quieter machine:
7.24 / 7.92 ns against 12.01 / 13.44 for the best pure-Go candidate, i.e.
1.66x / 1.70x — the absolute numbers are lower with fewer candidates in the
process, the ratio is unchanged. Only ratios travel.

Correctness: naive-DFT cross-check, round-trip and in-place all pass under both
QEMU and the M5.

Converting this kernel to the `WORD`-encoded arithmetic described above — same
algorithm, only the instruction idiom changed — took it to **6.22 / 6.76 ns**,
i.e. **1.95x / 1.99x** vs the best pure-Go candidate. That is ~15% off an
already-winning kernel purely from not synthesizing adds.

### Sizes 4, 8 and 16 — five more kernels, and where the crossover is

Five further kernels were rewritten: `neon_f32_size4_radix4.s`,
`neon_f32_size8_radix8.s`, `neon_f64_size4_radix4.s`,
`neon_f64_size8_radix4.s` and `neon_f64_size16_radix4.s`, all in the
`WORD`-encoded idiom. State after that round (M5, `count=10`, same process,
mean ns/op, ratio against the best non-NEON candidate at the cell):

| cell             | best pure-Go |      NEON | verdict       |
| ---------------- | -----------: | --------: | ------------- |
| c64 size 16 fwd  |        11.99 |  **6.19** | **win 1.94x** |
| c64 size 16 inv  |        13.41 |  **6.51** | **win 2.06x** |
| c128 size 16 fwd |        12.23 | **10.58** | **win 1.16x** |
| c128 size 16 inv |        13.61 | **11.59** | **win 1.17x** |
| c64 size 8 fwd   |         3.75 |      4.05 | loss 1.08x    |
| c64 size 8 inv   |         5.88 |  **4.23** | **win 1.39x** |
| c128 size 8 fwd  |         3.64 |      5.16 | loss 1.42x    |
| c128 size 8 inv  |         5.76 |  **5.36** | **win 1.07x** |
| c64 size 4 fwd   |         1.78 |      2.86 | loss 1.61x    |
| c64 size 4 inv   |         2.09 |      2.94 | loss 1.40x    |
| c128 size 4 fwd  |         1.76 |      3.64 | loss 2.07x    |
| c128 size 4 inv  |         2.13 |      3.80 | loss 1.78x    |

c128 size 16 is the notable one: it went from a **1.95x / 2.36x loss to a
1.16x / 1.17x win**, and since it was already selected, that loss was a live
regression rather than a missed opportunity.

**Size 4 looks structural.** The whole transform is sixteen real adds, which the
pure-Go codelet does in 1.8 ns with **no call boundary** — the Go compiler
inlines it. An assembly codelet pays a call, a length-validation preamble and
its prologue before doing any arithmetic, and on this host that is most of the
budget. Both precisions lose at size 4 and both are selected. The crossover
where assembly can win at all sits between 8 and 16 here. This is an argument
of exactly the mechanistic kind §2.2 warns about, so it wants a second host
before anyone acts on it — but unlike the usual case it is corroborated by the
measurement rather than substituting for one.

### Sizes 64–16384, complex64 — one looped core replaces five unrolled files

`neon_f32_size{64,256,1024,4096,16384}_*.s` were 28,503 lines of fully-unrolled
scalar code (18,399 of them in the size-16384 file alone) plus 175 KB of
bit-reversal tables, and all five lost. They are now one 586-line looped radix-4
Stockham core (`neon_f32_radix4_loop.s`) with ten thin Go wrappers keeping the
old exported names — a 48x reduction. Stage 0 (m=1) vectorizes along j with
strided twiddle gathers and a 4x4 transpose before the store; every later stage
(m>=4) vectorizes along k with scalar twiddles broadcast by `VLD1R`, which is
where most of the work is.

Every one of the ten cells now wins, and every one previously lost:

| cell      | best pure-Go |     NEON | verdict       |
| --------- | -----------: | -------: | ------------- |
| 64 fwd    |        85.19 |    52.21 | **win 1.63x** |
| 64 inv    |        99.31 |    54.60 | **win 1.82x** |
| 256 fwd   |       398.65 |   204.99 | **win 1.94x** |
| 256 inv   |       408.70 |   214.74 | **win 1.90x** |
| 1024 fwd  |      2085.00 |  1046.38 | **win 1.99x** |
| 1024 inv  |      2174.38 |  1077.88 | **win 2.02x** |
| 4096 fwd  |      9520.62 |  4638.25 | **win 2.05x** |
| 4096 inv  |      9899.62 |  4754.12 | **win 2.08x** |
| 16384 fwd |     47629.25 | 24926.00 | **win 1.91x** |
| 16384 inv |     48000.75 | 24984.50 | **win 1.92x** |

Same host, same benchmark, size 4096 before and after:

| direction | unrolled scalar | looped NEON | kernel speedup | ratio swing            |
| --------- | --------------: | ----------: | -------------: | ---------------------- |
| forward   |        15807.88 |     4638.25 |      **3.41x** | 1.62x loss → 2.05x win |
| inverse   |        17779.50 |     4754.12 |      **3.74x** | 1.80x loss → 2.08x win |

Part of that is vectorization and part is simply not executing an 18,000-line
straight-line body: the unrolled files were almost certainly I-cache hostile,
which is why the largest sizes improved as much as the middle ones rather than
less.

Note what is still registered and still losing at these sizes:
`dit256_radix2_neon` (2.67x/2.86x), `dit1024_radix2_neon` (2.85x/3.13x). They
rank below the converted radix-4 codelets so they are not selected, but they are
dead weight.

### The selection problem this exposes

Registry ordering is **SIMD-level major**, so a NEON codelet is selected over a
faster pure-Go one at the same size. Every losing cell above is therefore not a
missed opportunity but an active regression for arm64 users: the library picks
the slower kernel. With ~50 of 56 NEON cells still losing, that is the largest
single arm64 performance issue in the tree — larger than any individual kernel
rewrite.

Neither existing mechanism could express the fix. `RankLevel` cannot demote
below the generic tier, and `Priority < 0` removes the codelet from
`LookupBySignature` and from the registry-driven correctness tests as well as
from lookup — it stops being _verified_, which is far more than "measured slower
on one host" warrants, and it also puts the row out of the wisdom tuner's reach
on the very machines where it might win.

**Fixed by `CodeletEntry.RankBelowGeneric`** (2026-08-02), a third disposition
between "registered candidate" and "disabled": the entry ranks under every
pure-Go codelet, so `Lookup` never returns it while a generic sibling exists —
and one exists at every NEON size — but it stays compiled, stays covered by the
reference tests, and stays reachable by signature, so the wisdom tuner can still
select it on a different microarchitecture. That is what keeps the M5 result
re-measurable instead of turning it into folklore, per PLAN.md §2.2.

Applied to the **36 NEON spec rows** the sweep measured as losing — every NEON
row except the nine that now win: c64 `dit8_radix8`, c64/c128 `dit16_radix4`,
c128 `dit8_radix4`, and c64 `dit{64,256,1024,4096,16384}_radix4`. Those nine
keep normal NEON rank. Sizes 8 (both precisions) are the judgement call: each
loses forward and wins inverse, and one entry carries both directions, so they
are kept at NEON rank on the ≤1.5x arm of §2.2.

The demotions are recorded as a **verdict, not a deletion**. Re-measure and lift
`RankBelowGeneric` as each kernel is vectorized — the c64 sizes 64–16384 rows
above are exactly what that looks like, and they were demoted-equivalent losses
one round earlier.
