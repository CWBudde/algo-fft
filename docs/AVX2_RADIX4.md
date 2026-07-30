# The 256-bit AVX2 radix-4 kernels

How the size-generic radix-4 kernels (`internal/asm/amd64/avx2_f32_radix4.s`,
`avx2_f64_radix4.s`) replaced the per-size AVX2 radix-4 codelet family, what
their design constraints are, and what the radix-2 tail at `n = 2·4^k` costs.

Moved out of `PLAN.md` on 2026-07-30. `PLAN.md` remains the source of
truth for status and open work; this file is the record behind it.

## The 256-bit radix-4 kernels (2026-07-28)

**The whole AVX2 radix-4 codelet family was XMM-width, not 256-bit.** Every
`avx2_f32_size*_radix4*.s` loaded operands with `VMOVSD` — one complex64 — and
did all butterfly arithmetic in `X` registers; `Y` registers appeared only in
the trailing copy and 1/n loops. These were scalar radix-4 kernels in VEX
clothing: they got the three-operand form and freedom from the transition
penalty, but none of the width.

Replaced by **two size-generic kernels**, `internal/asm/amd64/avx2_f32_radix4.s`
and `avx2_f64_radix4.s`, rather than twenty more hand-rolled files:

| n     | c64 before → after     | c128 before → after  |
| ----- | ---------------------- | -------------------- |
| 16    | —                      | 21 → 14 (1.5×)       |
| 64    | —                      | 156 → 55 (2.8×)      |
| 128   | 320 → 88 (3.6×)        | 367 → 129 (2.8×)     |
| 256   | 426 → 199 (2.1×)       | 644 → 262 (2.5×)     |
| 512   | 797 → 430 (1.9×)       | 1478 → 607 (2.4×)    |
| 1024  | 3200 → 918 (3.5×)      | 3698 → 1324 (2.8×)   |
| 4096  | 16848 → 4320 (3.9×)    | 20234 → 9165 (2.2×)  |
| 16384 | 73712 → 23599 (3.1×)   | 98044 → 39375 (2.5×) |
| 65536 | 519000 → 130594 (4.0×) | — → 334106           |

(forward, ns, best-of-5 pinned; the ranking tests re-derive this ordering from
measurement so the `Priority` values cannot silently rot.)

Design notes, since they are what made one kernel cover every size:

- The twiddle for butterfly `j` depends only on `j`, not on the group, so each
  stage needs `3*m` twiddles held as three contiguous planes. Every twiddle
  load is then a plain 256-bit read and the per-butterfly index arithmetic
  disappears.
- `n = 2*4^k` needs no separate kernel: running the radix-4 stages only to `n/2`
  transforms the even- and odd-indexed halves independently, and one radix-2
  tail combines them.
- The permutation table stores only `p[4g]`, as `int32` — 16 KiB at n = 16384
  against the 128 KiB `DATA` blob the old kernel embedded. It is taken from
  `internal/math` rather than rederived: a self-derived permutation table is
  the one bug class that has actually escaped review here. It is also
  precision-independent, so both kernels share `radix4GroupIndices`.
- The ±i rotation is `permute + xor` (2 ops) instead of
  `permute + xor-zero + sub + blend` (4). Forward and inverse differ only in
  which mask feeds which output; the inverse 1/n is exact and folds into
  stage 1.
- **Permutation fused into stage 1.** At n = 16384 the permutation pass alone
  had been a third of the kernel while doing no arithmetic. On the complex64
  side `VPGATHERDQ` delivers a0..a3 already separated, removing a full
  store-then-load _and_ the input transpose. There is no 128-bit-element
  gather, so the complex128 kernel builds its groups with
  `VMOVUPD` + `VINSERTF128` instead — the fusion is what mattered, not the
  gather. Net at n = 16384: 29.1 → 23.6 µs.
- **Twiddle broadcasts belong on the load ports, not port 5.** The inner loop
  broadcast each twiddle's real and imaginary part with the _register_ form of
  `VMOVSLDUP`/`VMOVSHDUP` (f32) or `VMOVDDUP`/`VPERMILPD` (f64) — six port-5
  shuffles per iteration, on the one port the loop is bound by. The **memory**
  forms are pure load uops for a re-broadcast scalar. For f64 the imaginary
  broadcast needs no instruction at all: offsetting the address by 8 bytes
  makes `VMOVDDUP` duplicate the high float64 instead. That reads 8 bytes past
  the last plane, which the `n+4` twiddle padding covers — and the kernel's
  length check _enforces_ rather than assumes. Plus: both ±i rotations permute
  the same `t3`, so permute once and branch with two `VXOR`s. Port-5 traffic
  11 → 4 per iteration; c64 −4…−13% at every size, c128 20–24% at 256–16384.
- **A dedicated `.s` file per size would buy very little.** `stage2Generic`
  (m = 4, 1024 group iterations) and `stage7YMM` (m = 4096, one group) cost the
  same 2.5–2.6 µs for the same 4096 butterflies, so the loop structure costs
  essentially nothing and constant-folding the bounds has nothing to reclaim.

Superseded kernels are **removed**, not left registered alongside: thirteen
`.s` files deleted across the two precisions, together with the 8192 "params"
twiddle layout. Shared `bitrev*` tables moved to
`internal/asm/amd64/bitrev_radix4_tables.s` where other-precision kernels still
reference them by symbol. Sizes that stay do so because something other than
the registry calls them — the six-step row FFTs, the size-384 decomposition,
and the `KernelStrategy` dispatch in
`internal/fft/kernels_amd64_size_specific.go`, which selects by strategy rather
than through the registry and so has no way to obtain a prepared twiddle table
(§4).

**The port-5 pattern does not transfer to the fused mixed-radix stages.** Tried
and reverted 2026-07-28. Those kernels run the same three-shuffle
complex-multiply idiom per row, so they looked like the same opportunity, but
the dup source is the _data_, not a scalar twiddle: one input vector feeds both
duplicates, so a memory-operand form does not replace a load, it adds one — and
`VMOVSLDUP ymm, m256` is not a load-only uop the way a 64-bit broadcast is, so
nothing moves off port 5 either. Measured: 2–28% _slower_, all 32 cases
regressing. Port 5 is not the bottleneck there anyway — a probe that deleted
the ten table swaps from the radix-11 stage outright (wrong results, right
instruction mix) moved the time by 0–7% with no consistent sign. The radix-4
kernel is shuffle-bound because it retires many butterflies per load out of a
small working set; the stages are streaming kernels at 2 reads + 1 write per
row. **Do not retry it on a kernel whose broadcast operand comes from the data
stream.**

## The n = 2048 radix-2 tail (2026-07-30)

§4 read the 2048 dip as pointing "at the route rather than at the arithmetic".
It does, and the route has a name: for `n = 2*4^k` the size-generic radix-4
kernel runs its radix-4 stages only to `n/2` — transforming the even- and
odd-indexed halves independently — and then makes **a separate full pass over
the buffer** to combine them (`r4d_radix2_tail`). Counting passes, 2048 costs 6
for 11 butterfly levels where 1024 costs 5 for 10 and 4096 costs 6 for 12:
0.545 passes per level against 0.50 at both neighbours. 2048 is also the only
one of the three with no second AVX2 candidate at all.

**What the tail costs, measured rather than reasoned.** A probe that skips the
combine outright — wrong answers, right instruction mix, the radix-11
idiom — puts it at roughly **8–15% of the kernel for n ≥ 512, and 9–20% at
n = 128**, in both precisions and both directions. Read those as ranges, not
readings: the same cell moves a few points between sweeps, and one 8192 inverse
cell read 2.8% against 10–15% in the other three. The ordering is the stable
part, and it tracks the pass model — the tail is one pass of `k+1`, so 25%
predicted at 128, 17% at 2048, 12.5% at 32768, and the measurements fall below
each of those in the same order. The probe needed no assembly: the kernel's only
shape knob is `r4End`, and passing `n` instead of `n/2` leaves the executed
stages bit-identical (the next stage would overrun either way) while the tail's
own guard then skips it. `TestRadix4AVX2NoTailProbeIsStagesOnly` proves that
equivalence by applying the missing combine in Go and requiring the real
kernel's output back, rather than asserting it from the loop bounds.

That number is the ceiling on any fusion, and it is worth having on its own:
**the tail is a tax on every odd power of two, not a 2048-specific defect.** It
costs a comparable fraction at 512, 2048 and 8192 — which sit at 0.97, 0.91 and
1.24 against FFTW3 (tag v0.7.4). So it explains why the `2*4^k` sizes are all
~10% below where they could be; it does not explain why 2048 is the one that
lands under parity. Nor is the tail the whole of the mid-band softness: 1024 is
a power of four, has no tail at all, and still measures 0.97.

**Fusing it into the last radix-4 stage works, and mostly does not pay.** The
last stage always has `4m = r4End = n/2` and therefore exactly two groups — the
even half and the odd half — and the tail pairs one output of each at the same
position, so running the two groups in lockstep leaves both operands of four
radix-2 butterflies in registers. Output addresses, the permutation table and
the packed twiddle layout are all unchanged; only the loop structure moves. The
register file ends up exactly full (four outputs per group, six scratch, two
rotation masks), which is why group 1 re-loads its twiddle broadcasts instead of
keeping them.

Fused as a ratio to the separate tail, forward/inverse, canary-gated, pinned,
7–10 accepted groups per cell:

| n     | stride (c64/c128) | complex64         | complex128        |
| ----- | ----------------- | ----------------- | ----------------- |
| 128   | 128 B / 256 B     | **0.955 / 0.979** | **0.935 / 0.934** |
| 512   | 512 B / 1 KiB     | 0.971 / 1.005     | 1.002 / 1.020     |
| 2048  | 2 KiB / 4 KiB     | **0.943 / 0.974** | **1.110 / 1.104** |
| 8192  | 8 KiB / 16 KiB    | 1.034 / 1.021     | 1.006 / 1.077     |
| 32768 | 32 KiB / 64 KiB   | 1.004 / 1.013     | 1.020 / 1.000     |

Fusing doubles the live streams from four to eight, and past a point that costs
more than the pass it saves. **The single worst cell is complex128 at n = 2048 —
11% slower — which is the size the fusion was written for.** Its last-stage
stride is exactly 4 KiB there, so all eight streams land on one L1 set; that the
target size is also the pathological one is arithmetic, not luck.

Two corrections to how this was read at the time, both from adding data:

- A first pass over six cells suggested "loses whenever the stride is a multiple
  of 4 KiB". Sweeping n = 128 and n = 32768 kept the small-stride win but made
  32768 **neutral** rather than the predicted loss — at 256–512 KiB both variants
  are bandwidth-bound and an L1 effect cannot decide anything. The trend is not
  monotonic in stride either (2 KiB wins, 1 KiB slightly loses), because the
  complex64 loop retires four butterflies per iteration to complex128's two and
  amortises the doubled stream count better. **Six points supported a rule that
  four more falsified.**
- The fused kernel is correct, and known to be so cheaply: fusing reorders no
  arithmetic, so `TestRadix4AVX2FusedMatchesUnfused` demands **bit-identical**
  output at every size 16…32768. An approximate comparison would have waved
  through a real defect as rounding.

**Landed per size, in the registry.** Rather than a runtime predicate over an
empirical rule, the three cells that win — `dit128_radix4fused_avx2` in both
precisions and `dit2048_radix4fused_avx2` at complex64 — are ordinary
`specs.go` rows, which is where every other per-size measured fact in this
library already lives. `TestRadix4AVX2FusedSelection` pins the choice so a
regenerate cannot widen it silently, and the existing ranking tests re-derive it
from measurement. The `-tags fftprobe` harness stays in tree and registers both
variants plus the no-tail probe side by side, so the table above can be
re-derived on other hardware instead of being trusted (the rule in `BENCHMARKING.md` about numbers
that cannot be re-measured where they are quoted).

_Not fixed:_ complex128 at n = 2048, the item this round was opened for, is
unchanged — the fusion is the wrong instrument there. The tail still costs
12–13% at that cell and reclaiming it needs a shape with a different access
pattern; the candidate is a twiddle-free radix-8 first stage, which reads
through the permutation and would not meet the 4 KiB stride at all.

**Incumbent audit, discharged for seven sizes.** The same sweeps cover §4's open
audit at 128/512/1024/2048/4096/8192/32768 in both precisions. Every incumbent
was confirmed as the fastest correct candidate **except the three this round
then changed** — n = 128 in both precisions and n = 2048 complex64, where the
fused variant won and took the row. Two side observations worth the record:
`dit4096_sixstep_avx2` runs 4.6× its size's incumbent, and
`dit1024_radix32x32_avx2` 8.1×/9.8×, which independently confirms the VEX round's
deletion item.
