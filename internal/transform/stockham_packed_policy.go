package transform

import (
	"math"
	"sync/atomic"

	"github.com/cwbudde/algo-fft/internal/cpu"
)

// This file decides *when* the packed radix-4 Stockham engine is the route a
// plan takes. The engine itself lives in stockham_packed.go and is correct on
// every build; what varies is whether it beats the alternative at a given size,
// precision and instruction-set tier.
//
// It replaces a pair of build-tagged constants that disabled the engine
// outright on amd64/arm64/386, on the stated grounds that "the hand-written
// codelet path is checked first and supersedes it". That rationale was false as
// written: every registered codelet carries Algorithm: KernelDIT, so the
// planner's estimate already resolves to KernelDIT wherever a codelet binds and
// the packed branch was never reachable there anyway. What the constant
// actually suppressed was the sizes with no codelet at all — where the SIMD
// build then fell through to a radix-2 Stockham kernel measurably slower than
// the radix-4 route being disabled (PLAN.md P3).

// packedTier is the instruction-set tier the packed-vs-kernel decision is
// keyed on. The competing kernel differs per tier, so the crossover does too.
type packedTier uint8

const (
	// tierGeneric covers pure-Go builds: -tags purego, and architectures with
	// no SIMD kernels at all. There the packed engine is the fastest Stockham
	// route available and has been the default for as long as it has existed.
	tierGeneric packedTier = iota
	tierSSE
	tierAVX2
	tierAVX512
	tierNEON

	packedTierCount
)

// packedOff is a minimum size no transform can reach, i.e. "never take the
// packed route on this tier".
const packedOff = math.MaxInt

// packedStockhamMinSize is the smallest transform length at which the packed
// radix-4 route is taken, per tier and precision ([0] = complex64,
// [1] = complex128).
//
// The AVX2 row was measured on an i7-1255U, pinned, 5 interleaved rounds per
// cell, forced KernelStockham, both arms in one binary via the override. The
// statistic is the median of the *within-round* packed/kernel ratio: the
// round-to-round spread of the null control (forced six-step) reached 1.69, so
// cross-round medians would have been measuring the machine. Ratios below 1
// mean the packed route wins:
//
//	           2^16    2^17    2^18    2^19    2^20    2^21
//	c64  fwd   1.625   1.518   1.175   0.934   0.672   0.515
//	c64  inv   1.647   1.513   1.176   0.976   0.729   0.514
//	c128 fwd   0.972   0.803   0.626   0.565   0.437   0.374
//	c128 inv   1.009   0.831   0.625   0.514   0.481   0.397
//
// Thresholds are the smallest size where both directions win and every larger
// size does too. complex128 crosses at 2^17 (1.25x/1.20x); complex64 not until
// 2^20 (1.49x/1.37x).
//
// 2^19 complex64 is deliberately excluded although its first five rounds
// averaged 0.900/0.950, nominally clearing a 0.95 bar. Ten further rounds put
// the ratios at 0.474-1.245 (forward) and 0.429-1.220 (inverse), medians
// 0.934/0.976 — a wash, not a 5% win. It is the one size where this table
// gives up something real, and it is given up because the measurement does not
// support it.
//
// The threshold is per precision because the data demands it: complex128 wins
// from 2^17 while complex64 loses by 1.5x there. The padShapes convention of
// admitting a shape only where it wins at *both* precisions would forfeit
// 2^17-2^19 complex128, i.e. most of the benefit.
//
// The other SIMD tiers stay packedOff: §2.1 rule 5 forbids landing a route that
// has not beaten the incumbent on its own hardware tier, and of the four only
// AVX2 is measurable here — the SSE-only host is weekend-access and arm64 runs
// under QEMU, where timing is meaningless (§2.3). Their uncovered range is one
// octave wider than AVX2's, since their codelet ladder stops at 32768.
//
//nolint:gochecknoglobals // threshold table, read-only after init
var packedStockhamMinSize = [packedTierCount][2]int{
	tierGeneric: {4, 4},
	tierSSE:     {packedOff, packedOff},
	tierAVX2:    {1 << 20, 1 << 17},
	tierAVX512:  {packedOff, packedOff},
	tierNEON:    {packedOff, packedOff},
}

// PackedOverride forces the packed-route decision regardless of the table.
type PackedOverride int32

const (
	// PackedOverrideDefault consults packedStockhamMinSize (production path).
	PackedOverrideDefault PackedOverride = iota
	// PackedOverrideForceOn takes the packed route wherever it is applicable.
	PackedOverrideForceOn
	// PackedOverrideForceOff never takes it.
	PackedOverrideForceOff
)

//nolint:gochecknoglobals // process-wide test/measurement knob, see SetPackedStockhamOverride
var packedOverride atomic.Int32

// SetPackedStockhamOverride forces or restores the packed-route decision.
//
// This exists so both arms of the crossover measurement live in one binary.
// Comparing a default build against -tags purego would confound the routing
// change with every other difference between the two builds — and code layout
// alone has repeatedly moved results here by more than the effect being chased
// (§2.2: "prefer a single binary with an env knob over two builds").
//
// It is read once per plan construction, never per transform. Set it before
// building the plans under test and leave it alone afterwards: it is atomic, so
// concurrent reads are safe, but a plan captures the decision at construction
// and will not observe a later change.
func SetPackedStockhamOverride(mode PackedOverride) {
	packedOverride.Store(int32(mode))
}

// PackedStockhamOverride reports the current override.
func PackedStockhamOverride() PackedOverride {
	return PackedOverride(packedOverride.Load())
}

// packedTierFor maps detected CPU features to the tier whose threshold applies.
//
// It answers tierGeneric whenever the build has no SIMD kernels compiled in,
// and that check must come first: internal/cpu/detect_amd64.go is tagged
// `//go:build amd64` with no `!purego`, so a purego amd64 build still reports
// HasAVX2. Keying the decision on features alone would silently change purego
// behaviour, which §2.1 rule 4 forbids.
func packedTierFor(features cpu.Features) packedTier {
	if !packedBuildHasSIMDKernels || features.ForceGeneric {
		return tierGeneric
	}

	switch {
	case features.HasAVX512:
		return tierAVX512
	case features.HasAVX2:
		return tierAVX2
	case features.HasNEON:
		return tierNEON
	case features.HasSSE2:
		return tierSSE
	default:
		return tierGeneric
	}
}

// PackedStockhamEnabled reports whether a plan of length n and the given
// precision should take the packed radix-4 Stockham route.
//
// wide selects the complex128 column. Callers pass it explicitly rather than
// having this function be generic: a type switch inside a generic body compiles
// every branch into every shape instantiation (PLAN.md §2.3).
func PackedStockhamEnabled(n int, wide bool, features cpu.Features) bool {
	switch PackedStockhamOverride() {
	case PackedOverrideForceOn:
		return true
	case PackedOverrideForceOff:
		return false
	case PackedOverrideDefault:
	}

	column := 0
	if wide {
		column = 1
	}

	return n >= packedStockhamMinSize[packedTierFor(features)][column]
}
