package main

// codeletSpec is one declarative row of the codelet registration table. It is
// the single source of truth for the built-in codelets; the generator turns it
// into the register*DITCodelets{64,128} functions in internal/kernels.
//
// This table was seeded by AST-extracting the original hand-written
// codelet_init*.go files. To add or retune a codelet, edit a row here (or add
// one) and run `go generate ./internal/kernels/...`.
type codeletSpec struct {
	// Target selects the output file / build tag: "generic", "avx2", "avx512", "sse2", "neon".
	Target string
	// Prec is 64 or 128 (complex64 / complex128).
	Prec int
	// Size is the FFT length the codelet handles.
	Size int
	// Forward / Inverse are kernel expressions matching the CodeletFunc
	// signature (bool-returning; the codelet reports whether it handled the
	// transform).
	Forward string
	Inverse string
	// Algorithm, SIMDLevel, KernelType are planner enum identifiers.
	Algorithm  string
	SIMDLevel  string
	KernelType string
	// Signature is the human-readable codelet name (used for wisdom lookups).
	Signature string
	// Priority breaks ties within a (Size, rank level); higher wins, negative disables.
	Priority int
	// RankLevel optionally overrides the SIMD level used for ordering only
	// (see registry.CodeletEntry.RankLevel). Empty = rank at SIMDLevel. Use it
	// to demote an AVX2-encoded but SSE-width codelet into the SSE2 tier so
	// its priority is actually comparable with the SSE2 codelets it loses to.
	RankLevel string
	// RankBelowGeneric orders the codelet under every pure-Go codelet (see
	// registry.CodeletEntry.RankBelowGeneric). Set it when a measurement found
	// the codelet slower than pure Go: it stops being selected but stays
	// compiled, correctness-tested and reachable by wisdom, so the result can
	// be rechecked on another microarchitecture. Mutually exclusive with
	// RankLevel, which only moves a codelet between SIMD tiers.
	RankBelowGeneric bool
	// TwiddleSize / PrepareTwiddle are optional identifiers for codelets that
	// need a custom twiddle layout; empty when unused.
	TwiddleSize    string
	PrepareTwiddle string
}

//nolint:gochecknoglobals // the codelet table is the generator's declarative input
var codeletSpecs = []codeletSpec{
	{
		Target: "generic", Prec: 64, Size: 4,
		Forward:   "forwardDIT4Radix4Complex64",
		Inverse:   "inverseDIT4Radix4Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeCore",
		Signature: "dit4_radix4_generic", Priority: 0,
	},
	{
		Target: "generic", Prec: 64, Size: 8,
		Forward:   "forwardDIT8Radix2Complex64",
		Inverse:   "inverseDIT8Radix2Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit8_radix2_generic", Priority: 20,
	},
	{
		Target: "generic", Prec: 64, Size: 8,
		Forward:   "forwardDIT8Radix8Complex64",
		Inverse:   "inverseDIT8Radix8Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeCore",
		Signature: "dit8_radix8_generic", Priority: 30,
	},
	{
		Target: "generic", Prec: 64, Size: 8,
		Forward:   "forwardDIT8Radix4Complex64",
		Inverse:   "inverseDIT8Radix4Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit8_mixedradix_generic", Priority: 25,
	},
	{
		Target: "generic", Prec: 64, Size: 16,
		Forward:   "forwardDIT16Radix2Complex64",
		Inverse:   "inverseDIT16Radix2Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit16_radix2_generic", Priority: 10,
	},
	{
		Target: "generic", Prec: 64, Size: 16,
		Forward:   "forwardDIT16Radix4Complex64",
		Inverse:   "inverseDIT16Radix4Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit16_radix4_generic", Priority: 20,
	},
	{
		Target: "generic", Prec: 64, Size: 16,
		Forward:   "forwardDIT16Radix16Complex64",
		Inverse:   "inverseDIT16Radix16Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit16_radix16_generic", Priority: 30,
	},
	{
		Target: "generic", Prec: 64, Size: 32,
		Forward:   "forwardDIT32Radix2Complex64",
		Inverse:   "inverseDIT32Radix2Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit32_radix2_generic", Priority: 0,
	},
	{
		Target: "generic", Prec: 64, Size: 32,
		Forward:   "forwardDIT32Radix4Then2Complex64",
		Inverse:   "inverseDIT32Radix4Then2Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit32_radix4_then2_generic", Priority: 10,
	},
	{
		Target: "generic", Prec: 64, Size: 32,
		Forward:   "forwardDIT32Radix32Complex64",
		Inverse:   "inverseDIT32Radix32Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeCore",
		Signature: "dit32_radix32_generic", Priority: 5,
	},
	{
		Target: "generic", Prec: 64, Size: 64,
		Forward:   "forwardDIT64Radix2Complex64",
		Inverse:   "inverseDIT64Radix2Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit64_radix2_generic", Priority: 0,
	},
	{
		Target: "generic", Prec: 64, Size: 64,
		Forward:   "forwardDIT64Radix4Complex64",
		Inverse:   "inverseDIT64Radix4Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit64_radix4_generic", Priority: 20,
	},
	{
		Target: "generic", Prec: 64, Size: 128,
		Forward:   "forwardDIT128Radix2Complex64",
		Inverse:   "inverseDIT128Radix2Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit128_radix2_generic", Priority: 0,
	},
	{
		Target: "generic", Prec: 64, Size: 128,
		Forward:   "forwardDIT128Radix4Then2Complex64",
		Inverse:   "inverseDIT128Radix4Then2Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit128_radix4_then2_generic", Priority: 15,
	},
	{
		Target: "generic", Prec: 64, Size: 256,
		Forward:   "forwardDIT256Complex64",
		Inverse:   "inverseDIT256Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit256_radix2_generic", Priority: 10,
	},
	{
		Target: "generic", Prec: 64, Size: 256,
		Forward:   "forwardDIT256Radix4Complex64",
		Inverse:   "inverseDIT256Radix4Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit256_radix4_generic", Priority: 20,
	},
	{
		// Measured 1.014 forward / 0.784 inverse against dit256_radix4_generic in
		// the 2026-07-30 canary-gated purego re-sweep (46/48 groups accepted, both
		// sides post-scaling-sweep). Forward is a tie inside noise; the inverse
		// win pays for it. See docs/CODELET_BENCHMARKS.md.
		Target: "generic", Prec: 64, Size: 256,
		Forward:   "forwardRadix8Complex64",
		Inverse:   "inverseRadix8Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit256_radix8ladder_generic", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex64",
	},
	{
		Target: "generic", Prec: 64, Size: 384,
		Forward:   "forwardDIT384MixedComplex64",
		Inverse:   "inverseDIT384MixedComplex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit384_mixed_generic", Priority: 20,
	},
	{
		Target: "generic", Prec: 64, Size: 512,
		Forward:   "forwardDIT512Complex64",
		Inverse:   "inverseDIT512Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit512_radix2_generic", Priority: 10,
	},
	{
		Target: "generic", Prec: 64, Size: 512,
		Forward:   "forwardDIT512Radix8Complex64",
		Inverse:   "inverseDIT512Radix8Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit512_radix8_generic", Priority: 30,
	},
	{
		Target: "generic", Prec: 64, Size: 512,
		Forward:   "forwardDIT512Radix4Then2Complex64",
		Inverse:   "inverseDIT512Radix4Then2Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit512_radix4_then2_generic", Priority: 45,
	},
	{
		// Demoted 35 -> 1 on 2026-08-02. Measured 1.230 fwd / 1.339 inv
		// (complex64) and 1.470 / 1.527 (complex128) against
		// dit512_radix8ladder_generic; Xeon, purego, canary-gated. It was never
		// selected at 35 either -- the ladder holds 50 -- but 35 read as a
		// contender, and PLAN.md §2.2 keeps a beaten kernel registered only at a
		// priority that says so. See the sibling 32x32 row for the shared
		// forward/inverse asymmetry that is the real finding.
		Target: "generic", Prec: 64, Size: 512,
		Forward:   "forwardDIT512Mixed16x32Complex64",
		Inverse:   "inverseDIT512Mixed16x32Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit512_radix16x32_generic", Priority: 1,
	},
	{
		// Measured 0.807 forward / 0.823 inverse against dit512_radix4_then2_generic
		// in the 2026-07-30 canary-gated purego sweep; see docs/CODELET_BENCHMARKS.md.
		Target: "generic", Prec: 64, Size: 512,
		Forward:   "forwardRadix8Complex64",
		Inverse:   "inverseRadix8Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit512_radix8ladder_generic", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex64",
	},
	{
		Target: "generic", Prec: 64, Size: 1024,
		Forward:   "forwardDIT1024Radix4Complex64",
		Inverse:   "inverseDIT1024Radix4Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit1024_radix4_generic", Priority: 30,
	},
	{
		// Demoted 25 -> 1 on 2026-08-02. Measured 1.264 fwd / 1.794 inv
		// (complex64) and 1.522 / 1.979 (complex128) against the group
		// incumbent; Xeon, purego, canary-gated.
		//
		// The forward/inverse asymmetry is the finding, and it is why this row
		// is demoted rather than probe-gated: a decomposition that loses 1.26x
		// forward and 1.79x inverse has an inverse-path defect, not a bad
		// decomposition. PLAN.md §2.2 -- a poor implementation disqualifies the
		// file, not the algorithm.
		//
		// Note also what this row is NOT: PLAN.md long blamed the family for
		// having "only one of two stages vectorised" and losing 7.2x/5.2x to
		// dit1024_radix4_generic. The vectorised files were deleted (08c8e7b,
		// 1f7977b) and the surviving pure-Go row measures 1.255x against
		// radix4, not 7.2x. Both halves of that premise were stale.
		Target: "generic", Prec: 64, Size: 1024,
		Forward:   "forwardDIT1024Mixed32x32Complex64",
		Inverse:   "inverseDIT1024Mixed32x32Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit1024_radix32x32_generic", Priority: 1,
	},
	{
		// Measured 0.900 forward / 0.933 inverse against dit1024_radix4_generic
		// in the 2026-07-30 canary-gated purego sweep; see docs/CODELET_BENCHMARKS.md.
		Target: "generic", Prec: 64, Size: 1024,
		Forward:   "forwardRadix8Complex64",
		Inverse:   "inverseRadix8Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit1024_radix8ladder_generic", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex64",
	},
	{
		Target: "generic", Prec: 64, Size: 2048,
		Forward:   "forwardDIT2048Radix4Then2Complex64",
		Inverse:   "inverseDIT2048Radix4Then2Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit2048_radix4_then2_generic", Priority: 20,
	},
	{
		// Measured 0.968 forward / 0.764 inverse against dit2048_radix4_then2_generic
		// in the 2026-07-30 canary-gated purego re-sweep; see docs/CODELET_BENCHMARKS.md.
		Target: "generic", Prec: 64, Size: 2048,
		Forward:   "forwardRadix8Complex64",
		Inverse:   "inverseRadix8Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit2048_radix8ladder_generic", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex64",
	},
	{
		Target: "generic", Prec: 64, Size: 4096,
		Forward:   "forwardDIT4096Radix4Complex64",
		Inverse:   "inverseDIT4096Radix4Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit4096_radix4_generic", Priority: 30,
	},
	{
		// Measured 0.859 forward / 0.766 inverse against dit4096_radix4_generic
		// in the 2026-07-30 canary-gated purego sweep; see docs/CODELET_BENCHMARKS.md.
		Target: "generic", Prec: 64, Size: 4096,
		Forward:   "forwardRadix8Complex64",
		Inverse:   "inverseRadix8Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit4096_radix8ladder_generic", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex64",
	},
	{
		Target: "generic", Prec: 64, Size: 8192,
		Forward:   "forwardDIT8192Radix4Then2Complex64",
		Inverse:   "inverseDIT8192Radix4Then2Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit8192_radix4_then2_generic", Priority: 35,
	},
	{
		// Measured 0.889 forward / 0.882 inverse against dit8192_radix4_then2_generic
		// in the 2026-07-30 canary-gated purego sweep; see docs/CODELET_BENCHMARKS.md.
		Target: "generic", Prec: 64, Size: 8192,
		Forward:   "forwardRadix8Complex64",
		Inverse:   "inverseRadix8Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit8192_radix8ladder_generic", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex64",
	},
	{
		Target: "generic", Prec: 64, Size: 16384,
		Forward:   "forwardDIT16384Radix4Complex64",
		Inverse:   "inverseDIT16384Radix4Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit16384_radix4_generic", Priority: 30,
	},
	{
		// Measured 0.790 forward / 0.773 inverse against dit16384_radix4_generic
		// in the 2026-07-30 canary-gated purego sweep; see docs/CODELET_BENCHMARKS.md.
		Target: "generic", Prec: 64, Size: 16384,
		Forward:   "forwardRadix8Complex64",
		Inverse:   "inverseRadix8Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit16384_radix8ladder_generic", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex64",
	},
	{
		Target: "generic", Prec: 64, Size: 32768,
		Forward:   "forwardDIT32768Radix4Then2Complex64",
		Inverse:   "inverseDIT32768Radix4Then2Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit32768_radix4_then2_generic", Priority: 20,
	},

	{
		Target: "generic", Prec: 64, Size: 65536,
		Forward:   "forwardDIT65536Radix4Complex64",
		Inverse:   "inverseDIT65536Radix4Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit65536_radix4_generic", Priority: 20,
	},

	// Split-radix (PLAN.md §1.2, "Give split-radix a fair measurement").
	//
	// These rows exist so the family can be *measured*. Split-radix is the
	// largest untested cell in the algorithm × ISA matrix: it has a full
	// strategy plumbing and a pure-Go kernel, but no codelet row anywhere, and
	// the canary-gated sweep can only see registered candidates
	// (scripts/bench_gated.sh drives BenchmarkCodeletCandidates<prec>). No row,
	// no measurement.
	//
	// One function serves every size — ForwardSplitRadixComplex64 is
	// size-generic and declines any non-power-of-two via splitRadixValid — so
	// the rows differ only in Size and Signature.
	//
	// Priority 1 is deliberately non-selectable: every size below has a generic
	// row at 20 or above, so these are ranked last in the pure-Go tier while
	// staying visible to the sweep (the bench skips Priority < 0, not 1) and to
	// the registry-driven reference tests, which is correctness coverage this
	// kernel has never had. §2.1 gate 5 forbids a selectable priority before a
	// measurement; the numbers set the final value.
	//
	// The split-radix ladder stops at 32768, and the reason has changed.
	// It used to be a safety property: n = 65536 had no generic row at all,
	// so a split-radix row there would have been the only pure-Go candidate
	// and would have become the selected purego route for a kernel nobody
	// had measured. `dit65536_radix4_generic` above removes that hazard — a
	// priority-1 row now ranks below it and cannot be selected.
	//
	// What remains is simply that split-radix has not been measured at
	// 65536 against a real opponent. Its apparent win there was against the
	// coverage hole the row above just filled, so the comparison has to be
	// re-run before a row is added; PLAN.md's Phase 3 ladder item owns that
	// order of operations. Meanwhile the 16384-131072 band is measured at
	// plan level by the KernelSplitRadix arm in BenchmarkStepCrossover.
	{
		Target: "generic", Prec: 64, Size: 256,
		Forward:   "ForwardSplitRadixComplex64",
		Inverse:   "InverseSplitRadixComplex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit256_splitradix_generic", Priority: 1,
	},
	{
		Target: "generic", Prec: 64, Size: 512,
		Forward:   "ForwardSplitRadixComplex64",
		Inverse:   "InverseSplitRadixComplex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit512_splitradix_generic", Priority: 1,
	},
	{
		Target: "generic", Prec: 64, Size: 1024,
		Forward:   "ForwardSplitRadixComplex64",
		Inverse:   "InverseSplitRadixComplex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit1024_splitradix_generic", Priority: 1,
	},
	{
		Target: "generic", Prec: 64, Size: 2048,
		Forward:   "ForwardSplitRadixComplex64",
		Inverse:   "InverseSplitRadixComplex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit2048_splitradix_generic", Priority: 1,
	},
	{
		Target: "generic", Prec: 64, Size: 4096,
		Forward:   "ForwardSplitRadixComplex64",
		Inverse:   "InverseSplitRadixComplex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit4096_splitradix_generic", Priority: 1,
	},
	{
		Target: "generic", Prec: 64, Size: 8192,
		Forward:   "ForwardSplitRadixComplex64",
		Inverse:   "InverseSplitRadixComplex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit8192_splitradix_generic", Priority: 1,
	},
	{
		Target: "generic", Prec: 64, Size: 16384,
		Forward:   "ForwardSplitRadixComplex64",
		Inverse:   "InverseSplitRadixComplex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit16384_splitradix_generic", Priority: 1,
	},
	{
		Target: "generic", Prec: 64, Size: 32768,
		Forward:   "ForwardSplitRadixComplex64",
		Inverse:   "InverseSplitRadixComplex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit32768_splitradix_generic", Priority: 1,
	},

	{
		Target: "generic", Prec: 128, Size: 4,
		Forward:   "forwardDIT4Radix4Complex128",
		Inverse:   "inverseDIT4Radix4Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeCore",
		Signature: "dit4_radix4_generic", Priority: 0,
	},
	{
		Target: "generic", Prec: 128, Size: 8,
		Forward:   "forwardDIT8Radix2Complex128",
		Inverse:   "inverseDIT8Radix2Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit8_radix2_generic", Priority: 20,
	},
	{
		Target: "generic", Prec: 128, Size: 8,
		Forward:   "forwardDIT8Radix8Complex128",
		Inverse:   "inverseDIT8Radix8Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeCore",
		Signature: "dit8_radix8_generic", Priority: 30,
	},
	{
		Target: "generic", Prec: 128, Size: 8,
		Forward:   "forwardDIT8Radix4Complex128",
		Inverse:   "inverseDIT8Radix4Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit8_mixedradix_generic", Priority: 25,
	},
	{
		Target: "generic", Prec: 128, Size: 16,
		Forward:   "forwardDIT16Radix2Complex128",
		Inverse:   "inverseDIT16Radix2Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit16_radix2_generic", Priority: 10,
	},
	{
		Target: "generic", Prec: 128, Size: 16,
		Forward:   "forwardDIT16Radix4Complex128",
		Inverse:   "inverseDIT16Radix4Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit16_radix4_generic", Priority: 20,
	},
	{
		Target: "generic", Prec: 128, Size: 16,
		Forward:   "forwardDIT16Radix16Complex128",
		Inverse:   "inverseDIT16Radix16Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit16_radix16_generic", Priority: 30,
	},
	{
		Target: "generic", Prec: 128, Size: 32,
		Forward:   "forwardDIT32Radix2Complex128",
		Inverse:   "inverseDIT32Radix2Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit32_radix2_generic", Priority: 0,
	},
	{
		Target: "generic", Prec: 128, Size: 32,
		Forward:   "forwardDIT32Radix4Then2Complex128",
		Inverse:   "inverseDIT32Radix4Then2Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit32_radix4_then2_generic", Priority: 15,
	},
	{
		Target: "generic", Prec: 128, Size: 32,
		Forward:   "forwardDIT32Radix32Complex128",
		Inverse:   "inverseDIT32Radix32Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeCore",
		Signature: "dit32_radix32_generic", Priority: 5,
	},
	{
		Target: "generic", Prec: 128, Size: 64,
		Forward:   "forwardDIT64Radix2Complex128",
		Inverse:   "inverseDIT64Radix2Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit64_radix2_generic", Priority: 0,
	},
	{
		Target: "generic", Prec: 128, Size: 64,
		Forward:   "forwardDIT64Radix4Complex128",
		Inverse:   "inverseDIT64Radix4Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit64_radix4_generic", Priority: 20,
	},
	{
		Target: "generic", Prec: 128, Size: 128,
		Forward:   "forwardDIT128Radix2Complex128",
		Inverse:   "inverseDIT128Radix2Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit128_radix2_generic", Priority: 0,
	},
	{
		Target: "generic", Prec: 128, Size: 128,
		Forward:   "forwardDIT128Radix4Then2Complex128",
		Inverse:   "inverseDIT128Radix4Then2Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit128_radix4_then2_generic", Priority: 15,
	},
	{
		Target: "generic", Prec: 128, Size: 256,
		Forward:   "forwardDIT256Complex128",
		Inverse:   "inverseDIT256Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit256_radix2_generic", Priority: 10,
	},
	{
		Target: "generic", Prec: 128, Size: 256,
		Forward:   "forwardDIT256Radix4Complex128",
		Inverse:   "inverseDIT256Radix4Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit256_radix4_generic", Priority: 20,
	},
	{
		// Measured 1.003 forward / 0.888 inverse against dit256_radix4_generic in
		// the 2026-07-30 canary-gated purego re-sweep; see docs/CODELET_BENCHMARKS.md.
		Target: "generic", Prec: 128, Size: 256,
		Forward:   "forwardRadix8Complex128",
		Inverse:   "inverseRadix8Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit256_radix8ladder_generic", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex128",
	},
	{
		Target: "generic", Prec: 128, Size: 384,
		Forward:   "forwardDIT384MixedComplex128",
		Inverse:   "inverseDIT384MixedComplex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit384_mixed_generic", Priority: 20,
	},
	{
		Target: "generic", Prec: 128, Size: 512,
		Forward:   "forwardDIT512Complex128",
		Inverse:   "inverseDIT512Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit512_radix2_generic", Priority: 10,
	},
	{
		Target: "generic", Prec: 128, Size: 512,
		Forward:   "forwardDIT512Radix8Complex128",
		Inverse:   "inverseDIT512Radix8Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit512_radix8_generic", Priority: 35,
	},
	{
		Target: "generic", Prec: 128, Size: 512,
		Forward:   "forwardDIT512Radix4Then2Complex128",
		Inverse:   "inverseDIT512Radix4Then2Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit512_radix4_then2_generic", Priority: 45,
	},
	{
		Target: "generic", Prec: 128, Size: 512,
		Forward:   "forwardDIT512Mixed16x32Complex128",
		Inverse:   "inverseDIT512Mixed16x32Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit512_radix16x32_generic", Priority: 1,
	},
	{
		// Measured 1.002 forward / 0.886 inverse against dit512_radix4_then2_generic
		// in the 2026-07-30 canary-gated purego re-sweep. An earlier partial re-sweep
		// put radix-4-then-2 ahead here (0.921 forward); that run was taken on a
		// contended machine and only this gated one is trustworthy.
		Target: "generic", Prec: 128, Size: 512,
		Forward:   "forwardRadix8Complex128",
		Inverse:   "inverseRadix8Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit512_radix8ladder_generic", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex128",
	},
	{
		Target: "generic", Prec: 128, Size: 1024,
		Forward:   "forwardDIT1024Radix4Complex128",
		Inverse:   "inverseDIT1024Radix4Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit1024_radix4_generic", Priority: 30,
	},
	{
		Target: "generic", Prec: 128, Size: 1024,
		Forward:   "forwardDIT1024Mixed32x32Complex128",
		Inverse:   "inverseDIT1024Mixed32x32Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit1024_radix32x32_generic", Priority: 1,
	},
	{
		Target: "generic", Prec: 128, Size: 2048,
		Forward:   "forwardDIT2048Radix4Then2Complex128",
		Inverse:   "inverseDIT2048Radix4Then2Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit2048_radix4_then2_generic", Priority: 20,
	},
	{
		// Measured 0.928 forward / 0.829 inverse against dit2048_radix4_then2_generic
		// in the 2026-07-30 canary-gated purego sweep; see docs/CODELET_BENCHMARKS.md.
		Target: "generic", Prec: 128, Size: 2048,
		Forward:   "forwardRadix8Complex128",
		Inverse:   "inverseRadix8Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit2048_radix8ladder_generic", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex128",
	},
	{
		Target: "generic", Prec: 128, Size: 4096,
		Forward:   "forwardDIT4096Radix4Complex128",
		Inverse:   "inverseDIT4096Radix4Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit4096_radix4_generic", Priority: 30,
	},
	{
		// Measured 0.792 forward / 0.738 inverse against dit4096_radix4_generic
		// in the 2026-07-30 canary-gated purego sweep; see docs/CODELET_BENCHMARKS.md.
		Target: "generic", Prec: 128, Size: 4096,
		Forward:   "forwardRadix8Complex128",
		Inverse:   "inverseRadix8Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit4096_radix8ladder_generic", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex128",
	},
	{
		Target: "generic", Prec: 128, Size: 8192,
		Forward:   "forwardDIT8192Radix4Then2Complex128",
		Inverse:   "inverseDIT8192Radix4Then2Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit8192_radix4_then2_generic", Priority: 35,
	},
	{
		// Measured 0.934 forward / 0.835 inverse against dit8192_radix4_then2_generic
		// in the 2026-07-30 canary-gated purego sweep; see docs/CODELET_BENCHMARKS.md.
		Target: "generic", Prec: 128, Size: 8192,
		Forward:   "forwardRadix8Complex128",
		Inverse:   "inverseRadix8Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit8192_radix8ladder_generic", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex128",
	},
	{
		Target: "generic", Prec: 128, Size: 16384,
		Forward:   "forwardDIT16384Radix4Complex128",
		Inverse:   "inverseDIT16384Radix4Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit16384_radix4_generic", Priority: 30,
	},
	{
		// Measured 0.834 forward / 0.807 inverse against dit16384_radix4_generic
		// in the 2026-07-30 canary-gated purego sweep; see docs/CODELET_BENCHMARKS.md.
		Target: "generic", Prec: 128, Size: 16384,
		Forward:   "forwardRadix8Complex128",
		Inverse:   "inverseRadix8Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit16384_radix8ladder_generic", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex128",
	},
	{
		Target: "generic", Prec: 128, Size: 32768,
		Forward:   "forwardDIT32768Radix4Then2Complex128",
		Inverse:   "inverseDIT32768Radix4Then2Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit32768_radix4_then2_generic", Priority: 20,
	},

	{
		Target: "generic", Prec: 128, Size: 65536,
		Forward:   "forwardDIT65536Radix4Complex128",
		Inverse:   "inverseDIT65536Radix4Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit65536_radix4_generic", Priority: 20,
	},

	// Split-radix, complex128 — see the complex64 block above for why these
	// rows exist, why the priority is 1 and why the ladder stops at 32768.
	{
		Target: "generic", Prec: 128, Size: 256,
		Forward:   "ForwardSplitRadixComplex128",
		Inverse:   "InverseSplitRadixComplex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit256_splitradix_generic", Priority: 1,
	},
	{
		Target: "generic", Prec: 128, Size: 512,
		Forward:   "ForwardSplitRadixComplex128",
		Inverse:   "InverseSplitRadixComplex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit512_splitradix_generic", Priority: 1,
	},
	{
		Target: "generic", Prec: 128, Size: 1024,
		Forward:   "ForwardSplitRadixComplex128",
		Inverse:   "InverseSplitRadixComplex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit1024_splitradix_generic", Priority: 1,
	},
	{
		Target: "generic", Prec: 128, Size: 2048,
		Forward:   "ForwardSplitRadixComplex128",
		Inverse:   "InverseSplitRadixComplex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit2048_splitradix_generic", Priority: 1,
	},
	{
		Target: "generic", Prec: 128, Size: 4096,
		Forward:   "ForwardSplitRadixComplex128",
		Inverse:   "InverseSplitRadixComplex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit4096_splitradix_generic", Priority: 1,
	},
	{
		Target: "generic", Prec: 128, Size: 8192,
		Forward:   "ForwardSplitRadixComplex128",
		Inverse:   "InverseSplitRadixComplex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit8192_splitradix_generic", Priority: 1,
	},
	{
		Target: "generic", Prec: 128, Size: 16384,
		Forward:   "ForwardSplitRadixComplex128",
		Inverse:   "InverseSplitRadixComplex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit16384_splitradix_generic", Priority: 1,
	},
	{
		Target: "generic", Prec: 128, Size: 32768,
		Forward:   "ForwardSplitRadixComplex128",
		Inverse:   "InverseSplitRadixComplex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDNone", KernelType: "KernelTypeDIT",
		Signature: "dit32768_splitradix_generic", Priority: 1,
	},

	{
		Target: "avx2", Prec: 64, Size: 4,
		Forward:   "amd64.ForwardAVX2Size4Radix4Complex64Asm",
		Inverse:   "amd64.InverseAVX2Size4Radix4Complex64Asm",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit4_radix4_avx2", Priority: 5,
	},
	{
		Target: "avx2", Prec: 64, Size: 8,
		Forward:   "amd64.ForwardAVX2Size8Radix2Complex64Asm",
		Inverse:   "amd64.InverseAVX2Size8Radix2Complex64Asm",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit8_radix2_avx2", Priority: 12,
	},
	{
		Target: "avx2", Prec: 64, Size: 8,
		Forward:   "amd64.ForwardAVX2Size8Radix4Complex64Asm",
		Inverse:   "amd64.InverseAVX2Size8Radix4Complex64Asm",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit8_radix4_avx2", Priority: 10,
	},
	{
		Target: "avx2", Prec: 64, Size: 8,
		Forward:   "amd64.ForwardAVX2Size8Radix8Complex64Asm",
		Inverse:   "amd64.InverseAVX2Size8Radix8Complex64Asm",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeCore",
		Signature: "dit8_radix8_avx2", Priority: 11,
	},
	{
		Target: "avx2", Prec: 64, Size: 16,
		Forward:   "amd64.ForwardAVX2Size16Radix2Complex64Asm",
		Inverse:   "amd64.InverseAVX2Size16Radix2Complex64Asm",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit16_radix2_avx2", Priority: 55,
	},
	{
		Target: "avx2", Prec: 64, Size: 16,
		Forward:   "amd64.ForwardAVX2Size16Radix16Complex64Asm",
		Inverse:   "amd64.InverseAVX2Size16Radix16Complex64Asm",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit16_radix16_avx2", Priority: 50,
	},
	{
		Target: "avx2", Prec: 64, Size: 32,
		Forward:   "amd64.ForwardAVX2Size32Radix32Complex64Asm",
		Inverse:   "amd64.InverseAVX2Size32Radix32Complex64Asm",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeCore",
		Signature: "dit32_radix32_avx2", Priority: 25,
	},
	{
		Target: "avx2", Prec: 64, Size: 32,
		Forward:   "amd64.ForwardAVX2Size32Radix2Complex64Asm",
		Inverse:   "amd64.InverseAVX2Size32Radix2Complex64Asm",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit32_radix2_avx2", Priority: 30,
	},
	{
		Target: "avx2", Prec: 64, Size: 64,
		Forward:   "amd64.ForwardAVX2Size64Radix2Complex64Asm",
		Inverse:   "amd64.InverseAVX2Size64Radix2Complex64Asm",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		// 54.6/56.5 ns vs 124.6/133.0 for dit64_radix4_avx2 (i7-1255U, AVX2):
		// this is the only size-64 codelet that is genuinely 256-bit wide.
		Signature: "dit64_radix2_avx2", Priority: 26,
	},
	{
		Target: "avx2", Prec: 64, Size: 384,
		Forward:   "forwardDIT384MixedComplex64",
		Inverse:   "inverseDIT384MixedComplex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit384_mixed_avx2", Priority: 25,
	},
	// The AVX2 complex64 radix-4 DIT (internal/asm/amd64/avx2_f32_radix4.s):
	// one size-generic kernel for every n = 4^k and n = 2*4^k below, replacing
	// the per-size dit*_radix4*_avx2 files it superseded. Those were VEX-encoded
	// but XMM-width -- one complex64 per operation -- where this one runs four
	// butterflies per instruction in Y registers, measured 2.9-4.0x faster at
	// every size here (see TestRadix4AVX2Ranking in internal/kernels).
	//
	// Not registered for n = 32 or 64: stage 1 is fixed overhead that does not
	// amortise there, and dit32_radix2_avx2 (22 vs 34 ns) and dit64_radix2_avx2
	// (50 vs 53 ns) stay ahead.
	{
		Target: "avx2", Prec: 64, Size: 256,
		Forward:   "forwardRadix4AVX2Complex64",
		Inverse:   "inverseRadix4AVX2Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit256_radix4_avx2", Priority: 90,
		TwiddleSize: "twiddleSizeRadix4AVX2", PrepareTwiddle: "prepareTwiddleRadix4AVX2",
	},
	{
		// The size-generic AVX2 radix-8 ladder (internal/asm/amd64/avx2_f32_radix8.s).
		//
		// It wins exactly the complex64 cells whose last radix-8 stage strides
		// 512 bytes or less between its eight streams, and loses every cell
		// that strides 4 KiB or more -- 256/512/1024/2048 against
		// 4096/8192/16384/32768, with no exceptions either way. Eight streams a
		// multiple of 4 KiB apart all land on one L1 set; that is the same
		// collision forwardRadix4AVX2FusedComplex64 documents, and radix-8 is
		// twice as exposed to it because it doubles the live streams.
		//
		// Measured 0.953 forward / 0.940 inverse here, in the 2026-07-30
		// canary-gated sweep (118/128 groups accepted; 6 for this cell).
		Target: "avx2", Prec: 64, Size: 256,
		Forward:   "forwardRadix8AVX2Complex64",
		Inverse:   "inverseRadix8AVX2Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit256_radix8ladder_avx2", Priority: 95,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex64",
	},
	{
		Target: "avx2", Prec: 64, Size: 1024,
		Forward:   "forwardRadix4AVX2Complex64",
		Inverse:   "inverseRadix4AVX2Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit1024_radix4_avx2", Priority: 90,
		TwiddleSize: "twiddleSizeRadix4AVX2", PrepareTwiddle: "prepareTwiddleRadix4AVX2",
	},
	{
		// Measured 0.983 forward / 0.940 inverse. The forward column is inside
		// noise on its own; the inverse win is what carries the cell, and both
		// sides of the comparison fold 1/n into stage 1, so the two directions
		// differ only in the butterfly.
		Target: "avx2", Prec: 64, Size: 1024,
		Forward:   "forwardRadix8AVX2Complex64",
		Inverse:   "inverseRadix8AVX2Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit1024_radix8ladder_avx2", Priority: 95,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex64",
	},
	{
		Target: "avx2", Prec: 64, Size: 4096,
		Forward:   "forwardRadix4AVX2Complex64",
		Inverse:   "inverseRadix4AVX2Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit4096_radix4_avx2", Priority: 90,
		TwiddleSize: "twiddleSizeRadix4AVX2", PrepareTwiddle: "prepareTwiddleRadix4AVX2",
	},
	{
		Target: "avx2", Prec: 64, Size: 16384,
		Forward:   "forwardRadix4AVX2Complex64",
		Inverse:   "inverseRadix4AVX2Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit16384_radix4_avx2", Priority: 90,
		TwiddleSize: "twiddleSizeRadix4AVX2", PrepareTwiddle: "prepareTwiddleRadix4AVX2",
	},
	{
		// Fused tail: measured 0.955/0.979 against the separate-tail variant
		// (see forwardRadix4AVX2FusedComplex64 for the whole table). Only
		// n = 128 and n = 2048 take it at this precision; the fused loop holds
		// eight live streams instead of four and loses that trade at the
		// larger strides.
		Target: "avx2", Prec: 64, Size: 128,
		Forward:   "forwardRadix4AVX2FusedComplex64",
		Inverse:   "inverseRadix4AVX2FusedComplex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit128_radix4fused_avx2", Priority: 90,
		TwiddleSize: "twiddleSizeRadix4AVX2", PrepareTwiddle: "prepareTwiddleRadix4AVX2",
	},
	{
		Target: "avx2", Prec: 64, Size: 512,
		Forward:   "forwardRadix4AVX2Complex64",
		Inverse:   "inverseRadix4AVX2Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit512_radix4_avx2", Priority: 90,
		TwiddleSize: "twiddleSizeRadix4AVX2", PrepareTwiddle: "prepareTwiddleRadix4AVX2",
	},
	{
		// Measured 0.903 forward / 0.934 inverse -- the ladder's best complex64
		// cell, and it also clears dit512_radix4fused_avx2 (0.952/0.987).
		Target: "avx2", Prec: 64, Size: 512,
		Forward:   "forwardRadix8AVX2Complex64",
		Inverse:   "inverseRadix8AVX2Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit512_radix8ladder_avx2", Priority: 95,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex64",
	},
	{
		// Fused tail: measured 0.943/0.974. The complex128 row at this size
		// deliberately does NOT fuse -- there the stride is exactly 4 KiB and
		// fusing costs 11%.
		Target: "avx2", Prec: 64, Size: 2048,
		Forward:   "forwardRadix4AVX2FusedComplex64",
		Inverse:   "inverseRadix4AVX2FusedComplex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit2048_radix4fused_avx2", Priority: 90,
		TwiddleSize: "twiddleSizeRadix4AVX2", PrepareTwiddle: "prepareTwiddleRadix4AVX2",
	},
	{
		// Measured 0.953 forward / 0.946 inverse against dit2048_radix4fused_avx2,
		// which is itself the tuned incumbent here rather than the plain
		// radix-4 row.
		Target: "avx2", Prec: 64, Size: 2048,
		Forward:   "forwardRadix8AVX2Complex64",
		Inverse:   "inverseRadix8AVX2Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit2048_radix8ladder_avx2", Priority: 95,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex64",
	},
	{
		Target: "avx2", Prec: 64, Size: 8192,
		Forward:   "forwardRadix4AVX2Complex64",
		Inverse:   "inverseRadix4AVX2Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit8192_radix4_avx2", Priority: 90,
		TwiddleSize: "twiddleSizeRadix4AVX2", PrepareTwiddle: "prepareTwiddleRadix4AVX2",
	},
	{
		Target: "avx2", Prec: 64, Size: 32768,
		Forward:   "forwardRadix4AVX2Complex64",
		Inverse:   "inverseRadix4AVX2Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit32768_radix4_avx2", Priority: 90,
		TwiddleSize: "twiddleSizeRadix4AVX2", PrepareTwiddle: "prepareTwiddleRadix4AVX2",
	},
	// n = 65536 had no codelet at all before this; the planner fell back to
	// Stockham.
	{
		Target: "avx2", Prec: 64, Size: 65536,
		Forward:   "forwardRadix4AVX2Complex64",
		Inverse:   "inverseRadix4AVX2Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit65536_radix4_avx2", Priority: 90,
		TwiddleSize: "twiddleSizeRadix4AVX2", PrepareTwiddle: "prepareTwiddleRadix4AVX2",
	},
	// AVX-512 codelets: the generic AVX-512 radix-2 DIT kernel. Registered at
	// the sizes where it once beat the best AVX2 codelet; measured on a Xeon
	// Gold 5218 it now loses to plain AVX2 by 2.9-4.1x at all four, so each is
	// demoted with RankLevel (ordering only -- SIMDLevel still gates execution).
	// Do not delete without re-measuring: only Skylake-SP has been seen. Evidence
	// in docs/CODELET_BENCHMARKS.md; kernel notes in dit_avx512_amd64.go.
	{
		Target: "avx512", Prec: 64, Size: 1024,
		Forward:   "forwardAVX512Radix2Complex64",
		Inverse:   "inverseAVX512Radix2Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit1024_radix2_avx512", Priority: 10, RankLevel: "SIMDAVX2",
	},
	{
		Target: "avx512", Prec: 64, Size: 4096,
		Forward:   "forwardAVX512Radix2Complex64",
		Inverse:   "inverseAVX512Radix2Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit4096_radix2_avx512", Priority: 10, RankLevel: "SIMDAVX2",
	},
	{
		Target: "avx512", Prec: 64, Size: 8192,
		Forward:   "forwardAVX512Radix2Complex64",
		Inverse:   "inverseAVX512Radix2Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit8192_radix2_avx512", Priority: 10, RankLevel: "SIMDAVX2",
	},
	{
		Target: "avx512", Prec: 64, Size: 16384,
		Forward:   "forwardAVX512Radix2Complex64",
		Inverse:   "inverseAVX512Radix2Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit16384_radix2_avx512", Priority: 10, RankLevel: "SIMDAVX2",
	},
	// Size-specific AVX-512 complex64 codelets. A ZMM holds 8 complex64, so at
	// these lengths the whole transform is register-resident: the kernels load
	// once, run every stage in registers and store once, never touching
	// scratch. The input permutation is absorbed into which load lands in which
	// register, so none of them needs a bit-reversal table. All are AVX512F
	// only. Measured on a Xeon Gold 5218 against the best pre-existing codelet
	// at each size; see docs/CODELET_BENCHMARKS.md.
	{
		Target: "avx512", Prec: 64, Size: 8,
		Forward:   "amd64.ForwardAVX512Size8Radix8Complex64Asm",
		Inverse:   "amd64.InverseAVX512Size8Radix8Complex64Asm",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeCore",
		Signature: "dit8_radix8_avx512", Priority: 9,
	},
	{
		Target: "avx512", Prec: 64, Size: 16,
		Forward:   "amd64.ForwardAVX512Size16Radix16Complex64Asm",
		Inverse:   "amd64.InverseAVX512Size16Radix16Complex64Asm",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit16_radix16_avx512", Priority: 50,
	},
	{
		Target: "avx512", Prec: 64, Size: 32,
		Forward:   "amd64.ForwardAVX512Size32Radix4Then2Complex64Asm",
		Inverse:   "amd64.InverseAVX512Size32Radix4Then2Complex64Asm",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit32_radix4_then2_avx512", Priority: 22,
	},
	// The two size-64 kernels are 8x8 four-step transforms differing only in how
	// the vertical 8-point sub-FFT is decomposed; they emit the same 148
	// instructions and measure the same within 2%, so radix2 holds the higher
	// priority arbitrarily. Note the suffixes name that sub-FFT decomposition,
	// not three literal radix-4 stages (not expressible over 8 registers x 8
	// lanes); the file headers state the real algorithm.
	{
		Target: "avx512", Prec: 64, Size: 64,
		Forward:   "amd64.ForwardAVX512Size64Radix2Complex64Asm",
		Inverse:   "amd64.InverseAVX512Size64Radix2Complex64Asm",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit64_radix2_avx512", Priority: 30,
	},
	{
		Target: "avx512", Prec: 64, Size: 64,
		Forward:   "amd64.ForwardAVX512Size64Radix4Complex64Asm",
		Inverse:   "amd64.InverseAVX512Size64Radix4Complex64Asm",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit64_radix4_avx512", Priority: 25,
	},
	// Register-resident radix-2 DIT with a fused in-register radix-8 leaf. Size
	// 128 keeps all 16 ZMM of data live from load to store; size 256 runs two
	// such sub-transforms plus a final radix-2 stage.
	{
		Target: "avx512", Prec: 64, Size: 128,
		Forward:   "amd64.ForwardAVX512Size128Radix8Then2Complex64Asm",
		Inverse:   "amd64.InverseAVX512Size128Radix8Then2Complex64Asm",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit128_radix8_then2_avx512", Priority: 30,
	},
	{
		Target: "avx512", Prec: 64, Size: 256,
		Forward:   "amd64.ForwardAVX512Size256Radix8Then2Complex64Asm",
		Inverse:   "amd64.InverseAVX512Size256Radix8Then2Complex64Asm",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit256_radix8_then2_avx512", Priority: 30,
	},
	// Size-specific AVX-512 complex128 codelets. A ZMM holds 4 complex128, so
	// these transforms are likewise fully register-resident and never spill
	// between stages. Before this set, complex128 had no AVX-512 codelets at
	// all; see internal/asm/amd64/avx512_f64_size*.s.
	// Size 4 is DISABLED (negative priority) on purpose. Measured on a Xeon Gold
	// 5218, median of 7 x 300ms on an idle host: forward 11.54 ns vs 8.27 ns for
	// the pure-Go codelet (10.86 ns for SSE2), inverse 12.07 ns vs 11.18 ns.
	// The kernel is only 11 vector ops with no multiply, so this is not an
	// instruction-count problem: a 4-point butterfly network over the four lanes
	// of one ZMM needs two levels of lane-crossing VSHUFF64X2 (3 cycles each),
	// while the SSE2 kernel keeps each complex128 in its own XMM and needs no
	// shuffle at all for stage 1. Packing n = 4 into one register trades free
	// register-level parallelism for ~7 cycles of serial shuffle latency, and at
	// 64 bytes there is no data-movement win to pay for it. Codelet selection
	// prefers the higher SIMD level over priority, so registering this would
	// make AVX-512 hosts slower; the row is kept so the kernel is not lost.
	// NOTE: negative-priority rows are skipped by the behavioural sweeps, so this
	// kernel was verified at a positive priority before being disabled.
	{
		Target: "avx512", Prec: 128, Size: 4,
		Forward:   "amd64.ForwardAVX512Size4Radix4Complex128Asm",
		Inverse:   "amd64.InverseAVX512Size4Radix4Complex128Asm",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeCore",
		Signature: "dit4_radix4_avx512", Priority: -1,
	},
	{
		Target: "avx512", Prec: 128, Size: 8,
		Forward:   "amd64.ForwardAVX512Size8Radix8Complex128Asm",
		Inverse:   "amd64.InverseAVX512Size8Radix8Complex128Asm",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeCore",
		Signature: "dit8_radix8_avx512", Priority: 10,
	},
	{
		Target: "avx512", Prec: 128, Size: 16,
		Forward:   "amd64.ForwardAVX512Size16Radix4Complex128Asm",
		Inverse:   "amd64.InverseAVX512Size16Radix4Complex128Asm",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit16_radix4_avx512", Priority: 30,
	},
	{
		Target: "avx512", Prec: 128, Size: 32,
		Forward:   "amd64.ForwardAVX512Size32Radix4Then2Complex128Asm",
		Inverse:   "amd64.InverseAVX512Size32Radix4Then2Complex128Asm",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit32_radix4_then2_avx512", Priority: 22,
	},
	{
		Target: "avx512", Prec: 128, Size: 64,
		Forward:   "amd64.ForwardAVX512Size64Radix4Complex128Asm",
		Inverse:   "amd64.InverseAVX512Size64Radix4Complex128Asm",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit64_radix4_avx512", Priority: 25,
	},
	{
		Target: "avx512", Prec: 128, Size: 128,
		Forward:   "amd64.ForwardAVX512Size128Radix4Then2Complex128Asm",
		Inverse:   "amd64.InverseAVX512Size128Radix4Then2Complex128Asm",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit128_radix4_then2_avx512", Priority: 25,
	},
	{
		Target: "avx2", Prec: 128, Size: 384,
		Forward:   "forwardDIT384MixedComplex128",
		Inverse:   "inverseDIT384MixedComplex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit384_mixed_avx2", Priority: 25,
	},
	{
		Target: "avx2", Prec: 128, Size: 4,
		Forward:   "forwardDIT4Radix4Complex128",
		Inverse:   "inverseDIT4Radix4Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit4_radix4_avx2", Priority: 5,
	},
	{
		Target: "avx2", Prec: 128, Size: 8,
		Forward:   "amd64.ForwardAVX2Size8Radix8Complex128Asm",
		Inverse:   "amd64.InverseAVX2Size8Radix8Complex128Asm",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeCore",
		// Promoted over dit8_radix4_avx2 by the 2026-07-30 incumbent audit
		// (docs/CODELET_BENCHMARKS.md): 0.970 forward / 0.859 inverse against it, medianed
		// over 16 canary-gated groups. The inverse gap is the substantive one
		// -- 8.2 ns to 7.0 ns; the forward gap is 0.2 ns and on its own would
		// not justify a change. dit8_radix2_avx2 is a near-tie on inverse
		// (0.866) but loses forward, so radix-8 wins both directions.
		Signature: "dit8_radix8_avx2", Priority: 35,
	},
	{
		Target: "avx2", Prec: 128, Size: 8,
		Forward:   "amd64.ForwardAVX2Size8Radix2Complex128Asm",
		Inverse:   "amd64.InverseAVX2Size8Radix2Complex128Asm",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit8_radix2_avx2", Priority: 25,
	},
	{
		Target: "avx2", Prec: 128, Size: 8,
		Forward:   "amd64.ForwardAVX2Size8Radix4Complex128Asm",
		Inverse:   "amd64.InverseAVX2Size8Radix4Complex128Asm",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit8_radix4_avx2", Priority: 30,
	},
	// The AVX2 complex128 radix-4 DIT (internal/asm/amd64/avx2_f64_radix4.s):
	// the complex128 twin of the size-generic kernel above, replacing the
	// per-size dit*_radix4*_avx2 files it superseded. Those were VEX-encoded
	// but XMM-width -- one complex128 per operation -- where this one runs two
	// butterflies per instruction in Y registers, measured 1.4-2.5x faster at
	// every size here (see TestRadix4AVX2Complex128Ranking in
	// internal/kernels).
	//
	// Unlike the complex64 side, this one is registered down to n = 16: the
	// codelets it replaces are relatively weaker at this precision, so stage
	// 1's fixed overhead still amortises at the small sizes.
	{
		Target: "avx2", Prec: 128, Size: 16,
		Forward:   "forwardRadix4AVX2Complex128",
		Inverse:   "inverseRadix4AVX2Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit16_radix4_avx2", Priority: 90,
		TwiddleSize: "twiddleSizeRadix4AVX2Complex128", PrepareTwiddle: "prepareTwiddleRadix4AVX2Complex128",
	},
	{
		Target: "avx2", Prec: 128, Size: 32,
		Forward:   "forwardRadix4AVX2Complex128",
		Inverse:   "inverseRadix4AVX2Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit32_radix4_avx2", Priority: 90,
		TwiddleSize: "twiddleSizeRadix4AVX2Complex128", PrepareTwiddle: "prepareTwiddleRadix4AVX2Complex128",
	},
	{
		Target: "avx2", Prec: 128, Size: 64,
		Forward:   "forwardRadix4AVX2Complex128",
		Inverse:   "inverseRadix4AVX2Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit64_radix4_avx2", Priority: 90,
		TwiddleSize: "twiddleSizeRadix4AVX2Complex128", PrepareTwiddle: "prepareTwiddleRadix4AVX2Complex128",
	},
	{
		// Fused tail: measured 0.935/0.934, the largest fusion win in either
		// precision. n = 128 is the only complex128 size that takes it.
		Target: "avx2", Prec: 128, Size: 128,
		Forward:   "forwardRadix4AVX2FusedComplex128",
		Inverse:   "inverseRadix4AVX2FusedComplex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit128_radix4fused_avx2", Priority: 90,
		TwiddleSize: "twiddleSizeRadix4AVX2Complex128", PrepareTwiddle: "prepareTwiddleRadix4AVX2Complex128",
	},
	{
		Target: "avx2", Prec: 128, Size: 256,
		Forward:   "forwardRadix4AVX2Complex128",
		Inverse:   "inverseRadix4AVX2Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit256_radix4_avx2", Priority: 90,
		TwiddleSize: "twiddleSizeRadix4AVX2Complex128", PrepareTwiddle: "prepareTwiddleRadix4AVX2Complex128",
	},
	{
		Target: "avx2", Prec: 128, Size: 512,
		Forward:   "forwardRadix4AVX2Complex128",
		Inverse:   "inverseRadix4AVX2Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit512_radix4_avx2", Priority: 90,
		TwiddleSize: "twiddleSizeRadix4AVX2Complex128", PrepareTwiddle: "prepareTwiddleRadix4AVX2Complex128",
	},
	{
		// Measured 0.933 forward / 0.979 inverse against dit512_radix4_avx2.
		// It also beats dit512_radix4fused_avx2 (0.940/0.961), so the fused
		// variant is not the row that should have been here either.
		Target: "avx2", Prec: 128, Size: 512,
		Forward:   "forwardRadix8AVX2Complex128",
		Inverse:   "inverseRadix8AVX2Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit512_radix8ladder_avx2", Priority: 95,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex128",
	},
	{
		Target: "avx2", Prec: 128, Size: 1024,
		Forward:   "forwardRadix4AVX2Complex128",
		Inverse:   "inverseRadix4AVX2Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit1024_radix4_avx2", Priority: 90,
		TwiddleSize: "twiddleSizeRadix4AVX2Complex128", PrepareTwiddle: "prepareTwiddleRadix4AVX2Complex128",
	},
	{
		Target: "avx2", Prec: 128, Size: 2048,
		Forward:   "forwardRadix4AVX2Complex128",
		Inverse:   "inverseRadix4AVX2Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit2048_radix4_avx2", Priority: 90,
		TwiddleSize: "twiddleSizeRadix4AVX2Complex128", PrepareTwiddle: "prepareTwiddleRadix4AVX2Complex128",
	},
	{
		// Measured 0.942 forward / 0.922 inverse against dit2048_radix4_avx2 in
		// the 2026-07-30 canary-gated AVX2 radix-8 sweep (118/128 groups
		// accepted, 8 groups this cell).
		Target: "avx2", Prec: 128, Size: 2048,
		Forward:   "forwardRadix8AVX2Complex128",
		Inverse:   "inverseRadix8AVX2Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit2048_radix8ladder_avx2", Priority: 95,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex128",
	},
	{
		Target: "avx2", Prec: 128, Size: 4096,
		Forward:   "forwardRadix4AVX2Complex128",
		Inverse:   "inverseRadix4AVX2Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit4096_radix4_avx2", Priority: 90,
		TwiddleSize: "twiddleSizeRadix4AVX2Complex128", PrepareTwiddle: "prepareTwiddleRadix4AVX2Complex128",
	},
	{
		Target: "avx2", Prec: 128, Size: 8192,
		Forward:   "forwardRadix4AVX2Complex128",
		Inverse:   "inverseRadix4AVX2Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit8192_radix4_avx2", Priority: 90,
		TwiddleSize: "twiddleSizeRadix4AVX2Complex128", PrepareTwiddle: "prepareTwiddleRadix4AVX2Complex128",
	},
	{
		Target: "avx2", Prec: 128, Size: 16384,
		Forward:   "forwardRadix4AVX2Complex128",
		Inverse:   "inverseRadix4AVX2Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit16384_radix4_avx2", Priority: 90,
		TwiddleSize: "twiddleSizeRadix4AVX2Complex128", PrepareTwiddle: "prepareTwiddleRadix4AVX2Complex128",
	},
	{
		Target: "avx2", Prec: 128, Size: 32768,
		Forward:   "forwardRadix4AVX2Complex128",
		Inverse:   "inverseRadix4AVX2Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit32768_radix4_avx2", Priority: 90,
		TwiddleSize: "twiddleSizeRadix4AVX2Complex128", PrepareTwiddle: "prepareTwiddleRadix4AVX2Complex128",
	},
	{
		// Measured 0.931 forward / 0.985 inverse against dit32768_radix4_avx2,
		// and ahead of dit32768_radix4fused_avx2 (0.963/0.994) as well.
		//
		// This is the one complex128 cell that wins at a large last-stage
		// stride (m = 4096, so 64 KiB between the eight streams). At 512 KiB the
		// working set is far past L2 and the ladder's third fewer passes over
		// the buffer decides the cell on memory traffic alone -- which is the
		// opposite regime from the 4-8 KiB strides that lose.
		Target: "avx2", Prec: 128, Size: 32768,
		Forward:   "forwardRadix8AVX2Complex128",
		Inverse:   "inverseRadix8AVX2Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit32768_radix8ladder_avx2", Priority: 95,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex128",
	},
	{
		Target: "avx2", Prec: 128, Size: 65536,
		Forward:   "forwardRadix4AVX2Complex128",
		Inverse:   "inverseRadix4AVX2Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX2", KernelType: "KernelTypeDIT",
		Signature: "dit65536_radix4_avx2", Priority: 90,
		TwiddleSize: "twiddleSizeRadix4AVX2Complex128", PrepareTwiddle: "prepareTwiddleRadix4AVX2Complex128",
	},
}
