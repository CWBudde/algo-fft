package main

// AVX-512 rows for the size-generic radix-8 ladder. Kept in a separate file
// from specs.go only to respect the repository file-length limit; the table is
// still one logical unit and is concatenated in init().
//
// Evidence: Xeon Gold 5218 sweep of 2026-07-31, 8 passes x 20 groups,
// 160 accepted / 0 rejected, `benchmarks/gated-avx512r8`. Ratios below are the
// median within-group ratio against the incumbent that size then selected;
// under 1.00 means the ladder won. Full table and the register-budget
// diagnosis are in docs/CODELET_BENCHMARKS.md.
//
//	         c64 fwd / inv     c128 fwd / inv
//	   256   0.947 / 0.921     0.708 / 0.743
//	   512   0.777 / 0.774     0.814 / 0.766
//	  1024   0.740 / 0.748     0.697 / 0.753
//	  2048   0.703 / 0.751     0.884 / 0.882
//	  4096   0.790 / 0.807     0.702 / 0.695
//	  8192   0.900 / 0.890     0.700 / 0.696
//	 16384   0.808 / 0.813     0.708 / 0.711
//	 32768   0.883 / 0.887     0.869 / 0.865
//
// n = 64 and n = 128 are deliberately absent in both precisions and stay
// probe-only in internal/kernels/radix8_avx512_probe_amd64.go: at 64 the ladder
// loses outright (1.968/1.861 c64, 1.464/1.455 c128) to dit64_radix2_avx512 and
// dit64_radix4_avx512, and at 128 it is a c128 loss (1.256/1.283) and c64
// parity (1.039/0.997). Do not add rows for them without a fresh sweep, and do
// not leave a size registered both here and in the probe -- a duplicated
// signature puts the kernel in a sweep group twice and reports it against
// itself.
//
// Priority 50 matches the generic-tier ladder rows. It has to clear 30, which
// is dit256_radix8_then2_avx512, the only production AVX-512 row at any size
// covered here that is not RankLevel-demoted; the radix-2 rows at
// 1024/4096/8192/16384 rank at SIMDAVX2 and are outranked by level alone.
// RankLevel is deliberately empty: these rows are meant to be the incumbent.

//nolint:gochecknoglobals // the codelet table is the generator's declarative input
var codeletSpecsAVX512Radix8 = []codeletSpec{
	{
		Target: "avx512", Prec: 64, Size: 256,
		Forward:   "forwardRadix8AVX512Complex64",
		Inverse:   "inverseRadix8AVX512Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit256_radix8ladder_avx512", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex64",
	},
	{
		Target: "avx512", Prec: 64, Size: 512,
		Forward:   "forwardRadix8AVX512Complex64",
		Inverse:   "inverseRadix8AVX512Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit512_radix8ladder_avx512", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex64",
	},
	{
		Target: "avx512", Prec: 64, Size: 1024,
		Forward:   "forwardRadix8AVX512Complex64",
		Inverse:   "inverseRadix8AVX512Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit1024_radix8ladder_avx512", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex64",
	},
	{
		Target: "avx512", Prec: 64, Size: 2048,
		Forward:   "forwardRadix8AVX512Complex64",
		Inverse:   "inverseRadix8AVX512Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit2048_radix8ladder_avx512", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex64",
	},
	{
		Target: "avx512", Prec: 64, Size: 4096,
		Forward:   "forwardRadix8AVX512Complex64",
		Inverse:   "inverseRadix8AVX512Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit4096_radix8ladder_avx512", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex64",
	},
	{
		Target: "avx512", Prec: 64, Size: 8192,
		Forward:   "forwardRadix8AVX512Complex64",
		Inverse:   "inverseRadix8AVX512Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit8192_radix8ladder_avx512", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex64",
	},
	{
		Target: "avx512", Prec: 64, Size: 16384,
		Forward:   "forwardRadix8AVX512Complex64",
		Inverse:   "inverseRadix8AVX512Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit16384_radix8ladder_avx512", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex64",
	},
	{
		Target: "avx512", Prec: 64, Size: 32768,
		Forward:   "forwardRadix8AVX512Complex64",
		Inverse:   "inverseRadix8AVX512Complex64",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit32768_radix8ladder_avx512", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex64",
	},
	{
		Target: "avx512", Prec: 128, Size: 256,
		Forward:   "forwardRadix8AVX512Complex128",
		Inverse:   "inverseRadix8AVX512Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit256_radix8ladder_avx512", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex128",
	},
	{
		Target: "avx512", Prec: 128, Size: 512,
		Forward:   "forwardRadix8AVX512Complex128",
		Inverse:   "inverseRadix8AVX512Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit512_radix8ladder_avx512", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex128",
	},
	{
		Target: "avx512", Prec: 128, Size: 1024,
		Forward:   "forwardRadix8AVX512Complex128",
		Inverse:   "inverseRadix8AVX512Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit1024_radix8ladder_avx512", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex128",
	},
	{
		Target: "avx512", Prec: 128, Size: 2048,
		Forward:   "forwardRadix8AVX512Complex128",
		Inverse:   "inverseRadix8AVX512Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit2048_radix8ladder_avx512", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex128",
	},
	{
		Target: "avx512", Prec: 128, Size: 4096,
		Forward:   "forwardRadix8AVX512Complex128",
		Inverse:   "inverseRadix8AVX512Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit4096_radix8ladder_avx512", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex128",
	},
	{
		Target: "avx512", Prec: 128, Size: 8192,
		Forward:   "forwardRadix8AVX512Complex128",
		Inverse:   "inverseRadix8AVX512Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit8192_radix8ladder_avx512", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex128",
	},
	{
		Target: "avx512", Prec: 128, Size: 16384,
		Forward:   "forwardRadix8AVX512Complex128",
		Inverse:   "inverseRadix8AVX512Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit16384_radix8ladder_avx512", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex128",
	},
	{
		Target: "avx512", Prec: 128, Size: 32768,
		Forward:   "forwardRadix8AVX512Complex128",
		Inverse:   "inverseRadix8AVX512Complex128",
		Algorithm: "KernelDIT", SIMDLevel: "SIMDAVX512", KernelType: "KernelTypeDIT",
		Signature: "dit32768_radix8ladder_avx512", Priority: 50,
		TwiddleSize: "twiddleSizeRadix8", PrepareTwiddle: "prepareTwiddleRadix8Complex128",
	},
}

func init() {
	codeletSpecs = append(codeletSpecs, codeletSpecsAVX512Radix8...)
}
