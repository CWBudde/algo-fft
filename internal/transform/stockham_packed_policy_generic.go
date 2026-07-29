//go:build (!amd64 && !arm64 && !386) || purego

package transform

// packedBuildHasSIMDKernels is false here: this build has no SIMD kernels, so
// the packed radix-4 Stockham route is the fastest one available and
// packedTierFor answers tierGeneric without consulting CPU features (which on a
// `-tags purego` amd64 build would still report HasAVX2). See the SIMD
// counterpart in stockham_packed_policy_simd.go.
const packedBuildHasSIMDKernels = false
