//go:build (amd64 || arm64 || 386) && !purego

package transform

// packedBuildHasSIMDKernels reports whether this build has SIMD kernels
// compiled in at all, which decides whether packedTierFor may consult CPU
// features or must answer tierGeneric outright.
//
// It cannot be replaced by a feature check: internal/cpu/detect_amd64.go is
// tagged `//go:build amd64` with no `!purego`, so a `-tags purego` amd64 build
// still reports HasAVX2: true. See the pure-Go counterpart in
// stockham_packed_policy_generic.go.
const packedBuildHasSIMDKernels = true
