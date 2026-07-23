package kernels

//go:generate go run ../../cmd/gencodelets .
//go:generate go run ../../cmd/gencodelets -inventory ../../docs/IMPLEMENTATION_INVENTORY.md
//go:generate go run ../../cmd/genkernels .

// This file registers all built-in codelets with the global registries in
// internal/registry. Registration happens at init time so codelets are
// available when plans are created. The register*DITCodelets* functions are
// generated from cmd/gencodelets/specs.go; run
// `go generate ./internal/kernels/...` after editing the spec table.

//nolint:gochecknoinits
func init() {
	// Register complex64 DIT codelets
	registerDITCodelets64()

	// Register complex128 DIT codelets
	registerDITCodelets128()

	// Register NEON codelets (conditional on build tags)
	registerNEONDITCodelets64()
	registerNEONDITCodelets128()

	// Register SSE2 codelets (conditional on build tags)
	registerSSE2DITCodelets64()
	registerSSE2DITCodelets128()

	// Register AVX2 codelets (conditional on build tags)
	registerAVX2DITCodelets64()
	registerAVX2DITCodelets128()

	// Register AVX-512 codelets (conditional on build tags)
	registerAVX512DITCodelets64()
	registerAVX512DITCodelets128()
}
