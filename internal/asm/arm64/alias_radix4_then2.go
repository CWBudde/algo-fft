//go:build arm64 && !purego

package arm64

// Aliases for radix-4-then-2 kernels (backed by mixed-radix asm symbols).
//
// The complex64 32/128 aliases used to live here too, but they now call the
// shared NEON Stockham radix-4 core directly (see neon_radix4_loop.go) rather
// than the retired size-specific mixed-radix asm, so they moved there with
// their same-size Go-only siblings (512, 2048, 8192, 32768).

// The complex128 Size32/Size128 Radix4Then2 wrappers used to alias
// MixedRadix24 asm symbols in neon_f64_size{32,128}_mixed24.s (scalar, no
// vector instructions). They are now defined directly in
// neon_radix4_loop_f64.go over the shared vectorized radix-4 core, so no
// alias is needed here for complex128 any more.
