//go:build arm64 && !purego && fftprobe

package arm64

// NOTE: These are Go declarations for the retired scalar-NEON *.s routines
// that live behind the fftprobe build tag. Each contains zero vector
// instructions and measured 2.7x-5.6x slower than pure Go on an Apple M5 —
// see AGENTS.md "Losing on one machine is not grounds for deletion" §2.2
// ("Measured loss >= 1.5x, or a research kernel — keep, unregistered") and
// docs/CODELET_BENCHMARKS.md. Kept compiled and correctness-tested under
// this tag so the question stays re-measurable instead of becoming
// folklore.

//go:noescape
func ForwardNEONSize8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseNEONSize8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardNEONSize8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseNEONSize8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardNEONSize8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseNEONSize8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardNEONSize16Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseNEONSize16Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardNEONSize16Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseNEONSize16Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardNEONSize32Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseNEONSize32Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardNEONSize32Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseNEONSize32Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardNEONSize64Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseNEONSize64Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardNEONSize128Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseNEONSize128Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardNEONSize128Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseNEONSize128Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardNEONSize256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseNEONSize256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardNEONSize256Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseNEONSize256Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool
