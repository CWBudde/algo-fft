//go:build (amd64 || arm64) && asm && !purego

package kernels

// wrapAsmDIT64 creates a 4-parameter kernel from a 5-parameter assembly function.
// The assembly function still takes a bitrev parameter, which we provide here.
func wrapAsmDIT64(asmFunc func(dst, src, twiddle, scratch []complex64, bitrev []int) bool, bitrev []int) func(dst, src, twiddle, scratch []complex64) bool {
	return func(dst, src, twiddle, scratch []complex64) bool {
		return asmFunc(dst, src, twiddle, scratch, bitrev)
	}
}

// wrapAsmDIT128 creates a 4-parameter kernel from a 5-parameter assembly function.
// The assembly function still takes a bitrev parameter, which we provide here.
func wrapAsmDIT128(asmFunc func(dst, src, twiddle, scratch []complex128, bitrev []int) bool, bitrev []int) func(dst, src, twiddle, scratch []complex128) bool {
	return func(dst, src, twiddle, scratch []complex128) bool {
		return asmFunc(dst, src, twiddle, scratch, bitrev)
	}
}
