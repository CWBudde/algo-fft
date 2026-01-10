//go:build (amd64 || arm64) && asm && !purego

package kernels

// wrapAsmDIT64 creates a 4-parameter CoreKernelFunc64 from a 5-parameter assembly function.
func wrapAsmDIT64(asmFunc func(dst, src, twiddle, scratch []complex64, bitrev []int) bool, bitrev []int) CoreKernelFunc64 {
	return func(dst, src, twiddle, scratch []complex64) bool {
		return asmFunc(dst, src, twiddle, scratch, bitrev)
	}
}

// wrapAsmDIT128 creates a 4-parameter CoreKernelFunc128 from a 5-parameter assembly function.
func wrapAsmDIT128(asmFunc func(dst, src, twiddle, scratch []complex128, bitrev []int) bool, bitrev []int) CoreKernelFunc128 {
	return func(dst, src, twiddle, scratch []complex128) bool {
		return asmFunc(dst, src, twiddle, scratch, bitrev)
	}
}
