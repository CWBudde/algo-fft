//go:build !arm64 || !fft_asm || purego

package kernels

func registerNEONDITCodelets64()  {}
func registerNEONDITCodelets128() {}
