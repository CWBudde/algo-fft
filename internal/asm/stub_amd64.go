//go:build amd64 && !purego

package asm

//go:noescape
func stubAsm()

func Stub() {
	stubAsm()
}
