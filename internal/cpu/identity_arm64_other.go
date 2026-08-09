//go:build arm64 && !linux

package cpu

func detectARM64Identity() (vendor string, family, model uint32) {
	return "", 0, 0
}
