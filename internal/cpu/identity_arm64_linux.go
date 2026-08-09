//go:build arm64 && linux

package cpu

import "os"

func detectARM64Identity() (vendor string, family, model uint32) {
	data, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return "", 0, 0
	}

	return parseARM64CPUInfo(string(data))
}
