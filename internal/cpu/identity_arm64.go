//go:build arm64

package cpu

import (
	"strconv"
	"strings"
)

// parseARM64CPUInfo maps Linux's MIDR-derived cpuinfo fields to the generic
// vendor/family/model tuple: implementer, architecture, and part respectively.
func parseARM64CPUInfo(data string) (vendor string, family, model uint32) {
	var implementer uint64

	for line := range strings.SplitSeq(data, "\n") {
		key, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}

		key = strings.TrimSpace(key)
		value = strings.TrimSpace(value)

		switch key {
		case "CPU implementer":
			implementer, _ = strconv.ParseUint(value, 0, 32)
		case "CPU architecture":
			parsed, err := strconv.ParseUint(value, 0, 32)
			if err == nil {
				family = uint32(parsed)
			}
		case "CPU part":
			parsed, err := strconv.ParseUint(value, 0, 32)
			if err == nil {
				model = uint32(parsed)
			}
		}

		if implementer != 0 && family != 0 && model != 0 {
			break
		}
	}

	if implementer == 0 {
		return "", family, model
	}

	return "arm" + strconv.FormatUint(implementer, 16), family, model
}
