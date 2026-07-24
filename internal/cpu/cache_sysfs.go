package cpu

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// sysfsCacheRoot is the per-CPU cache topology exposed by the Linux kernel.
// cpu0 is used as representative for all cores.
const sysfsCacheRoot = "/sys/devices/system/cpu/cpu0/cache"

// readCachesFromSysfsDir reads L1 data and L2 cache sizes from a sysfs-style
// cache directory (index*/{level,type,size}). Missing or malformed entries
// leave the corresponding field zero; the caller substitutes defaults.
//
// The function is separate from the Linux detection hook so it can be unit
// tested against fixture directories on any platform.
func readCachesFromSysfsDir(root string) CacheInfo {
	var caches CacheInfo

	entries, err := os.ReadDir(root)
	if err != nil {
		return caches
	}

	for _, entry := range entries {
		if !strings.HasPrefix(entry.Name(), "index") {
			continue
		}

		dir := filepath.Join(root, entry.Name())

		level, err := readSysfsString(filepath.Join(dir, "level"))
		if err != nil {
			continue
		}

		cacheType, err := readSysfsString(filepath.Join(dir, "type"))
		if err != nil {
			continue
		}

		sizeRaw, err := readSysfsString(filepath.Join(dir, "size"))
		if err != nil {
			continue
		}

		size := parseSysfsCacheSize(sizeRaw)
		if size <= 0 {
			continue
		}

		switch {
		case level == "1" && (cacheType == "Data" || cacheType == "Unified"):
			caches.L1DataBytes = size
		case level == "2" && (cacheType == "Data" || cacheType == "Unified"):
			caches.L2Bytes = size
		}
	}

	return caches
}

// readSysfsString reads a single-value sysfs file and trims whitespace.
func readSysfsString(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read sysfs cache file: %w", err)
	}

	return strings.TrimSpace(string(data)), nil
}

// parseSysfsCacheSize parses the sysfs cache size format: a decimal number
// with an optional K/M/G suffix (e.g. "32K", "1280K", "8M"). Returns 0 for
// malformed or non-positive input.
func parseSysfsCacheSize(s string) int {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0
	}

	multiplier := 1

	switch s[len(s)-1] {
	case 'K', 'k':
		multiplier = 1024
		s = s[:len(s)-1]
	case 'M', 'm':
		multiplier = 1024 * 1024
		s = s[:len(s)-1]
	case 'G', 'g':
		multiplier = 1024 * 1024 * 1024
		s = s[:len(s)-1]
	}

	value, err := strconv.Atoi(s)
	if err != nil || value <= 0 {
		return 0
	}

	return value * multiplier
}
