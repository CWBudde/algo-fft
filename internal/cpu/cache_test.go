package cpu

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDetectCachesReturnsPositiveSizes(t *testing.T) {
	ResetCacheDetection()

	t.Cleanup(ResetCacheDetection)

	caches := DetectCaches()
	if caches.L1DataBytes <= 0 {
		t.Errorf("L1DataBytes = %d, want > 0", caches.L1DataBytes)
	}

	if caches.L2Bytes <= 0 {
		t.Errorf("L2Bytes = %d, want > 0", caches.L2Bytes)
	}

	if caches.L2Bytes < caches.L1DataBytes {
		t.Errorf("L2Bytes = %d < L1DataBytes = %d", caches.L2Bytes, caches.L1DataBytes)
	}
}

func TestDetectCachesIsStable(t *testing.T) {
	ResetCacheDetection()

	t.Cleanup(ResetCacheDetection)

	first := DetectCaches()

	second := DetectCaches()
	if first != second {
		t.Errorf("DetectCaches not stable: first %+v, second %+v", first, second)
	}
}

func TestSetForcedCachesOverridesDetection(t *testing.T) {
	forced := CacheInfo{L1DataBytes: 1 << 14, L2Bytes: 1 << 19}
	SetForcedCaches(forced)

	t.Cleanup(ResetCacheDetection)

	if got := DetectCaches(); got != forced {
		t.Errorf("DetectCaches() = %+v, want forced %+v", got, forced)
	}

	ResetCacheDetection()

	if got := DetectCaches(); got == forced {
		t.Errorf("DetectCaches() still returns forced value after reset")
	}
}

func TestParseSysfsCacheSize(t *testing.T) {
	tests := []struct {
		in   string
		want int
	}{
		{"32K", 32 * 1024},
		{"48K\n", 48 * 1024},
		{"1024K", 1024 * 1024},
		{"8M", 8 * 1024 * 1024},
		{"1G", 1024 * 1024 * 1024}, // largest G value that still fits a 32-bit int (386 build)
		{"65536", 65536},
		{"", 0},
		{"garbage", 0},
		{"-32K", 0},
	}

	for _, tt := range tests {
		if got := parseSysfsCacheSize(tt.in); got != tt.want {
			t.Errorf("parseSysfsCacheSize(%q) = %d, want %d", tt.in, got, tt.want)
		}
	}
}

func TestReadCachesFromSysfsDir(t *testing.T) {
	root := t.TempDir()

	writeCacheIndex(t, root, "index0", "1", "Data", "48K")
	writeCacheIndex(t, root, "index1", "1", "Instruction", "32K")
	writeCacheIndex(t, root, "index2", "2", "Unified", "1280K")
	writeCacheIndex(t, root, "index3", "3", "Unified", "12M")

	caches := readCachesFromSysfsDir(root)
	if caches.L1DataBytes != 48*1024 {
		t.Errorf("L1DataBytes = %d, want %d", caches.L1DataBytes, 48*1024)
	}

	if caches.L2Bytes != 1280*1024 {
		t.Errorf("L2Bytes = %d, want %d", caches.L2Bytes, 1280*1024)
	}
}

func TestReadCachesFromSysfsDirMissing(t *testing.T) {
	caches := readCachesFromSysfsDir(filepath.Join(t.TempDir(), "nonexistent"))
	if caches != (CacheInfo{}) {
		t.Errorf("expected zero CacheInfo for missing sysfs dir, got %+v", caches)
	}
}

// writeCacheIndex creates a fake sysfs cache index directory with the given
// level, type, and size files.
func writeCacheIndex(t *testing.T, root, name, level, cacheType, size string) {
	t.Helper()

	dir := filepath.Join(root, name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}

	for file, content := range map[string]string{
		"level": level,
		"type":  cacheType,
		"size":  size,
	} {
		if err := os.WriteFile(filepath.Join(dir, file), []byte(content+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}
