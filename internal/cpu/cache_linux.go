//go:build linux

package cpu

// detectCachesImpl reads cache sizes from the kernel's sysfs cache topology.
func detectCachesImpl() CacheInfo {
	return readCachesFromSysfsDir(sysfsCacheRoot)
}
