//go:build !amd64 && !arm64

package cpu

import "time"

// readCycleCounter falls back to time.Now() on platforms without assembly support.
// Returns nanoseconds since an arbitrary point in time.
func readCycleCounter() int64 {
	return time.Now().UnixNano()
}

// getCounterFrequencyHz returns 0 for generic platforms.
// Generic platforms use time.Now() which returns nanoseconds directly.
func getCounterFrequencyHz() int64 {
	return 0
}
