//go:build !amd64 && !arm64

package cpu

import "time"

// readCycleCounter falls back to time.Now() on platforms without assembly support.
// Returns nanoseconds since an arbitrary point in time.
func readCycleCounter() int64 {
	return time.Now().UnixNano()
}
