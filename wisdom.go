package algofft

import (
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/planner"
)

// Wisdom caches planning decisions for fast lookup. It wraps the internal
// wisdom cache behind a root-owned type so internal refactors cannot change
// the public API; keys and entries convert at this boundary.
//
// Wisdom implements the WisdomStore interface and is safe for concurrent use.
type Wisdom struct {
	inner *planner.Wisdom
}

// NewWisdom creates a new empty wisdom cache.
func NewWisdom() *Wisdom {
	return &Wisdom{inner: planner.NewWisdom()}
}

// CurrentCPUIdentifier returns the stable architecture, microarchitecture, and
// cache identity used to scope built-in Wisdom decisions. It is primarily
// useful to custom MicroarchitectureWisdomStore implementations and callers
// that construct WisdomKey values directly.
func CurrentCPUIdentifier() string {
	return cpu.WisdomCPUIdentifier(cpu.DetectFeatures())
}

// LookupWisdom returns the algorithm name for a given FFT configuration on the
// current CPU, falling back to an identifier-free entry stored through the
// legacy API. Returns empty string and false if no wisdom is available.
func (w *Wisdom) LookupWisdom(size int, precision uint8, cpuFeatures uint64) (string, bool) {
	return w.inner.LookupWisdom(size, precision, cpuFeatures)
}

// LookupWisdomForCPU returns the algorithm for an exact CPU context. This is
// the lookup path used by plans so measurements from different
// microarchitectures do not collide.
func (w *Wisdom) LookupWisdomForCPU(
	size int, precision uint8, cpuFeatures uint64, cpuIdentifier string,
) (string, bool) {
	return w.inner.LookupWisdomForCPU(size, precision, cpuFeatures, cpuIdentifier)
}

// Lookup returns the full wisdom entry for a given key.
func (w *Wisdom) Lookup(key WisdomKey) (WisdomEntry, bool) {
	entry, found := w.inner.Lookup(planner.WisdomKey{
		Size:          key.Size,
		Precision:     key.Precision,
		CPUFeatures:   key.CPUFeatures,
		CPUIdentifier: key.CPUIdentifier,
	})
	if !found {
		return WisdomEntry{}, false
	}

	return wisdomEntryFromInternal(entry), true
}

// Store saves a planning decision to the wisdom cache.
func (w *Wisdom) Store(entry WisdomEntry) {
	w.inner.Store(planner.WisdomEntry{
		Key: planner.WisdomKey{
			Size:          entry.Key.Size,
			Precision:     entry.Key.Precision,
			CPUFeatures:   entry.Key.CPUFeatures,
			CPUIdentifier: entry.Key.CPUIdentifier,
		},
		Algorithm: entry.Algorithm,
		Timestamp: entry.Timestamp,
	})
}

// Clear removes all entries from the wisdom cache.
func (w *Wisdom) Clear() {
	w.inner.Clear()
}

// Len returns the number of entries in the wisdom cache.
func (w *Wisdom) Len() int {
	return w.inner.Len()
}

// Export writes the wisdom cache to writer in the textual wisdom format.
func (w *Wisdom) Export(writer io.Writer) error {
	err := w.inner.Export(writer)
	if err != nil {
		return fmt.Errorf("export wisdom: %w", err)
	}

	return nil
}

// Import loads wisdom data from reader, merging it into the cache.
// The import is atomic: on any parse or validation error the cache is left
// unchanged.
func (w *Wisdom) Import(reader io.Reader) error {
	err := w.inner.Import(reader)
	if err != nil {
		return fmt.Errorf("import wisdom: %w", err)
	}

	return nil
}

// ImportWithMaxAge loads wisdom data from reader, dropping entries whose
// recorded timestamp is older than maxAge. A maxAge <= 0 imports every entry.
// The import is atomic: on any parse or validation error the cache is left
// unchanged.
func (w *Wisdom) ImportWithMaxAge(reader io.Reader, maxAge time.Duration) error {
	err := w.inner.ImportWithMaxAge(reader, maxAge)
	if err != nil {
		return fmt.Errorf("import wisdom: %w", err)
	}

	return nil
}

// EvictOlderThan removes entries whose timestamp is older than maxAge and
// returns the number of evicted entries.
func (w *Wisdom) EvictOlderThan(maxAge time.Duration) int {
	return w.inner.EvictOlderThan(maxAge)
}

func wisdomEntryFromInternal(entry planner.WisdomEntry) WisdomEntry {
	return WisdomEntry{
		Key: WisdomKey{
			Size:          entry.Key.Size,
			Precision:     entry.Key.Precision,
			CPUFeatures:   entry.Key.CPUFeatures,
			CPUIdentifier: entry.Key.CPUIdentifier,
		},
		Algorithm: entry.Algorithm,
		Timestamp: entry.Timestamp,
	}
}

// ImportWisdom loads wisdom data from a file into the process-wide default
// wisdom cache. The file should be in the format produced by ExportWisdom.
func ImportWisdom(filename string) error {
	return ImportWisdomWithMaxAge(filename, 0)
}

// ImportWisdomWithMaxAge loads wisdom data from a file, dropping entries whose
// recorded timestamp is older than maxAge. A maxAge <= 0 imports every entry.
// The import is atomic: on any parse or validation error the wisdom cache is left
// unchanged. The file should be in the format produced by ExportWisdom.
func ImportWisdomWithMaxAge(filename string, maxAge time.Duration) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open wisdom file: %w", err)
	}

	defer file.Close()

	err = planner.DefaultWisdom.ImportWithMaxAge(file, maxAge)
	if err != nil {
		return fmt.Errorf("failed to import wisdom: %w", err)
	}

	return nil
}

// ExportWisdom saves the process-wide default wisdom cache to a file.
// The file can be loaded later with ImportWisdom.
func ExportWisdom(filename string) error {
	return ExportWisdomTo(filename, &Wisdom{inner: planner.DefaultWisdom})
}

// ExportWisdomTo saves a specific wisdom cache to a file.
// This is useful for exporting benchmark results from custom wisdom instances.
func ExportWisdomTo(filename string, wisdom *Wisdom) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create wisdom file: %w", err)
	}

	defer file.Close()

	err = wisdom.Export(file)
	if err != nil {
		return fmt.Errorf("failed to export wisdom: %w", err)
	}

	return nil
}

// ImportWisdomFromString loads wisdom data from a string into the
// process-wide default wisdom cache. This is useful for embedding wisdom data
// in compiled binaries.
func ImportWisdomFromString(data string) error {
	err := planner.DefaultWisdom.Import(strings.NewReader(data))
	if err != nil {
		return fmt.Errorf("failed to import wisdom from string: %w", err)
	}

	return nil
}

// ClearWisdom removes all entries from the process-wide default wisdom cache.
func ClearWisdom() {
	planner.DefaultWisdom.Clear()
}

// WisdomLen returns the number of entries in the process-wide default wisdom
// cache.
func WisdomLen() int {
	return planner.DefaultWisdom.Len()
}
