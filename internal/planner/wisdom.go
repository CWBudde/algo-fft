package planner

import (
	"bufio"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// WisdomKey uniquely identifies a planning decision.
type WisdomKey struct {
	Size        int    // FFT size
	Precision   uint8  // 0 = complex64, 1 = complex128
	CPUFeatures uint64 // Bitmask of relevant CPU features
}

// WisdomEntry stores a planning decision.
type WisdomEntry struct {
	Key       WisdomKey
	Algorithm string    // e.g., "dit64_generic", "stockham"
	Timestamp time.Time // When this entry was recorded
}

// Wisdom caches planning decisions for fast lookup.
// It is thread-safe and can be used from multiple goroutines.
type Wisdom struct {
	mu      sync.RWMutex
	entries map[WisdomKey]WisdomEntry
}

// NewWisdom creates a new empty wisdom cache.
func NewWisdom() *Wisdom {
	return &Wisdom{
		entries: make(map[WisdomKey]WisdomEntry),
	}
}

// Lookup finds a cached planning decision.
// Returns the entry and true if found, zero value and false otherwise.
func (w *Wisdom) Lookup(key WisdomKey) (WisdomEntry, bool) {
	w.mu.RLock()
	defer w.mu.RUnlock()

	entry, ok := w.entries[key]

	return entry, ok
}

// LookupWisdom returns the algorithm name for a given FFT configuration.
// This method provides a simplified interface for the planner.
//
//nolint:nonamedreturns
func (w *Wisdom) LookupWisdom(size int, precision uint8, cpuFeatures uint64) (algorithm string, found bool) {
	key := WisdomKey{
		Size:        size,
		Precision:   precision,
		CPUFeatures: cpuFeatures,
	}

	entry, ok := w.Lookup(key)
	if !ok {
		return "", false
	}

	return entry.Algorithm, true
}

// Store saves a planning decision to the cache.
func (w *Wisdom) Store(entry WisdomEntry) {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.entries[entry.Key] = entry
}

// Clear removes all entries from the wisdom cache.
func (w *Wisdom) Clear() {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.entries = make(map[WisdomKey]WisdomEntry)
}

// Len returns the number of entries in the wisdom cache.
func (w *Wisdom) Len() int {
	w.mu.RLock()
	defer w.mu.RUnlock()

	return len(w.entries)
}

// Export writes the wisdom cache to a writer in a portable text format.
// Format: one entry per line as "size:precision:features:algorithm:timestamp"
// Entries are sorted by size, precision, and CPU features for deterministic output.
func (w *Wisdom) Export(writer io.Writer) error {
	w.mu.RLock()
	defer w.mu.RUnlock()

	// Collect entries into a slice for sorting
	entries := make([]WisdomEntry, 0, len(w.entries))
	for _, entry := range w.entries {
		entries = append(entries, entry)
	}

	// Sort by size, then precision, then CPU features for deterministic output
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].Key.Size != entries[j].Key.Size {
			return entries[i].Key.Size < entries[j].Key.Size
		}

		if entries[i].Key.Precision != entries[j].Key.Precision {
			return entries[i].Key.Precision < entries[j].Key.Precision
		}

		return entries[i].Key.CPUFeatures < entries[j].Key.CPUFeatures
	})

	// Write sorted entries
	for _, entry := range entries {
		line := fmt.Sprintf("%d:%d:%d:%s:%d\n",
			entry.Key.Size,
			entry.Key.Precision,
			entry.Key.CPUFeatures,
			entry.Algorithm,
			entry.Timestamp.Unix())

		_, err := writer.Write([]byte(line))
		if err != nil {
			return fmt.Errorf("failed to write wisdom entry: %w", err)
		}
	}

	return nil
}

// Import reads wisdom entries from a reader and adds them to the cache.
// Existing entries with the same key are overwritten.
func (w *Wisdom) Import(reader io.Reader) error {
	scanner := bufio.NewScanner(reader)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue // Skip empty lines and comments
		}

		entry, err := parseWisdomLine(line)
		if err != nil {
			return fmt.Errorf("wisdom import: %w", err)
		}

		w.Store(entry)
	}

	err := scanner.Err()
	if err != nil {
		return fmt.Errorf("failed to scan wisdom entries: %w", err)
	}

	return nil
}

// parseWisdomLine parses a single line of wisdom format.
func parseWisdomLine(line string) (WisdomEntry, error) {
	parts := strings.Split(line, ":")
	if len(parts) != 5 {
		return WisdomEntry{}, fmt.Errorf("invalid format: expected 5 fields, got %d", len(parts))
	}

	size, err := strconv.Atoi(parts[0])
	if err != nil {
		return WisdomEntry{}, fmt.Errorf("invalid size: %w", err)
	}

	precision, err := strconv.ParseUint(parts[1], 10, 8)
	if err != nil {
		return WisdomEntry{}, fmt.Errorf("invalid precision: %w", err)
	}

	features, err := strconv.ParseUint(parts[2], 10, 64)
	if err != nil {
		return WisdomEntry{}, fmt.Errorf("invalid features: %w", err)
	}

	algorithm := parts[3]

	timestamp, err := strconv.ParseInt(parts[4], 10, 64)
	if err != nil {
		return WisdomEntry{}, fmt.Errorf("invalid timestamp: %w", err)
	}

	return WisdomEntry{
		Key: WisdomKey{
			Size:        size,
			Precision:   uint8(precision),
			CPUFeatures: features,
		},
		Algorithm: algorithm,
		Timestamp: time.Unix(timestamp, 0),
	}, nil
}

// DefaultWisdom is the global wisdom cache used by default planning.
//
//nolint:gochecknoglobals
var DefaultWisdom = NewWisdom()

// PrecisionComplex64 is the precision value for complex64.
const PrecisionComplex64 uint8 = 0

// PrecisionComplex128 is the precision value for complex128.
const PrecisionComplex128 uint8 = 1

// MakeWisdomKey creates a wisdom key from the given parameters.
func MakeWisdomKey[T Complex](size int, hasSSE2, hasAVX2, hasAVX512, hasNEON bool) WisdomKey {
	var zero T

	precision := PrecisionComplex64
	if _, ok := any(zero).(complex128); ok {
		precision = PrecisionComplex128
	}

	return WisdomKey{
		Size:        size,
		Precision:   precision,
		CPUFeatures: CPUFeatureMask(hasSSE2, hasAVX2, hasAVX512, hasNEON),
	}
}
