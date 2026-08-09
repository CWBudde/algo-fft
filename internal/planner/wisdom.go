package planner

import (
	"bufio"
	"fmt"
	"io"
	"maps"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/cwbudde/algo-fft/internal/cpu"
)

// WisdomKey uniquely identifies a planning decision.
type WisdomKey struct {
	Size          int    // FFT size
	Precision     uint8  // 0 = complex64, 1 = complex128
	CPUFeatures   uint64 // Bitmask of relevant CPU features
	CPUIdentifier string // Stable architecture, microarchitecture, and cache identity
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

// LookupWisdom returns the algorithm name for a given FFT configuration. It
// first checks the current CPU context, then an identifier-free legacy entry.
// Context-aware planner code calls LookupWisdomForCPU directly.
//
//nolint:nonamedreturns
func (w *Wisdom) LookupWisdom(size int, precision uint8, cpuFeatures uint64) (algorithm string, found bool) {
	algorithm, found = w.LookupWisdomForCPU(
		size,
		precision,
		cpuFeatures,
		cpu.WisdomCPUIdentifier(cpu.DetectFeatures()),
	)
	if found {
		return algorithm, true
	}

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

// LookupWisdomForCPU returns the algorithm for an exact CPU context. Unlike
// LookupWisdom, this distinguishes processors that expose the same SIMD bits.
func (w *Wisdom) LookupWisdomForCPU(
	size int, precision uint8, cpuFeatures uint64, cpuIdentifier string,
) (algorithm string, found bool) {
	key := WisdomKey{
		Size:          size,
		Precision:     precision,
		CPUFeatures:   cpuFeatures,
		CPUIdentifier: cpuIdentifier,
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

// wisdomMagic is the required first line of a wisdom file. It doubles as a magic
// marker and a version header: Import rejects any file that does not start with
// this exact line, so unversioned or future-format files fail loudly instead of
// being mis-parsed.
//
// v4 adds the CPU identifier to the key so results measured on different
// microarchitectures are not silently shared merely because their SIMD feature
// masks match. Older files cannot safely supply that field and are rejected.
// A wisdom entry outranks the codelet
// registry (see EstimatePlan), and that is only sound because
// internal/fft.MeasureAndSelect times the registry's codelets as candidates.
// It did not under v2, so a v2 strategy entry records a comparison that never
// included the codelet it would now displace. Re-measure to regenerate.
const wisdomMagic = "# algofft-wisdom v4"

// wisdomLegend is a human-readable column legend written after the magic header.
const wisdomLegend = "# size:precision:features:cpu:algorithm:timestamp"

// maxWisdomSize is a sanity cap on the FFT size stored in a wisdom entry.
const maxWisdomSize = 1 << 30

// Export writes the wisdom cache to a writer in a portable, versioned text format.
// The first line is the magic/version header (wisdomMagic), followed by a column
// legend, then one entry per line as
// "size:precision:features:cpu:algorithm:timestamp". Entries are sorted by
// size, precision, CPU features, and CPU identifier for deterministic output.
func (w *Wisdom) Export(writer io.Writer) error {
	w.mu.RLock()
	defer w.mu.RUnlock()

	// Write the versioned header and column legend first.
	header := wisdomMagic + "\n" + wisdomLegend + "\n"

	_, err := writer.Write([]byte(header))
	if err != nil {
		return fmt.Errorf("failed to write wisdom header: %w", err)
	}

	// Collect entries into a slice for sorting
	entries := make([]WisdomEntry, 0, len(w.entries))
	for _, entry := range w.entries {
		entries = append(entries, entry)
	}

	// Sort by size, precision, CPU features, then CPU identifier.
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].Key.Size != entries[j].Key.Size {
			return entries[i].Key.Size < entries[j].Key.Size
		}

		if entries[i].Key.Precision != entries[j].Key.Precision {
			return entries[i].Key.Precision < entries[j].Key.Precision
		}

		if entries[i].Key.CPUFeatures != entries[j].Key.CPUFeatures {
			return entries[i].Key.CPUFeatures < entries[j].Key.CPUFeatures
		}

		return entries[i].Key.CPUIdentifier < entries[j].Key.CPUIdentifier
	})

	// Write sorted entries
	for _, entry := range entries {
		cpuIdentifier := entry.Key.CPUIdentifier
		if cpuIdentifier == "" {
			cpuIdentifier = "-"
		}

		line := fmt.Sprintf("%d:%d:%d:%s:%s:%d\n",
			entry.Key.Size,
			entry.Key.Precision,
			entry.Key.CPUFeatures,
			cpuIdentifier,
			entry.Algorithm,
			entry.Timestamp.Unix())

		_, err = writer.Write([]byte(line))
		if err != nil {
			return fmt.Errorf("failed to write wisdom entry: %w", err)
		}
	}

	return nil
}

// Import reads wisdom entries from a reader and merges them into the cache.
// Existing entries with the same key are overwritten. See ImportWithMaxAge for
// the full contract; Import applies no age-based eviction.
func (w *Wisdom) Import(reader io.Reader) error {
	return w.ImportWithMaxAge(reader, 0)
}

// ImportWithMaxAge reads wisdom entries from a reader and merges them into the
// cache atomically: the file is fully parsed and validated into a temporary map
// before any live entry is touched, so a malformed line aborts the whole import
// with the cache left unchanged.
//
// The reader must begin with the versioned magic header (wisdomMagic); an
// unrecognized or missing header is rejected rather than mis-parsed.
//
// If maxAge > 0, entries whose Timestamp is older than now-maxAge are dropped
// before the merge, so a stale file cannot reintroduce outdated choices. A
// maxAge <= 0 disables age-based eviction.
func (w *Wisdom) ImportWithMaxAge(reader io.Reader, maxAge time.Duration) error {
	scanner := bufio.NewScanner(reader)

	headerSeen := false
	staged := make(map[WisdomKey]WisdomEntry)

	var cutoff time.Time
	if maxAge > 0 {
		cutoff = time.Now().Add(-maxAge)
	}

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		// The first non-blank line must be the magic/version header.
		if !headerSeen {
			consumed, err := consumeHeader(line)
			if err != nil {
				return err
			}

			headerSeen = consumed

			continue
		}

		err := stageWisdomLine(line, staged, cutoff, maxAge > 0)
		if err != nil {
			return fmt.Errorf("wisdom import: %w", err)
		}
	}

	err := scanner.Err()
	if err != nil {
		return fmt.Errorf("failed to scan wisdom entries: %w", err)
	}

	// headerSeen is false only when the input had no non-blank lines at all (a
	// genuinely empty file), which is a harmless no-op. Any non-blank content is
	// forced through the header check above, so a headerless data file errors there.
	if !headerSeen {
		return nil
	}

	// Merge atomically only after the whole file parsed and validated.
	w.mu.Lock()
	defer w.mu.Unlock()

	maps.Copy(w.entries, staged)

	return nil
}

// EvictOlderThan removes entries whose Timestamp is older than now-maxAge and
// returns the number of entries removed. A maxAge <= 0 removes nothing.
func (w *Wisdom) EvictOlderThan(maxAge time.Duration) int {
	if maxAge <= 0 {
		return 0
	}

	cutoff := time.Now().Add(-maxAge)

	w.mu.Lock()
	defer w.mu.Unlock()

	removed := 0

	for key, entry := range w.entries {
		if entry.Timestamp.Before(cutoff) {
			removed++

			delete(w.entries, key)
		}
	}

	return removed
}

// consumeHeader inspects the first non-blank line of a wisdom file. It reports
// whether the header was consumed (true once the magic line is seen) and errors
// if a non-blank line is present but is not the expected magic/version header.
func consumeHeader(line string) (bool, error) {
	if line == "" {
		return false, nil
	}

	if line != wisdomMagic {
		return false, fmt.Errorf("wisdom import: unrecognized header %q, expected %q", line, wisdomMagic)
	}

	return true, nil
}

// stageWisdomLine parses one body line and, unless it is blank/comment or a
// stale entry (when useCutoff is set), stages it into dst. Parse/validation
// errors are returned unwrapped for the caller to contextualize.
func stageWisdomLine(line string, dst map[WisdomKey]WisdomEntry, cutoff time.Time, useCutoff bool) error {
	if line == "" || strings.HasPrefix(line, "#") {
		return nil // Skip empty lines and comments.
	}

	entry, err := parseWisdomLine(line)
	if err != nil {
		return err
	}

	if useCutoff && entry.Timestamp.Before(cutoff) {
		return nil // Drop stale entries.
	}

	dst[entry.Key] = entry

	return nil
}

// parseWisdomLine parses and validates a single line of wisdom format.
func parseWisdomLine(line string) (WisdomEntry, error) {
	parts := strings.Split(line, ":")
	if len(parts) != 6 {
		return WisdomEntry{}, fmt.Errorf("invalid format: expected 6 fields, got %d", len(parts))
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

	cpuIdentifier := parts[3]
	if cpuIdentifier == "-" {
		cpuIdentifier = ""
	}

	algorithm := parts[4]

	timestamp, err := strconv.ParseInt(parts[5], 10, 64)
	if err != nil {
		return WisdomEntry{}, fmt.Errorf("invalid timestamp: %w", err)
	}

	entry := WisdomEntry{
		Key: WisdomKey{
			Size:          size,
			Precision:     uint8(precision),
			CPUFeatures:   features,
			CPUIdentifier: cpuIdentifier,
		},
		Algorithm: algorithm,
		Timestamp: time.Unix(timestamp, 0),
	}

	err = validateEntry(entry)
	if err != nil {
		return WisdomEntry{}, err
	}

	return entry, nil
}

// validateEntry rejects wisdom entries whose fields fall outside the supported
// ranges of the current format.
func validateEntry(entry WisdomEntry) error {
	if entry.Key.Size <= 0 || entry.Key.Size > maxWisdomSize {
		return fmt.Errorf("invalid size %d: out of range (1..%d)", entry.Key.Size, maxWisdomSize)
	}

	if entry.Key.Precision != PrecisionComplex64 && entry.Key.Precision != PrecisionComplex128 {
		return fmt.Errorf("invalid precision %d: expected %d or %d",
			entry.Key.Precision, PrecisionComplex64, PrecisionComplex128)
	}

	if entry.Key.CPUFeatures&^featMaskAll != 0 {
		return fmt.Errorf("invalid feature mask %#x: bits set outside supported width %#x",
			entry.Key.CPUFeatures, featMaskAll)
	}

	if !isValidCPUIdentifier(entry.Key.CPUIdentifier) {
		return fmt.Errorf("invalid CPU identifier %q: expected [A-Za-z0-9_]", entry.Key.CPUIdentifier)
	}

	if !isValidAlgorithmName(entry.Algorithm) {
		return fmt.Errorf("invalid algorithm name %q: expected non-empty [A-Za-z0-9_]", entry.Algorithm)
	}

	return nil
}

func isValidCPUIdentifier(identifier string) bool {
	return strings.IndexFunc(identifier, isNotAlgorithmNameRune) < 0
}

// isValidAlgorithmName reports whether s is a plausible algorithm/codelet name:
// non-empty and restricted to a safe charset. Codelet signatures are
// size-specific (e.g. "dit8_avx2"), so a closed enum is infeasible; this rejects
// empty or garbage/injected values.
func isValidAlgorithmName(s string) bool {
	if s == "" {
		return false
	}

	return strings.IndexFunc(s, isNotAlgorithmNameRune) < 0
}

// isNotAlgorithmNameRune reports whether r is outside the safe algorithm-name
// charset ([A-Za-z0-9_]).
func isNotAlgorithmNameRune(r rune) bool {
	switch {
	case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '_':
		return false
	default:
		return true
	}
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
func MakeWisdomKey[T Complex](size int, hasSSE2, hasSSE3, hasAVX2, hasAVX512, hasNEON bool) WisdomKey {
	var zero T

	precision := PrecisionComplex64
	if _, ok := any(zero).(complex128); ok {
		precision = PrecisionComplex128
	}

	return WisdomKey{
		Size:          size,
		Precision:     precision,
		CPUFeatures:   CPUFeatureMask(hasSSE2, hasSSE3, hasAVX2, hasAVX512, hasNEON),
		CPUIdentifier: cpu.WisdomCPUIdentifier(cpu.DetectFeatures()),
	}
}
