package planner

import (
	"bytes"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestWisdomStoreAndLookup(t *testing.T) {
	t.Parallel()

	w := NewWisdom()

	key := WisdomKey{Size: 1024, Precision: PrecisionComplex64, CPUFeatures: 0x3}
	entry := WisdomEntry{
		Key:       key,
		Algorithm: "dit1024_avx2",
		Timestamp: time.Now(),
	}

	// Lookup before store should fail
	_, found := w.Lookup(key)
	if found {
		t.Error("expected not found before store")
	}

	// Store and lookup
	w.Store(entry)

	got, found := w.Lookup(key)
	if !found {
		t.Fatal("expected found after store")
	}

	if got.Algorithm != entry.Algorithm {
		t.Errorf("expected algorithm %q, got %q", entry.Algorithm, got.Algorithm)
	}

	if got.Key != entry.Key {
		t.Errorf("expected key %v, got %v", entry.Key, got.Key)
	}
}

func TestWisdomLen(t *testing.T) {
	t.Parallel()

	w := NewWisdom()

	if w.Len() != 0 {
		t.Errorf("expected len 0, got %d", w.Len())
	}

	w.Store(WisdomEntry{
		Key:       WisdomKey{Size: 8, Precision: 0, CPUFeatures: 0},
		Algorithm: "test",
		Timestamp: time.Now(),
	})

	if w.Len() != 1 {
		t.Errorf("expected len 1, got %d", w.Len())
	}

	w.Store(WisdomEntry{
		Key:       WisdomKey{Size: 16, Precision: 0, CPUFeatures: 0},
		Algorithm: "test2",
		Timestamp: time.Now(),
	})

	if w.Len() != 2 {
		t.Errorf("expected len 2, got %d", w.Len())
	}
}

func TestWisdomClear(t *testing.T) {
	t.Parallel()

	w := NewWisdom()

	w.Store(WisdomEntry{
		Key:       WisdomKey{Size: 8, Precision: 0, CPUFeatures: 0},
		Algorithm: "test",
		Timestamp: time.Now(),
	})

	w.Clear()

	if w.Len() != 0 {
		t.Errorf("expected len 0 after clear, got %d", w.Len())
	}
}

func TestWisdomExportImport(t *testing.T) {
	t.Parallel()

	w := NewWisdom()

	now := time.Now().Truncate(time.Second) // Truncate for comparison

	entries := []WisdomEntry{
		{Key: WisdomKey{Size: 8, Precision: 0, CPUFeatures: 1}, Algorithm: "dit8_generic", Timestamp: now},
		{Key: WisdomKey{Size: 16, Precision: 1, CPUFeatures: 3}, Algorithm: "dit16_avx2", Timestamp: now},
		{Key: WisdomKey{Size: 1024, Precision: 0, CPUFeatures: 7}, Algorithm: "stockham", Timestamp: now},
	}

	for _, e := range entries {
		w.Store(e)
	}

	// Export
	var buf bytes.Buffer

	err := w.Export(&buf)
	if err != nil {
		t.Fatalf("export failed: %v", err)
	}

	exported := buf.String()
	if exported == "" {
		t.Fatal("exported data is empty")
	}

	// Import into new wisdom
	w2 := NewWisdom()

	err = w2.Import(strings.NewReader(exported))
	if err != nil {
		t.Fatalf("import failed: %v", err)
	}

	if w2.Len() != len(entries) {
		t.Errorf("expected %d entries after import, got %d", len(entries), w2.Len())
	}

	// Verify entries
	for _, entry := range entries {
		got, found := w2.Lookup(entry.Key)
		if !found {
			t.Errorf("entry for size %d not found after import", entry.Key.Size)
			continue
		}

		if got.Algorithm != entry.Algorithm {
			t.Errorf("size %d: expected algorithm %q, got %q", entry.Key.Size, entry.Algorithm, got.Algorithm)
		}
	}
}

func TestWisdomExportImportDirectionalAlgorithm(t *testing.T) {
	t.Parallel()

	w := NewWisdom()
	key := WisdomKey{Size: 32768, Precision: PrecisionComplex128, CPUFeatures: 7}
	w.Store(WisdomEntry{
		Key:       key,
		Algorithm: "dit32768_radix4_avx2/dit32768_radix8_avx2",
		Timestamp: time.Unix(1700000000, 0),
	})

	var buf bytes.Buffer
	if err := w.Export(&buf); err != nil {
		t.Fatalf("Export: %v", err)
	}

	replayed := NewWisdom()
	if err := replayed.Import(&buf); err != nil {
		t.Fatalf("Import: %v", err)
	}

	entry, ok := replayed.Lookup(key)
	if !ok || entry.Algorithm != "dit32768_radix4_avx2/dit32768_radix8_avx2" {
		t.Fatalf("replayed entry = (%+v, %v), want directional algorithm", entry, ok)
	}
}

func TestWisdomImportComments(t *testing.T) {
	t.Parallel()

	w := NewWisdom()

	data := wisdomMagic + `
# This is a comment
8:0:1:cpu_a:dit8_generic:1700000000
# Another comment

16:1:3:cpu_a:dit16_avx2:1700000000
`

	err := w.Import(strings.NewReader(data))
	if err != nil {
		t.Fatalf("import with comments failed: %v", err)
	}

	if w.Len() != 2 {
		t.Errorf("expected 2 entries, got %d", w.Len())
	}
}

func TestWisdomImportInvalid(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		data string
	}{
		{"wrong_field_count", "8:0:1:cpu:test"},
		{"invalid_size", "abc:0:1:cpu:test:1700000000"},
		{"invalid_precision", "8:xyz:1:cpu:test:1700000000"},
		{"invalid_features", "8:0:xyz:cpu:test:1700000000"},
		{"invalid_timestamp", "8:0:1:cpu:test:abc"},
		{"size_zero", "0:0:1:cpu:test:1700000000"},
		{"size_negative", "-8:0:1:cpu:test:1700000000"},
		{"precision_out_of_range", "8:2:1:cpu:test:1700000000"},
		{"feature_mask_too_wide", "8:0:64:cpu:test:1700000000"},
		{"invalid_cpu", "8:0:1:bad cpu:test:1700000000"},
		{"empty_algorithm", "8:0:1:cpu::1700000000"},
		{"garbage_algorithm", "8:0:1:cpu:bad name:1700000000"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			w := NewWisdom()

			err := w.Import(strings.NewReader(wisdomMagic + "\n" + tt.data + "\n"))
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

// TestWisdomImportHeader verifies the magic/version header is required.
func TestWisdomImportHeader(t *testing.T) {
	t.Parallel()

	valid := "8:0:1:cpu:dit8_generic:1700000000\n"

	tests := []struct {
		name    string
		data    string
		wantErr bool
	}{
		{"missing_header", valid, true},
		{"wrong_header", "# algofft-wisdom v1\n" + valid, true},
		{"superseded_header", "# algofft-wisdom v2\n" + valid, true},
		{"v3_header", "# algofft-wisdom v3\n" + valid, true},
		{"v4_header", "# algofft-wisdom v4\n" + valid, true},
		{"future_header", "# algofft-wisdom v6\n" + valid, true},
		{"only_comments_no_magic", "# just a comment\n" + valid, true},
		{"valid_header", wisdomMagic + "\n" + valid, false},
		{"blank_lines_before_header", "\n\n" + wisdomMagic + "\n" + valid, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			w := NewWisdom()

			err := w.Import(strings.NewReader(tt.data))
			if tt.wantErr && err == nil {
				t.Error("expected error, got nil")
			}

			if !tt.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

// TestWisdomImportAtomic verifies a malformed line leaves the cache untouched.
func TestWisdomImportAtomic(t *testing.T) {
	t.Parallel()

	w := NewWisdom()

	// Pre-populate with a good entry.
	w.Store(WisdomEntry{
		Key:       WisdomKey{Size: 4, Precision: 0, CPUFeatures: 1},
		Algorithm: "existing",
		Timestamp: time.Unix(1700000000, 0),
	})

	// One good line followed by a bad one: nothing should be applied.
	data := wisdomMagic + "\n" +
		"8:0:1:cpu:dit8_generic:1700000000\n" +
		"16:0:99:cpu:bad:1700000000\n" // feature mask too wide

	err := w.Import(strings.NewReader(data))
	if err == nil {
		t.Fatal("expected error from malformed line, got nil")
	}

	if w.Len() != 1 {
		t.Errorf("expected cache untouched (len 1), got len %d", w.Len())
	}

	// The staged good line must not have leaked in.
	if _, found := w.Lookup(WisdomKey{Size: 8, Precision: 0, CPUFeatures: 1}); found {
		t.Error("staged entry leaked into cache after failed import")
	}
}

// TestWisdomEvictOlderThan verifies age-based eviction.
func TestWisdomEvictOlderThan(t *testing.T) {
	t.Parallel()

	w := NewWisdom()

	now := time.Now()
	w.Store(WisdomEntry{
		Key:       WisdomKey{Size: 8, Precision: 0, CPUFeatures: 1},
		Algorithm: "fresh",
		Timestamp: now,
	})
	w.Store(WisdomEntry{
		Key:       WisdomKey{Size: 16, Precision: 0, CPUFeatures: 1},
		Algorithm: "stale",
		Timestamp: now.Add(-48 * time.Hour),
	})

	removed := w.EvictOlderThan(24 * time.Hour)
	if removed != 1 {
		t.Errorf("expected 1 removed, got %d", removed)
	}

	if w.Len() != 1 {
		t.Errorf("expected 1 remaining, got %d", w.Len())
	}

	if _, found := w.Lookup(WisdomKey{Size: 16, Precision: 0, CPUFeatures: 1}); found {
		t.Error("stale entry should have been evicted")
	}

	// Non-positive maxAge is a no-op.
	if n := w.EvictOlderThan(0); n != 0 {
		t.Errorf("EvictOlderThan(0) removed %d, want 0", n)
	}
}

// TestWisdomImportWithMaxAge verifies stale entries are dropped during import.
func TestWisdomImportWithMaxAge(t *testing.T) {
	t.Parallel()

	w := NewWisdom()

	now := time.Now()
	fresh := now.Unix()
	stale := now.Add(-72 * time.Hour).Unix()

	data := wisdomMagic + "\n" +
		"8:0:1:-:fresh:" + strconv.FormatInt(fresh, 10) + "\n" +
		"16:0:1:-:stale:" + strconv.FormatInt(stale, 10) + "\n"

	err := w.ImportWithMaxAge(strings.NewReader(data), 24*time.Hour)
	if err != nil {
		t.Fatalf("import failed: %v", err)
	}

	if w.Len() != 1 {
		t.Errorf("expected 1 entry after stale drop, got %d", w.Len())
	}

	if _, found := w.Lookup(WisdomKey{Size: 8, Precision: 0, CPUFeatures: 1}); !found {
		t.Error("fresh entry missing after import")
	}

	if _, found := w.Lookup(WisdomKey{Size: 16, Precision: 0, CPUFeatures: 1}); found {
		t.Error("stale entry should not have been imported")
	}
}

func TestMakeWisdomKey(t *testing.T) {
	t.Parallel()

	key64 := MakeWisdomKey[complex64](1024, true, false, true, false, false)
	if key64.Precision != PrecisionComplex64 {
		t.Errorf("expected precision %d, got %d", PrecisionComplex64, key64.Precision)
	}

	if key64.Size != 1024 {
		t.Errorf("expected size 1024, got %d", key64.Size)
	}

	if key64.CPUIdentifier == "" {
		t.Error("expected non-empty CPU identifier")
	}

	key128 := MakeWisdomKey[complex128](1024, true, false, true, false, false)
	if key128.Precision != PrecisionComplex128 {
		t.Errorf("expected precision %d, got %d", PrecisionComplex128, key128.Precision)
	}
}

func TestCPUFeatureMask(t *testing.T) {
	t.Parallel()

	tests := []struct {
		sse2, sse3, avx2, avx512, neon bool
		expected                       uint64
	}{
		{false, false, false, false, false, 0},
		{true, false, false, false, false, 1},  // SSE2
		{false, true, false, false, false, 2},  // SSE3
		{true, true, false, false, false, 3},   // SSE2+SSE3
		{true, false, true, false, false, 5},   // SSE2+AVX2
		{true, true, true, false, false, 7},    // SSE2+SSE3+AVX2
		{true, true, true, true, false, 15},    // + AVX512
		{false, false, false, false, true, 16}, // NEON
		{true, true, true, true, true, 31},     // all
	}

	for _, tt := range tests {
		got := CPUFeatureMask(tt.sse2, tt.sse3, tt.avx2, tt.avx512, tt.neon)
		if got != tt.expected {
			t.Errorf("CPUFeatureMask(%v,%v,%v,%v,%v) = %d, want %d",
				tt.sse2, tt.sse3, tt.avx2, tt.avx512, tt.neon, got, tt.expected)
		}
	}
}

// TestCPUFeatureMaskSSE3Distinct verifies SSE3 is tracked separately from SSE2.
func TestCPUFeatureMaskSSE3Distinct(t *testing.T) {
	t.Parallel()

	sse2Only := CPUFeatureMask(true, false, false, false, false)
	sse2AndSSE3 := CPUFeatureMask(true, true, false, false, false)

	if sse2Only == sse2AndSSE3 {
		t.Errorf("SSE3 not distinguished from SSE2: both masks = %d", sse2Only)
	}
}
