package planner

import (
	"bytes"
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

func TestWisdomImportComments(t *testing.T) {
	t.Parallel()

	w := NewWisdom()

	data := `# This is a comment
8:0:1:dit8_generic:1700000000
# Another comment

16:1:3:dit16_avx2:1700000000
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
		{"wrong_field_count", "8:0:1:test"},
		{"invalid_size", "abc:0:1:test:1700000000"},
		{"invalid_precision", "8:xyz:1:test:1700000000"},
		{"invalid_features", "8:0:xyz:test:1700000000"},
		{"invalid_timestamp", "8:0:1:test:abc"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			w := NewWisdom()

			err := w.Import(strings.NewReader(tt.data))
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestMakeWisdomKey(t *testing.T) {
	t.Parallel()

	key64 := MakeWisdomKey[complex64](1024, true, true, false, false)
	if key64.Precision != PrecisionComplex64 {
		t.Errorf("expected precision %d, got %d", PrecisionComplex64, key64.Precision)
	}

	if key64.Size != 1024 {
		t.Errorf("expected size 1024, got %d", key64.Size)
	}

	key128 := MakeWisdomKey[complex128](1024, true, true, false, false)
	if key128.Precision != PrecisionComplex128 {
		t.Errorf("expected precision %d, got %d", PrecisionComplex128, key128.Precision)
	}
}

func TestCPUFeatureMask(t *testing.T) {
	t.Parallel()

	tests := []struct {
		sse2, avx2, avx512, neon bool
		expected                 uint64
	}{
		{false, false, false, false, 0},
		{true, false, false, false, 1},
		{true, true, false, false, 3},
		{true, true, true, false, 7},
		{false, false, false, true, 8},
		{true, true, true, true, 15},
	}

	for _, tt := range tests {
		got := CPUFeatureMask(tt.sse2, tt.avx2, tt.avx512, tt.neon)
		if got != tt.expected {
			t.Errorf("CPUFeatureMask(%v,%v,%v,%v) = %d, want %d",
				tt.sse2, tt.avx2, tt.avx512, tt.neon, got, tt.expected)
		}
	}
}
