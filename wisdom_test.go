package algofft

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/MeKo-Christian/algo-fft/internal/planner"
)

func TestNewWisdom(t *testing.T) {
	t.Parallel()

	wisdom := NewWisdom()
	if wisdom == nil {
		t.Fatal("NewWisdom() returned nil")
	}

	// New wisdom should be empty
	if wisdom.Len() != 0 {
		t.Errorf("New wisdom length = %d, want 0", wisdom.Len())
	}
}

func TestWisdom_StoreAndLookup(t *testing.T) {
	t.Parallel()

	wisdom := NewWisdom()

	// Store an entry
	entry := planner.WisdomEntry{
		Key: planner.WisdomKey{
			Size:        128,
			Precision:   0, // complex64
			CPUFeatures: 0,
		},
		Algorithm: "dit64",
		Timestamp: time.Now(),
	}

	wisdom.Store(entry)

	// Verify it was stored
	if wisdom.Len() != 1 {
		t.Errorf("Wisdom length = %d, want 1 after Store", wisdom.Len())
	}

	// Lookup the entry
	retrieved, found := wisdom.Lookup(entry.Key)
	if !found {
		t.Fatal("Lookup() failed to find stored entry")
	}

	if retrieved.Algorithm != entry.Algorithm {
		t.Errorf("Retrieved algorithm = %s, want %s", retrieved.Algorithm, entry.Algorithm)
	}

	// Lookup with LookupWisdom method
	algorithm, found := wisdom.LookupWisdom(128, 0, 0)
	if !found {
		t.Fatal("LookupWisdom() failed to find stored entry")
	}

	if algorithm != "dit64" {
		t.Errorf("LookupWisdom algorithm = %s, want dit64", algorithm)
	}
}

func TestWisdom_LookupNonExistent(t *testing.T) {
	t.Parallel()

	wisdom := NewWisdom()

	// Lookup non-existent entry
	_, found := wisdom.Lookup(planner.WisdomKey{Size: 256, Precision: 0, CPUFeatures: 0})
	if found {
		t.Error("Lookup() found non-existent entry")
	}

	// LookupWisdom for non-existent entry
	algorithm, found := wisdom.LookupWisdom(256, 0, 0)
	if found {
		t.Error("LookupWisdom() found non-existent entry")
	}

	if algorithm != "" {
		t.Errorf("LookupWisdom algorithm = %s, want empty string for non-existent entry", algorithm)
	}
}

func TestWisdom_OverwriteEntry(t *testing.T) {
	t.Parallel()

	wisdom := NewWisdom()

	key := planner.WisdomKey{Size: 64, Precision: 1, CPUFeatures: 123}

	// Store first entry
	entry1 := planner.WisdomEntry{
		Key:       key,
		Algorithm: "stockham",
		Timestamp: time.Now(),
	}
	wisdom.Store(entry1)

	// Store second entry with same key
	entry2 := planner.WisdomEntry{
		Key:       key,
		Algorithm: "dit64",
		Timestamp: time.Now(),
	}
	wisdom.Store(entry2)

	// Should have only one entry (overwritten)
	if wisdom.Len() != 1 {
		t.Errorf("Wisdom length = %d, want 1 after overwrite", wisdom.Len())
	}

	// Should retrieve the second entry
	retrieved, found := wisdom.Lookup(key)
	if !found {
		t.Fatal("Lookup() failed after overwrite")
	}

	if retrieved.Algorithm != "dit64" {
		t.Errorf("Retrieved algorithm = %s, want dit64 (should be overwritten)", retrieved.Algorithm)
	}
}

func TestClearWisdom(t *testing.T) {
	t.Parallel()

	// Note: This test uses the global DefaultWisdom, so we need to be careful
	// Save initial state
	initialLen := WisdomLen()

	// Add some entries
	wisdom := NewWisdom()
	wisdom.Store(planner.WisdomEntry{
		Key:       planner.WisdomKey{Size: 128, Precision: 0, CPUFeatures: 0},
		Algorithm: "test",
		Timestamp: time.Now(),
	})

	// For this test, we'll use a local wisdom instance instead of the global one
	wisdom.Clear()

	if wisdom.Len() != 0 {
		t.Errorf("Wisdom length = %d after Clear(), want 0", wisdom.Len())
	}

	// Verify global wisdom was not affected (should still have initial length)
	if WisdomLen() != initialLen {
		t.Errorf("Global wisdom length changed from %d to %d", initialLen, WisdomLen())
	}
}

func TestWisdomLen(t *testing.T) {
	t.Parallel()

	wisdom := NewWisdom()

	// Initially empty
	if wisdom.Len() != 0 {
		t.Errorf("Initial length = %d, want 0", wisdom.Len())
	}

	// Add entries
	for i := 0; i < 5; i++ {
		wisdom.Store(planner.WisdomEntry{
			Key: planner.WisdomKey{
				Size:        128 << i,
				Precision:   0,
				CPUFeatures: 0,
			},
			Algorithm: "test",
			Timestamp: time.Now(),
		})
	}

	if wisdom.Len() != 5 {
		t.Errorf("Length after 5 stores = %d, want 5", wisdom.Len())
	}

	// Clear
	wisdom.Clear()

	if wisdom.Len() != 0 {
		t.Errorf("Length after Clear = %d, want 0", wisdom.Len())
	}
}

func TestExportWisdom(t *testing.T) {
	t.Parallel()

	// Create temporary directory for test files
	tmpDir := t.TempDir()
	filename := filepath.Join(tmpDir, "wisdom_test.txt")

	// Create a wisdom instance with some entries
	wisdom := NewWisdom()
	wisdom.Store(planner.WisdomEntry{
		Key: planner.WisdomKey{
			Size:        128,
			Precision:   0,
			CPUFeatures: 0,
		},
		Algorithm: "dit64",
		Timestamp: time.Unix(1234567890, 0),
	})

	wisdom.Store(planner.WisdomEntry{
		Key: planner.WisdomKey{
			Size:        256,
			Precision:   1,
			CPUFeatures: 123,
		},
		Algorithm: "stockham",
		Timestamp: time.Unix(1234567891, 0),
	})

	// Export to file
	err := ExportWisdomTo(filename, wisdom)
	if err != nil {
		t.Fatalf("ExportWisdomTo() failed: %v", err)
	}

	// Verify file exists
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		t.Fatal("ExportWisdomTo() did not create file")
	}

	// Read file content
	data, err := os.ReadFile(filename)
	if err != nil {
		t.Fatalf("Failed to read exported wisdom file: %v", err)
	}

	content := string(data)

	// Verify content format (should have two lines)
	lines := strings.Split(strings.TrimSpace(content), "\n")
	if len(lines) != 2 {
		t.Errorf("Exported wisdom has %d lines, want 2", len(lines))
	}

	// Verify format: size:precision:features:algorithm:timestamp
	// Should be sorted by size
	if !strings.Contains(lines[0], "128:0:0:dit64:1234567890") {
		t.Errorf("First line = %s, want 128:0:0:dit64:1234567890", lines[0])
	}

	if !strings.Contains(lines[1], "256:1:123:stockham:1234567891") {
		t.Errorf("Second line = %s, want 256:1:123:stockham:1234567891", lines[1])
	}
}

func TestImportWisdom(t *testing.T) {
	t.Parallel()

	// Create temporary directory for test files
	tmpDir := t.TempDir()
	filename := filepath.Join(tmpDir, "wisdom_import_test.txt")

	// Create wisdom file
	wisdomData := `128:0:0:dit64:1234567890
256:1:123:stockham:1234567891
# This is a comment
512:0:456:sixstep:1234567892

1024:1:0:bluestein:1234567893`

	err := os.WriteFile(filename, []byte(wisdomData), 0644)
	if err != nil {
		t.Fatalf("Failed to create test wisdom file: %v", err)
	}

	// Create a new wisdom and import
	wisdom := NewWisdom()

	// Use a custom import (not the global ImportWisdom)
	file, err := os.Open(filename)
	if err != nil {
		t.Fatalf("Failed to open wisdom file: %v", err)
	}
	defer file.Close()

	err = wisdom.Import(file)
	if err != nil {
		t.Fatalf("Import() failed: %v", err)
	}

	// Verify entries were imported (should have 4 entries, skipping comment and empty line)
	if wisdom.Len() != 4 {
		t.Errorf("Imported wisdom length = %d, want 4", wisdom.Len())
	}

	// Verify first entry
	entry, found := wisdom.Lookup(planner.WisdomKey{Size: 128, Precision: 0, CPUFeatures: 0})
	if !found {
		t.Error("Failed to find first imported entry")
	} else if entry.Algorithm != "dit64" {
		t.Errorf("First entry algorithm = %s, want dit64", entry.Algorithm)
	}

	// Verify second entry
	entry, found = wisdom.Lookup(planner.WisdomKey{Size: 256, Precision: 1, CPUFeatures: 123})
	if !found {
		t.Error("Failed to find second imported entry")
	} else if entry.Algorithm != "stockham" {
		t.Errorf("Second entry algorithm = %s, want stockham", entry.Algorithm)
	}

	// Verify third entry
	entry, found = wisdom.Lookup(planner.WisdomKey{Size: 512, Precision: 0, CPUFeatures: 456})
	if !found {
		t.Error("Failed to find third imported entry")
	} else if entry.Algorithm != "sixstep" {
		t.Errorf("Third entry algorithm = %s, want sixstep", entry.Algorithm)
	}

	// Verify fourth entry
	entry, found = wisdom.Lookup(planner.WisdomKey{Size: 1024, Precision: 1, CPUFeatures: 0})
	if !found {
		t.Error("Failed to find fourth imported entry")
	} else if entry.Algorithm != "bluestein" {
		t.Errorf("Fourth entry algorithm = %s, want bluestein", entry.Algorithm)
	}
}

func TestImportWisdomFromString(t *testing.T) {
	// Note: This uses the global DefaultWisdom, so we can't run in parallel
	// and need to clean up afterward

	// Save initial state
	initialLen := WisdomLen()

	wisdomData := `2048:0:0:test_algo:1234567890`

	err := ImportWisdomFromString(wisdomData)
	if err != nil {
		t.Fatalf("ImportWisdomFromString() failed: %v", err)
	}

	// Verify it was imported to global wisdom
	expectedLen := initialLen + 1
	if WisdomLen() != expectedLen {
		t.Errorf("Global wisdom length = %d, want %d after import", WisdomLen(), expectedLen)
	}

	// Clean up - remove our test entry
	// (We can't easily do this without clearing all wisdom, so we'll just note the pollution)
	// In a real scenario, this test would need isolation
}

func TestImportWisdom_InvalidFormat(t *testing.T) {
	t.Parallel()

	tmpDir := t.TempDir()
	filename := filepath.Join(tmpDir, "invalid_wisdom.txt")

	// Create wisdom file with invalid format
	invalidData := `128:0:0:dit64` // Missing timestamp field

	err := os.WriteFile(filename, []byte(invalidData), 0644)
	if err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	wisdom := NewWisdom()
	file, err := os.Open(filename)
	if err != nil {
		t.Fatalf("Failed to open file: %v", err)
	}
	defer file.Close()

	err = wisdom.Import(file)
	if err == nil {
		t.Error("Import() should fail with invalid format")
	}
}

func TestExportImportRoundTrip(t *testing.T) {
	t.Parallel()

	tmpDir := t.TempDir()
	filename := filepath.Join(tmpDir, "roundtrip_wisdom.txt")

	// Create original wisdom
	original := NewWisdom()

	entries := []planner.WisdomEntry{
		{
			Key:       planner.WisdomKey{Size: 64, Precision: 0, CPUFeatures: 0},
			Algorithm: "dit32",
			Timestamp: time.Unix(1000000000, 0),
		},
		{
			Key:       planner.WisdomKey{Size: 128, Precision: 0, CPUFeatures: 1},
			Algorithm: "dit64",
			Timestamp: time.Unix(1000000001, 0),
		},
		{
			Key:       planner.WisdomKey{Size: 256, Precision: 1, CPUFeatures: 2},
			Algorithm: "stockham",
			Timestamp: time.Unix(1000000002, 0),
		},
	}

	for _, entry := range entries {
		original.Store(entry)
	}

	// Export
	err := ExportWisdomTo(filename, original)
	if err != nil {
		t.Fatalf("ExportWisdomTo() failed: %v", err)
	}

	// Import to new wisdom
	imported := NewWisdom()
	file, err := os.Open(filename)
	if err != nil {
		t.Fatalf("Failed to open exported file: %v", err)
	}
	defer file.Close()

	err = imported.Import(file)
	if err != nil {
		t.Fatalf("Import() failed: %v", err)
	}

	// Verify all entries match
	if imported.Len() != original.Len() {
		t.Errorf("Imported wisdom length = %d, want %d", imported.Len(), original.Len())
	}

	for _, originalEntry := range entries {
		importedEntry, found := imported.Lookup(originalEntry.Key)
		if !found {
			t.Errorf("Entry not found after import: %+v", originalEntry.Key)
			continue
		}

		if importedEntry.Algorithm != originalEntry.Algorithm {
			t.Errorf("Algorithm mismatch: got %s, want %s", importedEntry.Algorithm, originalEntry.Algorithm)
		}

		if importedEntry.Timestamp.Unix() != originalEntry.Timestamp.Unix() {
			t.Errorf("Timestamp mismatch: got %v, want %v", importedEntry.Timestamp, originalEntry.Timestamp)
		}
	}
}

func TestExportWisdom_NonExistentPath(t *testing.T) {
	t.Parallel()

	wisdom := NewWisdom()

	// Try to export to invalid path
	err := ExportWisdomTo("/nonexistent/path/wisdom.txt", wisdom)
	if err == nil {
		t.Error("ExportWisdomTo() should fail with non-existent path")
	}
}

func TestImportWisdom_NonExistentFile(t *testing.T) {
	// Can't run in parallel because it uses global ImportWisdom

	err := ImportWisdom("/nonexistent/wisdom.txt")
	if err == nil {
		t.Error("ImportWisdom() should fail with non-existent file")
	}
}

func TestWisdom_ConcurrentAccess(t *testing.T) {
	t.Parallel()

	wisdom := NewWisdom()

	done := make(chan bool, 3)

	// Concurrent stores
	go func() {
		for i := 0; i < 100; i++ {
			wisdom.Store(planner.WisdomEntry{
				Key: planner.WisdomKey{
					Size:        128 + i,
					Precision:   0,
					CPUFeatures: 0,
				},
				Algorithm: "test1",
				Timestamp: time.Now(),
			})
		}
		done <- true
	}()

	// Concurrent lookups
	go func() {
		for i := 0; i < 100; i++ {
			_, _ = wisdom.Lookup(planner.WisdomKey{Size: 128 + i, Precision: 0, CPUFeatures: 0})
		}
		done <- true
	}()

	// Concurrent len checks
	go func() {
		for i := 0; i < 100; i++ {
			_ = wisdom.Len()
		}
		done <- true
	}()

	<-done
	<-done
	<-done

	// Verify wisdom is in a consistent state
	length := wisdom.Len()
	if length < 0 || length > 100 {
		t.Errorf("Wisdom length = %d after concurrent access, should be between 0 and 100", length)
	}
}

func TestExportWisdom_EmptyWisdom(t *testing.T) {
	t.Parallel()

	tmpDir := t.TempDir()
	filename := filepath.Join(tmpDir, "empty_wisdom.txt")

	wisdom := NewWisdom()

	err := ExportWisdomTo(filename, wisdom)
	if err != nil {
		t.Fatalf("ExportWisdomTo() failed for empty wisdom: %v", err)
	}

	// Read file - should be empty or have no entries
	data, err := os.ReadFile(filename)
	if err != nil {
		t.Fatalf("Failed to read exported file: %v", err)
	}

	content := strings.TrimSpace(string(data))
	if content != "" {
		t.Errorf("Empty wisdom exported content = %q, want empty", content)
	}
}

func TestImportWisdom_EmptyFile(t *testing.T) {
	t.Parallel()

	tmpDir := t.TempDir()
	filename := filepath.Join(tmpDir, "empty.txt")

	// Create empty file
	err := os.WriteFile(filename, []byte(""), 0644)
	if err != nil {
		t.Fatalf("Failed to create empty file: %v", err)
	}

	wisdom := NewWisdom()
	file, err := os.Open(filename)
	if err != nil {
		t.Fatalf("Failed to open file: %v", err)
	}
	defer file.Close()

	err = wisdom.Import(file)
	if err != nil {
		t.Fatalf("Import() failed for empty file: %v", err)
	}

	if wisdom.Len() != 0 {
		t.Errorf("Wisdom length = %d after importing empty file, want 0", wisdom.Len())
	}
}

func TestImportWisdom_GlobalFunction(t *testing.T) {
	// Note: Uses global DefaultWisdom, cannot run in parallel

	tmpDir := t.TempDir()
	filename := filepath.Join(tmpDir, "global_import_test.txt")

	// Create wisdom file
	wisdomData := `4096:0:0:global_test:1234567890`

	err := os.WriteFile(filename, []byte(wisdomData), 0644)
	if err != nil {
		t.Fatalf("Failed to create test wisdom file: %v", err)
	}

	// Save initial global wisdom length
	initialLen := WisdomLen()

	// Import using global function
	err = ImportWisdom(filename)
	if err != nil {
		t.Fatalf("ImportWisdom() failed: %v", err)
	}

	// Verify entry was added to global wisdom
	newLen := WisdomLen()
	if newLen <= initialLen {
		t.Errorf("Global wisdom length = %d, expected > %d after import", newLen, initialLen)
	}

	// Note: We leave the entry in global wisdom as cleanup is difficult
}

func TestExportWisdom_GlobalFunction(t *testing.T) {
	// Note: Uses global DefaultWisdom, cannot run in parallel

	tmpDir := t.TempDir()
	filename := filepath.Join(tmpDir, "global_export_test.txt")

	// Export global wisdom using the wrapper function
	err := ExportWisdom(filename)
	if err != nil {
		t.Fatalf("ExportWisdom() failed: %v", err)
	}

	// Verify file was created
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		t.Fatal("ExportWisdom() did not create file")
	}

	// File should exist and be readable
	data, err := os.ReadFile(filename)
	if err != nil {
		t.Fatalf("Failed to read exported file: %v", err)
	}

	// Should have some content if global wisdom has entries
	// (may be empty if no global entries exist)
	_ = data
}

func TestClearWisdom_GlobalFunction(t *testing.T) {
	// Note: This test modifies global state, so it cannot run in parallel

	// Save current state by reading all entries first
	// (This is the best we can do without being able to restore state)

	// Add a known entry
	err := ImportWisdomFromString("8192:0:0:test_clear:1234567890")
	if err != nil {
		t.Fatalf("ImportWisdomFromString() failed: %v", err)
	}

	lengthBefore := WisdomLen()

	// Call ClearWisdom
	ClearWisdom()

	// Verify it was cleared
	lengthAfter := WisdomLen()
	if lengthAfter != 0 {
		t.Errorf("WisdomLen() = %d after ClearWisdom(), want 0", lengthAfter)
	}

	// Note: This will affect other tests that rely on global wisdom,
	// but since we're testing the global function, we need to accept this
	_ = lengthBefore
}

func TestWisdomLen_GlobalFunction(t *testing.T) {
	// Note: Uses global DefaultWisdom, cannot run in parallel

	// Verify WisdomLen returns a non-negative value
	length := WisdomLen()
	if length < 0 {
		t.Errorf("WisdomLen() = %d, should be >= 0", length)
	}
}
