package algofft

import (
	"fmt"
	"os"
	"strings"

	"github.com/cwbudde/algo-fft/internal/fft"
)

// ImportWisdom loads wisdom data from a file.
// The file should be in the format produced by ExportWisdom.
func ImportWisdom(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open wisdom file: %w", err)
	}

	defer file.Close()

	err = fft.DefaultWisdom.Import(file)
	if err != nil {
		return fmt.Errorf("failed to import wisdom: %w", err)
	}

	return nil
}

// ExportWisdom saves the current wisdom cache to a file.
// The file can be loaded later with ImportWisdom.
func ExportWisdom(filename string) error {
	return ExportWisdomTo(filename, fft.DefaultWisdom)
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

// Wisdom is a type alias for the internal wisdom cache.
// It provides the WisdomStore interface for storing and retrieving
// optimal kernel choices.
type Wisdom = fft.Wisdom

// NewWisdom creates a new empty wisdom cache.
func NewWisdom() *Wisdom {
	return fft.NewWisdom()
}

// ImportWisdomFromString loads wisdom data from a string.
// This is useful for embedding wisdom data in compiled binaries.
func ImportWisdomFromString(data string) error {
	err := fft.DefaultWisdom.Import(strings.NewReader(data))
	if err != nil {
		return fmt.Errorf("failed to import wisdom from string: %w", err)
	}

	return nil
}

// ClearWisdom removes all entries from the wisdom cache.
func ClearWisdom() {
	fft.DefaultWisdom.Clear()
}

// WisdomLen returns the number of entries in the wisdom cache.
func WisdomLen() int {
	return fft.DefaultWisdom.Len()
}
