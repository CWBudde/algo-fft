package algoforge

import (
	"os"
	"strings"

	"github.com/MeKo-Christian/algoforge/internal/fft"
)

// ImportWisdom loads wisdom data from a file.
// The file should be in the format produced by ExportWisdom.
func ImportWisdom(filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		return err
	}

	defer f.Close()

	return fft.DefaultWisdom.Import(f)
}

// ExportWisdom saves the current wisdom cache to a file.
// The file can be loaded later with ImportWisdom.
func ExportWisdom(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}

	defer f.Close()

	return fft.DefaultWisdom.Export(f)
}

// ImportWisdomFromString loads wisdom data from a string.
// This is useful for embedding wisdom data in compiled binaries.
func ImportWisdomFromString(data string) error {
	return fft.DefaultWisdom.Import(strings.NewReader(data))
}

// ClearWisdom removes all entries from the wisdom cache.
func ClearWisdom() {
	fft.DefaultWisdom.Clear()
}

// WisdomLen returns the number of entries in the wisdom cache.
func WisdomLen() int {
	return fft.DefaultWisdom.Len()
}
