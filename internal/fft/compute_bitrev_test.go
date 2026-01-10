package fft

import (
	"fmt"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/math"
)

func TestComputeBitRevForMigration(t *testing.T) {
	// Size 4 Radix-4
	indices4 := math.ComputePermutationIndices(4, 4)
	fmt.Println("Size 4 Radix-4:", indices4)

	// Size 8 Radix-4 (Mixed24)
	const mixed24 = -24
	indices8 := math.ComputePermutationIndices(8, mixed24)
	fmt.Println("Size 8 Radix-4 (Mixed24):", indices8)

	// Size 16 Radix-4
	indices16 := math.ComputePermutationIndices(16, 4)
	fmt.Println("Size 16 Radix-4:", indices16)

	// Size 64 Radix-4
	indices64 := math.ComputePermutationIndices(64, 4)
	fmt.Println("Size 64 Radix-4:", indices64)

	// Size 256 Radix-4
	indices256 := math.ComputePermutationIndices(256, 4)
	fmt.Println("Size 256 Radix-4:", indices256)
}
