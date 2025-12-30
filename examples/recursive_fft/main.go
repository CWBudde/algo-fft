// Package main demonstrates using the recursive FFT decomposition strategy.
//
// The recursive FFT implementation splits large transforms into smaller
// codelet-sized sub-problems, enabling better cache utilization and
// SIMD codelet reuse.
package main

import (
	"fmt"
	"math"

	algofft "github.com/MeKo-Christian/algo-fft"
)

func main() {
	fmt.Println("Recursive FFT Decomposition Example")
	fmt.Println("====================================")
	fmt.Println()

	// Set the global kernel strategy to use recursive decomposition
	algofft.SetKernelStrategy(algofft.KernelRecursive)

	// Create a plan for 8192-point FFT
	size := 8192

	plan, err := algofft.NewPlan32(size)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Created plan for size %d\n", plan.Len())
	fmt.Printf("Strategy: %v (Recursive)\n", plan.KernelStrategy())
	fmt.Printf("Algorithm: %s\n\n", plan.Algorithm())

	// Generate a test signal: sum of two sine waves
	input := make([]complex64, size)
	for i := range input {
		t := float64(i) / float64(size)
		// 10 Hz + 50 Hz sine waves
		signal := math.Sin(2*math.Pi*10*t) + 0.5*math.Sin(2*math.Pi*50*t)
		input[i] = complex(float32(signal), 0)
	}

	// Compute forward FFT
	output := make([]complex64, size)

	err = plan.Forward(output, input)
	if err != nil {
		panic(err)
	}

	// Find the two strongest frequency components
	type peak struct {
		index int
		mag   float32
	}

	peaks := make([]peak, 0)

	for i := range size / 2 {
		mag := float32(math.Sqrt(float64(real(output[i])*real(output[i]) + imag(output[i])*imag(output[i]))))
		if mag > 100 { // Threshold to filter noise
			peaks = append(peaks, peak{i, mag})
		}
	}

	fmt.Println("Detected frequency peaks:")

	for _, p := range peaks {
		freq := float64(p.index) // Frequency bin (assuming sample rate = size)
		fmt.Printf("  Bin %4d: magnitude %.1f (frequency %.1f Hz)\n", p.index, p.mag, freq)
	}

	// Verify inverse transform recovers original signal
	inverse := make([]complex64, size)

	err = plan.Inverse(inverse, output)
	if err != nil {
		panic(err)
	}

	// Calculate round-trip error
	maxError := float32(0)

	for i := range input {
		diff := inverse[i] - input[i]

		errMag := float32(math.Sqrt(float64(real(diff)*real(diff) + imag(diff)*imag(diff))))
		if errMag > maxError {
			maxError = errMag
		}
	}

	fmt.Printf("\nRound-trip accuracy:\n")
	fmt.Printf("  Max error: %.2e (relative: %.2e)\n", maxError, maxError/float32(size))

	if maxError < 1e-3 {
		fmt.Println("  ✓ Inverse FFT successfully recovered original signal!")
	} else {
		fmt.Println("  ✗ Warning: Round-trip error exceeds threshold")
	}

	// Demonstrate performance advantage: recursive decomposition reuses codelets
	fmt.Println("\nRecursive decomposition benefits:")
	fmt.Println("  • Reuses SIMD-optimized codelets (sizes 512, 256, etc.)")
	fmt.Println("  • Better cache locality (small sub-problems fit in L1/L2)")
	fmt.Println("  • Fewer stages than pure radix-2 (radix-4/8 combines)")
	fmt.Println("  • Zero allocations after plan creation")
}
