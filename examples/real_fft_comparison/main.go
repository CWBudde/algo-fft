package main

import (
	"fmt"
	"math"

	algofft "github.com/cwbudde/algo-fft"
)

//nolint:cyclop
func main() {
	fmt.Println("algofft - Real FFT Precision Comparison")
	fmt.Println("========================================")
	fmt.Println()

	const n = 4096

	// Example 1: Float32 Real FFT (backward compatible)
	fmt.Println("Example 1: Float32 Real FFT (standard precision)")
	fmt.Println("-------------------------------------------------")

	plan32, err := algofft.NewPlanReal32(n) // or NewPlanReal(n)
	if err != nil {
		panic(err)
	}

	input32 := make([]float32, n)
	for i := range input32 {
		input32[i] = float32(math.Sin(2 * math.Pi * float64(i) / float64(n) * 5))
	}

	spectrum32 := make([]complex64, plan32.SpectrumLen())

	err = plan32.Forward(spectrum32, input32)
	if err != nil {
		panic(err)
	}

	recovered32 := make([]float32, n)

	err = plan32.Inverse(recovered32, spectrum32)
	if err != nil {
		panic(err)
	}

	error32 := computeError32(input32, recovered32)

	fmt.Printf("  Precision: ~7 decimal digits\n")
	fmt.Printf("  Round-trip error: %.6e\n", error32)
	fmt.Printf("  Memory: spectrum uses %.1f KB\n", float64(len(spectrum32)*8)/1024)
	fmt.Println()

	// Example 2: Float64 Real FFT (high precision)
	fmt.Println("Example 2: Float64 Real FFT (high precision)")
	fmt.Println("----------------------------------------------")

	plan64, err := algofft.NewPlanReal64(n)
	if err != nil {
		panic(err)
	}

	input64 := make([]float64, n)
	for i := range input64 {
		input64[i] = math.Sin(2 * math.Pi * float64(i) / float64(n) * 5)
	}

	spectrum64 := make([]complex128, plan64.SpectrumLen())

	err = plan64.Forward(spectrum64, input64)
	if err != nil {
		panic(err)
	}

	recovered64 := make([]float64, n)

	err = plan64.Inverse(recovered64, spectrum64)
	if err != nil {
		panic(err)
	}

	error64 := computeError64(input64, recovered64)

	fmt.Printf("  Precision: ~15 decimal digits\n")
	fmt.Printf("  Round-trip error: %.15e\n", error64)
	fmt.Printf("  Memory: spectrum uses %.1f KB\n", float64(len(spectrum64)*16)/1024)
	fmt.Println()

	// Example 3: Generic API (type-safe)
	fmt.Println("Example 3: Generic API (type-safe)")
	fmt.Println("-----------------------------------")

	// Explicit type parameters
	planGeneric, err := algofft.NewPlanRealT[float64, complex128](n)
	if err != nil {
		panic(err)
	}

	spectrumGeneric := make([]complex128, planGeneric.SpectrumLen())

	err = planGeneric.Forward(spectrumGeneric, input64)
	if err != nil {
		panic(err)
	}

	fmt.Printf("  Type-safe: compiler enforces float64 → complex128\n")
	fmt.Printf("  Same performance as explicit constructor\n")
	fmt.Println()

	// Summary
	fmt.Println("Summary")
	fmt.Println("-------")
	fmt.Printf("  Precision improvement: %.1fx\n", error32/float32(error64))
	fmt.Printf("  Memory overhead: %.1fx\n", 16.0/8.0)
	fmt.Println()

	fmt.Println("When to use float32:")
	fmt.Println("  ✓ Standard audio processing")
	fmt.Println("  ✓ Real-time applications")
	fmt.Println("  ✓ Memory-constrained systems")
	fmt.Println()

	fmt.Println("When to use float64:")
	fmt.Println("  ✓ Scientific computing")
	fmt.Println("  ✓ High-precision measurements")
	fmt.Println("  ✓ Error-sensitive calculations")
}

func computeError32(a, b []float32) float32 {
	maxErr := float32(0.0)

	for i := range a {
		err := abs32(a[i] - b[i])
		if err > maxErr {
			maxErr = err
		}
	}

	return maxErr
}

func computeError64(a, b []float64) float64 {
	maxErr := 0.0

	for i := range a {
		err := math.Abs(a[i] - b[i])
		if err > maxErr {
			maxErr = err
		}
	}

	return maxErr
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}

	return x
}
