package main

import (
	"fmt"
	"math"

	algofft "github.com/cwbudde/algo-fft"
)

//nolint:cyclop
func main() {
	// Example: High-precision audio processing with float64
	const sampleRate = 48000 // 48 kHz

	const fftSize = 4096

	fmt.Println("algofft - High-Precision Real FFT Example (float64)")
	fmt.Println("===================================================")
	fmt.Println()

	// Create a float64 real FFT plan
	plan, err := algofft.NewPlanReal64(fftSize)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Created plan: size=%d, spectrum length=%d\n", plan.Len(), plan.SpectrumLen())
	fmt.Println()

	// Generate a test signal: 440 Hz sine wave (A4 musical note)
	input := make([]float64, fftSize)
	freq := 440.0 // Hz

	for i := range input {
		t := float64(i) / float64(sampleRate)
		input[i] = math.Sin(2 * math.Pi * freq * t)
	}

	// Perform forward FFT
	spectrum := make([]complex128, plan.SpectrumLen())

	err = plan.Forward(spectrum, input)
	if err != nil {
		panic(err)
	}

	// Find the peak frequency
	maxMagnitude := 0.0
	peakBin := 0

	for k, value := range spectrum {
		magnitude := math.Sqrt(real(value)*real(value) + imag(value)*imag(value))
		if magnitude > maxMagnitude {
			maxMagnitude = magnitude
			peakBin = k
		}
	}

	peakFreq := float64(peakBin) * float64(sampleRate) / float64(fftSize)
	fmt.Printf("Peak found at bin %d (%.2f Hz)\n", peakBin, peakFreq)
	fmt.Printf("Expected frequency: %.2f Hz\n", freq)
	fmt.Printf("Error: %.6f Hz\n", math.Abs(peakFreq-freq))
	fmt.Println()

	// Perform inverse FFT to reconstruct signal
	recovered := make([]float64, fftSize)

	err = plan.Inverse(recovered, spectrum)
	if err != nil {
		panic(err)
	}

	// Measure round-trip error
	maxError := 0.0

	for i := range input {
		err := math.Abs(recovered[i] - input[i])
		if err > maxError {
			maxError = err
		}
	}

	fmt.Printf("Round-trip reconstruction:\n")
	fmt.Printf("  Max error: %.15e\n", maxError)
	fmt.Printf("  Precision: ~%.1f decimal digits\n", -math.Log10(maxError))
	fmt.Println()

	// Compare with float32 precision
	plan32, _ := algofft.NewPlanReal32(fftSize)

	input32 := make([]float32, fftSize)
	for i := range input {
		input32[i] = float32(input[i])
	}

	spectrum32 := make([]complex64, plan32.SpectrumLen())
	_ = plan32.Forward(spectrum32, input32)

	recovered32 := make([]float32, fftSize)
	_ = plan32.Inverse(recovered32, spectrum32)

	maxError32 := float32(0.0)

	for i := range input32 {
		err := abs32(recovered32[i] - input32[i])
		if err > maxError32 {
			maxError32 = err
		}
	}

	fmt.Printf("Precision comparison:\n")
	fmt.Printf("  float32 error: %.6e (~%.1f digits)\n", maxError32, -math.Log10(float64(maxError32)))
	fmt.Printf("  float64 error: %.15e (~%.1f digits)\n", maxError, -math.Log10(maxError))
	fmt.Printf("  Improvement: %.1fx better precision\n", float64(maxError32)/maxError)
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}

	return x
}
