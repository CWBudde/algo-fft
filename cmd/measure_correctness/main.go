package main

import (
	"fmt"
	"math/cmplx"
	"math/rand"

	algofft "github.com/MeKo-Christian/algo-fft"
	"github.com/cwbudde/algo-fft/internal/reference"
)

func main() {
	sizes := []int{8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}
	numTrials := 100

	fmt.Println("Correctness Validation: Max Relative Error vs Reference DFT")
	fmt.Println("============================================================")
	fmt.Println()
	fmt.Printf("Testing %d random vectors per size\n\n", numTrials)
	fmt.Printf("%-10s %-20s %-20s\n", "Size", "complex64 Max Err", "complex128 Max Err")
	fmt.Println("------------------------------------------------------------")

	for _, n := range sizes {
		maxErr32 := measureMaxError32(n, numTrials)
		maxErr64 := measureMaxError64(n, numTrials)
		fmt.Printf("%-10d %-20.2e %-20.2e\n", n, maxErr32, maxErr64)
	}
}

func measureMaxError32(n, trials int) float64 {
	plan, err := algofft.NewPlan(n)
	if err != nil {
		panic(err)
	}

	maxRelErr := 0.0
	rng := rand.New(rand.NewSource(42))

	for trial := 0; trial < trials; trial++ {
		// Generate random input
		src := make([]complex64, n)
		for i := range src {
			src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
		}

		// Compute FFT using algo-fft
		got := make([]complex64, n)
		if err := plan.Forward(got, src); err != nil {
			panic(err)
		}

		// Compute reference DFT
		want := reference.NaiveDFT(src)

		// Compute max relative error
		for i := range got {
			refMag := cmplx.Abs(complex128(want[i]))
			if refMag < 1e-10 {
				continue // Skip near-zero values
			}
			diff := cmplx.Abs(complex128(got[i] - want[i]))
			relErr := diff / refMag
			if relErr > maxRelErr {
				maxRelErr = relErr
			}
		}
	}

	return maxRelErr
}

func measureMaxError64(n, trials int) float64 {
	plan, err := algofft.NewPlan64(n)
	if err != nil {
		panic(err)
	}

	maxRelErr := 0.0
	rng := rand.New(rand.NewSource(42))

	for trial := 0; trial < trials; trial++ {
		// Generate random input
		src := make([]complex128, n)
		for i := range src {
			src[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
		}

		// Compute FFT using algo-fft
		got := make([]complex128, n)
		if err := plan.Forward(got, src); err != nil {
			panic(err)
		}

		// Compute reference DFT
		want := reference.NaiveDFT128(src)

		// Compute max relative error
		for i := range got {
			refMag := cmplx.Abs(want[i])
			if refMag < 1e-15 {
				continue // Skip near-zero values
			}
			diff := cmplx.Abs(got[i] - want[i])
			relErr := diff / refMag
			if relErr > maxRelErr {
				maxRelErr = relErr
			}
		}
	}

	return maxRelErr
}
