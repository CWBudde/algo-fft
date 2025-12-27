package main

import (
	"fmt"
	"time"

	"github.com/MeKo-Christian/algofft"
	"gonum.org/v1/gonum/dsp/fourier"
)

func main() {
	sizes := []int{8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}
	iterations := 10000

	fmt.Println("algo-fft vs gonum FFT Performance Comparison")
	fmt.Println("==============================================")
	fmt.Println()
	fmt.Printf("%-10s %-15s %-15s %-10s\n", "Size", "algo-fft", "gonum", "Speedup")
	fmt.Println("----------------------------------------------------------")

	for _, n := range sizes {
		iters := iterations
		if n >= 2048 {
			iters = 1000
		}
		if n >= 8192 {
			iters = 100
		}

		algoFftTime := benchmarkAlgoFft(n, iters)
		gonumTime := benchmarkGonum(n, iters)
		speedup := float64(gonumTime) / float64(algoFftTime)

		fmt.Printf("%-10d %-15s %-15s %.2fx\n",
			n,
			formatDuration(algoFftTime),
			formatDuration(gonumTime),
			speedup)
	}
}

func benchmarkAlgoFft(n, iterations int) time.Duration {
	plan, err := algofft.NewPlan(n)
	if err != nil {
		panic(err)
	}

	src := make([]complex64, n)
	dst := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i+1), float32(-i))
	}

	start := time.Now()
	for i := 0; i < iterations; i++ {
		if err := plan.Forward(dst, src); err != nil {
			panic(err)
		}
	}
	return time.Since(start) / time.Duration(iterations)
}

func benchmarkGonum(n, iterations int) time.Duration {
	fft := fourier.NewCmplxFFT(n)

	src := make([]complex128, n)
	for i := range src {
		src[i] = complex(float64(i+1), float64(-i))
	}

	start := time.Now()
	for i := 0; i < iterations; i++ {
		_ = fft.Coefficients(nil, src)
	}
	return time.Since(start) / time.Duration(iterations)
}

func formatDuration(d time.Duration) string {
	if d < time.Microsecond {
		return fmt.Sprintf("%d ns", d.Nanoseconds())
	}
	if d < time.Millisecond {
		return fmt.Sprintf("%.2f µs", float64(d.Nanoseconds())/1000.0)
	}
	return fmt.Sprintf("%.2f ms", float64(d.Nanoseconds())/1000000.0)
}
