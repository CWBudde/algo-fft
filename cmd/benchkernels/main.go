package main

import (
	"flag"
	"fmt"
	"math/rand"
	"runtime"
	"sort"
	"strings"
	"time"

	algofft "github.com/MeKo-Christian/algo-fft"
	"github.com/MeKo-Christian/algo-fft/internal/cpu"
	"github.com/MeKo-Christian/algo-fft/internal/fft"
)

const modeInverse = "inverse"

type benchResult struct {
	size     int
	strategy algofft.KernelStrategy
	nsPerOp  float64
}

func main() {
	var (
		sizeList   = flag.String("sizes", "1024,4096,16384,65536", "comma-separated sizes")
		iters      = flag.Int("iters", 50, "benchmark iterations")
		warmup     = flag.Int("warmup", 5, "warmup iterations")
		emit       = flag.Bool("emit", false, "emit RecordBenchmarkDecision lines")
		wisdomFile = flag.String("wisdom", "", "export wisdom to file (portable format)")
		mode       = flag.String("mode", "forward", "benchmark mode: forward, inverse, roundtrip, all")
		seed       = flag.Int64("seed", 1, "rng seed")
	)
	flag.Parse()

	sizes := parseSizes(*sizeList)
	if len(sizes) == 0 {
		fmt.Println("no sizes specified")
		return
	}

	rnd := rand.New(rand.NewSource(*seed))

	algofft.SetKernelStrategy(algofft.KernelAuto)
	defer algofft.SetKernelStrategy(algofft.KernelAuto)

	fmt.Printf("iters=%d warmup=%d\n", *iters, *warmup)
	fmt.Printf("%8s  %10s  %12s  %12s\n", "size", "mode", "kernel", "ns/op")

	// Collect best results for wisdom export
	var bestResults []benchResult

	for _, n := range sizes {
		modes := resolveModes(*mode)
		for _, runMode := range modes {
			results := benchmarkSize(rnd, n, *iters, *warmup, runMode)
			if len(results) == 0 {
				continue
			}

			sort.Slice(results, func(i, j int) bool {
				return results[i].nsPerOp < results[j].nsPerOp
			})

			for _, res := range results {
				fmt.Printf("%8d  %10s  %12s  %12.1f\n", n, runMode, strategyName(res.strategy), res.nsPerOp)
			}

			if runMode == "forward" {
				best := results[0]
				best.size = n
				bestResults = append(bestResults, best)

				if *emit {
					fmt.Printf("algofft.RecordBenchmarkDecision(%d, algofft.%s)\n", n, strategyConst(best.strategy))
				}
			}
		}
	}

	// Export wisdom if requested
	if *wisdomFile != "" {
		err := exportWisdom(*wisdomFile, bestResults)
		if err != nil {
			fmt.Printf("error exporting wisdom: %v\n", err)
			return
		}

		fmt.Printf("\nWisdom exported to: %s\n", *wisdomFile)
	}
}

//nolint:cyclop
func benchmarkSize(rnd *rand.Rand, n, iters, warmup int, mode string) []benchResult {
	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(rnd.Float32(), rnd.Float32())
	}

	dst := make([]complex64, n)
	freq := make([]complex64, n)

	strategies := []algofft.KernelStrategy{
		algofft.KernelDIT,
		algofft.KernelStockham,
		algofft.KernelSixStep,
		algofft.KernelEightStep,
	}

	results := make([]benchResult, 0, len(strategies))

	for _, strategy := range strategies {
		algofft.SetKernelStrategy(strategy)

		plan, err := algofft.NewPlanT[complex64](n)
		if err != nil {
			continue
		}

		ok := true

		if mode == modeInverse {
			err := plan.Forward(freq, src)
			if err != nil {
				continue
			}
		}

		for range warmup {
			err := runPlanMode(plan, dst, src, freq, mode)
			if err != nil {
				ok = false
				break
			}
		}

		if !ok {
			continue
		}

		runtime.GC()

		start := cpu.ReadCycleCounter()

		for range iters {
			err := runPlanMode(plan, dst, src, freq, mode)
			if err != nil {
				ok = false
				break
			}
		}

		if !ok {
			continue
		}

		elapsedCycles := cpu.CyclesSince(start)
		elapsedNanos := cpu.CyclesToNanoseconds(elapsedCycles)

		results = append(results, benchResult{
			strategy: strategy,
			nsPerOp:  float64(elapsedNanos) / float64(iters),
		})
	}

	algofft.SetKernelStrategy(algofft.KernelAuto)

	return results
}

func runPlanMode(plan *algofft.Plan[complex64], dst, src, freq []complex64, mode string) error {
	switch mode {
	case modeInverse:
		err := plan.Inverse(dst, freq)
		if err != nil {
			return fmt.Errorf("plan inverse: %w", err)
		}

		return nil
	case "roundtrip":
		err := plan.Forward(freq, src)
		if err != nil {
			return fmt.Errorf("plan forward: %w", err)
		}

		err = plan.Inverse(dst, freq)
		if err != nil {
			return fmt.Errorf("plan inverse: %w", err)
		}

		return nil
	default:
		err := plan.Forward(dst, src)
		if err != nil {
			return fmt.Errorf("plan forward: %w", err)
		}

		return nil
	}
}

func resolveModes(mode string) []string {
	switch mode {
	case "all":
		return []string{"forward", "inverse", "roundtrip"}
	case "inverse", "roundtrip", "forward":
		return []string{mode}
	default:
		return []string{"forward"}
	}
}

func parseSizes(list string) []int {
	parts := strings.Split(list, ",")

	out := make([]int, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		var n int

		_, err := fmt.Sscanf(part, "%d", &n)
		if err != nil || n <= 0 {
			continue
		}

		out = append(out, n)
	}

	return out
}

func strategyName(strategy algofft.KernelStrategy) string {
	switch strategy {
	case algofft.KernelDIT:
		return "DIT"
	case algofft.KernelStockham:
		return "Stockham"
	case algofft.KernelSixStep:
		return "SixStep"
	case algofft.KernelEightStep:
		return "EightStep"
	default:
		return "Auto"
	}
}

func strategyConst(strategy algofft.KernelStrategy) string {
	switch strategy {
	case algofft.KernelDIT:
		return "KernelDIT"
	case algofft.KernelStockham:
		return "KernelStockham"
	case algofft.KernelSixStep:
		return "KernelSixStep"
	case algofft.KernelEightStep:
		return "KernelEightStep"
	default:
		return "KernelAuto"
	}
}

// exportWisdom writes benchmark results to a wisdom file.
func exportWisdom(filename string, results []benchResult) error {
	wisdom := fft.NewWisdom()
	features := cpu.DetectFeatures()
	cpuMask := fft.CPUFeatureMask(
		features.HasSSE2,
		features.HasAVX2,
		features.HasAVX512,
		features.HasNEON,
	)

	for _, res := range results {
		entry := fft.WisdomEntry{
			Key: fft.WisdomKey{
				Size:        res.size,
				Precision:   fft.PrecisionComplex64, // benchkernels uses complex64
				CPUFeatures: cpuMask,
			},
			Algorithm: strategyToAlgorithmName(res.strategy),
			Timestamp: time.Now(),
		}
		wisdom.Store(entry)
	}

	// Use internal wisdom directly since algofft.Wisdom is a type alias
	err := algofft.ExportWisdomTo(filename, wisdom)
	if err != nil {
		return fmt.Errorf("export wisdom to %s: %w", filename, err)
	}

	return nil
}

// strategyToAlgorithmName converts strategy to the algorithm name used in wisdom files.
func strategyToAlgorithmName(strategy algofft.KernelStrategy) string {
	switch strategy {
	case algofft.KernelDIT:
		return "dit_fallback"
	case algofft.KernelStockham:
		return "stockham"
	case algofft.KernelSixStep:
		return "sixstep"
	case algofft.KernelEightStep:
		return "eightstep"
	default:
		return "unknown"
	}
}
