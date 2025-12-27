package main

import (
	"flag"
	"fmt"
	"math/rand"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/MeKo-Christian/algoforge"
	"github.com/MeKo-Christian/algoforge/internal/cpu"
	"github.com/MeKo-Christian/algoforge/internal/fft"
)

const modeInverse = "inverse"

type benchResult struct {
	size     int
	strategy algoforge.KernelStrategy
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

	algoforge.SetKernelStrategy(algoforge.KernelAuto)
	defer algoforge.SetKernelStrategy(algoforge.KernelAuto)

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
					fmt.Printf("algoforge.RecordBenchmarkDecision(%d, algoforge.%s)\n", n, strategyConst(best.strategy))
				}
			}
		}
	}

	// Export wisdom if requested
	if *wisdomFile != "" {
		if err := exportWisdom(*wisdomFile, bestResults); err != nil {
			fmt.Printf("error exporting wisdom: %v\n", err)
			return
		}

		fmt.Printf("\nWisdom exported to: %s\n", *wisdomFile)
	}
}

func benchmarkSize(rnd *rand.Rand, n, iters, warmup int, mode string) []benchResult {
	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(rnd.Float32(), rnd.Float32())
	}

	dst := make([]complex64, n)
	freq := make([]complex64, n)

	strategies := []algoforge.KernelStrategy{
		algoforge.KernelDIT,
		algoforge.KernelStockham,
		algoforge.KernelSixStep,
		algoforge.KernelEightStep,
	}

	results := make([]benchResult, 0, len(strategies))

	for _, strategy := range strategies {
		algoforge.SetKernelStrategy(strategy)

		plan, err := algoforge.NewPlanT[complex64](n)
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

		start := time.Now()

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

		elapsed := time.Since(start)

		results = append(results, benchResult{
			strategy: strategy,
			nsPerOp:  float64(elapsed.Nanoseconds()) / float64(iters),
		})
	}

	algoforge.SetKernelStrategy(algoforge.KernelAuto)

	return results
}

func runPlanMode(plan *algoforge.Plan[complex64], dst, src, freq []complex64, mode string) error {
	switch mode {
	case modeInverse:
		return plan.Inverse(dst, freq)
	case "roundtrip":
		err := plan.Forward(freq, src)
		if err != nil {
			return err
		}

		return plan.Inverse(dst, freq)
	default:
		return plan.Forward(dst, src)
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

func strategyName(strategy algoforge.KernelStrategy) string {
	switch strategy {
	case algoforge.KernelDIT:
		return "DIT"
	case algoforge.KernelStockham:
		return "Stockham"
	case algoforge.KernelSixStep:
		return "SixStep"
	case algoforge.KernelEightStep:
		return "EightStep"
	default:
		return "Auto"
	}
}

func strategyConst(strategy algoforge.KernelStrategy) string {
	switch strategy {
	case algoforge.KernelDIT:
		return "KernelDIT"
	case algoforge.KernelStockham:
		return "KernelStockham"
	case algoforge.KernelSixStep:
		return "KernelSixStep"
	case algoforge.KernelEightStep:
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

	// Use internal wisdom directly since algoforge.Wisdom is a type alias
	return algoforge.ExportWisdomTo(filename, wisdom)
}

// strategyToAlgorithmName converts strategy to the algorithm name used in wisdom files.
func strategyToAlgorithmName(strategy algoforge.KernelStrategy) string {
	switch strategy {
	case algoforge.KernelDIT:
		return "dit_fallback"
	case algoforge.KernelStockham:
		return "stockham"
	case algoforge.KernelSixStep:
		return "sixstep"
	case algoforge.KernelEightStep:
		return "eightstep"
	default:
		return "unknown"
	}
}
