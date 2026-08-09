package main

import (
	"flag"
	"fmt"
	"math/rand"
	"runtime"
	"sort"
	"strings"
	"time"

	algofft "github.com/cwbudde/algo-fft"
	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/planner"
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
				fmt.Printf("%8d  %10s  %12s  %12.1f\n", n, runMode, kernelStrategyLabel(res.strategy), res.nsPerOp)
			}

			if runMode == "forward" {
				best := results[0]
				best.size = n
				bestResults = append(bestResults, best)
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
		algofft.KernelFourStep,
	}

	results := make([]benchResult, 0, len(strategies))

	// Results are labeled by the strategy the plan resolved to, not the one
	// requested: at a non-power-of-two length every request above resolves to
	// the mixed-radix engine, so timing them all would report one transform
	// five times under four names that never run.
	seen := make(map[algofft.KernelStrategy]bool, len(strategies))

	for _, strategy := range strategies {
		plan, err := algofft.NewPlanWithOptions[complex64](n, algofft.PlanOptions{Strategy: strategy})
		if err != nil {
			continue
		}

		resolved := plan.KernelStrategy()
		if seen[resolved] {
			continue
		}

		seen[resolved] = true

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
			strategy: resolved,
			nsPerOp:  float64(elapsedNanos) / float64(iters),
		})
	}

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

// exportWisdom writes benchmark results to a wisdom file.
func exportWisdom(filename string, results []benchResult) error {
	wisdom := algofft.NewWisdom()
	features := cpu.DetectFeatures()
	cpuMask := planner.CPUFeatureMask(
		features.HasSSE2,
		features.HasSSE3,
		features.HasAVX2,
		features.HasAVX512,
		features.HasNEON,
	)

	for _, res := range results {
		entry := algofft.WisdomEntry{
			Key: algofft.WisdomKey{
				Size:          res.size,
				Precision:     uint8(algofft.PrecisionComplex64), // benchkernels uses complex64
				CPUFeatures:   cpuMask,
				CPUIdentifier: cpu.WisdomCPUIdentifier(features),
			},
			Algorithm: strategyToAlgorithmName(res.strategy),
			Timestamp: time.Now(),
		}
		wisdom.Store(entry)
	}

	err := algofft.ExportWisdomTo(filename, wisdom)
	if err != nil {
		return fmt.Errorf("export wisdom to %s: %w", filename, err)
	}

	return nil
}

// kernelStrategyLabel names a strategy for display, via the public String().
func kernelStrategyLabel(strategy algofft.KernelStrategy) string {
	return strategy.String()
}

// strategyToAlgorithmName converts strategy to the algorithm name used in
// wisdom files. The names must match the planner's strategy↔algorithm-name
// table (internal/planner/utils.go); benchkernels operates on the public
// enum, so it carries its own copy of the strategies it benchmarks.
func strategyToAlgorithmName(strategy algofft.KernelStrategy) string {
	switch strategy {
	case algofft.KernelDIT:
		return "dit_fallback"
	case algofft.KernelStockham:
		return "stockham"
	case algofft.KernelSixStep:
		return "sixstep"
	case algofft.KernelBluestein:
		return "bluestein"
	case algofft.KernelSplitRadix:
		return "splitradix"
	case algofft.KernelRecursive:
		return "recursive"
	case algofft.KernelFourStep:
		return "fourstep"
	case algofft.KernelMixedRadix:
		return "mixedradix"
	case algofft.KernelAuto:
		return "unknown"
	default:
		return "unknown"
	}
}
