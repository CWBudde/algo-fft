// Command tune measures algofft's registered implementations on the current
// host and writes the winners to a Wisdom file.
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	algofft "github.com/cwbudde/algo-fft"
)

const (
	precisionAll = "all"
	precision32  = "complex64"
	precision64  = "complex128"
)

func main() {
	var (
		minSize   = flag.Int("min", 8, "smallest power-of-two size")
		maxSize   = flag.Int("max", 32768, "largest power-of-two size")
		output    = flag.String("output", "algofft-wisdom.txt", "output Wisdom file")
		effort    = flag.String("effort", "patient", "planning effort: patient or exhaustive")
		precision = flag.String("precision", precisionAll, "precision: all, complex64, or complex128")
	)

	flag.Parse()

	mode, err := parseEffort(*effort)
	if err != nil {
		fatalf("%v", err)
	}

	precisions, err := parsePrecisions(*precision)
	if err != nil {
		fatalf("%v", err)
	}

	sizes, err := powerOfTwoSizes(*minSize, *maxSize)
	if err != nil {
		fatalf("%v", err)
	}

	wisdom := algofft.NewWisdom()
	fmt.Printf("tuning %d sizes with %s effort\n", len(sizes), strings.ToLower(*effort))
	fmt.Printf("%8s  %-10s  %-32s  %12s\n", "size", "precision", "winner", "plan time")

	for _, n := range sizes {
		for _, kind := range precisions {
			started := time.Now()
			algorithm, err := tuneOne(n, kind, mode, wisdom)
			if err != nil {
				fatalf("tune n=%d %s: %v", n, kind, err)
			}

			fmt.Printf("%8d  %-10s  %-32s  %12s\n", n, kind, algorithm, time.Since(started).Round(time.Microsecond))
		}
	}

	if err := algofft.ExportWisdomTo(*output, wisdom); err != nil {
		fatalf("export Wisdom: %v", err)
	}

	fmt.Printf("wrote %d decisions to %s\n", wisdom.Len(), *output)
}

func tuneOne(n int, precision string, mode algofft.PlannerMode, wisdom *algofft.Wisdom) (string, error) {
	opts := algofft.PlanOptions{Planner: mode, Wisdom: wisdom}

	switch precision {
	case precision32:
		plan, err := algofft.NewPlanWithOptions[complex64](n, opts)
		if err != nil {
			return "", err
		}
		defer plan.Close()

		return plan.Algorithm(), nil
	case precision64:
		plan, err := algofft.NewPlanWithOptions[complex128](n, opts)
		if err != nil {
			return "", err
		}
		defer plan.Close()

		return plan.Algorithm(), nil
	default:
		return "", fmt.Errorf("unsupported precision %q", precision)
	}
}

func parseEffort(value string) (algofft.PlannerMode, error) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "patient":
		return algofft.PlannerPatient, nil
	case "exhaustive":
		return algofft.PlannerExhaustive, nil
	default:
		return algofft.PlannerEstimate, fmt.Errorf("-effort must be patient or exhaustive, got %q", value)
	}
}

func parsePrecisions(value string) ([]string, error) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case precisionAll:
		return []string{precision32, precision64}, nil
	case precision32:
		return []string{precision32}, nil
	case precision64:
		return []string{precision64}, nil
	default:
		return nil, fmt.Errorf("-precision must be all, complex64, or complex128, got %q", value)
	}
}

func powerOfTwoSizes(minSize, maxSize int) ([]int, error) {
	if minSize < 1 || minSize&(minSize-1) != 0 {
		return nil, fmt.Errorf("-min must be a positive power of two, got %d", minSize)
	}

	if maxSize < minSize || maxSize&(maxSize-1) != 0 {
		return nil, fmt.Errorf("-max must be a power of two at least -min, got %d", maxSize)
	}

	sizes := make([]int, 0, 16)
	for n := minSize; n <= maxSize; n *= 2 {
		sizes = append(sizes, n)
		if n > maxSize/2 {
			break
		}
	}

	return sizes, nil
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "tune: "+format+"\n", args...)
	os.Exit(1)
}
