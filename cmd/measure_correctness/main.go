// Command measure_correctness measures FFT accuracy against a float64 naive DFT.
//
// It reports relative L2 error over the whole spectrum, as a mean and a max over
// trials, plus a peak-normalized max-per-bin error. See the header it prints for
// the metric definitions and the caveats that apply to each precision.
package main

import (
	"flag"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"strings"

	algofft "github.com/cwbudde/algo-fft"
	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/reference"
)

const (
	defaultSizes = "8,16,32,64,128,256,512,1024,2048,4096"

	// eps32 and eps64 are the float32 and float64 machine epsilons, printed as
	// the yardstick the reported errors should be read against.
	eps32 = 1.1920928955078125e-07
	eps64 = 2.220446049250313e-16
)

// sizeResult holds the aggregated metrics for one transform size and precision.
type sizeResult struct {
	size       int
	relL2Mean  float64
	relL2Max   float64
	peakRelMax float64
}

func main() {
	var (
		sizeList = flag.String("sizes", defaultSizes, "comma-separated transform sizes")
		trials   = flag.Int("trials", 100, "random vectors per size")
		seed     = flag.Int64("seed", 42, "rng seed, re-applied per size so one -sizes value reproduces its row from a full run")
	)

	flag.Parse()

	sizes := parseSizes(*sizeList)
	if len(sizes) == 0 {
		fatalf("no valid sizes in %q", *sizeList)
	}

	if *trials < 1 {
		fatalf("-trials must be at least 1, got %d", *trials)
	}

	results32 := make([]sizeResult, 0, len(sizes))
	results64 := make([]sizeResult, 0, len(sizes))

	for _, n := range sizes {
		// Progress goes to stderr so stdout stays a clean paste.
		fmt.Fprintf(os.Stderr, "measuring n=%d (%d trials)...\n", n, *trials)

		res32, res64 := measure(n, *trials, *seed)
		results32 = append(results32, res32)
		results64 = append(results64, res64)
	}

	printHeader(sizes, *trials, *seed)
	printBlock(complex64Caveat, results32)
	printBlock(complex128Caveat, results64)
}

// measure runs `trials` random vectors of length n through both precisions.
//
// Both arms transform the same mathematical vector: each draw is rounded to
// float32 once and the complex128 arm receives that rounded vector widened back,
// which is exact. So the two blocks are a genuine precision comparison of one
// problem rather than of two different random inputs.
func measure(n, trials int, seed int64) (sizeResult, sizeResult) {
	plan32, err := algofft.NewPlan[complex64](n)
	if err != nil {
		fatalf("NewPlan[complex64](%d): %v", n, err)
	}

	plan64, err := algofft.NewPlan64(n)
	if err != nil {
		fatalf("NewPlan64(%d): %v", n, err)
	}

	rng := rand.New(rand.NewSource(seed)) //nolint:gosec // reproducible test vectors, not crypto

	src32 := make([]complex64, n)
	src128 := make([]complex128, n)
	got32 := make([]complex64, n)
	got64 := make([]complex128, n)
	wide32 := make([]complex128, n)

	var acc32, acc64 accumulator

	for range trials {
		for i := range n {
			// Round to float32 first, then widen back: float64(float32(x)) is
			// exact, so both arms hold the bit-identical input vector.
			re := float32(rng.Float64()*2 - 1)
			im := float32(rng.Float64()*2 - 1)
			src32[i] = complex(re, im)
			src128[i] = complex(float64(re), float64(im))
		}

		if err := plan32.Forward(got32, src32); err != nil {
			fatalf("Plan[complex64].Forward(n=%d): %v", n, err)
		}

		if err := plan64.Forward(got64, src128); err != nil {
			fatalf("Plan[complex128].Forward(n=%d): %v", n, err)
		}

		// The complex64 reference takes the float32 vector directly. Referencing
		// an unrounded float64 draw instead would fold input quantization into
		// the error budget and install a floor at ~eps32/sqrt(12).
		want32 := reference.NaiveDFTWide(src32)
		want64 := reference.NaiveDFT128(src128)

		for i := range n {
			wide32[i] = complex128(got32[i]) // exact
		}

		acc32.add(relL2(wide32, want32), peakRel(wide32, want32))
		acc64.add(relL2(got64, want64), peakRel(got64, want64))
	}

	return acc32.result(n), acc64.result(n)
}

// accumulator aggregates per-trial metrics for one size and precision.
type accumulator struct {
	l2Sum      float64
	l2Max      float64
	peakRelMax float64
	trials     int
}

func (a *accumulator) add(l2, peak float64) {
	a.l2Sum += l2
	a.trials++

	if l2 > a.l2Max {
		a.l2Max = l2
	}

	if peak > a.peakRelMax {
		a.peakRelMax = peak
	}
}

func (a *accumulator) result(size int) sizeResult {
	mean := 0.0
	if a.trials > 0 {
		mean = a.l2Sum / float64(a.trials)
	}

	return sizeResult{
		size:       size,
		relL2Mean:  mean,
		relL2Max:   a.l2Max,
		peakRelMax: a.peakRelMax,
	}
}

// relL2 returns ||got-want||₂ / ||want||₂ over the whole spectrum.
//
// The whole-vector norm is the point. Normalizing each bin by its own magnitude
// — what this tool did before — divides by a quantity that can approach zero, so
// the result is decided by whichever bin happened to land nearest a zero rather
// than by the transform. There is no bin skipping here because there is nothing
// to skip.
func relL2(got, want []complex128) float64 {
	var num, den float64

	for i := range want {
		diff := cmplx.Abs(got[i] - want[i])
		num += diff * diff

		mag := cmplx.Abs(want[i])
		den += mag * mag
	}

	if den == 0 {
		return 0
	}

	return math.Sqrt(num) / math.Sqrt(den)
}

// peakRel returns max|got-want| / max|want|.
//
// Unlike relL2 this does not average over bins, so a single badly wrong bin
// stays visible at large n instead of being attenuated by ~1/sqrt(n) — which is
// the failure mode a broken codelet or a mis-permuted output actually produces.
// The denominator is the peak spectral magnitude, the best-conditioned quantity
// in the vector, so the statistic is stable.
//
// This is NOT the per-bin max relative error the tool used to report: that
// divided each bin by its own magnitude.
func peakRel(got, want []complex128) float64 {
	var maxDiff, maxRef float64

	for i := range want {
		if diff := cmplx.Abs(got[i] - want[i]); diff > maxDiff {
			maxDiff = diff
		}

		if mag := cmplx.Abs(want[i]); mag > maxRef {
			maxRef = mag
		}
	}

	if maxRef == 0 {
		return 0
	}

	return maxDiff / maxRef
}

const complex64Caveat = `complex64 (reference: reference.NaiveDFTWide, a float64 DFT of the same float32
           input vector, so this is the transform's own error and nothing else.
           The reference's own error is ~1e-16*n (see below), five or more orders
           below this column at every size here. Expect 0.4-1.0x float32 eps,
           rising slowly and monotonically with n.)`

const complex128Caveat = `complex128 (reference: reference.NaiveDFT128, which is also float64, so this
           column is the divergence between two float64 computations and NOT the
           FFT's error -- the reference dominates it from about n = 16 up. Read it
           as a fixed-reference regression tripwire: it is reproducible to the
           digit for a given size and seed, so a change that moves it has changed
           the arithmetic, but its absolute value is a property of the reference.
           It grows as O(n) -- measured at 2.00x per doubling of n from 16 to
           4096 -- because NaiveDFT128 forms its twiddle from an un-reduced angle
           -2*pi*k*m/n whose magnitude reaches ~2*pi*n, so the phase argument's
           own rounding grows in proportion. The FFT alone would sit near float64
           eps and grow like sqrt(log n), as the complex64 column does.)`

func printHeader(sizes []int, trials int, seed int64) {
	fmt.Println("Correctness: relative L2 error vs a float64 naive DFT")
	fmt.Println("=====================================================")
	fmt.Println()
	fmt.Printf("arch=%s  simd=%s  purego=%t\n", runtime.GOARCH, simdName(), puregoBuild)

	if puregoBuild {
		fmt.Println("  (purego build: the SIMD features above are detected but unused —")
		fmt.Println("   the pure-Go codelets are the transform here)")
	}

	fmt.Printf("trials=%d  seed=%d\n", trials, seed)
	fmt.Printf("sizes=%s\n", joinSizes(sizes))
	fmt.Println()
	fmt.Println("Per trial:")
	fmt.Println("  relL2 = ||got-want||2 / ||want||2            (whole-vector norm)")
	fmt.Println("  peak  = max_i|got_i-want_i| / max_i|want_i|   (normalized by the peak bin)")
	fmt.Println()
	fmt.Println("Both replace the per-bin max |got-want|/|want| this tool reported previously,")
	fmt.Println("which divided each bin by its own magnitude and was therefore dominated by")
	fmt.Println("whichever bin landed nearest a zero. Old numbers run 2-3 orders of magnitude")
	fmt.Println("higher than these and are not comparable to them.")
	fmt.Println()
	fmt.Printf("float32 eps = %.3e   float64 eps = %.3e\n", eps32, eps64)
}

func printBlock(caveat string, results []sizeResult) {
	fmt.Println()
	fmt.Println(caveat)
	fmt.Println()
	fmt.Printf("%8s %14s %14s %14s\n", "size", "relL2 mean", "relL2 max", "peak max")

	for _, r := range results {
		fmt.Printf("%8d %14.3e %14.3e %14.3e\n", r.size, r.relL2Mean, r.relL2Max, r.peakRelMax)
	}
}

// simdName reports the highest SIMD tier the host supports. On a purego build
// this is what the CPU has, not what the transform used.
func simdName() string {
	switch {
	case cpu.HasAVX512():
		return "AVX-512"
	case cpu.HasAVX2():
		return "AVX2"
	case cpu.HasNEON():
		return "NEON"
	case cpu.HasSSE41():
		return "SSE4.1"
	case cpu.HasSSE2():
		return "SSE2"
	default:
		return "none"
	}
}

// parseSizes splits a comma-separated size list, dropping entries that are not
// positive integers. Mirrors the helper in cmd/benchkernels.
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

func joinSizes(sizes []int) string {
	parts := make([]string, len(sizes))
	for i, n := range sizes {
		parts[i] = strconv.Itoa(n)
	}

	return strings.Join(parts, ",")
}

// fatalf reports a fatal error and exits. Sizes are user input now, so a bad
// one should not produce a goroutine dump.
func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "measure_correctness: "+format+"\n", args...)
	os.Exit(1)
}
