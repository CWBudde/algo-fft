package algofft

import (
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"testing"
	"time"
)

// TestStressLongRunning performs continuous FFT transforms for an extended period.
// This test is skipped in short mode and controlled via STRESS_DURATION env var.
func TestStressLongRunning(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping long-running stress test in short mode")
	}

	duration := getStressDuration()
	t.Logf("Running stress test for %v", duration)

	sizes := []int{256, 1024, 4096}
	for _, n := range sizes {
		t.Run(fmt.Sprintf("size_%d", n), func(t *testing.T) {
			runStressTest(t, n, duration)
		})
	}
}

func runStressTest(t *testing.T, n int, duration time.Duration) {
	plan, err := NewPlan(n)
	if err != nil {
		t.Fatalf("failed to create plan: %v", err)
	}

	src := make([]complex64, n)
	dst := make([]complex64, n)

	// Populate with random data
	for i := range src {
		src[i] = complex(rand.Float32(), rand.Float32())
	}

	start := time.Now()
	iterations := 0

	for time.Since(start) < duration {
		err := plan.Forward(dst, src)
		if err != nil {
			t.Fatalf("Forward failed at iteration %d: %v", iterations, err)
		}
		err = plan.Inverse(src, dst)
		if err != nil {
			t.Fatalf("Inverse failed at iteration %d: %v", iterations, err)
		}

		iterations++
	}

	t.Logf("Completed %d iterations in %v (%.2f iterations/sec)",
		iterations, duration, float64(iterations)/duration.Seconds())
}

// TestStressMemoryStability performs millions of transforms and tracks memory usage.
func TestStressMemoryStability(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping memory stress test in short mode")
	}

	const (
		totalIterations = 1_000_000
		sampleInterval  = 10_000
	)

	sizes := []int{256, 1024, 4096}
	for _, n := range sizes {
		t.Run(fmt.Sprintf("size_%d", n), func(t *testing.T) {
			testMemoryStability(t, n, totalIterations, sampleInterval)
		})
	}
}

func testMemoryStability(t *testing.T, n, totalIters, sampleInterval int) {
	plan, err := NewPlan(n)
	if err != nil {
		t.Fatalf("failed to create plan: %v", err)
	}

	src := make([]complex64, n)
	dst := make([]complex64, n)

	// Initialize
	for i := range src {
		src[i] = complex(rand.Float32(), rand.Float32())
	}

	var (
		memStats               runtime.MemStats
		initialAlloc, maxAlloc uint64
	)

	start := time.Now()

	for i := range totalIters {
		err := plan.Forward(dst, src)
		if err != nil {
			t.Fatalf("Forward failed at iteration %d: %v", i, err)
		}

		// Sample memory every N iterations
		if i%sampleInterval == 0 {
			runtime.ReadMemStats(&memStats)

			if i == 0 {
				initialAlloc = memStats.Alloc
				maxAlloc = initialAlloc
			} else {
				if memStats.Alloc > maxAlloc {
					maxAlloc = memStats.Alloc
				}

				// Check for memory growth
				growth := float64(memStats.Alloc-initialAlloc) / float64(initialAlloc)
				if growth > 0.5 { // Allow 50% jitter from GC
					t.Errorf("Memory growth detected at iteration %d: initial=%d, current=%d, growth=%.2f%%",
						i, initialAlloc, memStats.Alloc, growth*100)
				}
			}

			if i > 0 && i%(sampleInterval*10) == 0 {
				t.Logf("Progress: %d/%d iterations, alloc=%d KB, max=%d KB",
					i, totalIters, memStats.Alloc/1024, maxAlloc/1024)
			}
		}
	}

	elapsed := time.Since(start)
	t.Logf("Completed %d iterations in %v (%.2f iterations/sec)",
		totalIters, elapsed, float64(totalIters)/elapsed.Seconds())
	t.Logf("Memory: initial=%d KB, max=%d KB, final=%d KB",
		initialAlloc/1024, maxAlloc/1024, memStats.Alloc/1024)
}

// TestStressRandomSizes tests with random FFT sizes.
func TestStressRandomSizes(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping random size stress test in short mode")
	}

	duration := getStressDuration() / 2 // Shorter than long-running test
	powerOfTwo := []int{8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}

	t.Logf("Running random size stress test for %v", duration)

	start := time.Now()
	iterations := 0

	for time.Since(start) < duration {
		// Pick random size
		n := powerOfTwo[rand.Intn(len(powerOfTwo))]

		plan, err := NewPlan(n)
		if err != nil {
			t.Fatalf("failed to create plan for size %d: %v", n, err)
		}

		src := make([]complex64, n)
		dst := make([]complex64, n)

		for i := range src {
			src[i] = complex(rand.Float32(), rand.Float32())
		}

		if err := plan.Forward(dst, src); err != nil {
			t.Fatalf("Forward failed for size %d: %v", n, err)
		}

		iterations++

		if iterations%100 == 0 {
			t.Logf("Completed %d iterations", iterations)
		}
	}

	t.Logf("Completed %d random-size transforms in %v", iterations, duration)
}

// TestStressPlanPooled tests pooled plan creation under stress.
func TestStressPlanPooled(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping plan pool stress test in short mode")
	}

	const iterations = 10000

	sizes := []int{256, 1024, 4096}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("size_%d", n), func(t *testing.T) {
			testPlanPooledStress(t, n, iterations)
		})
	}
}

func testPlanPooledStress(t *testing.T, n, iters int) {
	src := make([]complex64, n)
	dst := make([]complex64, n)

	for i := range src {
		src[i] = complex(rand.Float32(), rand.Float32())
	}

	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	initialAlloc := memStats.Alloc

	start := time.Now()

	for i := range iters {
		// Create pooled plan
		plan, err := NewPlanPooled[complex64](n)
		if err != nil {
			t.Fatalf("NewPlanPooled failed at iteration %d: %v", i, err)
		}

		if err := plan.Forward(dst, src); err != nil {
			t.Fatalf("Forward failed at iteration %d: %v", i, err)
		}

		// Plan will be returned to pool when GC'd

		// Check memory every 1000 iterations
		if i > 0 && i%1000 == 0 {
			runtime.ReadMemStats(&memStats)

			growth := float64(memStats.Alloc-initialAlloc) / float64(initialAlloc)
			if growth > 1.0 { // Allow 100% growth for pooling overhead
				t.Errorf("Pool memory growth at iteration %d: %.2f%%", i, growth*100)
			}
		}
	}

	elapsed := time.Since(start)
	t.Logf("Completed %d pooled plan creations in %v (%.2f ops/sec)",
		iters, elapsed, float64(iters)/elapsed.Seconds())

	runtime.ReadMemStats(&memStats)
	t.Logf("Memory: initial=%d KB, final=%d KB", initialAlloc/1024, memStats.Alloc/1024)
}

// TestStressPrecisionSwitching tests switching between complex64 and complex128.
func TestStressPrecisionSwitching(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping precision switching stress test in short mode")
	}

	duration := getStressDuration() / 2
	n := 1024

	t.Logf("Running precision switching stress test for %v", duration)

	plan32, err := NewPlan32(n)
	if err != nil {
		t.Fatalf("failed to create complex64 plan: %v", err)
	}

	plan64, err := NewPlan64(n)
	if err != nil {
		t.Fatalf("failed to create complex128 plan: %v", err)
	}

	src32 := make([]complex64, n)
	dst32 := make([]complex64, n)
	src64 := make([]complex128, n)
	dst64 := make([]complex128, n)

	for i := range src32 {
		src32[i] = complex(rand.Float32(), rand.Float32())
		src64[i] = complex(rand.Float64(), rand.Float64())
	}

	start := time.Now()
	iterations := 0

	for time.Since(start) < duration {
		// Alternate between precisions
		if iterations%2 == 0 {
			err := plan32.Forward(dst32, src32)
			if err != nil {
				t.Fatalf("complex64 Forward failed: %v", err)
			}
		} else {
			err := plan64.Forward(dst64, src64)
			if err != nil {
				t.Fatalf("complex128 Forward failed: %v", err)
			}
		}

		iterations++
	}

	t.Logf("Completed %d precision-switching transforms in %v", iterations, duration)
}

// getStressDuration returns the stress test duration from env var or default.
func getStressDuration() time.Duration {
	envDuration := os.Getenv("STRESS_DURATION")
	if envDuration == "" {
		return 5 * time.Minute // Default: 5 minutes
	}

	seconds, err := strconv.Atoi(envDuration)
	if err != nil {
		return 5 * time.Minute
	}

	return time.Duration(seconds) * time.Second
}
