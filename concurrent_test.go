package algofft

import (
	"fmt"
	"math/rand"
	"sync"
	"testing"
)

// TestConcurrentSharedPlan tests concurrent use of a shared Plan with separate buffers.
// This is the recommended pattern for concurrent FFT transforms.
func TestConcurrentSharedPlan(t *testing.T) {
	t.Parallel()

	goroutineCounts := []int{2, 4, 8, 16}
	sizes := []int{256, 1024, 4096}

	for _, n := range sizes {
		for _, numGoroutines := range goroutineCounts {
			t.Run(fmt.Sprintf("size_%d_goroutines_%d", n, numGoroutines), func(t *testing.T) {
				t.Parallel()
				testSharedPlan(t, n, numGoroutines, 100)
			})
		}
	}
}

func testSharedPlan(t *testing.T, n, numGoroutines, itersPerGoroutine int) {
	t.Helper()
	// Create shared plan
	plan, err := NewPlan(n)
	if err != nil {
		t.Fatalf("failed to create plan: %v", err)
	}

	var waitGroup sync.WaitGroup

	errors := make(chan error, numGoroutines)

	for goroutine := range numGoroutines {
		waitGroup.Add(1)

		go func(goroutineID int) {
			defer waitGroup.Done()

			// Each goroutine has its own buffers
			src := make([]complex64, n)
			dst := make([]complex64, n)

			// Populate with unique data
			for i := range src {
				src[i] = complex(float32(goroutineID)+rand.Float32(), rand.Float32())
			}

			for iter := range itersPerGoroutine {
				// Forward transform
				err := plan.Forward(dst, src)
				if err != nil {
					errors <- fmt.Errorf("goroutine %d iteration %d Forward: %w", goroutineID, iter, err)
					return
				}

				// Inverse transform
				err = plan.Inverse(src, dst)
				if err != nil {
					errors <- fmt.Errorf("goroutine %d iteration %d Inverse: %w", goroutineID, iter, err)
					return
				}
			}
		}(goroutine)
	}

	waitGroup.Wait()
	close(errors)

	// Check for errors
	for err := range errors {
		t.Error(err)
	}

	totalOps := numGoroutines * itersPerGoroutine * 2 // Forward + Inverse
	t.Logf("Completed %d concurrent operations successfully", totalOps)
}

// TestConcurrentPooledPlans tests concurrent creation of pooled plans.
func TestConcurrentPooledPlans(t *testing.T) {
	t.Parallel()

	goroutineCounts := []int{4, 8, 16}
	n := 1024

	for _, numGoroutines := range goroutineCounts {
		t.Run(fmt.Sprintf("goroutines_%d", numGoroutines), func(t *testing.T) {
			t.Parallel()
			testConcurrentPooled(t, n, numGoroutines, 100)
		})
	}
}

func testConcurrentPooled(t *testing.T, n, numGoroutines, itersPerGoroutine int) {
	t.Helper()

	var waitGroup sync.WaitGroup

	errors := make(chan error, numGoroutines)

	for goroutine := range numGoroutines {
		waitGroup.Add(1)

		go func(goroutineID int) {
			defer waitGroup.Done()

			src := make([]complex64, n)
			dst := make([]complex64, n)

			for i := range src {
				src[i] = complex(rand.Float32(), rand.Float32())
			}

			for iter := range itersPerGoroutine {
				// Create pooled plan
				plan, err := NewPlanPooled[complex64](n)
				if err != nil {
					errors <- fmt.Errorf("goroutine %d iteration %d NewPlanPooled: %w", goroutineID, iter, err)
					return
				}

				err = plan.Forward(dst, src)
				if err != nil {
					errors <- fmt.Errorf("goroutine %d iteration %d Forward: %w", goroutineID, iter, err)
					return
				}
			}
		}(goroutine)
	}

	waitGroup.Wait()
	close(errors)

	for err := range errors {
		t.Error(err)
	}

	totalOps := numGoroutines * itersPerGoroutine
	t.Logf("Completed %d concurrent pooled plan creations", totalOps)
}

// TestConcurrentPlanCreation tests concurrent creation of multiple plans.
func TestConcurrentPlanCreation(t *testing.T) {
	t.Parallel()

	goroutineCounts := []int{4, 8, 16}
	sizes := []int{256, 1024}

	for _, numGoroutines := range goroutineCounts {
		t.Run(fmt.Sprintf("goroutines_%d", numGoroutines), func(t *testing.T) {
			t.Parallel()
			testConcurrentCreation(t, sizes, numGoroutines)
		})
	}
}

func testConcurrentCreation(t *testing.T, sizes []int, numGoroutines int) {
	t.Helper()

	var waitGroup sync.WaitGroup

	errors := make(chan error, numGoroutines)

	for goroutine := range numGoroutines {
		waitGroup.Add(1)

		go func(goroutineID int) {
			defer waitGroup.Done()

			// Each goroutine creates plans for different sizes
			for _, n := range sizes {
				plan, err := NewPlan(n)
				if err != nil {
					errors <- fmt.Errorf("goroutine %d size %d: %w", goroutineID, n, err)
					return
				}

				// Perform a transform to verify the plan works
				src := make([]complex64, n)
				dst := make([]complex64, n)

				for i := range src {
					src[i] = complex(rand.Float32(), rand.Float32())
				}

				err = plan.Forward(dst, src)
				if err != nil {
					errors <- fmt.Errorf("goroutine %d size %d Forward: %w", goroutineID, n, err)
					return
				}
			}
		}(goroutine)
	}

	waitGroup.Wait()
	close(errors)

	for err := range errors {
		t.Error(err)
	}

	totalPlans := numGoroutines * len(sizes)
	t.Logf("Created %d plans concurrently", totalPlans)
}

// TestConcurrentMixedOperations tests mixed Forward/Inverse operations concurrently.
func TestConcurrentMixedOperations(t *testing.T) {
	t.Parallel()

	n := 1024
	numGoroutines := 8
	itersPerGoroutine := 100

	plan, err := NewPlan(n)
	if err != nil {
		t.Fatalf("failed to create plan: %v", err)
	}

	var waitGroup sync.WaitGroup

	errors := make(chan error, numGoroutines)

	for goroutine := range numGoroutines {
		waitGroup.Add(1)

		go func(goroutineID int) {
			defer waitGroup.Done()

			src := make([]complex64, n)
			dst := make([]complex64, n)

			for i := range src {
				src[i] = complex(rand.Float32(), rand.Float32())
			}

			for iter := range itersPerGoroutine {
				// Randomly choose Forward or Inverse
				if iter%2 == 0 {
					err := plan.Forward(dst, src)
					if err != nil {
						errors <- fmt.Errorf("goroutine %d Forward: %w", goroutineID, err)
						return
					}
				} else {
					err := plan.Inverse(dst, src)
					if err != nil {
						errors <- fmt.Errorf("goroutine %d Inverse: %w", goroutineID, err)
						return
					}
				}
			}
		}(goroutine)
	}

	waitGroup.Wait()
	close(errors)

	for err := range errors {
		t.Error(err)
	}

	t.Logf("Completed %d mixed concurrent operations", numGoroutines*itersPerGoroutine)
}

// TestConcurrentDifferentPrecisions tests concurrent use with both complex64 and complex128.
//
//nolint:gocognit,paralleltest
func TestConcurrentDifferentPrecisions(t *testing.T) {
	n := 1024
	numGoroutines := 8
	itersPerGoroutine := 50

	plan64, err := NewPlan(n)
	if err != nil {
		t.Fatalf("failed to create complex64 plan: %v", err)
	}

	plan128, err := NewPlan64(n)
	if err != nil {
		t.Fatalf("failed to create complex128 plan: %v", err)
	}

	var waitGroup sync.WaitGroup

	errors := make(chan error, numGoroutines)

	for goroutine := range numGoroutines {
		waitGroup.Add(1)

		go func(goroutineID int) {
			defer waitGroup.Done()

			// Alternate between precisions
			if goroutineID%2 == 0 {
				// Use complex64
				src := make([]complex64, n)
				dst := make([]complex64, n)

				for i := range src {
					src[i] = complex(rand.Float32(), rand.Float32())
				}

				for range itersPerGoroutine {
					err := plan64.Forward(dst, src)
					if err != nil {
						errors <- fmt.Errorf("goroutine %d complex64: %w", goroutineID, err)
						return
					}
				}
			} else {
				// Use complex128
				src := make([]complex128, n)
				dst := make([]complex128, n)

				for i := range src {
					src[i] = complex(rand.Float64(), rand.Float64())
				}

				for range itersPerGoroutine {
					err := plan128.Forward(dst, src)
					if err != nil {
						errors <- fmt.Errorf("goroutine %d complex128: %w", goroutineID, err)
						return
					}
				}
			}
		}(goroutine)
	}

	waitGroup.Wait()
	close(errors)

	for err := range errors {
		t.Error(err)
	}

	t.Logf("Completed %d concurrent operations with mixed precisions", numGoroutines*itersPerGoroutine)
}

// TestConcurrentStress runs a high-concurrency stress test.
func TestConcurrentStress(t *testing.T) {
	t.Parallel()

	if testing.Short() {
		t.Skip("skipping concurrent stress test in short mode")
	}

	n := 1024
	numGoroutines := 32
	itersPerGoroutine := 1000

	plan, err := NewPlan(n)
	if err != nil {
		t.Fatalf("failed to create plan: %v", err)
	}

	var waitGroup sync.WaitGroup

	errors := make(chan error, numGoroutines)

	for goroutine := range numGoroutines {
		waitGroup.Add(1)

		go func(goroutineID int) {
			defer waitGroup.Done()

			src := make([]complex64, n)
			dst := make([]complex64, n)

			for i := range src {
				src[i] = complex(rand.Float32(), rand.Float32())
			}

			for iter := range itersPerGoroutine {
				err := plan.Forward(dst, src)
				if err != nil {
					errors <- fmt.Errorf("goroutine %d iteration %d: %w", goroutineID, iter, err)
					return
				}
			}
		}(goroutine)
	}

	waitGroup.Wait()
	close(errors)

	for err := range errors {
		t.Error(err)
	}

	totalOps := numGoroutines * itersPerGoroutine
	t.Logf("Stress test completed: %d goroutines × %d iterations = %d total operations",
		numGoroutines, itersPerGoroutine, totalOps)
}
