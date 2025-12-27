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
	goroutineCounts := []int{2, 4, 8, 16}
	sizes := []int{256, 1024, 4096}

	for _, n := range sizes {
		for _, numGoroutines := range goroutineCounts {
			t.Run(fmt.Sprintf("size_%d_goroutines_%d", n, numGoroutines), func(t *testing.T) {
				testSharedPlan(t, n, numGoroutines, 100)
			})
		}
	}
}

func testSharedPlan(t *testing.T, n, numGoroutines, itersPerGoroutine int) {
	// Create shared plan
	plan, err := NewPlan(n)
	if err != nil {
		t.Fatalf("failed to create plan: %v", err)
	}

	var wg sync.WaitGroup
	errors := make(chan error, numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		go func(goroutineID int) {
			defer wg.Done()

			// Each goroutine has its own buffers
			src := make([]complex64, n)
			dst := make([]complex64, n)

			// Populate with unique data
			for i := range src {
				src[i] = complex(float32(goroutineID)+rand.Float32(), rand.Float32())
			}

			for iter := 0; iter < itersPerGoroutine; iter++ {
				// Forward transform
				if err := plan.Forward(dst, src); err != nil {
					errors <- fmt.Errorf("goroutine %d iteration %d Forward: %w", goroutineID, iter, err)
					return
				}

				// Inverse transform
				if err := plan.Inverse(src, dst); err != nil {
					errors <- fmt.Errorf("goroutine %d iteration %d Inverse: %w", goroutineID, iter, err)
					return
				}
			}
		}(g)
	}

	wg.Wait()
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
	goroutineCounts := []int{4, 8, 16}
	n := 1024

	for _, numGoroutines := range goroutineCounts {
		t.Run(fmt.Sprintf("goroutines_%d", numGoroutines), func(t *testing.T) {
			testConcurrentPooled(t, n, numGoroutines, 100)
		})
	}
}

func testConcurrentPooled(t *testing.T, n, numGoroutines, itersPerGoroutine int) {
	var wg sync.WaitGroup
	errors := make(chan error, numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		go func(goroutineID int) {
			defer wg.Done()

			src := make([]complex64, n)
			dst := make([]complex64, n)

			for i := range src {
				src[i] = complex(rand.Float32(), rand.Float32())
			}

			for iter := 0; iter < itersPerGoroutine; iter++ {
				// Create pooled plan
				plan, err := NewPlanPooled[complex64](n)
				if err != nil {
					errors <- fmt.Errorf("goroutine %d iteration %d NewPlanPooled: %w", goroutineID, iter, err)
					return
				}

				if err := plan.Forward(dst, src); err != nil {
					errors <- fmt.Errorf("goroutine %d iteration %d Forward: %w", goroutineID, iter, err)
					return
				}
			}
		}(g)
	}

	wg.Wait()
	close(errors)

	for err := range errors {
		t.Error(err)
	}

	totalOps := numGoroutines * itersPerGoroutine
	t.Logf("Completed %d concurrent pooled plan creations", totalOps)
}

// TestConcurrentPlanCreation tests concurrent creation of multiple plans.
func TestConcurrentPlanCreation(t *testing.T) {
	goroutineCounts := []int{4, 8, 16}
	sizes := []int{256, 1024}

	for _, numGoroutines := range goroutineCounts {
		t.Run(fmt.Sprintf("goroutines_%d", numGoroutines), func(t *testing.T) {
			testConcurrentCreation(t, sizes, numGoroutines)
		})
	}
}

func testConcurrentCreation(t *testing.T, sizes []int, numGoroutines int) {
	var wg sync.WaitGroup
	errors := make(chan error, numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		go func(goroutineID int) {
			defer wg.Done()

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

				if err := plan.Forward(dst, src); err != nil {
					errors <- fmt.Errorf("goroutine %d size %d Forward: %w", goroutineID, n, err)
					return
				}
			}
		}(g)
	}

	wg.Wait()
	close(errors)

	for err := range errors {
		t.Error(err)
	}

	totalPlans := numGoroutines * len(sizes)
	t.Logf("Created %d plans concurrently", totalPlans)
}

// TestConcurrentMixedOperations tests mixed Forward/Inverse operations concurrently.
func TestConcurrentMixedOperations(t *testing.T) {
	n := 1024
	numGoroutines := 8
	itersPerGoroutine := 100

	plan, err := NewPlan(n)
	if err != nil {
		t.Fatalf("failed to create plan: %v", err)
	}

	var wg sync.WaitGroup
	errors := make(chan error, numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		go func(goroutineID int) {
			defer wg.Done()

			src := make([]complex64, n)
			dst := make([]complex64, n)

			for i := range src {
				src[i] = complex(rand.Float32(), rand.Float32())
			}

			for iter := 0; iter < itersPerGoroutine; iter++ {
				// Randomly choose Forward or Inverse
				if iter%2 == 0 {
					if err := plan.Forward(dst, src); err != nil {
						errors <- fmt.Errorf("goroutine %d Forward: %w", goroutineID, err)
						return
					}
				} else {
					if err := plan.Inverse(dst, src); err != nil {
						errors <- fmt.Errorf("goroutine %d Inverse: %w", goroutineID, err)
						return
					}
				}
			}
		}(g)
	}

	wg.Wait()
	close(errors)

	for err := range errors {
		t.Error(err)
	}

	t.Logf("Completed %d mixed concurrent operations", numGoroutines*itersPerGoroutine)
}

// TestConcurrentDifferentPrecisions tests concurrent use with both complex64 and complex128.
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

	var wg sync.WaitGroup
	errors := make(chan error, numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		go func(goroutineID int) {
			defer wg.Done()

			// Alternate between precisions
			if goroutineID%2 == 0 {
				// Use complex64
				src := make([]complex64, n)
				dst := make([]complex64, n)
				for i := range src {
					src[i] = complex(rand.Float32(), rand.Float32())
				}

				for iter := 0; iter < itersPerGoroutine; iter++ {
					if err := plan64.Forward(dst, src); err != nil {
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

				for iter := 0; iter < itersPerGoroutine; iter++ {
					if err := plan128.Forward(dst, src); err != nil {
						errors <- fmt.Errorf("goroutine %d complex128: %w", goroutineID, err)
						return
					}
				}
			}
		}(g)
	}

	wg.Wait()
	close(errors)

	for err := range errors {
		t.Error(err)
	}

	t.Logf("Completed %d concurrent operations with mixed precisions", numGoroutines*itersPerGoroutine)
}

// TestConcurrentStress runs a high-concurrency stress test.
func TestConcurrentStress(t *testing.T) {
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

	var wg sync.WaitGroup
	errors := make(chan error, numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		go func(goroutineID int) {
			defer wg.Done()

			src := make([]complex64, n)
			dst := make([]complex64, n)

			for i := range src {
				src[i] = complex(rand.Float32(), rand.Float32())
			}

			for iter := 0; iter < itersPerGoroutine; iter++ {
				if err := plan.Forward(dst, src); err != nil {
					errors <- fmt.Errorf("goroutine %d iteration %d: %w", goroutineID, iter, err)
					return
				}
			}
		}(g)
	}

	wg.Wait()
	close(errors)

	for err := range errors {
		t.Error(err)
	}

	totalOps := numGoroutines * itersPerGoroutine
	t.Logf("Stress test completed: %d goroutines × %d iterations = %d total operations",
		numGoroutines, itersPerGoroutine, totalOps)
}
