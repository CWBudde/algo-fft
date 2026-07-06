package algofft

import (
	"errors"
	"math/cmplx"
	"sync"
	"testing"
)

// These tests exercise the documented guarantee that a single plan instance
// is safe for concurrent transforms. Run with -race: a shared mutable
// scratch buffer makes them fail.

const (
	concurrentWorkers = 8
	concurrentIters   = 25
)

// runConcurrentWorkers runs transform from several goroutines and reports the first error.
func runConcurrentWorkers(t *testing.T, transform func() error) {
	t.Helper()

	var workers sync.WaitGroup

	errCh := make(chan error, concurrentWorkers)

	for range concurrentWorkers {
		workers.Go(func() {
			for range concurrentIters {
				err := transform()
				if err != nil {
					errCh <- err

					return
				}
			}
		})
	}

	workers.Wait()
	close(errCh)

	for err := range errCh {
		t.Fatalf("concurrent transform failed: %v", err)
	}
}

// spectrumMismatch compares two complex64 spectra and returns an error on the
// first divergence. It is goroutine-safe (unlike the t.Fatalf-based assert
// helpers) so worker goroutines can report failures back through the harness.
func spectrumMismatch(got, want []complex64, context string) error {
	const tol = 1e-3

	for i := range want {
		if cmplx.Abs(complex128(got[i]-want[i])) > tol {
			return errors.New(sprintf(context+" mismatch at index %d", i)) //nolint:err113 // descriptive test error, no fmt by convention
		}
	}

	return nil
}

// spectrumMismatch128 is the complex128 counterpart of spectrumMismatch.
func spectrumMismatch128(got, want []complex128, context string) error {
	const tol = 1e-10

	for i := range want {
		if cmplx.Abs(got[i]-want[i]) > tol {
			return errors.New(sprintf(context+" mismatch at index %d", i)) //nolint:err113 // descriptive test error, no fmt by convention
		}
	}

	return nil
}

func TestConcurrentSharedPlan2D(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan2D32(16, 8) // Non-square to exercise the strided column path
	if err != nil {
		t.Fatalf("NewPlan2D32 failed: %v", err)
	}

	src := make([]complex64, plan.Len())
	for i := range src {
		src[i] = complex(float32(i%13)-6, float32((i*7)%11)-5)
	}

	want := make([]complex64, plan.Len())

	err = plan.Forward(want, src)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	runConcurrentWorkers(t, func() error {
		dst := make([]complex64, plan.Len())

		err := plan.Forward(dst, src)
		if err != nil {
			return err
		}

		return spectrumMismatch(dst, want, "plan2d")
	})
}

func TestConcurrentSharedPlan3D(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan3D32(4, 6, 8)
	if err != nil {
		t.Fatalf("NewPlan3D32 failed: %v", err)
	}

	src := make([]complex64, plan.Len())
	for i := range src {
		src[i] = complex(float32(i%13)-6, float32((i*7)%11)-5)
	}

	want := make([]complex64, plan.Len())

	err = plan.Forward(want, src)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	runConcurrentWorkers(t, func() error {
		dst := make([]complex64, plan.Len())

		err := plan.Forward(dst, src)
		if err != nil {
			return err
		}

		return spectrumMismatch(dst, want, "plan3d")
	})
}

func TestConcurrentSharedPlanND(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanND32([]int{4, 4, 8})
	if err != nil {
		t.Fatalf("NewPlanND32 failed: %v", err)
	}

	src := make([]complex64, plan.Len())
	for i := range src {
		src[i] = complex(float32(i%13)-6, float32((i*7)%11)-5)
	}

	want := make([]complex64, plan.Len())

	err = plan.Forward(want, src)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	runConcurrentWorkers(t, func() error {
		dst := make([]complex64, plan.Len())

		err := plan.Forward(dst, src)
		if err != nil {
			return err
		}

		return spectrumMismatch(dst, want, "plannd")
	})
}

func TestConcurrentSharedPlanReal(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanReal(64)
	if err != nil {
		t.Fatalf("NewPlanReal failed: %v", err)
	}

	src := make([]float32, plan.Len())
	for i := range src {
		src[i] = float32(i%17) - 8
	}

	want := make([]complex64, plan.SpectrumLen())

	err = plan.Forward(want, src)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	runConcurrentWorkers(t, func() error {
		dst := make([]complex64, plan.SpectrumLen())

		err := plan.Forward(dst, src)
		if err != nil {
			return err
		}

		return spectrumMismatch(dst, want, "planreal")
	})
}

func TestConcurrentSharedPlanRealT(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanRealT[float64, complex128](64)
	if err != nil {
		t.Fatalf("NewPlanRealT failed: %v", err)
	}

	src := make([]float64, plan.Len())
	for i := range src {
		src[i] = float64(i%17) - 8
	}

	want := make([]complex128, plan.SpectrumLen())

	err = plan.Forward(want, src)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	runConcurrentWorkers(t, func() error {
		dst := make([]complex128, plan.SpectrumLen())

		err := plan.Forward(dst, src)
		if err != nil {
			return err
		}

		return spectrumMismatch128(dst, want, "planrealt")
	})
}

func TestConcurrentSharedPlanReal2D(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanReal2D(8, 16)
	if err != nil {
		t.Fatalf("NewPlanReal2D failed: %v", err)
	}

	src := make([]float32, plan.Len())
	for i := range src {
		src[i] = float32(i%17) - 8
	}

	want := make([]complex64, plan.SpectrumLen())

	err = plan.Forward(want, src)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	runConcurrentWorkers(t, func() error {
		dst := make([]complex64, plan.SpectrumLen())

		err := plan.Forward(dst, src)
		if err != nil {
			return err
		}

		return spectrumMismatch(dst, want, "planreal2d")
	})
}

func TestConcurrentSharedPlanReal3D(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanReal3D(4, 4, 8)
	if err != nil {
		t.Fatalf("NewPlanReal3D failed: %v", err)
	}

	src := make([]float32, plan.Len())
	for i := range src {
		src[i] = float32(i%17) - 8
	}

	want := make([]complex64, plan.SpectrumLen())

	err = plan.Forward(want, src)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	runConcurrentWorkers(t, func() error {
		dst := make([]complex64, plan.SpectrumLen())

		err := plan.Forward(dst, src)
		if err != nil {
			return err
		}

		return spectrumMismatch(dst, want, "planreal3d")
	})
}
