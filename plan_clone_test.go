package algofft

import (
	"math"
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

// Clone is the concurrent-use story for Plan: each goroutine transforms
// through its own clone, which shares the immutable executor tables but owns
// its scratch buffers. These tests replaced the tests of the deleted
// Executor[T] wrapper (a thin Clone shim).

func TestClone_Independent(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan32(64)
	if err != nil {
		t.Fatalf("NewPlan32(64) failed: %v", err)
	}

	clone := plan.Clone()
	if clone == plan {
		t.Fatal("Clone() returned the original plan")
	}

	if &clone.scratch[0] == &plan.twiddle[0] {
		t.Error("Clone() scratch aliases original data")
	}
}

func TestClone_Forward(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan64(128)
	if err != nil {
		t.Fatalf("NewPlan64(128) failed: %v", err)
	}

	clone := plan.Clone()

	// Broadband, not an impulse: an impulse would transform to all ones for
	// any twiddle table and any bin order, so it could not tell a clone that
	// carries the parent's tables from one that lost them.
	src := broadbandSrc128(128)
	want := reference.NaiveDFT128(src)

	dst := make([]complex128, 128)

	err = clone.Forward(dst, src)
	if err != nil {
		t.Fatalf("clone.Forward() failed: %v", err)
	}

	var peak float64
	for _, v := range want {
		peak = math.Max(peak, absComplex128(v))
	}

	for i, v := range dst {
		if diff := absComplex128(v - want[i]); diff > 1e-11*peak {
			t.Errorf("dst[%d] = %v, want %v (diff %.3e)", i, v, want[i], diff)
		}
	}
}

func TestClone_RoundTrip(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan32(512)
	if err != nil {
		t.Fatalf("NewPlan32(512) failed: %v", err)
	}

	clone := plan.Clone()

	// Create test signal
	original := make([]complex64, 512)
	for i := range original {
		original[i] = complex(float32(i), float32(-i))
	}

	// Forward transform
	freq := make([]complex64, 512)

	err = clone.Forward(freq, original)
	if err != nil {
		t.Fatalf("clone.Forward() failed: %v", err)
	}

	// Inverse transform
	recovered := make([]complex64, 512)

	err = clone.Inverse(recovered, freq)
	if err != nil {
		t.Fatalf("clone.Inverse() failed: %v", err)
	}

	// Verify round-trip (Inverse automatically normalizes by 1/N)
	for i := range original {
		expected := original[i]

		diff := absComplex64(recovered[i] - expected)
		if diff > 1e-3 { // Allow for floating point error
			t.Errorf("recovered[%d] = %v, want %v (diff=%v)", i, recovered[i], expected, diff)
		}
	}
}

func TestClone_InPlaceRoundTrip(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan64(128)
	if err != nil {
		t.Fatalf("NewPlan64(128) failed: %v", err)
	}

	clone := plan.Clone()

	// In-place round-trip on a broadband signal: an impulse round-trips
	// trivially, so it checks neither direction's twiddles.
	src := broadbandSrc128(128)
	data := make([]complex128, 128)
	copy(data, src)

	err = clone.ForwardInPlace(data)
	if err != nil {
		t.Fatalf("clone.ForwardInPlace() failed: %v", err)
	}

	err = clone.InverseInPlace(data)
	if err != nil {
		t.Fatalf("clone.InverseInPlace() failed: %v", err)
	}

	// Verify round-trip (Inverse automatically normalizes by 1/N)
	for i := range data {
		if diff := absComplex128(data[i] - src[i]); diff > 1e-12 {
			t.Errorf("data[%d] = %v, want %v (diff %.3e)", i, data[i], src[i], diff)
		}
	}
}

func TestClone_ConcurrentSafety(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan32(256)
	if err != nil {
		t.Fatalf("NewPlan32(256) failed: %v", err)
	}

	// Create multiple clones from the same plan
	clone1 := plan.Clone()
	clone2 := plan.Clone()

	// Verify their scratch buffers are independent
	if &clone1.scratch[0] == &clone2.scratch[0] {
		t.Error("clones share scratch buffers - not safe for concurrent use")
	}

	// Run transforms concurrently
	done := make(chan bool, 2)

	go func() {
		src := make([]complex64, 256)
		src[0] = 1

		dst := make([]complex64, 256)

		err := clone1.Forward(dst, src)
		if err != nil {
			t.Errorf("clone1.Forward() failed: %v", err)
		}

		done <- true
	}()

	go func() {
		src := make([]complex64, 256)
		src[1] = 1

		dst := make([]complex64, 256)

		err := clone2.Forward(dst, src)
		if err != nil {
			t.Errorf("clone2.Forward() failed: %v", err)
		}

		done <- true
	}()

	<-done
	<-done
}

func TestClone_Close(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan32(128)
	if err != nil {
		t.Fatalf("NewPlan32(128) failed: %v", err)
	}

	clone := plan.Clone()

	// Close on a clone is a no-op (clones are never pooled); it must not
	// panic and must not affect the clone or the original.
	clone.Close()
	clone.Close()

	src := make([]complex64, 128)
	src[0] = 1

	dst := make([]complex64, 128)

	if err := clone.Forward(dst, src); err != nil {
		t.Errorf("clone.Forward() after Close failed: %v", err)
	}

	if err := plan.Forward(dst, src); err != nil {
		t.Errorf("plan.Forward() after clone Close failed: %v", err)
	}
}

func TestClone_Complex64AndComplex128(t *testing.T) {
	t.Parallel()

	// Test with complex64
	plan32, err := NewPlan32(64)
	if err != nil {
		t.Fatalf("NewPlan32(64) failed: %v", err)
	}

	clone32 := plan32.Clone()
	src32 := make([]complex64, 64)
	src32[0] = 1
	dst32 := make([]complex64, 64)

	err = clone32.Forward(dst32, src32)
	if err != nil {
		t.Errorf("complex64 clone.Forward() failed: %v", err)
	}

	// Test with complex128
	plan64, err := NewPlan64(64)
	if err != nil {
		t.Fatalf("NewPlan64(64) failed: %v", err)
	}

	clone64 := plan64.Clone()
	src64 := make([]complex128, 64)
	src64[0] = 1
	dst64 := make([]complex128, 64)

	err = clone64.Forward(dst64, src64)
	if err != nil {
		t.Errorf("complex128 clone.Forward() failed: %v", err)
	}

	// Verify both produce similar results (accounting for precision)
	for i := range 64 {
		diff := complex128(dst32[i]) - dst64[i]
		if absComplex128(diff) > 1e-5 {
			t.Errorf("Results differ at index %d: complex64=%v, complex128=%v", i, dst32[i], dst64[i])
		}
	}
}

func TestClone_ErrorPropagation(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan32(128)
	if err != nil {
		t.Fatalf("NewPlan32(128) failed: %v", err)
	}

	clone := plan.Clone()

	// Test with wrong size slice
	src := make([]complex64, 64) // Wrong size
	dst := make([]complex64, 128)

	err = clone.Forward(dst, src)
	if err == nil {
		t.Error("clone.Forward() with wrong size did not return error")
	}

	// Test inverse with wrong size
	src = make([]complex64, 128)
	dst = make([]complex64, 64) // Wrong size

	err = clone.Inverse(dst, src)
	if err == nil {
		t.Error("clone.Inverse() with wrong size did not return error")
	}
}

// Helper function for complex128 absolute value.
func absComplex128(v complex128) float64 {
	return real(v)*real(v) + imag(v)*imag(v)
}
