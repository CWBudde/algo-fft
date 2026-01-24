package algofft

import (
	"testing"
)

func TestExecutor_NewExecutor(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan32(64)
	if err != nil {
		t.Fatalf("NewPlan32(64) failed: %v", err)
	}

	executor := plan.NewExecutor()
	if executor == nil {
		t.Fatal("NewExecutor() returned nil")
	}

	if executor.plan == nil {
		t.Fatal("NewExecutor() created executor with nil plan")
	}

	// Verify the executor has its own independent workspace (cloned plan)
	if executor.plan == plan {
		t.Error("NewExecutor() did not clone the plan - executor shares plan with original")
	}
}

func TestExecutor_Forward(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan64(128)
	if err != nil {
		t.Fatalf("NewPlan64(128) failed: %v", err)
	}

	executor := plan.NewExecutor()

	// Test with impulse signal
	src := make([]complex128, 128)
	src[0] = 1

	dst := make([]complex128, 128)

	err = executor.Forward(dst, src)
	if err != nil {
		t.Fatalf("executor.Forward() failed: %v", err)
	}

	// Impulse should produce all ones in frequency domain
	for i, v := range dst {
		expected := complex(1, 0)
		if absComplex128(v-expected) > 1e-10 {
			t.Errorf("dst[%d] = %v, want %v", i, v, expected)
		}
	}
}

func TestExecutor_Inverse(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan32(256)
	if err != nil {
		t.Fatalf("NewPlan32(256) failed: %v", err)
	}

	executor := plan.NewExecutor()

	// Test round-trip: Forward then Inverse should recover original (scaled by N)
	original := make([]complex64, 256)
	original[0] = 1

	freq := make([]complex64, 256)
	err = executor.Forward(freq, original)
	if err != nil {
		t.Fatalf("executor.Forward() failed: %v", err)
	}

	recovered := make([]complex64, 256)
	err = executor.Inverse(recovered, freq)
	if err != nil {
		t.Fatalf("executor.Inverse() failed: %v", err)
	}

	// Verify round-trip (Inverse automatically normalizes by 1/N)
	expectedFirst := complex(float32(1), 0)
	if absComplex64(recovered[0]-expectedFirst) > 1e-3 {
		t.Errorf("recovered[0] = %v, want %v", recovered[0], expectedFirst)
	}

	for i := 1; i < len(recovered); i++ {
		if absComplex64(recovered[i]) > 1e-3 {
			t.Errorf("recovered[%d] = %v, want ~0", i, recovered[i])
		}
	}
}

func TestExecutor_ForwardInPlace(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan32(64)
	if err != nil {
		t.Fatalf("NewPlan32(64) failed: %v", err)
	}

	executor := plan.NewExecutor()

	// Test in-place forward transform
	data := make([]complex64, 64)
	data[0] = 1

	err = executor.ForwardInPlace(data)
	if err != nil {
		t.Fatalf("executor.ForwardInPlace() failed: %v", err)
	}

	// Impulse should produce all ones in frequency domain
	for i, v := range data {
		expected := complex(float32(1), 0)
		if absComplex64(v-expected) > 1e-5 {
			t.Errorf("data[%d] = %v, want %v", i, v, expected)
		}
	}
}

func TestExecutor_InverseInPlace(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan64(128)
	if err != nil {
		t.Fatalf("NewPlan64(128) failed: %v", err)
	}

	executor := plan.NewExecutor()

	// Test in-place round-trip: start with impulse
	data := make([]complex128, 128)
	data[0] = 1

	// Forward transform
	err = executor.ForwardInPlace(data)
	if err != nil {
		t.Fatalf("executor.ForwardInPlace() failed: %v", err)
	}

	// Inverse transform (in-place)
	err = executor.InverseInPlace(data)
	if err != nil {
		t.Fatalf("executor.InverseInPlace() failed: %v", err)
	}

	// Verify round-trip (Inverse automatically normalizes by 1/N)
	expectedFirst := complex(1, 0)
	if absComplex128(data[0]-expectedFirst) > 1e-10 {
		t.Errorf("data[0] = %v, want %v", data[0], expectedFirst)
	}

	for i := 1; i < len(data); i++ {
		if absComplex128(data[i]) > 1e-10 {
			t.Errorf("data[%d] = %v, want ~0", i, data[i])
		}
	}
}

func TestExecutor_RoundTrip(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan32(512)
	if err != nil {
		t.Fatalf("NewPlan32(512) failed: %v", err)
	}

	executor := plan.NewExecutor()

	// Create test signal
	original := make([]complex64, 512)
	for i := range original {
		original[i] = complex(float32(i), float32(-i))
	}

	// Forward transform
	freq := make([]complex64, 512)
	err = executor.Forward(freq, original)
	if err != nil {
		t.Fatalf("executor.Forward() failed: %v", err)
	}

	// Inverse transform
	recovered := make([]complex64, 512)
	err = executor.Inverse(recovered, freq)
	if err != nil {
		t.Fatalf("executor.Inverse() failed: %v", err)
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

func TestExecutor_ConcurrentSafety(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan32(256)
	if err != nil {
		t.Fatalf("NewPlan32(256) failed: %v", err)
	}

	// Create multiple executors from the same plan
	exec1 := plan.NewExecutor()
	exec2 := plan.NewExecutor()

	// Verify they are independent
	if exec1.plan == exec2.plan {
		t.Error("Multiple executors share the same plan - not safe for concurrent use")
	}

	// Run transforms concurrently
	done := make(chan bool, 2)

	go func() {
		src := make([]complex64, 256)
		src[0] = 1
		dst := make([]complex64, 256)
		if err := exec1.Forward(dst, src); err != nil {
			t.Errorf("exec1.Forward() failed: %v", err)
		}
		done <- true
	}()

	go func() {
		src := make([]complex64, 256)
		src[1] = 1
		dst := make([]complex64, 256)
		if err := exec2.Forward(dst, src); err != nil {
			t.Errorf("exec2.Forward() failed: %v", err)
		}
		done <- true
	}()

	<-done
	<-done
}

func TestExecutor_Close(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan32(128)
	if err != nil {
		t.Fatalf("NewPlan32(128) failed: %v", err)
	}

	executor := plan.NewExecutor()

	// Close the executor
	executor.Close()

	// Verify plan is set to nil
	if executor.plan != nil {
		t.Error("executor.Close() did not set plan to nil")
	}

	// Multiple calls to Close should not panic
	executor.Close()
}

func TestExecutor_CloseNil(t *testing.T) {
	t.Parallel()

	var executor *Executor[complex64]

	// Closing a nil executor should not panic
	executor.Close()
}

func TestExecutor_Complex64AndComplex128(t *testing.T) {
	t.Parallel()

	// Test with complex64
	plan32, err := NewPlan32(64)
	if err != nil {
		t.Fatalf("NewPlan32(64) failed: %v", err)
	}

	exec32 := plan32.NewExecutor()
	src32 := make([]complex64, 64)
	src32[0] = 1
	dst32 := make([]complex64, 64)

	err = exec32.Forward(dst32, src32)
	if err != nil {
		t.Errorf("complex64 executor.Forward() failed: %v", err)
	}

	// Test with complex128
	plan64, err := NewPlan64(64)
	if err != nil {
		t.Fatalf("NewPlan64(64) failed: %v", err)
	}

	exec64 := plan64.NewExecutor()
	src64 := make([]complex128, 64)
	src64[0] = 1
	dst64 := make([]complex128, 64)

	err = exec64.Forward(dst64, src64)
	if err != nil {
		t.Errorf("complex128 executor.Forward() failed: %v", err)
	}

	// Verify both produce similar results (accounting for precision)
	for i := 0; i < 64; i++ {
		diff := complex128(dst32[i]) - dst64[i]
		if absComplex128(diff) > 1e-5 {
			t.Errorf("Results differ at index %d: complex64=%v, complex128=%v", i, dst32[i], dst64[i])
		}
	}
}

func TestExecutor_ErrorPropagation(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan32(128)
	if err != nil {
		t.Fatalf("NewPlan32(128) failed: %v", err)
	}

	executor := plan.NewExecutor()

	// Test with wrong size slice (should propagate error from plan)
	src := make([]complex64, 64) // Wrong size
	dst := make([]complex64, 128)

	err = executor.Forward(dst, src)
	if err == nil {
		t.Error("executor.Forward() with wrong size did not return error")
	}

	// Test inverse with wrong size
	src = make([]complex64, 128)
	dst = make([]complex64, 64) // Wrong size

	err = executor.Inverse(dst, src)
	if err == nil {
		t.Error("executor.Inverse() with wrong size did not return error")
	}
}

// Helper function for complex128 absolute value
func absComplex128(v complex128) float64 {
	return real(v)*real(v) + imag(v)*imag(v)
}
