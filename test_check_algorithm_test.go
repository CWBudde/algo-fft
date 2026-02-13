package algofft

import (
	"testing"
)

func TestWhichAlgorithm(t *testing.T) {
	sizes := []int{10, 20, 30, 40}
	
	for _, size := range sizes {
		plan, _ := NewPlanT[complex128](size)
		t.Logf("Size %d: algorithm=%s, strategy=%v", size, plan.Algorithm(), plan.kernelStrategy)
	}
}
