package kernels

import (
	"fmt"
	"testing"

	"github.com/cwbudde/algo-fft/internal/registry"
)

// TestCodeletsRegisterBothDirections64 asserts that no registered complex64
// codelet supplies only one direction.
//
// A registry entry carries Forward and Inverse together and Lookup returns one
// entry per (size, features, precision), so a plan's two directions always come
// from the same codelet. A half-registered entry would quietly break that: the
// nil direction falls through kernelExecutor to the generic kernel ladder while
// the other keeps the codelet, and the two directions of one plan would then be
// running different algorithms. That is invisible in correctness tests — both
// paths compute the right answer — and shows up only as an unexplained
// forward/inverse performance asymmetry, which cost a P5.0 investigation real
// time. Both directions must be present or the entry should not be registered.
func TestCodeletsRegisterBothDirections64(t *testing.T) {
	t.Parallel()

	for _, size := range registry.Registry64.Sizes() {
		for _, entry := range registry.Registry64.GetAllForSize(size) {
			t.Run(fmt.Sprintf("size%d/%s", size, entry.Signature), func(t *testing.T) {
				t.Parallel()

				if entry.Forward == nil {
					t.Errorf("codelet %s (size %d) registers no Forward", entry.Signature, size)
				}

				if entry.Inverse == nil {
					t.Errorf("codelet %s (size %d) registers no Inverse", entry.Signature, size)
				}
			})
		}
	}
}

// TestCodeletsRegisterBothDirections128 is the complex128 twin of
// TestCodeletsRegisterBothDirections64.
func TestCodeletsRegisterBothDirections128(t *testing.T) {
	t.Parallel()

	for _, size := range registry.Registry128.Sizes() {
		for _, entry := range registry.Registry128.GetAllForSize(size) {
			t.Run(fmt.Sprintf("size%d/%s", size, entry.Signature), func(t *testing.T) {
				t.Parallel()

				if entry.Forward == nil {
					t.Errorf("codelet %s (size %d) registers no Forward", entry.Signature, size)
				}

				if entry.Inverse == nil {
					t.Errorf("codelet %s (size %d) registers no Inverse", entry.Signature, size)
				}
			})
		}
	}
}
