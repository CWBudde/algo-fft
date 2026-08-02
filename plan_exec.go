package algofft

import (
	"strconv"

	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/kernels"
	"github.com/cwbudde/algo-fft/internal/transform"
)

// planExecutor is the strategy-specific engine behind Plan.Forward/Inverse.
// One implementation exists per strategy family (kernel/codelet, recursive,
// Bluestein, Rader), each owning only its own precomputed tables; the Plan
// keeps validation, scratch management, and introspection.
//
// Executors are immutable after construction, so a Plan and its clones share
// a single executor instance. The constructor guarantees the executor can run
// every transform for its plan configuration, so the methods do not return
// errors; an executor that still cannot proceed panics on the broken
// plan-construction invariant.
//
// scratch is the plan's main scratch buffer (length Plan.scratchLen); sub is
// the Bluestein/Rader sub-FFT scratch (length Plan.subScratchLen, nil for the
// other strategies).
type planExecutor[T Complex] interface {
	forward(dst, src, scratch, sub []T)
	inverse(dst, src, scratch, sub []T)
}

// Compile-time interface conformance for every executor, both precisions.
var (
	_ planExecutor[complex64]  = (*kernelExecutor[complex64])(nil)
	_ planExecutor[complex128] = (*kernelExecutor[complex128])(nil)
	_ planExecutor[complex64]  = (*bluesteinExecutor[complex64])(nil)
	_ planExecutor[complex128] = (*bluesteinExecutor[complex128])(nil)
	_ planExecutor[complex64]  = (*raderExecutor[complex64])(nil)
	_ planExecutor[complex128] = (*raderExecutor[complex128])(nil)
	_ planExecutor[complex64]  = (*recursiveExecutor[complex64])(nil)
	_ planExecutor[complex128] = (*recursiveExecutor[complex128])(nil)
)

// kernelExecutor runs the codelet/kernel strategy family (DIT, Stockham,
// six-step, split-radix, mixed-radix): a zero-dispatch codelet
// when one is bound, the pure-Go packed Stockham route when enabled, and the
// strategy-dispatched fallback kernel otherwise.
type kernelExecutor[T Complex] struct {
	// Zero-dispatch codelet bindings (nil = use fallback kernel) and their
	// twiddle layouts (alias twiddle when the codelet uses standard twiddles).
	forwardCodelet        fftypes.CodeletFunc[T]
	inverseCodelet        fftypes.CodeletFunc[T]
	codeletTwiddleForward []T
	codeletTwiddleInverse []T

	// backing buffers keep codelet-specific aligned twiddles alive for GC
	// (nil when the codelet layouts alias the standard table).
	codeletTwiddleForwardBacking []byte
	codeletTwiddleInverseBacking []byte

	// twiddle is the standard twiddle table shared with the owning Plan.
	twiddle []T

	forwardKernel kernels.Kernel[T]
	inverseKernel kernels.Kernel[T]

	// packed enables the packed radix-4 Stockham route; both directions share
	// it — the inverse conjugates the twiddles on load. It is non-nil only for
	// Stockham-resolved plans at sizes where that route was measured to beat
	// the bound kernel on this instruction-set tier and precision (see
	// transform.PackedStockhamEnabled).
	//
	// It can never coexist with a bound codelet: every registered codelet is
	// Algorithm: KernelDIT, so an estimate that bound one does not report
	// KernelStockham. The codelet-first ordering below is therefore not what
	// keeps the two apart, despite what the old build toggle claimed.
	packed *transform.PackedTwiddles[T]
}

func (e *kernelExecutor[T]) forward(dst, src, scratch, _ []T) {
	// Zero-dispatch codelet path (highest priority). A codelet reports false
	// when it bailed without doing any work; fall through to the generic
	// dispatch below then instead of returning unset output.
	if e.forwardCodelet != nil && e.forwardCodelet(dst, src, e.codeletTwiddleForward, scratch) {
		return
	}

	if e.packed != nil && transform.ForwardStockhamPacked(dst, src, e.twiddle, scratch, e.packed) {
		return
	}

	if e.forwardKernel(dst, src, e.twiddle, scratch) {
		return
	}

	panic("algofft: internal error: bound forward kernel rejected size " + strconv.Itoa(len(src)))
}

func (e *kernelExecutor[T]) inverse(dst, src, scratch, _ []T) {
	// Mirror of forward: codelet, then packed Stockham, then fallback kernel.
	if e.inverseCodelet != nil && e.inverseCodelet(dst, src, e.codeletTwiddleInverse, scratch) {
		return
	}

	if e.packed != nil && transform.InverseStockhamPacked(dst, src, e.twiddle, scratch, e.packed) {
		return
	}

	if e.inverseKernel(dst, src, e.twiddle, scratch) {
		return
	}

	panic("algofft: internal error: bound inverse kernel rejected size " + strconv.Itoa(len(src)))
}
