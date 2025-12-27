package algofft

// Executor runs transforms with its own workspace.
// Executors are safe for concurrent use as long as each goroutine
// uses a distinct Executor.
type Executor[T Complex] struct {
	plan *Plan[T]
}

// NewExecutor creates an executor with its own workspace.
func (p *Plan[T]) NewExecutor() *Executor[T] {
	return &Executor[T]{plan: p.Clone()}
}

// Forward computes the forward transform using the executor's workspace.
func (e *Executor[T]) Forward(dst, src []T) error {
	return e.plan.Forward(dst, src)
}

// Inverse computes the inverse transform using the executor's workspace.
func (e *Executor[T]) Inverse(dst, src []T) error {
	return e.plan.Inverse(dst, src)
}

// ForwardInPlace computes the forward transform in-place.
func (e *Executor[T]) ForwardInPlace(data []T) error {
	return e.plan.Forward(data, data)
}

// InverseInPlace computes the inverse transform in-place.
func (e *Executor[T]) InverseInPlace(data []T) error {
	return e.plan.Inverse(data, data)
}

// Close releases any pooled resources (no-op for executors).
func (e *Executor[T]) Close() {
	if e == nil || e.plan == nil {
		return
	}

	e.plan.Close()
	e.plan = nil
}
