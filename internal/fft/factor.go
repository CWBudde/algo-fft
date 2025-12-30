package fft

import m "github.com/MeKo-Christian/algo-fft/internal/math"

// Re-export public function from internal/math
var IsHighlyComposite = m.IsHighlyComposite

// Private re-exports for internal use
var (
	factorize         = m.Factorize
	isHighlyComposite = m.IsHighlyComposite
)
