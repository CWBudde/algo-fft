package fft

// ScratchSizeRecursive returns the scratch size required for a recursive strategy.
// The size accounts for holding all sub-results plus the maximum scratch needed
// by any single sub-FFT (subcalls are executed sequentially).
func ScratchSizeRecursive(strategy *DecomposeStrategy) int {
	if strategy == nil {
		return 0
	}

	if strategy.UseCodelet || strategy.Recursive == nil {
		return strategy.Size
	}

	subScratch := ScratchSizeRecursive(strategy.Recursive)

	return strategy.SplitFactor*strategy.SubSize + subScratch
}
