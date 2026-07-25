package transform

// ScratchSizeRecursive returns the scratch size required for a recursive strategy.
// The size accounts for holding all sub-results, one decimated sub-FFT input,
// and the maximum scratch needed by any single sub-FFT (subcalls are executed
// sequentially). splitScratch carves the buffer up in that order.
func ScratchSizeRecursive(strategy *DecomposeStrategy) int {
	if strategy == nil {
		return 0
	}

	if strategy.UseCodelet || strategy.Recursive == nil {
		return strategy.Size
	}

	subScratch := ScratchSizeRecursive(strategy.Recursive)

	// One shared decimated-input buffer serves every sub-FFT at this level:
	// each consumes its input before the next decimation overwrites it.
	return strategy.SplitFactor*strategy.SubSize + strategy.SubSize + subScratch
}
