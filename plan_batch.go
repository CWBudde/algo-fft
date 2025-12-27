package algoforge

func resolveBatchStride(size int, opts PlanOptions) (int, int, error) {
	batch := opts.Batch
	if batch < 1 {
		batch = 1
	}

	stride := opts.Stride
	if stride <= 0 {
		stride = size
	}

	if stride < size {
		return 0, 0, ErrInvalidStride
	}

	return batch, stride, nil
}

func resolveBatchStrideReal(inSize, outSize int, opts PlanOptions) (int, int, int, error) {
	batch := opts.Batch
	if batch < 1 {
		batch = 1
	}

	strideIn := opts.Stride
	if strideIn <= 0 {
		strideIn = inSize
	}

	strideOut := opts.Stride
	if strideOut <= 0 {
		strideOut = outSize
	}

	if strideIn < inSize || strideOut < outSize {
		return 0, 0, 0, ErrInvalidStride
	}

	return batch, strideIn, strideOut, nil
}
