package fft

import "github.com/MeKo-Christian/algo-fft/internal/cpu"

// recursive.go implements the recursive FFT algorithm using decomposition strategies.

// RecursiveForward executes a forward FFT using recursive decomposition.
// It splits the problem according to the strategy, calls codelets for base cases,
// and combines results using twiddle factors.
//
// This is the main entry point for recursive FFT transforms.
func RecursiveForward[T Complex](
	dst, src []T,
	strategy *DecomposeStrategy,
	twiddle []T,
	scratch []T,
	registry *CodeletRegistry[T],
	features cpu.Features,
) {
	recursiveForwardWithTwiddle(dst, src, strategy, twiddle, 0, scratch, registry, features)
}

// recursiveForward is an internal wrapper for tests in the same package.
func recursiveForward[T Complex](
	dst, src []T,
	strategy *DecomposeStrategy,
	twiddle []T,
	scratch []T,
	registry *CodeletRegistry[T],
	features cpu.Features,
) {
	RecursiveForward(dst, src, strategy, twiddle, scratch, registry, features)
}

func recursiveForwardWithTwiddle[T Complex](
	dst, src []T,
	strategy *DecomposeStrategy,
	twiddle []T,
	twiddleOffset int,
	scratch []T,
	registry *CodeletRegistry[T],
	features cpu.Features,
) int {
	n := len(src)

	// Base case: use codelet
	if strategy.UseCodelet {
		twiddleSlice := twiddle[twiddleOffset : twiddleOffset+n]
		codelet := registry.Lookup(n, features)
		if codelet != nil {
			// Call the codelet's forward function
			var bitrev []int
			if codelet.BitrevFunc != nil {
				bitrev = codelet.BitrevFunc(n)
			}
			codelet.Forward(dst, src, twiddleSlice, scratch, bitrev)
			return twiddleOffset + n
		}
		// Fallback to generic DIT if codelet missing (should not happen if registry is correct)
		ditForward(dst, src, twiddleSlice, scratch, ComputeBitReversalIndices(n))
		return twiddleOffset + n
	}

	// Recursive case: split and combine
	radix := strategy.SplitFactor
	subSize := strategy.SubSize

	// Allocate sub-result buffers from scratch space
	// Layout: [sub0 | sub1 | ... | subN | remaining scratch]
	subResults := make([][]T, radix)
	for i := 0; i < radix; i++ {
		subResults[i] = scratch[i*subSize : (i+1)*subSize]
	}
	remainingScratch := scratch[radix*subSize:]
	subScratchSize := ScratchSizeRecursive(strategy.Recursive)
	subScratch := remainingScratch[:subScratchSize]

	// Allocate sub-input buffers (temporary, could optimize with in-place decimation)
	subInputs := make([][]T, radix)
	for i := 0; i < radix; i++ {
		subInputs[i] = make([]T, subSize)
	}

	// Decimate input: extract strided sub-sequences
	// For radix-2: even/odd indices
	// For radix-4: indices mod 4
	// General: indices mod radix
	for i := 0; i < radix; i++ {
		for j := 0; j < subSize; j++ {
			subInputs[i][j] = src[i+j*radix]
		}
	}

	blockSize := radix * subSize
	combineBlock := twiddle[twiddleOffset : twiddleOffset+blockSize]
	twiddleOffset += blockSize

	// Recursively compute sub-FFTs
	for i := 0; i < radix; i++ {
		twiddleOffset = recursiveForwardWithTwiddle(
			subResults[i],
			subInputs[i],
			strategy.Recursive,
			twiddle,
			twiddleOffset,
			subScratch,
			registry,
			features,
		)
	}

	// Combine sub-results with twiddle factors
	switch radix {
	case 2:
		tw := combineBlock[subSize : 2*subSize]
		combineRadix2(dst, subResults[0], subResults[1], tw)
	case 4:
		tw1 := combineBlock[subSize : 2*subSize]
		tw2 := combineBlock[2*subSize : 3*subSize]
		tw3 := combineBlock[3*subSize : 4*subSize]
		combineRadix4(dst, subResults[0], subResults[1], subResults[2], subResults[3], tw1, tw2, tw3)
	case 8:
		twiddles := splitTwiddleBlock(combineBlock, radix, subSize)
		combineRadix8(dst, subResults, twiddles)
	default:
		twiddles := splitTwiddleBlock(combineBlock, radix, subSize)
		combineGeneral(dst, subResults, twiddles, radix)
	}

	return twiddleOffset
}

// RecursiveInverse executes an inverse FFT using recursive decomposition.
//
// This is the main entry point for recursive inverse FFT transforms.
func RecursiveInverse[T Complex](
	dst, src []T,
	strategy *DecomposeStrategy,
	twiddle []T,
	scratch []T,
	registry *CodeletRegistry[T],
	features cpu.Features,
) {
	recursiveInverseWithTwiddle(dst, src, strategy, twiddle, 0, scratch, registry, features)
}

// recursiveInverse is an internal wrapper for tests in the same package.
func recursiveInverse[T Complex](
	dst, src []T,
	strategy *DecomposeStrategy,
	twiddle []T,
	scratch []T,
	registry *CodeletRegistry[T],
	features cpu.Features,
) {
	RecursiveInverse(dst, src, strategy, twiddle, scratch, registry, features)
}

func recursiveInverseWithTwiddle[T Complex](
	dst, src []T,
	strategy *DecomposeStrategy,
	twiddle []T,
	twiddleOffset int,
	scratch []T,
	registry *CodeletRegistry[T],
	features cpu.Features,
) int {
	n := len(src)

	// Base case: use codelet
	if strategy.UseCodelet {
		twiddleSlice := twiddle[twiddleOffset : twiddleOffset+n]
		codelet := registry.Lookup(n, features)
		if codelet != nil {
			var bitrev []int
			if codelet.BitrevFunc != nil {
				bitrev = codelet.BitrevFunc(n)
			}
			codelet.Inverse(dst, src, twiddleSlice, scratch, bitrev)
			return twiddleOffset + n
		}
		// Fallback
		ditInverse(dst, src, twiddleSlice, scratch, ComputeBitReversalIndices(n))
		return twiddleOffset + n
	}

	// Recursive case: similar to forward, but use inverse twiddles
	radix := strategy.SplitFactor
	subSize := strategy.SubSize

	subResults := make([][]T, radix)
	for i := 0; i < radix; i++ {
		subResults[i] = scratch[i*subSize : (i+1)*subSize]
	}
	remainingScratch := scratch[radix*subSize:]
	subScratchSize := ScratchSizeRecursive(strategy.Recursive)
	subScratch := remainingScratch[:subScratchSize]

	subInputs := make([][]T, radix)
	for i := 0; i < radix; i++ {
		subInputs[i] = make([]T, subSize)
	}

	// Decimate input
	for i := 0; i < radix; i++ {
		for j := 0; j < subSize; j++ {
			subInputs[i][j] = src[i+j*radix]
		}
	}

	blockSize := radix * subSize
	combineBlock := twiddle[twiddleOffset : twiddleOffset+blockSize]
	twiddleOffset += blockSize

	// Recursively compute sub-IFFTs
	for i := 0; i < radix; i++ {
		twiddleOffset = recursiveInverseWithTwiddle(
			subResults[i],
			subInputs[i],
			strategy.Recursive,
			twiddle,
			twiddleOffset,
			subScratch,
			registry,
			features,
		)
	}

	// Combine with inverse twiddles (conjugated)
	switch radix {
	case 2:
		tw := combineBlock[subSize : 2*subSize]
		combineRadix2Conj(dst, subResults[0], subResults[1], tw)
	case 4:
		tw1 := combineBlock[subSize : 2*subSize]
		tw2 := combineBlock[2*subSize : 3*subSize]
		tw3 := combineBlock[3*subSize : 4*subSize]
		combineRadix4Conj(dst, subResults[0], subResults[1], subResults[2], subResults[3], tw1, tw2, tw3)
	case 8:
		twiddles := splitTwiddleBlock(combineBlock, radix, subSize)
		combineRadix8Conj(dst, subResults, twiddles)
	default:
		twiddles := splitTwiddleBlock(combineBlock, radix, subSize)
		combineGeneralConj(dst, subResults, twiddles, radix)
	}

	scaleComplexSlice(dst, 1.0/float64(radix))
	return twiddleOffset
}

// Helper functions for generating twiddle factors on-the-fly
// (These are temporary; we'll optimize with precomputation in twiddle_recursive.go)

func generateCombineTwiddles[T Complex](n, radix int) []T {
	subSize := n / radix
	twiddles := make([]T, subSize)
	for k := 0; k < subSize; k++ {
		angle := -2.0 * 3.14159265358979323846 * float64(k) / float64(n)
		twiddles[k] = makeComplex[T](angle)
	}
	return twiddles
}

func generateCombineTwiddlesInverse[T Complex](n, radix int) []T {
	subSize := n / radix
	twiddles := make([]T, subSize)
	for k := 0; k < subSize; k++ {
		angle := 2.0 * 3.14159265358979323846 * float64(k) / float64(n) // Note: positive for inverse
		twiddles[k] = makeComplex[T](angle)
	}
	return twiddles
}

func generateRadix4Twiddles[T Complex](n int) ([]T, []T, []T) {
	quarter := n / 4
	tw1 := make([]T, quarter)
	tw2 := make([]T, quarter)
	tw3 := make([]T, quarter)

	for k := 0; k < quarter; k++ {
		angle1 := -2.0 * 3.14159265358979323846 * float64(k) / float64(n)
		angle2 := -2.0 * 3.14159265358979323846 * float64(2*k) / float64(n)
		angle3 := -2.0 * 3.14159265358979323846 * float64(3*k) / float64(n)

		tw1[k] = makeComplex[T](angle1)
		tw2[k] = makeComplex[T](angle2)
		tw3[k] = makeComplex[T](angle3)
	}

	return tw1, tw2, tw3
}

func generateRadix4TwiddlesInverse[T Complex](n int) ([]T, []T, []T) {
	quarter := n / 4
	tw1 := make([]T, quarter)
	tw2 := make([]T, quarter)
	tw3 := make([]T, quarter)

	for k := 0; k < quarter; k++ {
		angle1 := 2.0 * 3.14159265358979323846 * float64(k) / float64(n)
		angle2 := 2.0 * 3.14159265358979323846 * float64(2*k) / float64(n)
		angle3 := 2.0 * 3.14159265358979323846 * float64(3*k) / float64(n)

		tw1[k] = makeComplex[T](angle1)
		tw2[k] = makeComplex[T](angle2)
		tw3[k] = makeComplex[T](angle3)
	}

	return tw1, tw2, tw3
}

func generateRadix8Twiddles[T Complex](n int) [][]T {
	eighth := n / 8
	twiddles := make([][]T, 8)

	twiddles[0] = make([]T, eighth)
	for k := 0; k < eighth; k++ {
		twiddles[0][k] = makeComplex[T](0) // W^0 = 1
	}

	for r := 1; r < 8; r++ {
		twiddles[r] = make([]T, eighth)
		for k := 0; k < eighth; k++ {
			angle := -2.0 * 3.14159265358979323846 * float64(r*k) / float64(n)
			twiddles[r][k] = makeComplex[T](angle)
		}
	}

	return twiddles
}

func generateGeneralTwiddles[T Complex](n, radix int) [][]T {
	subSize := n / radix
	twiddles := make([][]T, radix)

	twiddles[0] = make([]T, subSize)
	for k := 0; k < subSize; k++ {
		twiddles[0][k] = makeComplex[T](0) // W^0 = 1
	}

	for r := 1; r < radix; r++ {
		twiddles[r] = make([]T, subSize)
		for k := 0; k < subSize; k++ {
			angle := -2.0 * 3.14159265358979323846 * float64(r*k) / float64(n)
			twiddles[r][k] = makeComplex[T](angle)
		}
	}

	return twiddles
}

func generateGeneralTwiddlesInverse[T Complex](n, radix int) [][]T {
	subSize := n / radix
	twiddles := make([][]T, radix)

	twiddles[0] = make([]T, subSize)
	for k := 0; k < subSize; k++ {
		twiddles[0][k] = makeComplex[T](0) // W^0 = 1
	}

	for r := 1; r < radix; r++ {
		twiddles[r] = make([]T, subSize)
		for k := 0; k < subSize; k++ {
			angle := 2.0 * 3.14159265358979323846 * float64(r*k) / float64(n)
			twiddles[r][k] = makeComplex[T](angle)
		}
	}

	return twiddles
}

func splitTwiddleBlock[T Complex](block []T, radix, subSize int) [][]T {
	twiddles := make([][]T, radix)
	for r := 0; r < radix; r++ {
		start := r * subSize
		twiddles[r] = block[start : start+subSize]
	}
	return twiddles
}

func combineRadix2Conj[T Complex](dst []T, sub0, sub1 []T, twiddle []T) {
	half := len(sub0)
	for k := 0; k < half; k++ {
		t := conj(twiddle[k]) * sub1[k]
		dst[k] = sub0[k] + t
		dst[k+half] = sub0[k] - t
	}
}

func combineRadix4Conj[T Complex](
	dst []T,
	sub0, sub1, sub2, sub3 []T,
	twiddle1, twiddle2, twiddle3 []T,
) {
	quarter := len(sub0)

	for k := 0; k < quarter; k++ {
		t1 := conj(twiddle1[k]) * sub1[k]
		t2 := conj(twiddle2[k]) * sub2[k]
		t3 := conj(twiddle3[k]) * sub3[k]

		s0 := sub0[k]

		posIT1 := multiplyByI(t1)
		negIT1 := multiplyByNegI(t1)
		posIT3 := multiplyByI(t3)
		negIT3 := multiplyByNegI(t3)

		dst[k+0*quarter] = s0 + t1 + t2 + t3
		dst[k+1*quarter] = s0 + posIT1 - t2 + negIT3
		dst[k+2*quarter] = s0 - t1 + t2 - t3
		dst[k+3*quarter] = s0 + negIT1 - t2 + posIT3
	}
}

func combineRadix8Conj[T Complex](dst []T, subs [][]T, twiddles [][]T) {
	eighth := len(subs[0])

	for k := 0; k < eighth; k++ {
		t := make([]T, 8)
		t[0] = subs[0][k]
		for r := 1; r < 8; r++ {
			t[r] = conj(twiddles[r][k]) * subs[r][k]
		}

		for bin := 0; bin < 8; bin++ {
			sum := T(0)
			for r := 0; r < 8; r++ {
				angle := 2.0 * 3.14159265358979323846 * float64(bin*r) / 8.0
				w := T(complex(cos64(angle), sin64(angle)))
				sum += w * t[r]
			}
			dst[k+bin*eighth] = sum
		}
	}
}

func combineGeneralConj[T Complex](dst []T, subs [][]T, twiddles [][]T, radix int) {
	subSize := len(subs[0])

	for k := 0; k < subSize; k++ {
		t := make([]T, radix)
		t[0] = subs[0][k]
		for r := 1; r < radix; r++ {
			t[r] = conj(twiddles[r][k]) * subs[r][k]
		}

		for bin := 0; bin < radix; bin++ {
			sum := T(0)
			for r := 0; r < radix; r++ {
				angle := 2.0 * 3.14159265358979323846 * float64(bin*r) / float64(radix)
				w := T(complex(cos64(angle), sin64(angle)))
				sum += w * t[r]
			}
			dst[k+bin*subSize] = sum
		}
	}
}

func scaleComplexSlice[T Complex](dst []T, scale float64) {
	var zero T
	switch any(zero).(type) {
	case complex64:
		s := complex(float32(scale), 0)
		for i := range dst {
			dst[i] = any(any(dst[i]).(complex64) * s).(T)
		}
	case complex128:
		s := complex(scale, 0)
		for i := range dst {
			dst[i] = any(any(dst[i]).(complex128) * s).(T)
		}
	default:
		panic("unsupported complex type")
	}
}

func makeComplex[T Complex](angle float64) T {
	var zero T
	switch any(zero).(type) {
	case complex64:
		c := complex(float32(cos64(angle)), float32(sin64(angle)))
		return any(complex64(c)).(T)
	case complex128:
		c := complex(cos64(angle), sin64(angle))
		return any(complex128(c)).(T)
	default:
		panic("unsupported complex type")
	}
}
