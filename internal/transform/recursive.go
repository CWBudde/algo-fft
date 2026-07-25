package transform

import (
	"github.com/cwbudde/algo-fft/internal/cpu"
	imath "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/registry"
)

// recursive.go implements the recursive FFT algorithm using decomposition strategies.

// leafCodelet returns the registered codelet for a leaf of size n, but only
// when that codelet consumes the standard length-n DIT twiddle table.
//
// Codelets may instead declare a SIMD-friendly twiddle layout through
// TwiddleSize/PrepareTwiddle (registry.CodeletEntry); the non-recursive plan
// path materializes those tables up front (see plan_alloc.go). The recursive
// decomposition carries one standard table laid out by estimateTwiddleSize,
// so handing such a codelet that table would transform against the wrong
// factors and silently return a wrong spectrum — dit256_radix16_avx2, for
// example, asks for 748 elements where the leaf slice supplies 256.
//
// Returning nil routes the leaf to the generic DIT fallback, which is correct
// for every size. Binding prepared-twiddle codelets here would need per-leaf
// forward and inverse tables built at plan time; that is a performance
// opportunity, not a correctness requirement.
func leafCodelet[T Complex](
	reg *registry.CodeletRegistry[T],
	n int,
	features cpu.Features,
) *registry.CodeletEntry[T] {
	entry := reg.Lookup(n, features)
	if entry == nil || entry.TwiddleSize != nil || entry.PrepareTwiddle != nil {
		return nil
	}

	return entry
}

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
	registry *registry.CodeletRegistry[T],
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
	registry *registry.CodeletRegistry[T],
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
	registry *registry.CodeletRegistry[T],
	features cpu.Features,
) int {
	n := len(src)

	// Base case: use codelet
	if strategy.UseCodelet {
		twiddleSlice := twiddle[twiddleOffset : twiddleOffset+n]

		// Kernels are self-contained and handle permutation internally.
		// A codelet reports false when it bailed without doing any work;
		// fall back to generic DIT then (as when no codelet is registered).
		codelet := leafCodelet(registry, n, features)
		if codelet == nil || !codelet.Forward(dst, src, twiddleSlice, scratch) {
			ditForwardBitrev(dst, src, twiddleSlice, scratch, strategy.LeafBitrev)
		}

		return twiddleOffset + n
	}

	// Recursive case: split and combine
	radix := strategy.SplitFactor
	subSize := strategy.SubSize
	blockSize := radix * subSize

	subResults, subInput, subScratch := splitScratch(scratch, strategy)

	combineBlock := twiddle[twiddleOffset : twiddleOffset+blockSize]
	twiddleOffset += blockSize

	// Decimate and transform one sub-sequence at a time: extract the strided
	// sub-sequence (indices congruent to i mod radix), then recurse on it.
	// subInput is reused across the radix passes because each sub-FFT fully
	// consumes its input before the next decimation overwrites it.
	for i := range radix {
		for j := range subSize {
			subInput[j] = src[i+j*radix]
		}

		twiddleOffset = recursiveForwardWithTwiddle(
			subResults[i*subSize:(i+1)*subSize],
			subInput,
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
		combineRadix2(dst, subResults[:subSize], subResults[subSize:2*subSize], tw)
	case 4:
		tw1 := combineBlock[subSize : 2*subSize]
		tw2 := combineBlock[2*subSize : 3*subSize]
		tw3 := combineBlock[3*subSize : 4*subSize]
		combineRadix4(dst,
			subResults[:subSize], subResults[subSize:2*subSize],
			subResults[2*subSize:3*subSize], subResults[3*subSize:4*subSize],
			tw1, tw2, tw3)
	case 8:
		combineRadix8(dst, subResults, combineBlock, subSize)
	default:
		combineGeneral(dst, subResults, combineBlock, subSize, radix)
	}

	return twiddleOffset
}

// splitScratch carves one recursion level's working set out of scratch,
// matching the layout ScratchSizeRecursive reserves:
//
//	[0 : radix*subSize)          sub-FFT results, flat in [r][k] order
//	[radix*subSize : +subSize)   decimated input for the sub-FFT in flight
//	[remainder]                  scratch for the sub-FFT itself
//
// Only one decimated-input buffer is needed because sub-FFTs run one at a
// time. The three regions are disjoint, so a sub-FFT may write its results
// while reading its input.
//
// The returned slices are, in order: sub-results, decimated input, sub-scratch.
func splitScratch[T Complex](scratch []T, strategy *DecomposeStrategy) ([]T, []T, []T) {
	blockSize := strategy.SplitFactor * strategy.SubSize

	subResults := scratch[:blockSize]
	subInput := scratch[blockSize : blockSize+strategy.SubSize]
	subScratch := scratch[blockSize+strategy.SubSize:][:ScratchSizeRecursive(strategy.Recursive)]

	return subResults, subInput, subScratch
}

// RecursiveInverse executes an inverse FFT using recursive decomposition.
//
// This is the main entry point for recursive inverse FFT transforms.
func RecursiveInverse[T Complex](
	dst, src []T,
	strategy *DecomposeStrategy,
	twiddle []T,
	scratch []T,
	registry *registry.CodeletRegistry[T],
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
	registry *registry.CodeletRegistry[T],
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
	registry *registry.CodeletRegistry[T],
	features cpu.Features,
) int {
	n := len(src)

	// Base case: use codelet
	if strategy.UseCodelet {
		twiddleSlice := twiddle[twiddleOffset : twiddleOffset+n]

		// See the forward path: fall back to generic DIT when the codelet is
		// missing or bailed without doing any work.
		codelet := leafCodelet(registry, n, features)
		if codelet == nil || !codelet.Inverse(dst, src, twiddleSlice, scratch) {
			ditInverseBitrev(dst, src, twiddleSlice, scratch, strategy.LeafBitrev)
		}

		return twiddleOffset + n
	}

	// Recursive case: similar to forward, but use inverse twiddles
	radix := strategy.SplitFactor
	subSize := strategy.SubSize

	blockSize := radix * subSize

	subResults, subInput, subScratch := splitScratch(scratch, strategy)

	combineBlock := twiddle[twiddleOffset : twiddleOffset+blockSize]
	twiddleOffset += blockSize

	// Decimate and transform one sub-sequence at a time; see the forward path
	// for why a single reused input buffer is sufficient.
	for i := range radix {
		for j := range subSize {
			subInput[j] = src[i+j*radix]
		}

		twiddleOffset = recursiveInverseWithTwiddle(
			subResults[i*subSize:(i+1)*subSize],
			subInput,
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
		combineRadix2Conj(dst, subResults[:subSize], subResults[subSize:2*subSize], tw)
	case 4:
		tw1 := combineBlock[subSize : 2*subSize]
		tw2 := combineBlock[2*subSize : 3*subSize]
		tw3 := combineBlock[3*subSize : 4*subSize]
		combineRadix4Conj(dst,
			subResults[:subSize], subResults[subSize:2*subSize],
			subResults[2*subSize:3*subSize], subResults[3*subSize:4*subSize],
			tw1, tw2, tw3)
	case 8:
		combineRadix8Conj(dst, subResults, combineBlock, subSize)
	default:
		combineGeneralConj(dst, subResults, combineBlock, subSize, radix)
	}

	scaleComplexSlice(dst, 1.0/float64(radix))

	return twiddleOffset
}

func combineRadix2Conj[T Complex](dst []T, sub0, sub1 []T, twiddle []T) {
	half := len(sub0)
	for k := range half {
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

	for k := range quarter {
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

// combineRadix8Conj mirrors combineRadix8 with conjugated twiddles and
// rotations. subs and twiddles are flat blocks in [r][k] order.
func combineRadix8Conj[T Complex](dst, subs, twiddles []T, subSize int) {
	var roots [8]T

	for j := range roots {
		angle := imath.TwoPi * float64(j) / 8.0
		roots[j] = T(complex(cos64(angle), sin64(angle)))
	}

	var t [8]T

	for k := range subSize {
		t[0] = subs[k]
		for r := 1; r < 8; r++ {
			t[r] = conj(twiddles[r*subSize+k]) * subs[r*subSize+k]
		}

		for bin := range 8 {
			sum := t[0]
			for r := 1; r < 8; r++ {
				sum += roots[(bin*r)&7] * t[r]
			}

			dst[k+bin*subSize] = sum
		}
	}
}

// combineGeneralConj mirrors combineGeneral with conjugated twiddles and
// rotations. See combineGeneral for the loop ordering rationale.
func combineGeneralConj[T Complex](dst, subs, twiddles []T, subSize, radix int) {
	for bin := range radix {
		out := dst[bin*subSize : (bin+1)*subSize]
		copy(out, subs[:subSize])

		for r := 1; r < radix; r++ {
			angle := imath.TwoPi * float64((bin*r)%radix) / float64(radix)
			w := T(complex(cos64(angle), sin64(angle)))

			tw := twiddles[r*subSize : (r+1)*subSize]
			sub := subs[r*subSize : (r+1)*subSize]

			for k := range subSize {
				out[k] += w * (conj(tw[k]) * sub[k])
			}
		}
	}
}

func scaleComplexSlice[T Complex](dst []T, scale float64) {
	switch dt := any(dst).(type) {
	case []complex64:
		s := complex(float32(scale), 0)
		for i := range dt {
			dt[i] *= s
		}
	case []complex128:
		s := complex(scale, 0)
		for i := range dt {
			dt[i] *= s
		}
	default:
		panic("unsupported complex type")
	}
}
