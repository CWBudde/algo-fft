package fft

// StockhamPackedAvailable reports whether packed Stockham is enabled in this build.
func StockhamPackedAvailable() bool {
	return stockhamPackedEnabled
}

// ForwardStockhamPacked executes a mixed-radix (radix-4 + optional radix-2) Stockham FFT
// using packed radix-4 twiddles when available.
func ForwardStockhamPacked[T Complex](dst, src, twiddle, scratch []T, packed *PackedTwiddles[T]) bool {
	return stockhamPacked(dst, src, twiddle, scratch, packed, false)
}

// InverseStockhamPacked executes a mixed-radix (radix-4 + optional radix-2) Stockham inverse FFT
// using packed radix-4 twiddles when available.
func InverseStockhamPacked[T Complex](dst, src, twiddle, scratch []T, packed *PackedTwiddles[T]) bool {
	return stockhamPacked(dst, src, twiddle, scratch, packed, true)
}

func stockhamPacked[T Complex](dst, src, twiddle, scratch []T, packed *PackedTwiddles[T], inverse bool) bool {
	if !stockhamPackedEnabled {
		return false
	}

	switch any(*new(T)).(type) {
	case complex64:
		dst64, ok := any(dst).([]complex64)
		if !ok {
			return false
		}
		src64, ok := any(src).([]complex64)
		if !ok {
			return false
		}
		tw64, ok := any(twiddle).([]complex64)
		if !ok {
			return false
		}
		scratch64, ok := any(scratch).([]complex64)
		if !ok {
			return false
		}
		packed64, ok := any(packed).(*PackedTwiddles[complex64])
		if !ok {
			return false
		}
		return stockhamPackedComplex64(dst64, src64, tw64, scratch64, packed64, inverse)
	case complex128:
		dst128, ok := any(dst).([]complex128)
		if !ok {
			return false
		}
		src128, ok := any(src).([]complex128)
		if !ok {
			return false
		}
		tw128, ok := any(twiddle).([]complex128)
		if !ok {
			return false
		}
		scratch128, ok := any(scratch).([]complex128)
		if !ok {
			return false
		}
		packed128, ok := any(packed).(*PackedTwiddles[complex128])
		if !ok {
			return false
		}
		return stockhamPackedComplex128(dst128, src128, tw128, scratch128, packed128, inverse)
	default:
		return false
	}
}

func stockhamPackedComplex64(dst, src, twiddle, scratch []complex64, packed *PackedTwiddles[complex64], inverse bool) bool {
	n := len(src)
	if n == 0 {
		return true
	}

	if len(dst) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	if packed == nil || packed.Radix != 4 {
		return false
	}

	if !IsPowerOf2(n) {
		return false
	}

	if n == 1 {
		dst[0] = src[0]
		return true
	}

	remaining := n
	for remaining%4 == 0 {
		remaining /= 4
	}
	if remaining != 1 && remaining != 2 {
		return false
	}

	in := src
	out := dst
	same := sameSlice(dst, src)
	inIsDst := same
	outIsDst := true

	if same {
		out = scratch
		outIsDst = false
	}

	in = in[:n]
	out = out[:n]
	twiddle = twiddle[:n]

	m := n
	if remaining == 2 {
		if !stockhamRadix2StageComplex64(in, out, twiddle, n, m, inverse) {
			return false
		}

		in = out
		inIsDst = outIsDst
		if outIsDst {
			out = scratch
			outIsDst = false
		} else {
			out = dst
			outIsDst = true
		}
		out = out[:n]

		m = n / 2
	}

	for m >= 4 {
		stageIdx, ok := packedStageIndex(m)
		if !ok || stageIdx >= len(packed.StageOffsets) {
			return false
		}

		stageOffset := packed.StageOffsets[stageIdx]
		if !stockhamRadix4StageComplex64(in, out, packed.Values, n, m, stageOffset, inverse) {
			return false
		}

		in = out
		inIsDst = outIsDst
		if outIsDst {
			out = scratch
			outIsDst = false
		} else {
			out = dst
			outIsDst = true
		}
		out = out[:n]

		m /= 4
	}

	if !inIsDst {
		copy(dst, in)
	}

	if inverse {
		scale := float32(1.0 / float64(n))
		for i := range dst {
			dst[i] *= complex(scale, 0)
		}
	}

	return true
}

func stockhamPackedComplex128(dst, src, twiddle, scratch []complex128, packed *PackedTwiddles[complex128], inverse bool) bool {
	n := len(src)
	if n == 0 {
		return true
	}

	if len(dst) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	if packed == nil || packed.Radix != 4 {
		return false
	}

	if !IsPowerOf2(n) {
		return false
	}

	if n == 1 {
		dst[0] = src[0]
		return true
	}

	remaining := n
	for remaining%4 == 0 {
		remaining /= 4
	}
	if remaining != 1 && remaining != 2 {
		return false
	}

	in := src
	out := dst
	same := sameSlice(dst, src)
	inIsDst := same
	outIsDst := true

	if same {
		out = scratch
		outIsDst = false
	}

	in = in[:n]
	out = out[:n]
	twiddle = twiddle[:n]

	m := n
	if remaining == 2 {
		if !stockhamRadix2StageComplex128(in, out, twiddle, n, m, inverse) {
			return false
		}

		in = out
		inIsDst = outIsDst
		if outIsDst {
			out = scratch
			outIsDst = false
		} else {
			out = dst
			outIsDst = true
		}
		out = out[:n]

		m = n / 2
	}

	for m >= 4 {
		stageIdx, ok := packedStageIndex(m)
		if !ok || stageIdx >= len(packed.StageOffsets) {
			return false
		}

		stageOffset := packed.StageOffsets[stageIdx]
		if !stockhamRadix4StageComplex128(in, out, packed.Values, n, m, stageOffset, inverse) {
			return false
		}

		in = out
		inIsDst = outIsDst
		if outIsDst {
			out = scratch
			outIsDst = false
		} else {
			out = dst
			outIsDst = true
		}
		out = out[:n]

		m /= 4
	}

	if !inIsDst {
		copy(dst, in)
	}

	if inverse {
		scale := 1.0 / float64(n)
		for i := range dst {
			dst[i] *= complex(scale, 0)
		}
	}

	return true
}

func stockhamRadix2StageComplex64(in, out, twiddle []complex64, n, m int, inverse bool) bool {
	if m < 2 {
		return false
	}

	half := m >> 1
	step := n / m
	kLimit := n / m
	halfN := n >> 1

	for k := range kLimit {
		base := k * m

		outBase := k * half
		inBlock := in[base : base+m]
		outLo := out[outBase : outBase+half]
		outHi := out[outBase+halfN : outBase+halfN+half]
		for j := range half {
			a := inBlock[j]
			b := inBlock[j+half]
			tw := twiddle[j*step]
			if inverse {
				tw = complex(real(tw), -imag(tw))
			}
			outLo[j] = a + b
			outHi[j] = (a - b) * tw
		}
	}

	return true
}

func stockhamRadix2StageComplex128(in, out, twiddle []complex128, n, m int, inverse bool) bool {
	if m < 2 {
		return false
	}

	half := m >> 1
	step := n / m
	kLimit := n / m
	halfN := n >> 1

	for k := range kLimit {
		base := k * m

		outBase := k * half
		inBlock := in[base : base+m]
		outLo := out[outBase : outBase+half]
		outHi := out[outBase+halfN : outBase+halfN+half]
		for j := range half {
			a := inBlock[j]
			b := inBlock[j+half]
			tw := twiddle[j*step]
			if inverse {
				tw = complex(real(tw), -imag(tw))
			}
			outLo[j] = a + b
			outHi[j] = (a - b) * tw
		}
	}

	return true
}

func stockhamRadix4StageComplex64(in, out, packed []complex64, n, m, stageOffset int, inverse bool) bool {
	if m < 4 {
		return false
	}

	span := m / 4
	kLimit := n / m
	quarterN := n / 4

	for k := range kLimit {
		base := k * m

		outBase := k * span
		inBlock := in[base : base+m]
		out0 := out[outBase : outBase+span]
		out1 := out[outBase+quarterN : outBase+quarterN+span]
		out2 := out[outBase+2*quarterN : outBase+2*quarterN+span]
		out3 := out[outBase+3*quarterN : outBase+3*quarterN+span]

		for j := range span {
			a := inBlock[j]
			b := inBlock[j+span]
			c := inBlock[j+2*span]
			d := inBlock[j+3*span]

			t0 := a + c
			t1 := a - c
			t2 := b + d
			t3 := b - d

			y0 := t0 + t2
			y2 := t0 - t2

			var y1, y3 complex64
			if inverse {
				y1 = t1 + complex(-imag(t3), real(t3))
				y3 = t1 + complex(imag(t3), -real(t3))
			} else {
				y1 = t1 + complex(imag(t3), -real(t3))
				y3 = t1 + complex(-imag(t3), real(t3))
			}

			twOffset := stageOffset + j*3
			w1 := packed[twOffset]
			w2 := packed[twOffset+1]
			w3 := packed[twOffset+2]
			if inverse {
				w1 = complex(real(w1), -imag(w1))
				w2 = complex(real(w2), -imag(w2))
				w3 = complex(real(w3), -imag(w3))
			}

			out0[j] = y0
			out1[j] = y1 * w1
			out2[j] = y2 * w2
			out3[j] = y3 * w3
		}
	}

	return true
}

func stockhamRadix4StageComplex128(in, out, packed []complex128, n, m, stageOffset int, inverse bool) bool {
	if m < 4 {
		return false
	}

	span := m / 4
	kLimit := n / m
	quarterN := n / 4

	for k := range kLimit {
		base := k * m

		outBase := k * span
		inBlock := in[base : base+m]
		out0 := out[outBase : outBase+span]
		out1 := out[outBase+quarterN : outBase+quarterN+span]
		out2 := out[outBase+2*quarterN : outBase+2*quarterN+span]
		out3 := out[outBase+3*quarterN : outBase+3*quarterN+span]

		for j := range span {
			a := inBlock[j]
			b := inBlock[j+span]
			c := inBlock[j+2*span]
			d := inBlock[j+3*span]

			t0 := a + c
			t1 := a - c
			t2 := b + d
			t3 := b - d

			y0 := t0 + t2
			y2 := t0 - t2

			var y1, y3 complex128
			if inverse {
				y1 = t1 + complex(-imag(t3), real(t3))
				y3 = t1 + complex(imag(t3), -real(t3))
			} else {
				y1 = t1 + complex(imag(t3), -real(t3))
				y3 = t1 + complex(-imag(t3), real(t3))
			}

			twOffset := stageOffset + j*3
			w1 := packed[twOffset]
			w2 := packed[twOffset+1]
			w3 := packed[twOffset+2]
			if inverse {
				w1 = complex(real(w1), -imag(w1))
				w2 = complex(real(w2), -imag(w2))
				w3 = complex(real(w3), -imag(w3))
			}

			out0[j] = y0
			out1[j] = y1 * w1
			out2[j] = y2 * w2
			out3[j] = y3 * w3
		}
	}

	return true
}

func packedStageIndex(size int) (int, bool) {
	if size < 4 {
		return 0, false
	}

	idx := 0
	for step := 4; step < size; step <<= 2 {
		idx++
	}

	return idx, true
}
