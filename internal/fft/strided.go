package fft

// ForwardStridedDIT runs a radix-2 DIT FFT over strided data.
// dst and src must be large enough for n elements with the given stride.
func ForwardStridedDIT[T Complex](dst, src, twiddle []T, bitrev []int, stride, n int) bool {
	return ditForwardStrided(dst, src, twiddle, bitrev, stride, n)
}

// InverseStridedDIT runs a radix-2 inverse DIT FFT over strided data.
// dst and src must be large enough for n elements with the given stride.
func InverseStridedDIT[T Complex](dst, src, twiddle []T, bitrev []int, stride, n int) bool {
	return ditInverseStrided(dst, src, twiddle, bitrev, stride, n)
}

func ditForwardStrided[T Complex](dst, src, twiddle []T, bitrev []int, stride, n int) bool {
	if n == 0 {
		return true
	}

	if stride < 1 || len(twiddle) < n || len(bitrev) < n {
		return false
	}

	required := 1 + (n-1)*stride
	if len(dst) < required || len(src) < required {
		return false
	}

	for i := range n {
		dst[i*stride] = src[bitrev[i]*stride]
	}

	for size := 2; size <= n; size <<= 1 {
		half := size >> 1
		step := n / size

		for base := 0; base < n; base += size {
			index1 := base * stride

			index2 := (base + half) * stride
			for j := range half {
				tw := twiddle[j*step]
				a, b := butterfly2(dst[index1], dst[index2], tw)
				dst[index1] = a
				dst[index2] = b
				index1 += stride
				index2 += stride
			}
		}
	}

	return true
}

func ditInverseStrided[T Complex](dst, src, twiddle []T, bitrev []int, stride, n int) bool {
	if n == 0 {
		return true
	}

	if stride < 1 || len(twiddle) < n || len(bitrev) < n {
		return false
	}

	required := 1 + (n-1)*stride
	if len(dst) < required || len(src) < required {
		return false
	}

	for i := range n {
		dst[i*stride] = src[bitrev[i]*stride]
	}

	for size := 2; size <= n; size <<= 1 {
		half := size >> 1
		step := n / size

		for base := 0; base < n; base += size {
			index1 := base * stride

			index2 := (base + half) * stride
			for j := range half {
				tw := conj(twiddle[j*step])
				a, b := butterfly2(dst[index1], dst[index2], tw)
				dst[index1] = a
				dst[index2] = b
				index1 += stride
				index2 += stride
			}
		}
	}

	scale := complexFromFloat64[T](1.0/float64(n), 0)
	for i := range n {
		dst[i*stride] *= scale
	}

	return true
}
