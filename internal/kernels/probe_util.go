//go:build fftprobe

package kernels

// Helpers shared by the `-tags fftprobe` measurement harnesses. Nothing here
// is compiled into an ordinary build.

// itoa avoids pulling strconv into a probe file.
func itoa(v int) string {
	if v == 0 {
		return "0"
	}

	var buf [20]byte

	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}

	return string(buf[i:])
}
