package algoforge

import "math"

// PlanReal is a pre-computed real FFT plan for float32 input.
// The forward transform returns the non-redundant half-spectrum with length N/2+1.
//
// Output bins obey conjugate symmetry for real inputs:
//
//	X[k] = conj(X[N-k]) for k = 1..N/2-1
//
// Index 0 is DC and index N/2 is Nyquist (purely real for even N).
type PlanReal struct {
	n    int
	half int

	plan    *Plan[complex64]
	twiddle []complex64
	buf     []complex64
}

// NewPlanReal creates a new real FFT plan for length n.
// Currently, only even lengths are supported by the real FFT pack method.
func NewPlanReal(n int) (*PlanReal, error) {
	if n < 2 || n%2 != 0 {
		return nil, ErrInvalidLength
	}

	plan, err := NewPlan[complex64](n / 2)
	if err != nil {
		return nil, err
	}

	// Compute only the N/2+1 twiddle factors needed for real FFT recombination.
	// W_N^k = exp(-2πik/N) for k = 0..N/2
	twiddle := make([]complex64, n/2+1)
	for k := range twiddle {
		angle := -2 * math.Pi * float64(k) / float64(n)
		twiddle[k] = complex64(complex(math.Cos(angle), math.Sin(angle)))
	}

	return &PlanReal{
		n:       n,
		half:    n / 2,
		plan:    plan,
		twiddle: twiddle,
		buf:     make([]complex64, n/2),
	}, nil
}

// Len returns the number of real samples for this plan.
func (p *PlanReal) Len() int {
	return p.n
}

// SpectrumLen returns the number of complex frequency bins (N/2+1).
func (p *PlanReal) SpectrumLen() int {
	return p.half + 1
}

// Forward computes the real-to-complex FFT.
// dst must have length N/2+1 and src must have length N.
func (p *PlanReal) Forward(dst []complex64, src []float32) error {
	if dst == nil || src == nil {
		return ErrNilSlice
	}

	if len(src) != p.n || len(dst) != p.half+1 {
		return ErrLengthMismatch
	}

	for i := range p.half {
		p.buf[i] = complex(src[2*i], src[2*i+1])
	}

	err := p.plan.Forward(p.buf, p.buf)
	if err != nil {
		return err
	}

	y0 := p.buf[0]
	y0r := real(y0)
	y0i := imag(y0)
	dst[0] = complex(y0r+y0i, 0)
	dst[p.half] = complex(y0r-y0i, 0)

	// Recombination step: extract X[k] from the N/2-point FFT of packed data.
	// Given z[m] = x[2m] + i*x[2m+1], we computed Y = FFT(z).
	// The real FFT X[k] is recovered via:
	//   A[k] = Y[k], B[k] = conj(Y[N/2-k])
	//   X[k] = 0.5 * (A + B + W_N^k * (A - B) * (-i))
	// The 0.5 factor accounts for the averaging in the symmetric decomposition.
	for k := 1; k < p.half; k++ {
		a := p.buf[k]
		bSrc := p.buf[p.half-k]
		b := complex(real(bSrc), -imag(bSrc)) // conj(Y[N/2-k])

		t1 := a + b // A + B (symmetric part)
		t2 := a - b // A - B (antisymmetric part)
		w := p.twiddle[k]
		u := w * t2
		rot := complex(imag(u), -real(u)) // multiply by -i
		dst[k] = 0.5 * (t1 + rot)
	}

	return nil
}

// Inverse computes the complex-to-real inverse FFT.
// dst must have length N and src must have length N/2+1.
func (p *PlanReal) Inverse(dst []float32, src []complex64) error {
	if dst == nil || src == nil {
		return ErrNilSlice
	}

	if len(dst) != p.n || len(src) != p.half+1 {
		return ErrLengthMismatch
	}

	return ErrNotImplemented
}
