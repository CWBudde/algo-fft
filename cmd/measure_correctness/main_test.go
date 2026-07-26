package main

import (
	"math"
	"testing"
)

// The metric math is this tool's entire deliverable, and a dropped square root or
// a swapped numerator produces a table of plausible-looking numbers that nobody
// can falsify by inspection — which is how the metric it replaces survived for so
// long. These tests pin it against hand-computed cases.

func TestRelL2(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name string
		got  []complex128
		want []complex128
		exp  float64
		tol  float64
	}{
		{
			name: "identical is exactly zero",
			got:  []complex128{3 + 4i, -1 + 2i, 0},
			want: []complex128{3 + 4i, -1 + 2i, 0},
			exp:  0,
			tol:  0,
		},
		{
			name: "doubled is exactly one",
			got:  []complex128{2, 4, 6i},
			want: []complex128{1, 2, 3i},
			exp:  1,
			tol:  0,
		},
		{
			name: "uniform 1e-6 perturbation",
			got:  []complex128{1 * (1 + 1e-6), 2 * (1 + 1e-6), 3 * (1 + 1e-6)},
			want: []complex128{1, 2, 3},
			exp:  1e-6,
			tol:  1e-12,
		},
		{
			// ||diff||2 = 3 (only bin 1 differs), ||want||2 = sqrt(4^2+3^2) = 5.
			// The norms run over the whole vector, so this is 3/5, not 3/3.
			name: "hand computed two element",
			got:  []complex128{4, 0},
			want: []complex128{4, 3},
			exp:  0.6,
			tol:  1e-15,
		},
		{
			name: "all-zero reference yields zero not NaN",
			got:  []complex128{1, 2, 3},
			want: []complex128{0, 0, 0},
			exp:  0,
			tol:  0,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			got := relL2(testCase.got, testCase.want)

			if math.IsNaN(got) {
				t.Fatalf("relL2 = NaN, want %v", testCase.exp)
			}

			if math.Abs(got-testCase.exp) > testCase.tol {
				t.Errorf("relL2 = %v, want %v (tol %v)", got, testCase.exp, testCase.tol)
			}
		})
	}
}

func TestPeakRel(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name string
		got  []complex128
		want []complex128
		exp  float64
	}{
		{
			name: "identical is exactly zero",
			got:  []complex128{3 + 4i, 1},
			want: []complex128{3 + 4i, 1},
			exp:  0,
		},
		{
			// The largest |diff| (2, at bin 1) is deliberately NOT at the largest
			// |want| (10, at bin 0). Normalizing per-bin would give 2/1 = 2 here;
			// normalizing by the peak gives 2/10 = 0.2. That difference is the
			// whole reason this statistic is stable, and the easiest thing to
			// implement wrong.
			name: "max diff is not at the max reference bin",
			got:  []complex128{10.5, 3},
			want: []complex128{10, 1},
			exp:  0.2,
		},
		{
			name: "all-zero reference yields zero not NaN",
			got:  []complex128{1, 2},
			want: []complex128{0, 0},
			exp:  0,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			got := peakRel(testCase.got, testCase.want)

			if math.IsNaN(got) {
				t.Fatalf("peakRel = NaN, want %v", testCase.exp)
			}

			if math.Abs(got-testCase.exp) > 1e-15 {
				t.Errorf("peakRel = %v, want %v", got, testCase.exp)
			}
		})
	}
}

// TestPeakRelSeesWhatRelL2Dilutes is the justification for reporting both
// metrics: one badly wrong bin in a long spectrum barely moves relL2 but is
// plainly visible in peakRel.
func TestPeakRelSeesWhatRelL2Dilutes(t *testing.T) {
	t.Parallel()

	const n = 4096

	got := make([]complex128, n)
	want := make([]complex128, n)

	for i := range n {
		want[i] = 1
		got[i] = 1
	}

	got[7] = 1.5 // one bin wrong by 50%

	l2 := relL2(got, want)
	peak := peakRel(got, want)

	if l2 > 0.01 {
		t.Errorf("relL2 = %v, expected it to be diluted below 0.01", l2)
	}

	if peak < 0.49 {
		t.Errorf("peakRel = %v, expected it to expose the 0.5 error", peak)
	}
}

func TestParseSizes(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name string
		in   string
		want []int
	}{
		{"single", "8", []int{8}},
		{"spaces trimmed", "8, 16", []int{8, 16}},
		{"zero dropped", "0", nil},
		{"negative dropped", "-4", nil},
		{"non numeric dropped", "abc", nil},
		{"empty", "", nil},
		{"empty entry skipped", "8,,16", []int{8, 16}},
		{"mixed", "8,abc,0,16", []int{8, 16}},
		{"non power of two kept", "257,1009,2205", []int{257, 1009, 2205}},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			got := parseSizes(testCase.in)

			if len(got) != len(testCase.want) {
				t.Fatalf("parseSizes(%q) = %v, want %v", testCase.in, got, testCase.want)
			}

			for i := range got {
				if got[i] != testCase.want[i] {
					t.Errorf("parseSizes(%q)[%d] = %d, want %d", testCase.in, i, got[i], testCase.want[i])
				}
			}
		})
	}
}
