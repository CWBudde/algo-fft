package main

import (
	"reflect"
	"testing"

	algofft "github.com/cwbudde/algo-fft"
)

func TestParseEffort(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		input string
		want  algofft.PlannerMode
	}{
		{input: "patient", want: algofft.PlannerPatient},
		{input: "EXHAUSTIVE", want: algofft.PlannerExhaustive},
	} {
		got, err := parseEffort(test.input)
		if err != nil {
			t.Fatalf("parseEffort(%q): %v", test.input, err)
		}
		if got != test.want {
			t.Errorf("parseEffort(%q) = %v, want %v", test.input, got, test.want)
		}
	}

	if _, err := parseEffort("estimate"); err == nil {
		t.Fatal("parseEffort(estimate) succeeded, want error")
	}
}

func TestParsePrecisions(t *testing.T) {
	t.Parallel()

	got, err := parsePrecisions("all")
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{precision32, precision64}; !reflect.DeepEqual(got, want) {
		t.Fatalf("parsePrecisions(all) = %v, want %v", got, want)
	}

	if _, err := parsePrecisions("float32"); err == nil {
		t.Fatal("parsePrecisions(float32) succeeded, want error")
	}
}

func TestPowerOfTwoSizes(t *testing.T) {
	t.Parallel()

	got, err := powerOfTwoSizes(8, 64)
	if err != nil {
		t.Fatal(err)
	}
	if want := []int{8, 16, 32, 64}; !reflect.DeepEqual(got, want) {
		t.Fatalf("powerOfTwoSizes(8, 64) = %v, want %v", got, want)
	}

	for _, bounds := range [][2]int{{0, 64}, {12, 64}, {64, 32}, {8, 48}} {
		if _, err := powerOfTwoSizes(bounds[0], bounds[1]); err == nil {
			t.Errorf("powerOfTwoSizes(%d, %d) succeeded, want error", bounds[0], bounds[1])
		}
	}
}
