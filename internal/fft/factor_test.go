package fft

import (
	"reflect"
	"testing"
)

func TestFactorize(t *testing.T) {
	t.Parallel()

	tests := []struct {
		n    int
		want []int
	}{
		{n: -5, want: nil},
		{n: 0, want: nil},
		{n: 1, want: nil},
		{n: 2, want: []int{2}},
		{n: 3, want: []int{3}},
		{n: 4, want: []int{2, 2}},
		{n: 6, want: []int{2, 3}},
		{n: 12, want: []int{2, 2, 3}},
		{n: 45, want: []int{3, 3, 5}},
		{n: 97, want: []int{97}},
		{n: 100, want: []int{2, 2, 5, 5}},
		{n: 1024, want: []int{2, 2, 2, 2, 2, 2, 2, 2, 2, 2}},
	}

	for _, tt := range tests {
		got := factorize(tt.n)
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("factorize(%d) = %v, want %v", tt.n, got, tt.want)
		}
	}
}

func TestIsHighlyComposite(t *testing.T) {
	t.Parallel()

	tests := []struct {
		n    int
		want bool
	}{
		{n: -5, want: false},
		{n: 0, want: false},
		{n: 1, want: true},
		{n: 2, want: true},
		{n: 3, want: true},
		{n: 4, want: true},
		{n: 5, want: true},
		{n: 6, want: true},
		{n: 8, want: true},
		{n: 9, want: true},
		{n: 10, want: true},
		{n: 12, want: true},
		{n: 15, want: true},
		{n: 18, want: true},
		{n: 25, want: true},
		{n: 30, want: true},
		{n: 16, want: true},
		{n: 14, want: false},
		{n: 49, want: false},
		{n: 11, want: false},
	}

	for _, tt := range tests {
		got := isHighlyComposite(tt.n)
		if got != tt.want {
			t.Errorf("isHighlyComposite(%d) = %v, want %v", tt.n, got, tt.want)
		}
	}
}
