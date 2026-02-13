package algofft

import (
	"testing"
)

func TestMixedRadix40Debug(t *testing.T) {
	// Test what mixedRadix schedule produces for size 40
	n := 40
	var radices [64]int
	
	// Simplified schedule (mimic from mixedradix.go)
	count := 0
	for n > 1 {
		switch {
		case n%5 == 0:
			radices[count] = 5
			n /= 5
		case n%4 == 0:
			radices[count] = 4
			n /= 4
		case n%3 == 0:
			radices[count] = 3
			n /= 3
		case n%2 == 0:
			radices[count] = 2
			n /= 2
		default:
			t.Fatalf("Cannot decompose %d", n)
		}
		count++
	}
	
	t.Logf("Size 40 decomposition: %v (count=%d)", radices[:count], count)
	
	// Verify: 5 * 4 * 2 = 40
	product := 1
	for i := 0; i < count; i++ {
		product *= radices[i]
	}
	t.Logf("Product of radices: %d", product)
	if product != 40 {
		t.Errorf("Product %d != 40", product)
	}
}
