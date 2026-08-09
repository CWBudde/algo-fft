//go:build arm64 && !purego

package kernels

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/registry"
)

// TestNEONSelection pins the Apple M5 disposition: measured winners are
// selected, while losses below the 1.5x retention cutoff stay registered at a
// lower priority. Size-32 radix-8 is one such retained candidate; only the
// losing size-65536 rows are probe-only and intentionally absent.
func TestNEONSelection(t *testing.T) {
	features := cpu.DetectFeatures()
	if !features.HasNEON {
		t.Skip("NEON not available")
	}

	want64 := map[int]string{
		32:    "dit32_radix4fused_neon",
		64:    "dit64_radix2_neon",
		128:   "dit128_radix4fused_neon",
		256:   "dit256_radix4_neon",
		512:   "dit512_radix4fused_neon",
		1024:  "dit1024_radix4_neon",
		2048:  "dit2048_radix4fused_neon",
		4096:  "dit4096_radix4_neon",
		8192:  "dit8192_radix4fused_neon",
		16384: "dit16384_radix4_neon",
		32768: "dit32768_radix4_then2_neon",
	}

	want128 := map[int]string{
		32:    "dit32_radix4fused_neon",
		64:    "dit64_radix8ladder_neon",
		128:   "dit128_radix4fused_neon",
		256:   "dit256_radix4_neon",
		512:   "dit512_radix4fused_neon",
		1024:  "dit1024_radix4_neon",
		2048:  "dit2048_radix4fused_neon",
		4096:  "dit4096_radix4_neon",
		8192:  "dit8192_radix4fused_neon",
		16384: "dit16384_radix4_neon",
		32768: "dit32768_radix4_then2_neon",
	}

	for _, n := range []int{32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768} {
		entry64 := registry.Registry64.Lookup(n, features)
		if entry64 == nil {
			t.Fatalf("complex64 n=%d: no codelet", n)
		}
		if entry64.Signature != want64[n] {
			t.Errorf("complex64 n=%d: signature %q, want %q", n, entry64.Signature, want64[n])
		}

		entry128 := registry.Registry128.Lookup(n, features)
		if entry128 == nil {
			t.Fatalf("complex128 n=%d: no codelet", n)
		}
		if entry128.Signature != want128[n] {
			t.Errorf("complex128 n=%d: signature %q, want %q", n, entry128.Signature, want128[n])
		}
	}

	const alternate = "dit32_radix8ladder_neon"

	entry64 := registry.Registry64.LookupBySignature(32, alternate)
	if entry64 == nil {
		t.Fatalf("complex64 n=32: missing Wisdom candidate %q", alternate)
	}
	if entry64.Priority != 20 {
		t.Errorf("complex64 n=32: Wisdom candidate priority %d, want 20", entry64.Priority)
	}

	entry128 := registry.Registry128.LookupBySignature(32, alternate)
	if entry128 == nil {
		t.Fatalf("complex128 n=32: missing Wisdom candidate %q", alternate)
	}
	if entry128.Priority != 20 {
		t.Errorf("complex128 n=32: Wisdom candidate priority %d, want 20", entry128.Priority)
	}

	wantGroups := radix8GroupIndices(32)
	if len(wantGroups) != len(radix8NEONSize32Groups) {
		t.Fatalf("size-32 radix-8 group count %d, want %d", len(radix8NEONSize32Groups), len(wantGroups))
	}
	for i, want := range wantGroups {
		if got := radix8NEONSize32Groups[i]; got != want {
			t.Errorf("size-32 radix-8 group %d = %d, want %d", i, got, want)
		}
	}
}
