package algofft

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/planner"
)

type legacyWisdomStore struct {
	stored WisdomEntry
}

func (s *legacyWisdomStore) LookupWisdom(int, uint8, uint64) (string, bool) {
	return "", false
}

func (s *legacyWisdomStore) Lookup(WisdomKey) (WisdomEntry, bool) {
	return WisdomEntry{}, false
}

func (s *legacyWisdomStore) Store(entry WisdomEntry) {
	s.stored = entry
}

func TestWisdomAdapterPreservesLegacyStoreKeying(t *testing.T) {
	t.Parallel()

	legacy := &legacyWisdomStore{}
	adapter := wisdomAdapter{store: legacy}
	adapter.Store(planner.WisdomEntry{
		Key: planner.WisdomKey{
			Size:          2048,
			Precision:     planner.PrecisionComplex128,
			CPUFeatures:   7,
			CPUIdentifier: "amd64_AuthenticAMD_f23_m96_l1d32768_l2524288",
		},
		Algorithm: "stockham",
	})

	if legacy.stored.Key.CPUIdentifier != "" {
		t.Errorf("legacy store received CPU identifier %q", legacy.stored.Key.CPUIdentifier)
	}
}

func TestWisdomAdapterPreservesContextForBuiltInStore(t *testing.T) {
	t.Parallel()

	const identifier = "amd64_AuthenticAMD_f23_m96_l1d32768_l2524288"

	wisdom := NewWisdom()
	adapter := wisdomAdapter{store: wisdom}
	adapter.Store(planner.WisdomEntry{
		Key: planner.WisdomKey{
			Size:          2048,
			Precision:     planner.PrecisionComplex128,
			CPUFeatures:   7,
			CPUIdentifier: identifier,
		},
		Algorithm: "stockham",
	})

	_, found := wisdom.Lookup(WisdomKey{
		Size:          2048,
		Precision:     uint8(PrecisionComplex128),
		CPUFeatures:   7,
		CPUIdentifier: identifier,
	})
	if !found {
		t.Fatal("built-in store lost CPU identifier through adapter")
	}
}
