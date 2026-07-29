//go:build js && wasm

package main

import (
	"sync"
	"time"

	algofft "github.com/cwbudde/algo-fft"
)

// demoWisdom is the single wisdom store the demo owns and passes to every plan
// build.
//
// It has to be an explicit store: PlanOptions.Wisdom defaults to nil, the
// planner returns early when the store is nil, and the package exposes no
// accessor for a default store. Anything that writes into a store the plans do
// not reference would appear to work and have no effect whatsoever.
//
// Persistence, when it is added, must go through Export/Import over a
// bytes.Buffer. The file-based ImportWisdom/ExportWisdom helpers hit
// wasm_exec.js's stub filesystem and fail at runtime; never call them here.
var demoWisdom = algofft.NewWisdom()

// planKind distinguishes the plan families the demo builds.
type planKind uint8

const (
	planKind1D planKind = iota
	planKind2D
)

// String returns a stable name for the plan kind, used in cache readouts.
func (k planKind) String() string {
	switch k {
	case planKind1D:
		return "1d"
	case planKind2D:
		return "2d"
	default:
		return "unknown"
	}
}

// precisionKind selects the element type of a plan.
type precisionKind uint8

const (
	precision64  precisionKind = iota // complex64
	precision128                      // complex128
)

// String returns the Go type name for the precision.
func (p precisionKind) String() string {
	if p == precision128 {
		return "complex128"
	}

	return "complex64"
}

// precisionFromString maps the JS-visible precision name onto the enum,
// defaulting to complex64.
func precisionFromString(s string) precisionKind {
	switch s {
	case "complex128", "c128", "64":
		return precision128
	default:
		return precision64
	}
}

// planKey identifies a cached plan. Every field that can change which plan
// gets built has to be part of the key: the old int-keyed caches could not
// express precision, a forced strategy, or a planner mode, and would happily
// hand back a complex64 estimate plan for a complex128 exhaustive request.
type planKey struct {
	kind      planKind
	precision precisionKind
	strategy  algofft.KernelStrategy
	planner   algofft.PlannerMode
	d0        int
	d1        int
	d2        int
}

// planBuffers holds the per-size scratch the transform path needs, so the
// steady-state frame allocates nothing on the Go heap either. Slices are
// allocated on first use and then reused for the life of the cache entry.
type planBuffers struct {
	src64  []complex64
	dst64  []complex64
	src128 []complex128
	dst128 []complex128
	signal []float32
	mag    []float32
	phase  []float32

	// signal64 is the working buffer signals are generated and windowed into,
	// in float64, before being downcast to signal for display/transform. Doing
	// windowing in float64 keeps the roundtrip error comparison below
	// meaningful at complex128 precision — comparing against a float32 signal
	// would mask the ~1e-15 vs ~1e-6 gap the demo exists to show.
	signal64 []float64
	// windowShape64/windowF32 hold the window's own curve (not multiplied into
	// the signal), for the UI overlay.
	windowShape64 []float64
	windowF32     []float32

	// reconC64/reconC128 hold the inverse-transform result when a caller asks
	// for a round-trip check; reconReal64/reconF32 hold its real part, in
	// float64 for error math and float32 for display.
	reconC64  []complex64
	reconC128 []complex128
	reconReal []float64
	reconF32  []float32
}

// ensureFloat64 grows buf to length n, reusing its backing array when it
// already has enough capacity.
func ensureFloat64(buf []float64, n int) []float64 {
	if cap(buf) >= n {
		return buf[:n]
	}

	return make([]float64, n)
}

func ensureComplex64(buf []complex64, n int) []complex64 {
	if cap(buf) >= n {
		return buf[:n]
	}

	return make([]complex64, n)
}

func ensureComplex128(buf []complex128, n int) []complex128 {
	if cap(buf) >= n {
		return buf[:n]
	}

	return make([]complex128, n)
}

func ensureFloat32(buf []float32, n int) []float32 {
	if cap(buf) >= n {
		return buf[:n]
	}

	return make([]float32, n)
}

// planEntry is one cached plan plus its scratch buffers.
type planEntry struct {
	key     planKey
	info    algofft.PlanInfo
	plan    any // the concrete *algofft.Plan[T] / *algofft.Plan2D[T]
	bufs    planBuffers
	buildNs int64
	lastUse uint64
}

// describePlan reports the resolved algorithm name and kernel strategy of a
// cached plan's first axis, tolerating plan types that report nothing.
//
// The strategy here is the *resolved* one, which is not always the requested
// one, and it is not always the whole story either: a Rader plan reports
// Bluestein as its strategy and only names "rader" in the algorithm.
func describePlan(entry *planEntry) (algorithm, strategy string) {
	strategy = algofft.KernelAuto.String()

	if strategies := entry.info.KernelStrategies(); len(strategies) > 0 {
		strategy = strategies[0].String()
	}

	if algorithms := entry.info.Algorithms(); len(algorithms) > 0 {
		algorithm = algorithms[0]
	}

	return algorithm, strategy
}

// planCacheCapacity caps the number of live plans. Plans for large n hold
// twiddle tables and scratch proportional to n, Go's wasm heap never returns
// pages to the browser, and a slider drag can easily request dozens of
// distinct sizes in a second. The real defence against running the tab out of
// memory is the size clamp at the request boundary; this cap keeps the
// steady-state footprint bounded.
const planCacheCapacity = 48

// planCache is the process-wide plan cache.
type planCacheStore struct {
	mu      sync.Mutex
	entries map[planKey]*planEntry
	clock   uint64

	hits      uint64
	misses    uint64
	builds    uint64
	evictions uint64
	failures  uint64

	// convolvers holds the persistent Convolver/Correlator instances used by
	// the convolution panel, keyed by input lengths and mode. It is kept
	// alongside the plan cache (same store, same mutex) rather than built
	// fresh per call, per the "use the persistent types, not the one-shot
	// free functions" requirement. The set of distinct keys in practice is
	// small — one entry per (signal length, kernel length, correlate) the
	// user has actually selected — so this is capped generously and cleared
	// wholesale on overflow rather than tracking per-entry LRU.
	convolvers map[convolverKey]*convolverEntry
}

var planCache = &planCacheStore{
	entries:    make(map[planKey]*planEntry),
	convolvers: make(map[convolverKey]*convolverEntry),
}

// convolverKey identifies a cached Convolver/Correlator pair.
type convolverKey struct {
	lenA      int
	lenB      int
	correlate bool
}

// convolverEntry holds exactly one of conv or corr, matching the key's
// correlate flag.
type convolverEntry struct {
	conv *algofft.Convolver[complex64]
	corr *algofft.Correlator[complex64]
}

// convolverCacheCapacity bounds the convolver cache before it is cleared
// wholesale. The kernel set offered by the UI is fixed and small, so this
// should rarely if ever trigger.
const convolverCacheCapacity = 32

// getConvolver returns the cached Convolver or Correlator for the given input
// lengths, building it on miss.
func (c *planCacheStore) getConvolver(lenA, lenB int, correlate bool) (*convolverEntry, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	key := convolverKey{lenA: lenA, lenB: lenB, correlate: correlate}

	if entry, ok := c.convolvers[key]; ok {
		return entry, nil
	}

	if len(c.convolvers) >= convolverCacheCapacity {
		c.convolvers = make(map[convolverKey]*convolverEntry)
	}

	entry := &convolverEntry{}

	if correlate {
		corr, err := algofft.NewCorrelator[complex64](lenA, lenB)
		if err != nil {
			return nil, err
		}

		entry.corr = corr
	} else {
		conv, err := algofft.NewConvolver[complex64](lenA, lenB)
		if err != nil {
			return nil, err
		}

		entry.conv = conv
	}

	c.convolvers[key] = entry

	return entry, nil
}

// get returns the cached plan for key, building it on miss. The second result
// reports whether the plan was already cached, and the third the time spent
// building it (zero on a hit).
func (c *planCacheStore) get(key planKey) (*planEntry, bool, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.clock++

	if entry, ok := c.entries[key]; ok {
		c.hits++
		entry.lastUse = c.clock

		return entry, true, nil
	}

	c.misses++

	start := time.Now()

	info, plan, err := buildPlan(key)
	if err != nil {
		c.failures++

		return nil, false, err
	}

	c.builds++

	entry := &planEntry{
		key:     key,
		info:    info,
		plan:    plan,
		buildNs: time.Since(start).Nanoseconds(),
		lastUse: c.clock,
	}
	c.entries[key] = entry

	c.evictLocked()

	return entry, false, nil
}

// evictLocked drops least-recently-used entries until the cache is within
// capacity.
//
// Close() on a non-pooled Plan[T] is a no-op — it returns early when the plan
// has no pool — so eviction cannot rely on it to free anything. What actually
// reclaims the memory is dropping the map's reference and letting the GC run.
// Close is still called because the 2D and real plan types do release their
// child plans and scratch caches through it.
func (c *planCacheStore) evictLocked() {
	for len(c.entries) > planCacheCapacity {
		var (
			oldestKey planKey
			oldest    *planEntry
		)

		for key, entry := range c.entries {
			if oldest == nil || entry.lastUse < oldest.lastUse {
				oldestKey, oldest = key, entry
			}
		}

		if oldest == nil {
			return
		}

		oldest.info.Close()
		delete(c.entries, oldestKey)

		c.evictions++
	}
}

// stats returns a JS-friendly snapshot of the cache counters.
func (c *planCacheStore) stats() map[string]any {
	c.mu.Lock()
	defer c.mu.Unlock()

	live := make([]any, 0, len(c.entries))
	for key, entry := range c.entries {
		live = append(live, map[string]any{
			"kind":      key.kind.String(),
			"precision": key.precision.String(),
			"strategy":  key.strategy.String(),
			"planner":   plannerModeName(key.planner),
			"d0":        key.d0,
			"d1":        key.d1,
			"d2":        key.d2,
			"buildNs":   float64(entry.buildNs),
		})
	}

	return map[string]any{
		"entries":   len(c.entries),
		"capacity":  planCacheCapacity,
		"hits":      float64(c.hits),
		"misses":    float64(c.misses),
		"builds":    float64(c.builds),
		"evictions": float64(c.evictions),
		"failures":  float64(c.failures),
		"wisdom":    demoWisdom.Len(),
		"live":      live,
	}
}

// clear drops every cached plan.
func (c *planCacheStore) clear() int {
	c.mu.Lock()
	defer c.mu.Unlock()

	n := len(c.entries)
	for key, entry := range c.entries {
		entry.info.Close()
		delete(c.entries, key)
	}

	return n
}

// buildPlan constructs the plan named by key. Every build passes demoWisdom so
// measured decisions accumulate in one place.
func buildPlan(key planKey) (algofft.PlanInfo, any, error) {
	opts := algofft.PlanOptions{
		Planner:  key.planner,
		Strategy: key.strategy,
		Wisdom:   demoWisdom,
	}

	switch key.kind {
	case planKind2D:
		if key.precision == precision128 {
			plan, err := algofft.NewPlan2DWithOptions[complex128](key.d0, key.d1, opts)
			if err != nil {
				return nil, nil, err
			}

			return plan, plan, nil
		}

		plan, err := algofft.NewPlan2DWithOptions[complex64](key.d0, key.d1, opts)
		if err != nil {
			return nil, nil, err
		}

		return plan, plan, nil

	case planKind1D:
		fallthrough
	default:
		if key.precision == precision128 {
			plan, err := algofft.NewPlanWithOptions[complex128](key.d0, opts)
			if err != nil {
				return nil, nil, err
			}

			return plan, plan, nil
		}

		plan, err := algofft.NewPlanWithOptions[complex64](key.d0, opts)
		if err != nil {
			return nil, nil, err
		}

		return plan, plan, nil
	}
}
