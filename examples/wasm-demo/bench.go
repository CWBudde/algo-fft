//go:build js && wasm

package main

import (
	"errors"
	"math"
	"slices"
	"sync"
	"syscall/js"
	"time"

	algofft "github.com/cwbudde/algo-fft"
)

// Benchmarking under js/wasm, and why this file is shaped the way it is.
//
// Two constraints drive every decision here.
//
// 1. The clock is clamped. time.Now() is performance.now(), which browsers
// round as a Spectre mitigation — 100 microseconds in Chrome, roughly a
// millisecond in Firefox and Safari — and GitHub Pages cannot send the
// COOP/COEP headers that would lift the clamp. A single forward transform
// below n ~ 2^16 measures as zero ticks or one. So nothing is ever timed
// once: each case is calibrated by doubling an iteration count until a batch
// clears a target well above the clamp, and only whole batches are timed.
//
// 2. A Go call from JavaScript is synchronous and blocks the caller's event
// loop for its whole duration. A worker that enters a 20-second benchmark
// loop cannot dispatch the "cancel" message sitting in its queue. Therefore
// the loop does not live in Go at all: benchStart/benchStep/benchCancel
// expose the state machine, JS drives it one bounded chunk at a time, and the
// gap between chunks is where cancellation happens. The chunking *is* the
// cancellation mechanism.
//
// Everything else — separate plan-construction timing, the second build that
// shows the wisdom hit, the real (not all-zero) source signal, the returned
// checksum — exists so the numbers mean something.

const (
	// benchDefaultTrials is the number of timed batches per case. The median
	// of these is reported, with the relative standard deviation alongside so
	// a noisy machine is visible rather than averaged away.
	benchDefaultTrials = 5
	benchMinTrials     = 3
	benchMaxTrials     = 11

	// benchDefaultTargetMs is how long one timed batch should take. It sets
	// the chunk window: a calibration step at worst overshoots to twice this,
	// so every blocking Go call stays inside roughly 50-120 ms and the worker
	// gets to look at its message queue that often.
	benchDefaultTargetMs = 50
	benchMinTargetMs     = 10
	benchMaxTargetMs     = 250

	// benchReliableTicks is the multiple of the probed timer granularity a
	// timed batch must clear before its result is called reliable. Below it,
	// quantisation of the clock is a first-order term in the answer.
	benchReliableTicks = 200

	// benchMaxIterations caps the calibrated iteration count so a pathological
	// case (tiny n on a fast machine with a coarse clock) cannot spin forever.
	// Hitting the cap is what makes a row unreliable in practice.
	benchMaxIterations = 1 << 21

	// benchMaxCases bounds one job's cross product. The UI refuses larger
	// requests before it gets here; this is the backstop.
	benchMaxCases = 256

	// benchMaxJobs bounds how many job states are retained. Jobs are removed
	// on completion or cancel; this only matters if JS abandons one.
	benchMaxJobs = 8
)

var (
	errBenchNoJob      = errors.New("no such benchmark job")
	errBenchEmptyCases = errors.New("no benchmark cases requested")
	errBenchTooMany    = errors.New("too many benchmark cases requested")
)

// benchCase is one point in the requested cross product.
type benchCase struct {
	size         int
	precision    precisionKind
	strategy     algofft.KernelStrategy
	strategyName string
	planner      algofft.PlannerMode
}

// benchPhase names where a job's current case is in its lifecycle. The phase
// is reported to JS so the progress bar can be determinate.
type benchPhase uint8

const (
	phasePrepare benchPhase = iota
	phaseCalibrate
	phaseTrial
	phaseFinished
)

func (p benchPhase) String() string {
	switch p {
	case phasePrepare:
		return "prepare"
	case phaseCalibrate:
		return "calibrate"
	case phaseTrial:
		return "trial"
	case phaseFinished:
		return "finished"
	default:
		return "unknown"
	}
}

// benchRunner runs the timed inner loop for one precision.
//
// The point of the interface is that the type switch on plan/buffer types
// happens once per case, not once per iteration. forwardPlan's switch is fine
// for a single call from the analyze path; inside a loop that runs a million
// times it is measurable overhead attributed to the library.
type benchRunner interface {
	run(iterations int) error
	checksum() float64
}

type benchRunner64 struct {
	plan     *algofft.Plan[complex64]
	dst, src []complex64
}

func (r *benchRunner64) run(iterations int) error {
	for i := 0; i < iterations; i++ {
		if err := r.plan.Forward(r.dst, r.src); err != nil {
			return err
		}
	}

	return nil
}

func (r *benchRunner64) checksum() float64 { return complexChecksum(r.dst) }

type benchRunner128 struct {
	plan     *algofft.Plan[complex128]
	dst, src []complex128
}

func (r *benchRunner128) run(iterations int) error {
	for i := 0; i < iterations; i++ {
		if err := r.plan.Forward(r.dst, r.src); err != nil {
			return err
		}
	}

	return nil
}

func (r *benchRunner128) checksum() float64 { return complexChecksum(r.dst) }

// fillBenchSignal writes a deterministic, non-trivial signal into dst.
//
// The previous implementation benchmarked whatever a freshly made slice
// contained, i.e. all zeros. That is not obviously wrong for a fixed-schedule
// FFT, but it is not obviously right either: denormals, branch behaviour in
// mixed-radix twiddle paths and any future data-dependent shortcut all read
// differently on a zero buffer, and a zero buffer also makes the checksum
// useless as a "the loop actually ran" witness. Two tones plus a deterministic
// xorshift noise floor costs nothing and removes the question.
func fillBenchSignal[T algofft.Complex](dst []T) {
	n := len(dst)
	if n == 0 {
		return
	}

	state := uint32(0x9E3779B9)

	for i := range dst {
		state ^= state << 13
		state ^= state >> 17
		state ^= state << 5

		noise := float64(state)/float64(math.MaxUint32)*2 - 1
		t := float64(i) / float64(n)

		re := 0.6*math.Sin(2*math.Pi*7*t) + 0.3*math.Cos(2*math.Pi*23*t) + 0.08*noise
		im := 0.2*math.Sin(2*math.Pi*3*t) - 0.05*noise

		dst[i] = T(complex(re, im))
	}
}

// complexChecksum folds a transform result into one number.
//
// It is returned to JS purely as evidence the timed loop produced data: an
// identical checksum across trials means the same work ran each time, and a
// NaN means something went wrong that timing alone would not reveal. It is
// computed outside the timed region.
func complexChecksum[T algofft.Complex](buf []T) float64 {
	sum := 0.0

	for i, v := range buf {
		c := complex128(v)
		w := float64(i%17) + 1

		sum += (real(c) + imag(c)) * w
	}

	return sum
}

// benchPrep is everything a case needs before it can be timed.
type benchPrep struct {
	runner benchRunner
	info   algofft.PlanInfo

	algorithm string
	resolved  string

	// planNs times a cold plan build; planWisdomNs times an immediately
	// following identical build. Both plans see the demo's wisdom store, so
	// the second number is what wisdom buys once a decision has been recorded.
	// They are reported separately and are never part of the transform timing.
	planNs       int64
	planWisdomNs int64
}

// describePlanInfo reports the resolved algorithm and kernel strategy of a
// plan's first axis.
//
// Deliberately a local copy of plancache.go's describePlan: the benchmark
// builds its plans outside the cache (see prepareBenchCase) and so has a
// PlanInfo rather than a *planEntry.
func describePlanInfo(info algofft.PlanInfo) (algorithm, strategy string) {
	strategy = algofft.KernelAuto.String()

	if strategies := info.KernelStrategies(); len(strategies) > 0 {
		strategy = strategies[0].String()
	}

	if algorithms := info.Algorithms(); len(algorithms) > 0 {
		algorithm = algorithms[0]
	}

	return algorithm, strategy
}

// prepareBenchCase builds the plans for c, times both builds, allocates and
// fills the buffers, and runs one warmup transform.
//
// The plans are built directly rather than through planCache on purpose. A
// cache hit would report a plan-construction time of zero, and a sweep of
// dozens of sizes would evict everything the Signal Lab page had warmed. The
// benchmark's plans are throwaway.
func prepareBenchCase(c benchCase) (*benchPrep, error) {
	key := planKey{
		kind:      planKind1D,
		precision: c.precision,
		strategy:  c.strategy,
		planner:   c.planner,
		d0:        c.size,
	}

	coldStart := time.Now()

	info, _, err := buildPlan(key)
	if err != nil {
		return nil, err
	}

	planNs := time.Since(coldStart).Nanoseconds()

	warmStart := time.Now()

	warmInfo, warmPlan, err := buildPlan(key)
	if err != nil {
		info.Close()

		return nil, err
	}

	planWisdomNs := time.Since(warmStart).Nanoseconds()

	// The cold plan has served its purpose (its build time); the second one is
	// the one that gets used, so the reported wisdom-warm plan is also the
	// plan that ran.
	info.Close()

	algorithm, resolved := describePlanInfo(warmInfo)

	prep := &benchPrep{
		info:         warmInfo,
		algorithm:    algorithm,
		resolved:     resolved,
		planNs:       planNs,
		planWisdomNs: planWisdomNs,
	}

	switch p := warmPlan.(type) {
	case *algofft.Plan[complex64]:
		src := make([]complex64, c.size)
		dst := make([]complex64, c.size)

		fillBenchSignal(src)

		prep.runner = &benchRunner64{plan: p, dst: dst, src: src}

	case *algofft.Plan[complex128]:
		src := make([]complex128, c.size)
		dst := make([]complex128, c.size)

		fillBenchSignal(src)

		prep.runner = &benchRunner128{plan: p, dst: dst, src: src}

	default:
		warmInfo.Close()

		return nil, errUnsupportedPlan
	}

	// Warmup. The first transform through a fresh plan touches twiddle tables
	// and scratch that are cold in cache, and under wasm it is also the call
	// that gets the tier-up compiler interested.
	if err := prep.runner.run(1); err != nil {
		warmInfo.Close()

		return nil, err
	}

	return prep, nil
}

// benchJob is the server side of the JS-driven loop.
type benchJob struct {
	id    int
	cases []benchCase

	trials   int
	targetNs int64

	index int
	phase benchPhase

	prep   *benchPrep
	iters  int
	timing []float64 // per-iteration nanoseconds, one entry per completed trial
}

type benchJobStore struct {
	mu     sync.Mutex
	jobs   map[int]*benchJob
	nextID int
}

var benchJobs = &benchJobStore{jobs: make(map[int]*benchJob)}

func (s *benchJobStore) add(job *benchJob) int {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.nextID++
	job.id = s.nextID

	// Abandoned jobs (a worker terminated mid-run) are dropped oldest-first
	// rather than accumulating their plans and buffers forever.
	for len(s.jobs) >= benchMaxJobs {
		oldest := 0
		for id := range s.jobs {
			if oldest == 0 || id < oldest {
				oldest = id
			}
		}

		s.closeLocked(oldest)
	}

	s.jobs[job.id] = job

	return job.id
}

func (s *benchJobStore) get(id int) *benchJob {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.jobs[id]
}

func (s *benchJobStore) closeLocked(id int) {
	job, ok := s.jobs[id]
	if !ok {
		return
	}

	if job.prep != nil && job.prep.info != nil {
		job.prep.info.Close()
		job.prep = nil
	}

	delete(s.jobs, id)
}

func (s *benchJobStore) remove(id int) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, ok := s.jobs[id]
	s.closeLocked(id)

	return ok
}

// readIntArray reads opts[key] as an array of ints. A missing or non-array
// value yields nil, which callers turn into their own default.
func readIntArray(opts js.Value, key string) []int {
	val := opts.Get(key)
	if !isObject(val) {
		return nil
	}

	n := val.Length()
	out := make([]int, 0, n)

	for i := 0; i < n; i++ {
		item := val.Index(i)
		if item.Type() != js.TypeNumber {
			continue
		}

		out = append(out, item.Int())
	}

	return out
}

// readStringArray reads opts[key] as an array of strings.
func readStringArray(opts js.Value, key string) []string {
	val := opts.Get(key)
	if !isObject(val) {
		return nil
	}

	n := val.Length()
	out := make([]string, 0, n)

	for i := 0; i < n; i++ {
		item := val.Index(i)
		if item.Type() != js.TypeString {
			continue
		}

		out = append(out, item.String())
	}

	return out
}

// expandBenchCases builds the requested cross product of sizes, precisions and
// strategies. Order is size-major so a partially completed run still reads as
// a curve rather than a scatter.
func expandBenchCases(opts js.Value) ([]benchCase, error) {
	sizes := readIntArray(opts, "sizes")
	if len(sizes) == 0 {
		return nil, errBenchEmptyCases
	}

	precisionNames := readStringArray(opts, "precisions")
	if len(precisionNames) == 0 {
		precisionNames = []string{precision64.String()}
	}

	strategyList := readStringArray(opts, "strategies")
	if len(strategyList) == 0 {
		strategyList = []string{algofft.KernelAuto.String()}
	}

	planner := plannerModeFromString(readString(opts, "planner", "estimate"))

	total := len(sizes) * len(precisionNames) * len(strategyList)
	if total > benchMaxCases {
		return nil, errBenchTooMany
	}

	cases := make([]benchCase, 0, total)

	for _, rawSize := range sizes {
		size := clampInt(rawSize, minAnalyzeN, maxAnalyzeN)

		for _, precisionName := range precisionNames {
			for _, strategyName := range strategyList {
				cases = append(cases, benchCase{
					size:         size,
					precision:    precisionFromString(precisionName),
					strategy:     strategyFromString(strategyName),
					strategyName: strategyName,
					planner:      planner,
				})
			}
		}
	}

	return cases, nil
}

// jsBenchStart implements algofft.benchStart(). It expands the cross product
// and returns a job handle; it deliberately runs no transforms, so it returns
// promptly however large the request is.
//
// Shape:
//
//	benchStart({sizes: number[], precisions?: string[], strategies?: string[],
//	            planner?: string, trials?: number, targetMs?: number})
//	  -> {id, total, trials, targetMs, granularityNs,
//	      cases: [{size, precision, strategyRequested, planner}]}
func jsBenchStart(opts js.Value) any {
	if !isObject(opts) {
		return errorMessage("missing options object")
	}

	cases, err := expandBenchCases(opts)
	if err != nil {
		return errorResult(err)
	}

	trials := clampInt(readInt(opts, "trials", benchDefaultTrials), benchMinTrials, benchMaxTrials)
	targetMs := clampInt(readInt(opts, "targetMs", benchDefaultTargetMs), benchMinTargetMs, benchMaxTargetMs)

	// The target must clear the clock's own resolution by a wide margin or the
	// calibration is measuring quantisation noise. Where the browser clamp is
	// coarse (Firefox, Safari, ~1 ms) this raises the target above what the
	// user asked for, which is the right trade: slower, but true.
	targetNs := int64(targetMs) * int64(time.Millisecond)

	if floor := int64(benchReliableTicks) * int64(timerGranularityNs()); targetNs < floor {
		targetNs = floor
	}

	job := &benchJob{
		cases:    cases,
		trials:   trials,
		targetNs: targetNs,
		phase:    phasePrepare,
		iters:    1,
	}

	id := benchJobs.add(job)

	descriptors := make([]any, len(cases))
	for i, c := range cases {
		descriptors[i] = map[string]any{
			"size":              c.size,
			"precision":         c.precision.String(),
			"strategyRequested": c.strategyName,
			"planner":           plannerModeName(c.planner),
		}
	}

	return js.ValueOf(map[string]any{
		"id":            id,
		"total":         len(cases),
		"trials":        trials,
		"targetMs":      float64(targetNs) / float64(time.Millisecond),
		"granularityNs": timerGranularityNs(),
		"cases":         descriptors,
	})
}

// jsBenchStep implements algofft.benchStep(). One call performs exactly one
// unit of work — a plan build, one calibration batch, or one timed trial —
// and returns. Every unit is bounded to roughly the configured target time,
// which is what gives the worker a chance to see a queued cancel.
//
// Shape:
//
//	benchStep({id}) -> {id, done, phase, caseIndex, total,
//	                    iterations, prepared?, result?}
func jsBenchStep(opts js.Value) any {
	if !isObject(opts) {
		return errorMessage("missing options object")
	}

	id := readInt(opts, "id", 0)

	job := benchJobs.get(id)
	if job == nil {
		return errorResult(errBenchNoJob)
	}

	return js.ValueOf(job.step())
}

// jsBenchCancel implements algofft.benchCancel(). It drops the job state and
// its plans. Cancellation itself happens on the JS side simply by not calling
// benchStep again; this releases the memory.
//
// Shape: benchCancel({id}) -> {id, cancelled: bool}
func jsBenchCancel(opts js.Value) any {
	if !isObject(opts) {
		return errorMessage("missing options object")
	}

	id := readInt(opts, "id", 0)

	return js.ValueOf(map[string]any{
		"id":        id,
		"cancelled": benchJobs.remove(id),
	})
}

// step advances the job by one bounded unit of work.
func (j *benchJob) step() map[string]any {
	if j.index >= len(j.cases) {
		benchJobs.remove(j.id)

		return map[string]any{
			"id":        j.id,
			"done":      true,
			"phase":     phaseFinished.String(),
			"caseIndex": len(j.cases),
			"total":     len(j.cases),
		}
	}

	base := map[string]any{
		"id":        j.id,
		"done":      false,
		"caseIndex": j.index,
		"total":     len(j.cases),
	}

	switch j.phase {
	case phasePrepare:
		return j.stepPrepare(base)
	case phaseCalibrate:
		return j.stepCalibrate(base)
	case phaseTrial:
		return j.stepTrial(base)
	case phaseFinished:
		fallthrough
	default:
		j.advance()

		base["phase"] = phaseFinished.String()

		return base
	}
}

// advance moves to the next case and resets the per-case state.
func (j *benchJob) advance() {
	if j.prep != nil && j.prep.info != nil {
		j.prep.info.Close()
	}

	j.prep = nil
	j.iters = 1
	j.timing = j.timing[:0]
	j.index++
	j.phase = phasePrepare

	if j.index >= len(j.cases) {
		j.phase = phaseFinished
	}
}

// failCase records a failed case and moves on. A failure is a result, not the
// end of the run: forcing a strategy the planner cannot honour at some size is
// exactly the kind of thing this page exists to show.
func (j *benchJob) failCase(base map[string]any, err error) map[string]any {
	result := benchCaseHeader(j.cases[j.index])
	result["error"] = err.Error()

	base["phase"] = j.phase.String()
	base["result"] = result

	j.advance()

	if j.index >= len(j.cases) {
		base["done"] = true
	}

	return base
}

// benchCaseHeader is the part of a result that is known before any measuring.
func benchCaseHeader(c benchCase) map[string]any {
	return map[string]any{
		"size":              c.size,
		"precision":         c.precision.String(),
		"strategyRequested": c.strategyName,
		"planner":           plannerModeName(c.planner),
	}
}

func (j *benchJob) stepPrepare(base map[string]any) map[string]any {
	c := j.cases[j.index]

	prep, err := prepareBenchCase(c)
	if err != nil {
		return j.failCase(base, err)
	}

	j.prep = prep
	j.iters = 1
	j.timing = j.timing[:0]
	j.phase = phaseCalibrate

	prepared := benchCaseHeader(c)
	prepared["algorithm"] = prep.algorithm
	prepared["strategyResolved"] = prep.resolved
	prepared["planNs"] = float64(prep.planNs)
	prepared["planWisdomNs"] = float64(prep.planWisdomNs)
	prepared["wisdomEntries"] = demoWisdom.Len()

	base["phase"] = phasePrepare.String()
	base["prepared"] = prepared

	return base
}

// stepCalibrate runs one batch and doubles the iteration count until a batch
// clears the target time.
//
// Doubling rather than extrapolating is the point: extrapolating from a batch
// that measured zero ticks (which is what any small n does on the first pass)
// divides by zero or by one tick, and the old code's answer to that was to
// guess 100000 iterations. Doubling converges from below and never needs the
// clock to have resolved anything except the batch it is about to accept.
func (j *benchJob) stepCalibrate(base map[string]any) map[string]any {
	start := time.Now()

	if err := j.prep.runner.run(j.iters); err != nil {
		return j.failCase(base, err)
	}

	elapsed := time.Since(start).Nanoseconds()

	base["phase"] = phaseCalibrate.String()
	base["iterations"] = j.iters
	base["elapsedNs"] = float64(elapsed)

	if elapsed >= j.targetNs || j.iters >= benchMaxIterations {
		j.phase = phaseTrial

		return base
	}

	next := j.iters * 2

	// Jump straight to a count that should hit the target when the clock did
	// resolve something useful, but never shrink and never overshoot the cap.
	if elapsed > 0 {
		if scaled := int(float64(j.iters) * float64(j.targetNs) / float64(elapsed)); scaled > next {
			next = scaled
		}
	}

	j.iters = min(next, benchMaxIterations)

	return base
}

func (j *benchJob) stepTrial(base map[string]any) map[string]any {
	start := time.Now()

	if err := j.prep.runner.run(j.iters); err != nil {
		return j.failCase(base, err)
	}

	elapsed := time.Since(start).Nanoseconds()

	j.timing = append(j.timing, float64(elapsed))

	base["phase"] = phaseTrial.String()
	base["iterations"] = j.iters
	base["trial"] = len(j.timing)
	base["trials"] = j.trials

	if len(j.timing) < j.trials {
		return base
	}

	base["result"] = j.finishCase()

	j.advance()

	if j.index >= len(j.cases) {
		base["done"] = true
	}

	return base
}

// finishCase reduces the trial batches to the reported numbers.
func (j *benchJob) finishCase() map[string]any {
	c := j.cases[j.index]
	prep := j.prep

	batches := slices.Clone(j.timing)
	slices.Sort(batches)

	medianBatchNs := median(batches)
	meanBatchNs := mean(batches)
	stddevBatchNs := stddev(batches, meanBatchNs)

	iters := float64(j.iters)
	granularity := timerGranularityNs()

	perIterNs := medianBatchNs / iters

	relStddev := 0.0
	if meanBatchNs > 0 {
		relStddev = stddevBatchNs / meanBatchNs
	}

	// A batch is trustworthy when its total duration is a large multiple of
	// the smallest interval the clock can express. Below that the reported
	// figure is mostly a statement about the browser's rounding policy.
	reliable := medianBatchNs >= benchReliableTicks*granularity

	trialNs := make([]any, len(j.timing))
	for i, v := range j.timing {
		trialNs[i] = v / iters
	}

	out := benchCaseHeader(c)
	out["algorithm"] = prep.algorithm
	out["strategyResolved"] = prep.resolved
	out["planNs"] = float64(prep.planNs)
	out["planWisdomNs"] = float64(prep.planWisdomNs)
	out["iterations"] = j.iters
	out["trialNs"] = trialNs
	out["avgNs"] = perIterNs
	out["medianNs"] = perIterNs
	out["meanNs"] = meanBatchNs / iters
	out["stddevNs"] = stddevBatchNs / iters
	out["relStddev"] = relStddev
	out["totalNs"] = medianBatchNs
	out["totalTimeMs"] = medianBatchNs / float64(time.Millisecond)
	out["granularityNs"] = granularity
	out["reliable"] = reliable
	out["checksum"] = prep.runner.checksum()

	return out
}

func median(sorted []float64) float64 {
	n := len(sorted)
	if n == 0 {
		return 0
	}

	if n%2 == 1 {
		return sorted[n/2]
	}

	return (sorted[n/2-1] + sorted[n/2]) / 2
}

func mean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	sum := 0.0
	for _, v := range values {
		sum += v
	}

	return sum / float64(len(values))
}

func stddev(values []float64, m float64) float64 {
	if len(values) < 2 {
		return 0
	}

	sum := 0.0
	for _, v := range values {
		d := v - m
		sum += d * d
	}

	return math.Sqrt(sum / float64(len(values)-1))
}

// jsBenchmark implements the legacy algofft.benchmark(). It is retained only
// so the pre-worker globalThis.algofftBenchmark shim keeps working; it is a
// blocking main-thread call and the benchmark page no longer uses it.
//
// It now delegates to the same step machinery rather than carrying its own
// loop, so there is exactly one definition of how a case gets measured.
//
// Shape: benchmark({sizes, precision, strategy, planner, trials, targetMs})
func jsBenchmark(opts js.Value) any {
	if !isObject(opts) {
		return errorMessage("missing options object")
	}

	precision := readString(opts, "precision", precision64.String())
	strategy := readString(opts, "strategy", algofft.KernelAuto.String())

	// Reuse the array-shaped entry point by hand-building the singleton
	// precision/strategy lists the old scalar options imply.
	cases, err := expandBenchCases(opts)
	if err != nil {
		return errorResult(err)
	}

	planner := plannerModeFromString(readString(opts, "planner", "estimate"))

	trials := clampInt(readInt(opts, "trials", benchMinTrials), benchMinTrials, benchMaxTrials)
	targetMs := clampInt(readInt(opts, "targetMs", benchDefaultTargetMs), benchMinTargetMs, benchMaxTargetMs)

	targetNs := int64(targetMs) * int64(time.Millisecond)
	if floor := int64(benchReliableTicks) * int64(timerGranularityNs()); targetNs < floor {
		targetNs = floor
	}

	// The scalar precision/strategy options win over whatever expandBenchCases
	// defaulted to, preserving the old call shape exactly.
	for i := range cases {
		cases[i].precision = precisionFromString(precision)
		cases[i].strategy = strategyFromString(strategy)
		cases[i].strategyName = strategy
		cases[i].planner = planner
	}

	job := &benchJob{
		cases:    cases,
		trials:   trials,
		targetNs: targetNs,
		phase:    phasePrepare,
		iters:    1,
	}
	job.id = -1

	results := make([]any, 0, len(cases))

	for job.index < len(job.cases) {
		out := job.step()
		if result, ok := out["result"]; ok {
			results = append(results, result)
		}
	}

	if job.prep != nil && job.prep.info != nil {
		job.prep.info.Close()
		job.prep = nil
	}

	return js.ValueOf(results)
}
