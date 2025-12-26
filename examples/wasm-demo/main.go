//go:build js && wasm

package main

import (
	"math"
	"math/cmplx"
	"math/rand"
	"sync"
	"syscall/js"

	"github.com/MeKo-Christian/algoforge"
)

var (
	planMu    sync.Mutex
	planCache = map[int]*algoforge.Plan[complex64]{}
	plan2DMu  sync.Mutex
	plan2D    = map[int]*algoforge.Plan2D[complex64]{}
	fftFunc   js.Func
)

func main() {
	fftFunc = js.FuncOf(jsFFT)
	js.Global().Set("algoforgeFFT", fftFunc)

	js.Global().Set("algoforgeFFTInfo", js.ValueOf(map[string]any{
		"version": "wasm-demo",
	}))

	select {}
}

func jsFFT(this js.Value, args []js.Value) any {
	if len(args) == 0 || args[0].Type() != js.TypeObject {
		return js.ValueOf(map[string]any{
			"error": "missing options object",
		})
	}

	opts := args[0]
	n := readInt(opts, "n", 1024)
	if n < 16 {
		n = 16
	}
	if n > 4096 {
		n = 4096
	}

	freqA := readFloat(opts, "freqA", 6)
	freqB := readFloat(opts, "freqB", 20)
	noise := readFloat(opts, "noise", 0.08)
	phase := readFloat(opts, "phase", 0)
	gridSize := readInt(opts, "gridSize", 64)
	if gridSize < 16 {
		gridSize = 16
	}
	if gridSize > 256 {
		gridSize = 256
	}

	plan, err := getPlan(n)
	if err != nil {
		return js.ValueOf(map[string]any{
			"error": err.Error(),
		})
	}

	src := make([]complex64, n)
	signal := make([]float64, n)
	rng := rand.New(rand.NewSource(int64(math.Round(phase*1000)) + int64(n)*37))

	for i := 0; i < n; i++ {
		t := float64(i) / float64(n)
		s := math.Sin(2*math.Pi*freqA*t+phase) + 0.65*math.Sin(2*math.Pi*freqB*t+phase*0.7)
		if noise > 0 {
			s += (rng.Float64()*2 - 1) * noise
		}
		signal[i] = s
		src[i] = complex(float32(s), 0)
	}

	dst := make([]complex64, n)
	if err := plan.Forward(dst, src); err != nil {
		return js.ValueOf(map[string]any{
			"error": err.Error(),
		})
	}

	magCount := n / 2
	mags := make([]float64, magCount)
	for i := 0; i < magCount; i++ {
		mags[i] = cmplx.Abs(complex128(dst[i]))
	}

	gridSpectrum, gridErr := computeGridSpectrum(freqA, freqB, noise, phase, gridSize)
	if gridErr != nil {
		return js.ValueOf(map[string]any{
			"error": gridErr.Error(),
		})
	}

	signalArr := js.Global().Get("Float64Array").New(n)
	for i := 0; i < n; i++ {
		signalArr.SetIndex(i, signal[i])
	}

	spectrumArr := js.Global().Get("Float64Array").New(magCount)
	for i := 0; i < magCount; i++ {
		spectrumArr.SetIndex(i, mags[i])
	}

	gridArr := js.Global().Get("Float64Array").New(gridSize * gridSize)
	for i := 0; i < gridSize*gridSize; i++ {
		gridArr.SetIndex(i, gridSpectrum[i])
	}

	result := js.Global().Get("Object").New()
	result.Set("signal", signalArr)
	result.Set("spectrum", spectrumArr)
	result.Set("gridSpectrum", gridArr)
	result.Set("gridSize", gridSize)
	result.Set("n", n)
	return result
}

func getPlan(n int) (*algoforge.Plan[complex64], error) {
	planMu.Lock()
	defer planMu.Unlock()

	if plan, ok := planCache[n]; ok {
		return plan, nil
	}

	plan, err := algoforge.NewPlan32(n)
	if err != nil {
		return nil, err
	}

	planCache[n] = plan
	return plan, nil
}

func computeGridSpectrum(freqA, freqB, noise, phase float64, size int) ([]float64, error) {
	plan, err := getPlan2D(size)
	if err != nil {
		return nil, err
	}

	src := make([]complex64, size*size)
	dst := make([]complex64, size*size)
	rng := rand.New(rand.NewSource(int64(math.Round(phase*830)) + int64(size)*97))

	for y := 0; y < size; y++ {
		fy := float64(y) / float64(size)
		for x := 0; x < size; x++ {
			fx := float64(x) / float64(size)
			val := math.Sin(2*math.Pi*freqA*fx+phase*0.6) +
				0.8*math.Sin(2*math.Pi*freqB*fy+phase*0.4) +
				0.45*math.Sin(2*math.Pi*(freqA*0.5*fx+freqB*0.5*fy)+phase*0.2)
			if noise > 0 {
				val += (rng.Float64()*2 - 1) * noise
			}
			src[y*size+x] = complex(float32(val), 0)
		}
	}

	if err := plan.Forward(dst, src); err != nil {
		return nil, err
	}

	mags := make([]float64, size*size)
	for i := 0; i < size*size; i++ {
		mags[i] = cmplx.Abs(complex128(dst[i]))
	}
	return mags, nil
}

func getPlan2D(size int) (*algoforge.Plan2D[complex64], error) {
	plan2DMu.Lock()
	defer plan2DMu.Unlock()

	if plan, ok := plan2D[size]; ok {
		return plan, nil
	}

	plan, err := algoforge.NewPlan2D32(size, size)
	if err != nil {
		return nil, err
	}

	plan2D[size] = plan
	return plan, nil
}

func readInt(opts js.Value, key string, fallback int) int {
	val := opts.Get(key)
	if val.Type() != js.TypeNumber {
		return fallback
	}
	return val.Int()
}

func readFloat(opts js.Value, key string, fallback float64) float64 {
	val := opts.Get(key)
	if val.Type() != js.TypeNumber {
		return fallback
	}
	return val.Float()
}
