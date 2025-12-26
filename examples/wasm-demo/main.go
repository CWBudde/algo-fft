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

	signalArr := js.Global().Get("Float64Array").New(n)
	for i := 0; i < n; i++ {
		signalArr.SetIndex(i, signal[i])
	}

	spectrumArr := js.Global().Get("Float64Array").New(magCount)
	for i := 0; i < magCount; i++ {
		spectrumArr.SetIndex(i, mags[i])
	}

	result := js.Global().Get("Object").New()
	result.Set("signal", signalArr)
	result.Set("spectrum", spectrumArr)
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
