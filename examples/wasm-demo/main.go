//go:build js && wasm

package main

import (
	"syscall/js"
)

// exports lists every function the demo publishes, by its name on the
// namespaced globalThis.algofft object. Each one is wrapped by guard, which is
// the single rule this bridge has: nothing reaches JavaScript without a
// recover() in front of it (see bridge.go).
var exports = map[string]func(js.Value) any{
	"analyze":          jsAnalyze,
	"info":             jsInfo,
	"benchmark":        jsBenchmark,
	"benchStart":       jsBenchStart,
	"benchStep":        jsBenchStep,
	"benchCancel":      jsBenchCancel,
	"cacheStats":       jsCacheStats,
	"cacheClear":       jsCacheClear,
	"roundtripCompare": jsRoundtripCompare,
	"convolve":         jsConvolve,
}

// live keeps the js.Func values referenced so they are never released.
var live []js.Func

func main() {
	namespace := js.Global().Get("Object").New()

	for name, fn := range exports {
		wrapped := guard(name, fn)
		live = append(live, wrapped)
		namespace.Set(name, wrapped)
	}

	js.Global().Set("algofft", namespace)

	// Back-compat shims for the pre-namespace bridge. The existing app.js and
	// bench.js still call these; they are aliases, not separate code paths, and
	// can be dropped once the JS side moves to globalThis.algofft.
	js.Global().Set("algofftFFT", namespace.Get("analyze"))
	js.Global().Set("algofftBenchmark", namespace.Get("benchmark"))
	js.Global().Set("algofftFFTInfo", jsInfo(js.Undefined()))

	select {}
}
