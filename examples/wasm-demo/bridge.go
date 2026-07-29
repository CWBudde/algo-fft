//go:build js && wasm

package main

import (
	"fmt"
	"syscall/js"
)

// guard wraps a demo entry point so that no failure inside Go can ever reach
// the JavaScript side as a trap.
//
// This matters more under js/wasm than it would anywhere else: a Go panic that
// unwinds out of a js.Func aborts the whole wasm instance. Every subsequent
// call into the module then fails, so a single bad request permanently bricks
// the page until the user reloads. The library has at least one reachable
// panic on this path — plan_exec.go panics when a bound kernel rejects a size,
// and KernelSplitRadix is not size-guarded in the planner — so forcing a
// strategy at a length it cannot execute is a user-reachable abort.
//
// Every exported function must therefore go through guard. The wrapper hands
// fn the first argument (or js.Undefined when the caller passed none), and
// converts a panic into an ordinary result object:
//
//	{error: "<name>: <message>", panic: true}
//
// so the UI can render a failed row and carry on.
func guard(name string, fn func(js.Value) any) js.Func {
	return js.FuncOf(func(_ js.Value, args []js.Value) (result any) {
		defer func() {
			if r := recover(); r != nil {
				result = js.ValueOf(map[string]any{
					"error": fmt.Sprintf("%s: %v", name, r),
					"panic": true,
				})
			}
		}()

		opts := js.Undefined()
		if len(args) > 0 {
			opts = args[0]
		}

		return fn(opts)
	})
}

// errorResult builds the standard error shape returned by every export.
// panic is false here; only guard sets it to true.
func errorResult(err error) any {
	return js.ValueOf(map[string]any{
		"error": err.Error(),
		"panic": false,
	})
}

// errorMessage builds the standard error shape from a plain string.
func errorMessage(msg string) any {
	return js.ValueOf(map[string]any{
		"error": msg,
		"panic": false,
	})
}

// readInt returns opts[key] as an int, or fallback when it is missing or not
// a number.
func readInt(opts js.Value, key string, fallback int) int {
	val := opts.Get(key)
	if val.Type() != js.TypeNumber {
		return fallback
	}

	return val.Int()
}

// readFloat returns opts[key] as a float64, or fallback when it is missing or
// not a number.
func readFloat(opts js.Value, key string, fallback float64) float64 {
	val := opts.Get(key)
	if val.Type() != js.TypeNumber {
		return fallback
	}

	return val.Float()
}

// readString returns opts[key] as a string, or fallback when it is missing or
// not a string.
func readString(opts js.Value, key, fallback string) string {
	val := opts.Get(key)
	if val.Type() != js.TypeString {
		return fallback
	}

	return val.String()
}

// readBool returns opts[key] as a bool, or fallback when it is missing or not
// a boolean.
func readBool(opts js.Value, key string, fallback bool) bool {
	val := opts.Get(key)
	if val.Type() != js.TypeBoolean {
		return fallback
	}

	return val.Bool()
}

// isObject reports whether v is a usable JS object (and not null).
func isObject(v js.Value) bool {
	return v.Type() == js.TypeObject
}

// clampInt constrains v to [lo, hi].
func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}

	if v > hi {
		return hi
	}

	return v
}
