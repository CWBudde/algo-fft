//go:build js && wasm

package main

import (
	"syscall/js"
	"unsafe"
)

// Marshalling policy for the demo.
//
// The obvious way to hand a Go slice to JavaScript from wasm is a
// SetIndex loop, and that is what this demo used to do. It costs one JS
// boundary crossing per element: at n=4096 with a 64x64 grid that is roughly
// 10,240 crossings per animation frame, plausibly more time than the FFT the
// demo exists to show off. Everything below exists to turn that into four
// bulk copies.
//
// The mechanism is js.CopyBytesToJS, which does a single memcpy from the Go
// linear memory into a JS typed array. It only accepts Uint8Array, so the
// caller supplies view *pairs* over one JS-owned ArrayBuffer: Go writes
// through `u8`, JS reads the same bytes through `f32`.
//
// Why the buffer must be JS-owned: Go's wasm heap can grow, and growing the
// WebAssembly.Memory detaches every typed array that was created over its
// buffer. A view over a JS-allocated ArrayBuffer is unaffected, so it can be
// cached on the JS side and reused across frames.
//
// Why reinterpreting []float32 as []byte is sound here:
//   - wasm is little-endian by specification, and so is the JS typed-array
//     view of the same bytes, so no byte swapping is required;
//   - Go guarantees []float32 backing arrays are 4-byte aligned, which is all
//     a byte view needs;
//   - the byte slice never outlives the call, and the source slice is kept
//     alive by the caller's own reference for the duration of the copy.
//
// Output is Float32Array throughout. Canvas rendering has no use for more
// precision, and it halves the bytes moved.

// float32Sink is a caller-provided {f32, u8} view pair over one JS ArrayBuffer.
type float32Sink struct {
	f32      js.Value
	u8       js.Value
	capacity int // in float32 elements
}

// newFloat32Sink allocates a fresh JS ArrayBuffer of n float32s and returns
// the matching view pair. Used when the caller supplied no reusable buffer.
func newFloat32Sink(n int) float32Sink {
	buf := js.Global().Get("ArrayBuffer").New(n * 4)

	return float32Sink{
		f32:      js.Global().Get("Float32Array").New(buf),
		u8:       js.Global().Get("Uint8Array").New(buf),
		capacity: n,
	}
}

// sinkFromJS interprets a caller-supplied `{f32, u8}` object as a sink.
// It returns ok=false when the object is missing, malformed, or too small,
// in which case the caller should allocate instead.
func sinkFromJS(out js.Value, need int) (float32Sink, bool) {
	if !isObject(out) {
		return float32Sink{}, false
	}

	f32 := out.Get("f32")
	u8 := out.Get("u8")

	if !isObject(f32) || !isObject(u8) {
		return float32Sink{}, false
	}

	capacity := f32.Length()
	if capacity < need || u8.Length() < need*4 {
		return float32Sink{}, false
	}

	return float32Sink{f32: f32, u8: u8, capacity: capacity}, true
}

// write copies data into the sink and returns the JS Float32Array view that
// exactly covers it. When the sink is larger than the payload a subarray is
// returned, so JS never has to track a separate length.
func (s float32Sink) write(data []float32) js.Value {
	if len(data) > 0 {
		js.CopyBytesToJS(s.u8, float32Bytes(data))
	}

	if s.capacity == len(data) {
		return s.f32
	}

	return s.f32.Call("subarray", 0, len(data))
}

// writeFloat32 copies data into the caller-provided view pair `out` when it is
// usable, and otherwise allocates a fresh Float32Array. Either way it returns
// the JS Float32Array holding exactly len(data) elements.
func writeFloat32(out js.Value, data []float32) js.Value {
	sink, ok := sinkFromJS(out, len(data))
	if !ok {
		sink = newFloat32Sink(len(data))
	}

	return sink.write(data)
}

// outView returns opts.out[name], or undefined when the caller passed no
// output buffers.
func outView(opts js.Value, name string) js.Value {
	if !isObject(opts) {
		return js.Undefined()
	}

	out := opts.Get("out")
	if !isObject(out) {
		return js.Undefined()
	}

	return out.Get(name)
}

// float32Bytes returns a byte view over the same memory as data. See the
// soundness note at the top of this file. The result must not be retained
// beyond the immediate copy.
func float32Bytes(data []float32) []byte {
	if len(data) == 0 {
		return nil
	}

	return unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data)*4)
}
