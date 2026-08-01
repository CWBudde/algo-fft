// Vector floating-point arithmetic for Go's arm64 assembler.
//
// Go's assembler has no mnemonic for NEON vector FADD, FSUB or FMUL — they are
// rejected as "unrecognized instruction", in every spelling (FADD, VFADD). The
// only vector FP arithmetic mnemonics it accepts are VFMLA and VFMLS; VADD and
// VSUB are integer-only.
//
// That is a gap in the assembler's instruction table, NOT a gap in the
// hardware, and it can be closed by emitting the encodings directly with WORD.
// Doing so matters a lot here. The workaround these kernels used before was to
// synthesize a + b as a VMOV followed by VFMLA against a vector of 1.0, which
// costs TWO instructions per add or subtract and burns a register on the
// all-ones constant plus two instructions in every prologue to load it. A
// radix-4 butterfly is sixteen adds and subtracts, so on a small codelet that
// overhead is most of the kernel.
//
// The macros take register NUMBERS, not names, because the encoding embeds the
// number and Go's assembler preprocessor does not support token pasting (`##`
// is rejected — "'#' must be first item on line"), so V3 cannot be mapped to 3
// inside a macro. Callers pass the bare index and the identity V<n> <-> n holds
// everywhere: VADDF_S4(0, 2, 8) means "V0 + V2 -> V8". Keep that correspondence
// literal, since nothing checks it — a wrong number assembles happily and the
// registry-driven reference tests are what catch it.
//
// Verified against the hardware under QEMU and on an Apple M5 before use.
//
// Encodings (ARM DDI 0487, Advanced SIMD three-same):
//   FADD Vd.4S, Vn.4S, Vm.4S   0x4E20D400 | Rm<<16 | Rn<<5 | Rd
//   FSUB Vd.4S, Vn.4S, Vm.4S   0x4EA0D400 | ...
//   FMUL Vd.4S, Vn.4S, Vm.4S   0x6E20DC00 | ...
//   FADD Vd.2D, Vn.2D, Vm.2D   0x4E60D400 | ...
//   FSUB Vd.2D, Vn.2D, Vm.2D   0x4EE0D400 | ...
//   FMUL Vd.2D, Vn.2D, Vm.2D   0x6E60DC00 | ...
//   FMLA Vd.4S, Vn.4S, Vm.4S   0x4E20CC00 | ...   (Vd += Vn*Vm)
//   FMLS Vd.4S, Vn.4S, Vm.4S   0x4EA0CC00 | ...   (Vd -= Vn*Vm)
//   FMLA Vd.2D, Vn.2D, Vm.2D   0x4E60CC00 | ...
//   FMLS Vd.2D, Vn.2D, Vm.2D   0x4EE0CC00 | ...
//   ORR  Vd.16B, Vn.16B, Vn.16B  0x4EA01C00 | ... (register move)
//   EOR  Vd.16B, Vn.16B, Vm.16B  0x6E201C00 | ...
//
// VFMLA/VFMLS/VMOV/VEOR all have real mnemonics, but those take register NAMES,
// which cannot appear inside a macro whose other operands are numbers. The
// numeric forms below exist so a whole butterfly can be written in one idiom.

// float32 (.4S): d = a + b, d = a - b, d = a * b.
#define VADDF_S4(a, b, d) WORD $(0x4E20D400 | ((b)<<16) | ((a)<<5) | (d))
#define VSUBF_S4(a, b, d) WORD $(0x4EA0D400 | ((b)<<16) | ((a)<<5) | (d))
#define VMULF_S4(a, b, d) WORD $(0x6E20DC00 | ((b)<<16) | ((a)<<5) | (d))

// float64 (.2D): d = a + b, d = a - b, d = a * b.
#define VADDF_D2(a, b, d) WORD $(0x4E60D400 | ((b)<<16) | ((a)<<5) | (d))
#define VSUBF_D2(a, b, d) WORD $(0x4EE0D400 | ((b)<<16) | ((a)<<5) | (d))
#define VMULF_D2(a, b, d) WORD $(0x6E60DC00 | ((b)<<16) | ((a)<<5) | (d))

// Fused multiply-add/sub: d += a*b, d -= a*b.
#define VFMAF_S4(a, b, d) WORD $(0x4E20CC00 | ((b)<<16) | ((a)<<5) | (d))
#define VFMSF_S4(a, b, d) WORD $(0x4EA0CC00 | ((b)<<16) | ((a)<<5) | (d))
#define VFMAF_D2(a, b, d) WORD $(0x4E60CC00 | ((b)<<16) | ((a)<<5) | (d))
#define VFMSF_D2(a, b, d) WORD $(0x4EE0CC00 | ((b)<<16) | ((a)<<5) | (d))

// Whole-register move and xor (size-agnostic, .16B).
#define VMOVR(a, d)    WORD $(0x4EA01C00 | ((a)<<16) | ((a)<<5) | (d))
#define VEORR(a, b, d) WORD $(0x6E201C00 | ((b)<<16) | ((a)<<5) | (d))
