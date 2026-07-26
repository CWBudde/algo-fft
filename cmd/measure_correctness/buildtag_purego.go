//go:build purego

package main

// puregoBuild reports whether this binary was built with the purego tag, i.e.
// with the assembly codelets compiled out. CPU feature detection is not gated on
// the tag, so it still reports AVX2 on a purego build — an accuracy number is
// meaningless without knowing which codelets actually ran, hence this flag.
const puregoBuild = true
