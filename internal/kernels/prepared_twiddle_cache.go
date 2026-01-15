package kernels

import (
	"sync"

	"github.com/MeKo-Christian/algo-fft/internal/memory"
)

type preparedTwiddleKey struct {
	size      int
	inverse   bool
	signature string
}

type preparedTwiddleEntry[T Complex] struct {
	data    []T
	backing []byte
}

type preparedTwiddleCache[T Complex] struct {
	m sync.Map // map[preparedTwiddleKey]preparedTwiddleEntry[T]
}

func (c *preparedTwiddleCache[T]) get(
	n int,
	inverse bool,
	signature string,
	twiddleSize func(int) int,
	prepare func(n int, inverse bool, dst []T),
	alloc func(int) ([]T, []byte),
) []T {
	if twiddleSize == nil || prepare == nil {
		return nil
	}

	key := preparedTwiddleKey{
		size:      n,
		inverse:   inverse,
		signature: signature,
	}
	if v, ok := c.m.Load(key); ok {
		return v.(preparedTwiddleEntry[T]).data
	}

	size := twiddleSize(n)
	if size <= 0 {
		return nil
	}

	data, backing := alloc(size)
	prepare(n, inverse, data)

	entry := preparedTwiddleEntry[T]{
		data:    data,
		backing: backing,
	}

	actual, _ := c.m.LoadOrStore(key, entry)

	return actual.(preparedTwiddleEntry[T]).data
}

var (
	preparedTwiddleCache64  preparedTwiddleCache[complex64]
	preparedTwiddleCache128 preparedTwiddleCache[complex128]
)

func GetPreparedTwiddle64(entry *CodeletEntry[complex64], n int, inverse bool) []complex64 {
	if entry == nil {
		return nil
	}

	return preparedTwiddleCache64.get(
		n,
		inverse,
		entry.Signature,
		entry.TwiddleSize,
		entry.PrepareTwiddle,
		memory.AllocAlignedComplex64,
	)
}

func GetPreparedTwiddle128(entry *CodeletEntry[complex128], n int, inverse bool) []complex128 {
	if entry == nil {
		return nil
	}

	return preparedTwiddleCache128.get(
		n,
		inverse,
		entry.Signature,
		entry.TwiddleSize,
		entry.PrepareTwiddle,
		memory.AllocAlignedComplex128,
	)
}
