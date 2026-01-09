package algofft

import "testing"

func BenchmarkPlanForwardStrided(b *testing.B) {
	benchForwardStrided(b, false)
}

func BenchmarkPlanForwardStrided_Copy(b *testing.B) {
	benchForwardStrided(b, true)
}

//nolint:gocognit
func benchForwardStrided(b *testing.B, useCopy bool) {
	b.Helper()

	sizes := []int{256, 1024, 4096}
	cols := 256
	col := 7

	for _, n := range sizes {
		plan, err := NewPlan32(n)
		if err != nil {
			b.Fatalf("NewPlan32(%d) failed: %v", n, err)
		}

		total := n * cols

		data := make([]complex64, total)
		for i := range data {
			data[i] = complex(float32(i%97), float32(i%31))
		}

		dst := make([]complex64, total)
		stride := cols
		srcSlice := data[col:]
		dstSlice := dst[col:]

		tmp := make([]complex64, n)

		label := "Strided"
		if useCopy {
			label = "Copy"
		}

		b.Run(label+"/"+itoa(n), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(n * 8))
			b.ResetTimer()

			for range b.N {
				if useCopy {
					for j := range n {
						tmp[j] = srcSlice[j*stride]
					}

					err := plan.Forward(tmp, tmp)
					if err != nil {
						b.Fatalf("Forward failed: %v", err)
					}

					for j := range n {
						dstSlice[j*stride] = tmp[j]
					}
				} else {
					err := plan.ForwardStrided(dstSlice, srcSlice, stride)
					if err != nil {
						b.Fatalf("ForwardStrided failed: %v", err)
					}
				}
			}
		})
	}
}
