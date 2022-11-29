package faiss_test

import (
	"fmt"

	"github.com/matrixji/go-faiss/faiss"
)

func ExampleNewIDSelectorRange() {
	// create flat index with inner product metric, dimension 4
	index, _ := faiss.NewIndex(4, "Flat", faiss.MetricInnerProduct)

	// add 100 vectors(2*50) to index, [0.5, 0.5, 0.5, 0.5], [0.3, 0.9, 0.3, 0.1]
	for i := 0; i < 50; i++ {
		index.Add([]float32{0.5, 0.5, 0.5, 0.5, 0.3, 0.9, 0.3, 0.1})
	}
	fmt.Printf("Total %d\n", index.Ntotal())

	// remove id [10, 90)
	selector, _ := faiss.NewIDSelectorRange(10, 90)
	removed, _ := index.RemoveIDs(selector)

	fmt.Printf("Removed %d\n", removed)
	fmt.Printf("Total %d\n", index.Ntotal())

	// Output:
	// Total 100
	// Removed 80
	// Total 20
}
