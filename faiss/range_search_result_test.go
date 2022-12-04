package faiss_test

import (
	"fmt"
	"runtime"
	"testing"

	"github.com/matrixji/go-faiss/faiss"
	"github.com/stretchr/testify/assert"
)

func ExampleIndex_RangeSearch() {
	index, _ := faiss.NewIndexFlatIP(4)
	index.Add([]float32{
		0.8, 0.6, 0.0, 0.0,
		0.5, 0.5, 0.5, 0.5,
		1.0, 0.0, 0.0, 0.0,
		0.9, 0.3, 0.3, 0.1,
	})
	result, _ := index.RangeSearch([]float32{0.5, 0.5, 0.5, 0.5, 0.8, 0.6, 0.0, 0.0}, 0.8)

	lims := result.Lims()
	labels, distances := result.Labels()

	fmt.Printf("Number of query: %d\n", result.Nq())
	fmt.Printf("Lims: %d\n", lims)
	for i := 0; i < int(result.Nq()); i++ {
		fmt.Printf("Result#%d: ", i)
		fmt.Printf("ids=%d", labels[lims[i]:lims[i+1]])
		fmt.Printf(", distances=%.1f\n", distances[lims[i]:lims[i+1]])
	}

	// Output:
	// Number of query: 2
	// Lims: [0 1 3]
	// Result#0: ids=[1], distances=[1.0]
	// Result#1: ids=[0 3], distances=[1.0 0.9]
}

func TestRangeSearchResult_BufferSize(t *testing.T) {
	index, _ := faiss.NewIndexFlatIP(4)
	index.Add([]float32{
		0.8, 0.6, 0.0, 0.0,
		0.5, 0.5, 0.5, 0.5,
		1.0, 0.0, 0.0, 0.0,
		0.9, 0.3, 0.3, 0.1,
	})
	result, _ := index.RangeSearch([]float32{0.5, 0.5, 0.5, 0.5, 0.8, 0.6, 0.0, 0.0}, 0.8)
	assert.Greater(t, result.BufferSize(), 0)
	result = nil
	runtime.GC()

}
