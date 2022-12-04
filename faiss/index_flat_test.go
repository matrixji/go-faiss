package faiss_test

import (
	"fmt"
	"testing"

	"github.com/matrixji/go-faiss/faiss"
	"github.com/stretchr/testify/assert"
)

func ExampleNewIndexFlat() {
	// create flat index with inner product metric, dimension 4
	index, err := faiss.NewIndexFlat(4, faiss.MetricInnerProduct)
	fmt.Printf("Create index err=%v, dimension=%d, total=%d\n", err, index.D(), index.Ntotal())

	// add 2 vectors to index, [0.5, 0.5, 0.5, 0.5], [0.3, 0.9, 0.3, 0.1]
	err = index.Add([]float32{0.5, 0.5, 0.5, 0.5, 0.3, 0.9, 0.3, 0.1})
	fmt.Printf("Add to index err=%v, total=%d\n", err, index.Ntotal())

	// search with [0.5, 0.5, 0.5, 0.5]
	distances, ids, err := index.Search([]float32{0.5, 0.5, 0.5, 0.5}, 2)

	// output
	fmt.Printf("Search index err=%v, ", err)
	fmt.Print("ids:")
	for _, id := range ids {
		fmt.Printf(" %d", id)
	}
	fmt.Print(", ")

	fmt.Print("distances:")
	for _, distance := range distances {
		fmt.Printf(" %.2f", distance)
	}
	fmt.Print("\n")

	// Output:
	// Create index err=<nil>, dimension=4, total=0
	// Add to index err=<nil>, total=2
	// Search index err=<nil>, ids: 0 1, distances: 1.00 0.80
}
func TestFlatXb(t *testing.T) {
	index, err := faiss.NewIndexFlatIP(2)
	assert.Nil(t, err)
	index.Add([]float32{0.1, 0.1, 0.3, 0.3})
	floats := index.Xb()
	assert.Equal(t, floats, []float32{0.1, 0.1, 0.3, 0.3})
}

func TestIVFComputeDistanceSubset(t *testing.T) {
	index, _ := faiss.NewIndexFlatIP(2)
	index.Add([]float32{1.0, 0.0, 0.6, 0.8, 0.0, 1.0, 0.8, 0.6})
	distances, err := index.ComputeDistanceSubset([]float32{1.0, 0.0, 0.0, 1.0}, []int64{0, 1, 2, 1, 2, 3})
	assert.Nil(t, err)
	assert.Equal(t, len(distances), 6)
	assert.InDeltaSlice(t, distances, []float32{1.0, 0.6, 0.0, 0.8, 1.0, 0.6}, 0.001)
}

func TestNewIndexFlatL2(t *testing.T) {
	index, err := faiss.NewIndexFlatL2(100)
	assert.Nil(t, err)
	assert.Equal(t, index.MetricType(), faiss.MetricL2)
}

func TestNewIndexFlatIP(t *testing.T) {
	index, err := faiss.NewIndexFlatIP(100)
	assert.Nil(t, err)
	assert.Equal(t, index.MetricType(), faiss.MetricInnerProduct)
}
