package faiss_test

import (
	"fmt"
	"testing"

	"github.com/matrixji/go-faiss/faiss"
	"github.com/stretchr/testify/assert"
)

func ExampleNewIndex() {
	// create flat index with inner product metric, dimension 4
	index, err := faiss.NewIndex(4, "Flat", faiss.MetricInnerProduct)
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

func TestNonExistIndex(t *testing.T) {
	index, err := faiss.NewIndex(4, "NonExist", faiss.MetricInnerProduct)
	assert.Nil(t, index)
	assert.NotNil(t, err)
}

func TestNewIndexGeneric(t *testing.T) {
	index, err := faiss.NewIndex(4, "Flat,IDMap", faiss.MetricL2)
	assert.Nil(t, err)
	assert.NotNil(t, index)

	indexIDMap, ok := index.(*faiss.IndexIDMap)
	assert.True(t, ok)

	indexFlat, ok := indexIDMap.SubIndex().(*faiss.IndexFlat)
	assert.True(t, ok)
	assert.Equal(t, indexFlat.MetricType(), faiss.MetricL2)

}
