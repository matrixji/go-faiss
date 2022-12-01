package faiss_test

import (
	"fmt"

	"github.com/matrixji/go-faiss/faiss"
)

func ExampleNewIndexIDMap() {
	index, _ := faiss.NewIndexFlatIP(4)
	indexWithID, _ := faiss.NewIndexIDMap(index)
	indexWithID.AddWithIDs([]float32{0.5, 0.5, 0.5, 0.5, 0.3, 0.9, 0.3, 0.1}, []int64{10000, 20000})
	fmt.Printf("index have %d elements\n", index.Ntotal())
	fmt.Printf("indexWithID have %d elements\n", indexWithID.Ntotal())

	// Output:
	// index have 2 elements
	// indexWithID have 2 elements
}
