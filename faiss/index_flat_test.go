package faiss_test

import (
	"fmt"

	"github.com/matrixji/go-faiss/faiss"
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

func ExampleAsFlatIndex() {
	index, err := faiss.NewIndex(4, "Flat", faiss.MetricInnerProduct)
	fmt.Printf("Create index err=%v, dimension=%d, total=%d\n", err, index.D(), index.Ntotal())

	flatIndex := faiss.AsFlatIndex(index)
	// add to vectors by index
	index.Add([]float32{0.5, 0.5, 0.5, 0.5, 0.3, 0.9, 0.3, 0.1})

	fmt.Printf("Casted index type=%T, dimension=%d, total=%d\n", flatIndex, index.D(), index.Ntotal())

	// Output:
	// Create index err=<nil>, dimension=4, total=0
	// Casted index type=*faiss.IndexFlat, dimension=4, total=2
}
