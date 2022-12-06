package faiss_test

import (
	"fmt"
	"testing"

	"github.com/matrixji/go-faiss/faiss"
	"github.com/stretchr/testify/assert"
)

func TestIndexReset(t *testing.T) {
	index, _ := faiss.NewIndexFlatL2(2)
	_ = index.Add([]float32{0.1, 0.2, 0.3, 0.4})
	assert.EqualValues(t, index.Ntotal(), 2)
	_ = index.Reset()
	assert.EqualValues(t, index.Ntotal(), 0)
}

func ExampleIndex_Assign() {
	topk := 2
	index, _ := faiss.NewIndexFlatL2(2)
	_ = index.Add([]float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6})
	ids, _ := index.Assign([]float32{0.3, 0.4, 0.1, 0.2}, int64(topk))
	fmt.Printf("len(ids) = %d\n", len(ids))

	for i := 0; i < 2; i++ {
		fmt.Printf("result#%d: %d\n", i, ids[2*i:2*i+2])
	}

	// Output:
	// len(ids) = 4
	// result#0: [1 2]
	// result#1: [0 1]
}
