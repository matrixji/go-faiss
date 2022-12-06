package faiss_test

import (
	"testing"

	"github.com/matrixji/go-faiss/faiss"
	"github.com/stretchr/testify/assert"
)

func TestCloneIndexGeneric(t *testing.T) {
	index, err := faiss.NewIndex(128, "Flat", faiss.MetricL2)
	assert.Nil(t, err)

	index2, err := faiss.CloneIndex(index)

	assert.Nil(t, err)
	assert.Equal(t, index2.D(), 128)
}

func TestCloneIndexFromNil(t *testing.T) {
	index, err := faiss.CloneIndex(nil)
	assert.NotNil(t, err)
	assert.Nil(t, index)
}

func TestCloneIndexForIDMap(t *testing.T) {
	index, _ := faiss.NewIndex(4, "Flat,IDMap", faiss.MetricL2)
	indexIDMap, _ := index.(*faiss.IndexIDMap)
	_ = indexIDMap.AddWithIDs([]float32{0.1, 0.1, 0.1, 0.1}, []int64{10000})
	index2, _ := faiss.CloneIndex(index)
	indexIDMap2, _ := index2.(*faiss.IndexIDMap)
	assert.EqualValues(t, indexIDMap2.Ntotal(), 1)
}
