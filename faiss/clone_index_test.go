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
	index = nil

	assert.Nil(t, err)
	assert.Equal(t, index2.D(), 128)
}

func TestCloneIndexFromNil(t *testing.T) {
	index, err := faiss.CloneIndex(nil)
	assert.NotNil(t, err)
	assert.Nil(t, index)
}
