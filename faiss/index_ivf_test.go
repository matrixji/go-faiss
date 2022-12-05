package faiss_test

import (
	"math"
	"testing"

	"github.com/matrixji/go-faiss/faiss"
	"github.com/stretchr/testify/assert"
)

func TestIndexIVFGeneric(t *testing.T) {
	index, err := faiss.NewIndex(128, "IVF256,PQ16", faiss.MetricL2)
	assert.Nil(t, err)
	assert.NotNil(t, index)
	indexIVF, ok := index.(*faiss.IndexIVF)
	assert.True(t, ok)
	indexIVF.SetNProbe(16)
	assert.EqualValues(t, 256, indexIVF.NList())
	assert.EqualValues(t, 16, indexIVF.NProbe())
	assert.NotNil(t, indexIVF.Quantizer())
	assert.Equal(t, 0, indexIVF.QuantizerTrainsAlone())
	assert.EqualValues(t, 0, indexIVF.GetListSize(0))
	assert.EqualValues(t, []int64{}, indexIVF.InvlistsGetIds(0))
	assert.True(t, math.IsNaN(indexIVF.ImbalanceFactor()))
	indexIVF.PrintStats()
	t.Log(faiss.GetIndexIVFStats())
}
