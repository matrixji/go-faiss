package faiss_test

import (
	"testing"

	"github.com/matrixji/go-faiss/faiss"
	"github.com/stretchr/testify/assert"
)

func TestMetricTypeGeneric(t *testing.T) {
	assert.EqualValues(t, faiss.MetricInnerProduct, 0)
	assert.EqualValues(t, faiss.MetricL2, 1)
	assert.EqualValues(t, faiss.MetricL1, 2)
	assert.EqualValues(t, faiss.MetricLinf, 3)
	assert.EqualValues(t, faiss.MetricLp, 4)
	assert.EqualValues(t, faiss.MetricCanberra, 20)
	assert.EqualValues(t, faiss.MetricBraycurtis, 21)
	assert.EqualValues(t, faiss.MetricJensenshannon, 22)
}
