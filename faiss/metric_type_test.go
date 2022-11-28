package faiss_test

import (
	"github.com/matrixji/go-faiss/faiss"
	"testing"
)

func TestMetricTypeGeneric(t *testing.T) {
	testExpectEqual := func(metricType faiss.MetricType, value int) {
		if metricType != faiss.MetricType(value) {
			t.Errorf("%d != %d", metricType, value)
		}
	}

	testExpectEqual(faiss.MetricInnerProduct, 0)
	testExpectEqual(faiss.MetricL2, 1)
	testExpectEqual(faiss.MetricL1, 2)
	testExpectEqual(faiss.MetricLinf, 3)
	testExpectEqual(faiss.MetricLp, 4)
	testExpectEqual(faiss.MetricCanberra, 20)
	testExpectEqual(faiss.MetricBraycurtis, 21)
	testExpectEqual(faiss.MetricJensenshannon, 22)
}
