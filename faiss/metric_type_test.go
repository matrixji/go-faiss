package faiss

import (
	"testing"
)

func TestMetricTypeGeneric(t *testing.T) {
	testExpectEqual := func(metricType MetricType, value int) {
		if metricType != MetricType(value) {
			t.Errorf("%d != %d", metricType, value)
		}
	}

	testExpectEqual(MetricInnerProduct, 0)
	testExpectEqual(MetricL2, 1)
	testExpectEqual(MetricL1, 2)
	testExpectEqual(MetricLinf, 3)
	testExpectEqual(MetricLp, 4)
	testExpectEqual(MetricCanberra, 20)
	testExpectEqual(MetricBraycurtis, 21)
	testExpectEqual(MetricJensenshannon, 22)
}
