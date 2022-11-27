package faiss

// #include <faiss/c_api/Index_c.h>
import "C"

// MetricType wrapper for faiss MetricType
type MetricType int

const (
	MetricInnerProduct  MetricType = C.METRIC_INNER_PRODUCT
	MetricL2            MetricType = C.METRIC_L2
	MetricL1            MetricType = C.METRIC_L1
	MetricLinf          MetricType = C.METRIC_Linf
	MetricLp            MetricType = C.METRIC_Lp
	MetricCanberra      MetricType = C.METRIC_Canberra
	MetricBraycurtis    MetricType = C.METRIC_BrayCurtis
	MetricJensenshannon MetricType = C.METRIC_JensenShannon
)
