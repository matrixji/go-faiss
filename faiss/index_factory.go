package faiss

// #include <stdlib.h>
// #include <faiss/c_api/index_factory_c.h>
import "C"
import (
	"unsafe"
)

// NewIndex create index by metric with faiss's index_factory,
// d for dimensions, metric for metric type,
// description is a comma-separated list of components.
// Returns the created index and error.
//
// More details, see: https://github.com/facebookresearch/faiss/wiki/The-index-factory
func NewIndex(d int, description string, metric MetricType) (Index, error) {
	desc := C.CString(description)
	defer C.free(unsafe.Pointer(desc))
	var ptr *C.FaissIndex
	if ret := C.faiss_index_factory(
		&ptr,
		C.int(d),
		desc,
		C.FaissMetricType(metric),
	); ret != 0 {
		return nil, GetLastError()
	}
	return NewBaseIndex(ptr), nil
}
