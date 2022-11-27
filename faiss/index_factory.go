package faiss

// #include <stdlib.h>
// #include <faiss/c_api/index_factory_c.h>
import "C"
import (
	"runtime"
	"unsafe"
)

// NewIndex create index by metric with faiss's index_factory
func NewIndex(d int, description string, metric MetricType) (*Index, error) {
	desc := C.CString(description)
	defer C.free(unsafe.Pointer(desc))
	var index Index
	if ret := C.faiss_index_factory(
		&index.ptr,
		C.int(d),
		desc,
		C.FaissMetricType(metric),
	); ret != 0 {
		return nil, GetLastError()
	}
	runtime.SetFinalizer(index, func(r Index) { r.Free() })
	return &index, nil
}
