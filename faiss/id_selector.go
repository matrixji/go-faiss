package faiss

// #include <faiss/c_api/impl/AuxIndexStructures_c.h>
import "C"
import "runtime"

type IDSelector struct {
	ptr *C.FaissIDSelector
}

// Free destroy the resource for IDSelector
func (selector *IDSelector) Free() {
	if selector.ptr != nil {
		C.faiss_IDSelector_free(selector.ptr)
		selector.ptr = nil
	}
}

// NewIDSelectorRange creates a selector for remove IDs: [imin, imax)
func NewIDSelectorRange(imin, imax int64) (*IDSelector, error) {
	var ptr *C.FaissIDSelectorRange
	ret := C.faiss_IDSelectorRange_new(&ptr, C.idx_t(imin), C.idx_t(imax))
	if ret != 0 {
		return nil, GetLastError()
	}
	result := IDSelector{(*C.FaissIDSelector)(ptr)}
	runtime.SetFinalizer(result, func(r IDSelector) { r.Free() })
	return &result, nil
}

// NewIDSelectorBatch creates a new batch selector with indices.
func NewIDSelectorBatch(indices []int64) (*IDSelector, error) {
	var ptr *C.FaissIDSelectorBatch
	if ret := C.faiss_IDSelectorBatch_new(
		&ptr,
		C.size_t(len(indices)),
		(*C.idx_t)(&indices[0]),
	); ret != 0 {
		return nil, GetLastError()
	}
	result := IDSelector{(*C.FaissIDSelector)(ptr)}
	runtime.SetFinalizer(result, func(r IDSelector) { r.Free() })
	return &result, nil
}
