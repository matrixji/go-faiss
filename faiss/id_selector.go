package faiss

// #include <faiss/c_api/impl/AuxIndexStructures_c.h>
import "C"
import "runtime"

// IDSelector is intended to define a subset of vectors to handle (for removal
// * or as subset to search)
// Only IDSelectorRange and IDSelectorBatch ported by
// NewIDSelectorRange and NewIDSelectorBatch due to faiss's c_api
type IDSelector interface {
	Ptr() *C.FaissIDSelector
}

// abstract IDSelector
type baseIDSelector struct {
	ptr *C.FaissIDSelector
}

// Ptr return the internal FaissIDSelector pointer
func (selector *baseIDSelector) Ptr() *C.FaissIDSelector {
	return selector.ptr
}

// free destroy the resource for baseIDSelector
func (selector *baseIDSelector) free() {
	if selector.ptr != nil {
		C.faiss_IDSelector_free(selector.ptr)
		selector.ptr = nil
	}
}

// NewIDSelectorRange creates a selector for remove IDs: [imin, imax)
func NewIDSelectorRange(imin, imax int64) (IDSelector, error) {
	var ptr *C.FaissIDSelectorRange
	ret := C.faiss_IDSelectorRange_new(&ptr, C.idx_t(imin), C.idx_t(imax))
	if ret != 0 {
		return nil, GetLastError()
	}
	result := baseIDSelector{(*C.FaissIDSelector)(ptr)}
	runtime.SetFinalizer(&result, func(r baseIDSelector) { r.free() })
	return &result, nil
}

// NewIDSelectorBatch creates a new batch selector with indices.
func NewIDSelectorBatch(indices []int64) (IDSelector, error) {
	var ptr *C.FaissIDSelectorBatch
	if ret := C.faiss_IDSelectorBatch_new(
		&ptr,
		C.size_t(len(indices)),
		(*C.idx_t)(&indices[0]),
	); ret != 0 {
		return nil, GetLastError()
	}
	result := baseIDSelector{(*C.FaissIDSelector)(ptr)}
	runtime.SetFinalizer(&result, func(r baseIDSelector) { r.free() })
	return &result, nil
}
