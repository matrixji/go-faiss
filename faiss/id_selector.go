package faiss

// #include <faiss/c_api/impl/AuxIndexStructures_c.h>
import "C"
import "runtime"

// IDSelector is intended to define a subset of vectors to handle (for removal or as subset to search).
type IDSelector interface {
	// Ptr return the c pointer to FaissIDSelector
	Ptr() *C.FaissIDSelector
}

// baseIDSelector abstract IDSelector
type baseIDSelector struct {
	ptr *C.FaissIDSelector // internal pointer
}

// Ptr return the internal FaissIDSelector pointer
//
// Returns:
//   - *C.FaissIDSelector, the pointer to c api
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

// NewIDSelectorRange creates a selector for remove IDs
//
// the ids between [imax, imax) will be selected
//
// Parameters:
//   - imin, the min id for start
//   - imax, the max id for stop
//
// Returns:
//   - IDSelector, the selector
//   - error, error if failed, nil on success
func NewIDSelectorRange(imin, imax int64) (IDSelector, error) {
	var ptr *C.FaissIDSelectorRange
	ret := C.faiss_IDSelectorRange_new(&ptr, C.idx_t(imin), C.idx_t(imax))
	if ret != 0 {
		return nil, GetLastError()
	}
	selector := baseIDSelector{(*C.FaissIDSelector)(ptr)}
	runtime.SetFinalizer(&selector, func(selector *baseIDSelector) { selector.free() })
	return &selector, nil
}

// NewIDSelectorBatch creates a new batch selector with indices.
//
// Parameters:
//   - indices, the slice of indices to be selected
//
// Returns:
//   - IDSelector, the selector
//   - error, error if failed, nil on success
func NewIDSelectorBatch(indices []int64) (IDSelector, error) {
	var ptr *C.FaissIDSelectorBatch
	if ret := C.faiss_IDSelectorBatch_new(
		&ptr,
		C.size_t(len(indices)),
		(*C.idx_t)(&indices[0]),
	); ret != 0 {
		return nil, GetLastError()
	}
	selector := baseIDSelector{ptr: (*C.FaissIDSelector)(ptr)}
	runtime.SetFinalizer(&selector, func(selector *baseIDSelector) { selector.free() })
	return &selector, nil
}
