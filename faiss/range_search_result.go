package faiss

// #include <stdlib.h>
// #include <faiss/c_api/impl/AuxIndexStructures_c.h>
import "C"
import (
	"runtime"
	"unsafe"
)

// RangeSearchResult Wrapper for faiss c api RangeSearchResult
type RangeSearchResult struct {
	ptr *C.FaissRangeSearchResult // internal pointer to RangeSearchResult
}

// Ptr return raw c pointer of RangeSearchResult
func (result *RangeSearchResult) Ptr() *C.FaissRangeSearchResult {
	return result.ptr
}

// Free destroy the resource for RangeSearchResult
func (result *RangeSearchResult) Free() {
	if result.ptr != nil {
		C.faiss_RangeSearchResult_free(result.ptr)
		result.ptr = nil
	}
}

// NewRangeSearchResult create new RangeSearchResult
func NewRangeSearchResult(nq int) (*RangeSearchResult, error) {
	var ptr *C.FaissRangeSearchResult
	if ret := C.faiss_RangeSearchResult_new(&ptr, C.idx_t(nq)); ret != 0 {
		return nil, GetLastError()
	}
	result := RangeSearchResult{ptr: ptr}
	runtime.SetFinalizer(&result, func(r *RangeSearchResult) { r.Free() })
	return &result, nil
}

// Nq return the number of queries.
func (result *RangeSearchResult) Nq() int64 {
	return int64(C.faiss_RangeSearchResult_nq(result.ptr))
}

// Lims returns a slice contains for lims for returned labels/distances
func (result *RangeSearchResult) Lims() []int64 {
	var lims *C.size_t
	C.faiss_RangeSearchResult_lims(result.ptr, &lims)
	length := result.Nq() + 1
	return (*[1 << 30]int64)(unsafe.Pointer(lims))[:length:length]
}

// Labels returns labels/distances
// the result for query i is labels[lims[i]:lims[i+1]]
func (result *RangeSearchResult) Labels() ([]int64, []float32) {
	lims := result.Lims()
	length := lims[len(lims)-1]
	var cLabels *C.idx_t
	var cDistances *C.float
	C.faiss_RangeSearchResult_labels(result.ptr, &cLabels, &cDistances)
	labels := (*[1 << 30]int64)(unsafe.Pointer(cLabels))[:length:length]
	distances := (*[1 << 30]float32)(unsafe.Pointer(cDistances))[:length:length]
	return labels, distances
}

// BufferSize return the buffer_size
func (result *RangeSearchResult) BufferSize() int {
	return int(C.faiss_RangeSearchResult_buffer_size(result.ptr))
}
