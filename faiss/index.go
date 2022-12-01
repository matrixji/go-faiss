package faiss

// #include <stdlib.h>
// #include <faiss/c_api/Index_c.h>
import "C"
import "runtime"

// Index interface from faiss's Index class.
//
// The faiss's index an abstract class.
// We use interface to wrapper it, as faiss's implementation,
// not all methods are implemented for all indexes.
type Index interface {
	// D returns the dimension of the vectors.
	D() int

	// IsTrained returns if the index has been trained
	IsTrained() bool

	// Ntotal returns total number of vectors in index
	Ntotal() int64

	// MetricType returns the metric type for index
	MetricType() MetricType

	// Train trains index with the input x
	// 	the input x's length can be divided by D()
	// Returns error
	Train(x []float32) error

	// Add add to index with the input x
	// 	the input x's length can be divided by D()
	// Returns error
	Add(x []float32) error

	// AddWithIDs add index with the input x and related ids by xids
	//  the input x's length should be equals to len(xids) * D()
	// Returns error
	AddWithIDs(x []float32, xids []int64) error

	// Search do nn search for the input x and top k
	// Return the k nearest neighbors, corresponding distances and error
	Search(x []float32, k int64) ([]float32, []int64, error)

	// RangeSearch do range search for the input x and radius
	// Returns vectors with distance < radius and error
	RangeSearch(x []float32, radius float32) (*RangeSearchResult, error)

	// Assign similar to Search, but only return the neighbors
	// Returns the neighbors and error
	Assign(x []float32, k int64) ([]int64, error)

	// Reset clear vectors from the index.
	Reset() error

	// RemoveIDs removes the vectors specified by selector from the index.
	// Returns the number of elements removed and error
	RemoveIDs(selector IDSelector) (int, error)
}

type baseIndex struct {
	ptr           *C.FaissIndex // c pointer
	internalIndex *baseIndex    // internal index for support AsFooIndex
}

// NewBaseIndex new baseIndex from C faiss index pointer
func NewBaseIndex(ptr *C.FaissIndex) *baseIndex {
	ret := &baseIndex{ptr, nil}
	runtime.SetFinalizer(ret, func(index *baseIndex) { index.Free() })
	return ret
}

// Free destroy the resource for index
func (index *baseIndex) Free() {
	if index.ptr != nil {
		C.faiss_Index_free(index.ptr)
		index.ptr = nil
	}
	index.internalIndex = nil
}

// Ptr returns the raw c_api pointer
func (index *baseIndex) Ptr() *C.FaissIndex {
	if index.ptr != nil {
		return index.ptr
	}
	if index.internalIndex != nil {
		return index.internalIndex.Ptr()
	}
	return nil
}

// BaseIndex returns the base internal index which holding the c api index pointer
func (index *baseIndex) BaseIndex() *baseIndex {
	if index.internalIndex != nil {
		return index.internalIndex.BaseIndex()
	}
	return index
}

func (index *baseIndex) D() int {
	return int(C.faiss_Index_d(index.Ptr()))
}

func (index *baseIndex) IsTrained() bool {
	return C.faiss_Index_is_trained(index.Ptr()) != 0
}

func (index *baseIndex) Ntotal() int64 {
	return int64(C.faiss_Index_ntotal(index.Ptr()))
}

func (index *baseIndex) MetricType() MetricType {
	return MetricType(int(C.faiss_Index_metric_type(index.Ptr())))
}

func (index *baseIndex) Train(x []float32) error {
	n := len(x) / index.D()
	if n := C.faiss_Index_train(
		index.Ptr(),
		C.idx_t(n),
		(*C.float)(&x[0]),
	); n != 0 {
		return GetLastError()
	}
	return nil
}

func (index *baseIndex) Add(x []float32) error {
	n := len(x) / index.D()
	if n := C.faiss_Index_add(
		index.Ptr(),
		C.idx_t(n),
		(*C.float)(&x[0]),
	); n != 0 {
		return GetLastError()
	}
	return nil
}

func (index *baseIndex) AddWithIDs(x []float32, xids []int64) error {
	n := len(x) / index.D()
	if n := C.faiss_Index_add_with_ids(
		index.Ptr(),
		C.idx_t(n),
		(*C.float)(&x[0]),
		(*C.idx_t)(&xids[0]),
	); n != 0 {
		return GetLastError()
	}
	return nil
}

func (index *baseIndex) Search(x []float32, k int64) (
	[]float32, []int64, error,
) {
	n := len(x) / index.D()
	distances := make([]float32, int64(n)*k)
	labels := make([]int64, int64(n)*k)
	if c := C.faiss_Index_search(
		index.Ptr(),
		C.idx_t(n),
		(*C.float)(&x[0]),
		C.idx_t(k),
		(*C.float)(&distances[0]),
		(*C.idx_t)(&labels[0]),
	); c != 0 {
		return []float32{}, []int64{}, GetLastError()
	}
	return distances, labels, nil
}

func (index *baseIndex) RangeSearch(x []float32, radius float32) (
	*RangeSearchResult, error,
) {
	n := len(x) / index.D()
	result, err := NewRangeSearchResult(n)
	if err != nil {
		return nil, err
	}
	if c := C.faiss_Index_range_search(
		index.Ptr(),
		C.idx_t(n),
		(*C.float)(&x[0]),
		C.float(radius),
		result.Ptr(),
	); c != 0 {
		return nil, GetLastError()
	}
	return result, nil
}

func (index *baseIndex) Assign(x []float32, k int64) ([]int64, error) {
	n := len(x) / index.D()
	labels := make([]int64, int64(n)*k)
	if c := C.faiss_Index_assign(
		index.Ptr(),
		C.idx_t(n),
		(*C.float)(&x[0]),
		(*C.idx_t)(&labels[0]),
		C.idx_t(k),
	); c != 0 {
		return []int64{}, GetLastError()
	}
	return labels, nil
}

func (index *baseIndex) Reset() error {
	if c := C.faiss_Index_reset(index.Ptr()); c != 0 {
		return GetLastError()
	}
	return nil
}

func (index *baseIndex) RemoveIDs(selector IDSelector) (int, error) {
	var removed C.size_t
	if c := C.faiss_Index_remove_ids(index.Ptr(), selector.Ptr(), &removed); c != 0 {
		return 0, GetLastError()
	}
	return int(removed), nil
}

// The following functions are rarely used and will be implemented later
// TODO: reconstruct
// TODO: reconstruct_n
// TODO: compute_residual
// TODO: compute_residual_n
// TODO: sa_code_size
// TODO: sa_encode
// TODO: sa_decode
