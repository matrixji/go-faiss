package faiss

// #include <stdlib.h>
// #include <faiss/c_api/Index_c.h>
import "C"
import (
	"runtime"
)

// Index interface from faiss's Index class.
//
// The faiss's index an abstract class.
// We use interface to wrapper it, as faiss's implementation,
// not all methods are implemented for all indexes.
type Index interface {
	// D returns the dimension of the vectors.
	//
	// Returns:
	//	 - int, the dimension of the vectors.
	D() int

	// IsTrained returns if the index has been trained
	//
	// Returns:
	//   - bool, if the index has been trained
	IsTrained() bool

	// Ntotal returns total number of vectors in index
	//
	// Returns:
	//	 - int64, total number of vectors in index
	Ntotal() int64

	// MetricType returns the metric type for index
	//
	// Returns:
	//	 - MetricType, the metric type for index
	MetricType() MetricType

	// Train trains index with the input x
	//
	// Paramsters:
	//	 - x, the input data, x's length can be divided by D()
	//
	// Returns:
	//	 - error, failure reason, nil on success
	Train(x []float32) error

	// Add add to index with the input x
	//
	// Paramsters:
	//	 - x, the input data, x's length can be divided by D()
	//
	// Returns:
	//	 - error, failure reason, nil on success
	Add(x []float32) error

	// AddWithIDs add index with the input x and related ids by xids
	//
	// Paramsters:
	//	 - x, input x's length should be equals to len(xids) * D()
	// 	 - xids, input id slice
	//
	// Returns:
	//	 - error, failure reason, nil on success
	AddWithIDs(x []float32, xids []int64) error

	// Search do nn search for the input x and top k
	//
	// Paramsters:
	// 	 - x, input vectors for query, x's length can be divided by D()
	//   - k, top k
	//
	// Returns:
	//   - []float32, the distances, length should be len(x)/D()*k
	// 	 - []int64, the ret ids, ids length should be same as the didistances'
	// 	 - error, the failure reason, nil on success
	Search(x []float32, k int64) ([]float32, []int64, error)

	// RangeSearch do range search for the input x and radius
	// Paramsters:
	// 	 - x, input vectors for query, x's length can be divided by D()
	//   - radius, the distance threshold for query
	//
	// Returns:
	//	 - *RangeSearchResult, the result
	//	 - error, failure reason
	//
	// It vectors with distance < radius for L2 and distance > radius for IP
	RangeSearch(x []float32, radius float32) (*RangeSearchResult, error)

	// Assign similar to Search, but only return the neighbors
	//
	// Paramsters:
	// 	 - x, input vectors for query, x's length can be divided by D()
	//   - k, top k
	//
	// Returns:
	// 	 - []int64, the ret ids
	// 	 - error, the failure reason, nil on success
	Assign(x []float32, k int64) ([]int64, error)

	// Reset clear vectors from the index.
	Reset() error

	// RemoveIDs removes the vectors specified by selector from the index.
	//
	// Paramsters:
	// 	 - selector, selector for select IDs
	//
	// Returns:
	// 	 - int, removed items count
	//	 - error, failure reason
	RemoveIDs(selector IDSelector) (int, error)

	//
	// non faiss interface below
	//

	// Ptr returns raw c_api pointer
	Ptr() *C.FaissIndex
}

// base abstract index
type baseIndex struct {
	ptr *C.FaissIndex // c pointer
}

// newBaseIndex new baseIndex from C faiss index pointer
//
// Parameters:
//   - ptr, internal c api pointer to index
//
// Returns:
//   - *baseIndex, index
func newBaseIndex(ptr *C.FaissIndex) *baseIndex {
	ret := &baseIndex{ptr}
	runtime.SetFinalizer(ret, func(index *baseIndex) { index.Free() })
	return ret
}

// Free destroy the resource for index
func (index *baseIndex) Free() {
	if index.ptr != nil {
		C.faiss_Index_free(index.ptr)
		index.ptr = nil
	}
}

func (index *baseIndex) D() int {
	return int(C.faiss_Index_d(index.ptr))
}

func (index *baseIndex) IsTrained() bool {
	return C.faiss_Index_is_trained(index.ptr) != 0
}

func (index *baseIndex) Ntotal() int64 {
	return int64(C.faiss_Index_ntotal(index.ptr))
}

func (index *baseIndex) MetricType() MetricType {
	return MetricType(int(C.faiss_Index_metric_type(index.ptr)))
}

func (index *baseIndex) Train(x []float32) error {
	n := len(x) / index.D()
	if n := C.faiss_Index_train(
		index.ptr,
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
		index.ptr,
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
		index.ptr,
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
		index.ptr,
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
		index.ptr,
		C.idx_t(n),
		(*C.float)(&x[0]),
		C.float(radius),
		result.ptr,
	); c != 0 {
		return nil, GetLastError()
	}
	return result, nil
}

func (index *baseIndex) Assign(x []float32, k int64) ([]int64, error) {
	n := len(x) / index.D()
	labels := make([]int64, int64(n)*k)
	if c := C.faiss_Index_assign(
		index.ptr,
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
	if c := C.faiss_Index_reset(index.ptr); c != 0 {
		return GetLastError()
	}
	return nil
}

func (index *baseIndex) RemoveIDs(selector IDSelector) (int, error) {
	var removed C.size_t
	if c := C.faiss_Index_remove_ids(index.ptr, selector.Ptr(), &removed); c != 0 {
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

//
// below non faiss methods
//

func (index *baseIndex) Ptr() *C.FaissIndex {
	return index.ptr
}
