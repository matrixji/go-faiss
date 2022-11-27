package faiss

// #include <stdlib.h>
// #include <faiss/c_api/Index_c.h>
import "C"

type Index struct {
	ptr *C.FaissIndex
}

func (index *Index) Free() {
	if index.ptr != nil {
		C.faiss_Index_free(index.ptr)
		index.ptr = nil
	}
}

func (index *Index) D() int {
	return int(C.faiss_Index_d(index.ptr))
}

func (index *Index) IsTrained() bool {
	return C.faiss_Index_is_trained(index.ptr) != 0
}

func (index *Index) Ntotal() int64 {
	return int64(C.faiss_Index_ntotal(index.ptr))
}

func (index *Index) MetricType() MetricType {
	return MetricType(int(C.faiss_Index_metric_type(index.ptr)))
}

func (index *Index) Train(x []float32) error {
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

func (index *Index) Add(x []float32) error {
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

func (index *Index) AddWithIDs(x []float32, xids []int64) error {
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

func (index *Index) Search(x []float32, k int64) (
	distances []float32, labels []int64, err error,
) {
	n := len(x) / index.D()
	distances = make([]float32, int64(n)*k)
	labels = make([]int64, int64(n)*k)
	err = nil
	if c := C.faiss_Index_search(
		index.ptr,
		C.idx_t(n),
		(*C.float)(&x[0]),
		C.idx_t(k),
		(*C.float)(&distances[0]),
		(*C.idx_t)(&labels[0]),
	); c != 0 {
		err = GetLastError()
	}
	return
}

func (index *Index) RangeSearch(x []float32, radius float32) (
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
		result.Ptr(),
	); c != 0 {
		return nil, GetLastError()
	}
	return result, nil
}

func (index *Index) Assign(x []float32, k int64) (labels []int64, err error) {
	n := len(x) / index.D()
	labels = make([]int64, int64(n)*k)
	err = nil
	if c := C.faiss_Index_assign(
		index.ptr,
		C.idx_t(n),
		(*C.float)(&x[0]),
		(*C.idx_t)(&labels[0]),
		C.idx_t(k),
	); c != 0 {
		err = GetLastError()
	}
	return
}

func (index *Index) Reset() error {
	if c := C.faiss_Index_reset(index.ptr); c != 0 {
		return GetLastError()
	}
	return nil
}

func (index *Index) RemoveIDs(selector *IDSelector) (int, error) {
	var removed C.size_t
	if c := C.faiss_Index_remove_ids(index.ptr, selector.ptr, &removed); c != 0 {
		return 0, GetLastError()
	}
	return int(removed), nil
}

// TODO: reconstruct
// TODO: reconstruct_n
// TODO: compute_residual
// TODO: compute_residual_n
// TODO: sa_code_size
// TODO: sa_encode
// TODO: sa_decode
