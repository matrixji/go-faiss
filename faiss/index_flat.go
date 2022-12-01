package faiss

// #include <faiss/c_api/IndexFlat_c.h>
import "C"
import "unsafe"

// IndexFlat for broue force search
type IndexFlat struct {
	baseIndex
}

// NewIndexFlat creates a new flat index with dimension and metric.
// Returns the new index and error
func NewIndexFlat(d int, metric MetricType) (*IndexFlat, error) {
	var index baseIndex
	if ret := C.faiss_IndexFlat_new_with(
		&index.ptr,
		C.idx_t(d),
		C.FaissMetricType(metric),
	); ret != 0 {
		return nil, GetLastError()
	}
	return &IndexFlat{index}, nil
}

// Xb returns the index's vectors.
// The returned slice is mapping to c native memory,
// after add/delete some vectors from the index,
// this slice is invalid, need to fetch again
func (index *IndexFlat) Xb() []float32 {
	var size C.size_t
	var flaots *C.float
	C.faiss_IndexFlat_xb(index.Ptr(), &flaots, &size)
	return (*[1 << 30]float32)(unsafe.Pointer(flaots))[:size:size]
}

// ComputeDistanceSubset compute distance with a subset of vectors.
// Returns corresponding output distances, size n * k, and the error
func (index *IndexFlat) ComputeDistanceSubset(
	n int64, x []float32, k int64, labels []int64,
) ([]float32, error) {
	distances := make([]float32, n*k)
	if ret := C.faiss_IndexFlat_compute_distance_subset(
		index.Ptr(),
		C.idx_t(n),
		(*C.float)(&x[0]),
		C.idx_t(k),
		(*C.float)(&distances[0]),
		(*C.idx_t)(&labels[0])); ret == 0 {
		return distances, nil
	}
	return nil, GetLastError()
}

// AsFlatIndex casts index to flat index.
// Returns nil if not a flat index
func AsFlatIndex(index Index) *IndexFlat {
	myBaseIndex, ok := index.(*baseIndex)
	if !ok {
		return nil
	}
	ptr := C.faiss_IndexFlat_cast(myBaseIndex.ptr)
	if ptr == nil {
		return nil
	}
	return &IndexFlat{baseIndex{ptr: nil, internalIndex: myBaseIndex}}
}
