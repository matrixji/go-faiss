package faiss

// #include <faiss/c_api/IndexFlat_c.h>
import "C"
import "unsafe"

// IndexFlat for broue force search
type IndexFlat struct {
	baseIndex // base index
}

// NewIndexFlat creates a new flat index with dimension and metric.
//
// Parameters:
//   - d, dimension
//   - metric, metric type
//
// Returns:
//   - *IndexFlat, the flat index
//   - error, failure reason, nil on success
func NewIndexFlat(d int, metric MetricType) (*IndexFlat, error) {
	var ptr *C.FaissIndexFlat
	if ret := C.faiss_IndexFlat_new_with(
		&ptr,
		C.idx_t(d),
		C.FaissMetricType(metric),
	); ret != 0 {
		return nil, GetLastError()
	}
	return &IndexFlat{*newBaseIndex(ptr)}, nil
}

// Xb returns the index's vectors.
//
// Returns:
//   - []float32: the slice to vectors
//
// **Note**: The returned slice is mapping to c native memory,
// after add/delete some vectors from the index,
// this slice is invalid, need to fetch again.
func (index *IndexFlat) Xb() []float32 {
	var size C.size_t
	var flaots *C.float
	C.faiss_IndexFlat_xb(index.ptr, &flaots, &size)
	return (*[1 << 30]float32)(unsafe.Pointer(flaots))[:size:size]
}

// ComputeDistanceSubset compute distance with a subset of vectors.
//
// Parameters:
//   - x: the input vectors for calc
//   - labels: the array of labels to be compared for each element in x
//
// Returns:
//   - []float32, corresponding output distances, size n * k
//   - error, failure reason, nil on success
func (index *IndexFlat) ComputeDistanceSubset(x []float32, labels []int64) (
	[]float32, error,
) {
	n := len(x) / index.D()
	k := len(labels) / n
	distances := make([]float32, n*k)
	if ret := C.faiss_IndexFlat_compute_distance_subset(
		index.ptr,
		C.idx_t(n),
		(*C.float)(&x[0]),
		C.idx_t(k),
		(*C.float)(&distances[0]),
		(*C.idx_t)(&labels[0])); ret == 0 {
		return distances, nil
	}
	return nil, GetLastError()
}

// IndexFlatIP IndexFlat with IP metric
type IndexFlatIP struct {
	IndexFlat // flat index
}

// NewIndexFlatIP creates a new IP metric flat index with dimension.
//
// Parameters:
//   - d, dimension
//
// Returns:
//   - *IndexFlatIP, the flat index
//   - error, failure reason, nil on success
func NewIndexFlatIP(d int) (*IndexFlatIP, error) {
	index, err := NewIndexFlat(d, MetricInnerProduct)
	if err != nil {
		return nil, err
	}
	return &IndexFlatIP{*index}, nil
}

// IndexFlatL2 IndexFlat with L2 metric
type IndexFlatL2 struct {
	IndexFlat // flat index
}

// NewIndexFlatL2 creates a new L2 metric flat index with dimension.
//
// Parameters:
//   - d, dimension
//
// Returns:
//   - *IndexFlatL2, the flat index
//   - error, failure reason, nil on success
func NewIndexFlatL2(d int) (*IndexFlatL2, error) {
	index, err := NewIndexFlat(d, MetricL2)
	if err != nil {
		return nil, err
	}
	return &IndexFlatL2{*index}, nil
}

// TODO: RefineFlat
// TODO: Flat1D
