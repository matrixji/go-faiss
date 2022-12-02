package faiss

// #include <stdlib.h>
// #include <faiss/c_api/index_factory_c.h>
// #include <faiss/c_api/IndexFlat_c.h>
// #include <faiss/c_api/MetaIndexes_c.h>
import "C"
import (
	"errors"
	"unsafe"
)

func castFromNewFaissIndex(faissIndex *C.FaissIndex) (Index, error) {
	// Try cast from: faiss_XYZ_cast
	// - faiss_IndexFlat1D_cast
	// - faiss_IndexIDMap2_cast
	// - faiss_IndexIDMap_cast
	// - faiss_IndexIVF_cast
	// - faiss_IndexIVFFlat_cast
	// - faiss_IndexIVFScalarQuantizer_cast
	// - faiss_IndexLSH_cast
	// - faiss_IndexPreTransform_cast
	// - faiss_IndexRefineFlat_cast
	// - faiss_IndexScalarQuantizer_cast

	// IndexIDMap
	if ptr := C.faiss_IndexIDMap_cast(faissIndex); ptr != nil {
		// check if sub index is owned, should always true
		if C.faiss_IndexIDMap_own_fields(ptr) == 1 {
			C.faiss_IndexIDMap_set_own_fields(ptr, 0)
			subIndexPtr := C.faiss_IndexIDMap_sub_index(ptr)
			subIndex, err := castFromNewFaissIndex(subIndexPtr)
			if err != nil {
				return nil, err
			}
			return &IndexIDMap{*NewBaseIndex(faissIndex), subIndex}, nil
		}
		return &IndexIDMap{*NewBaseIndex(faissIndex), nil}, nil
	}

	// IndexFlat (go-faiss using IndexFlat for IndexFlatL2 and IndexFlatIP)
	if ptr := C.faiss_IndexFlat_cast(faissIndex); ptr != nil {
		return &IndexFlat{*NewBaseIndex(faissIndex)}, nil
	}

	return nil, errors.New("cast c index to index error")
}

func castFromFaissIndex(faissIndex *C.FaissIndex, fromIndex Index) (Index, error) {
	if fromIndex == nil {
		return castFromNewFaissIndex(faissIndex)
	}
	if _, ok := fromIndex.(*IndexFlat); ok {
		return &IndexFlat{*NewBaseIndex(faissIndex)}, nil
	}

	if index, ok := fromIndex.(*IndexIDMap); ok {
		clonedSubIndex, err := CloneIndex(index.subIndex)
		if err != nil {
			return nil, err
		}
		return &IndexIDMap{*NewBaseIndex(faissIndex), clonedSubIndex}, nil
	}

	return castFromNewFaissIndex(faissIndex)
}

// NewIndex create index by metric with faiss's index_factory,
// d for dimensions, metric for metric type,
// description is a comma-separated list of components.
// Returns the created index and error.
//
// More details, see: https://github.com/facebookresearch/faiss/wiki/The-index-factory
func NewIndex(d int, description string, metric MetricType) (Index, error) {

	desc := C.CString(description)
	defer C.free(unsafe.Pointer(desc))
	var faissIndex *C.FaissIndex
	if ret := C.faiss_index_factory(
		&faissIndex,
		C.int(d),
		desc,
		C.FaissMetricType(metric),
	); ret != 0 {
		return nil, GetLastError()
	}
	return castFromFaissIndex(faissIndex, nil)
}
