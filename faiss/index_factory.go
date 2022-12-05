package faiss

// #include <stdlib.h>
// #include <faiss/c_api/index_factory_c.h>
// #include <faiss/c_api/IndexFlat_c.h>
// #include <faiss/c_api/IndexIVF_c.h>
// #include <faiss/c_api/MetaIndexes_c.h>
import "C"
import (
	"errors"
	"unsafe"
)

// castFromNewFaissIndex cast to the final index class
//
// Parameters:
//   - faissIndexPtr, the just created new index
//
// Returns:
//   - Index, the casted index
//   - error, the failure reason, nil on success
func castFromNewFaissIndex(faissIndexPtr *C.FaissIndex) (Index, error) {
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
	if ptr := C.faiss_IndexIDMap_cast(faissIndexPtr); ptr != nil {
		// check if sub index is owned, should always true
		if C.faiss_IndexIDMap_own_fields(ptr) == 1 {
			C.faiss_IndexIDMap_set_own_fields(ptr, 0)
			subIndexPtr := C.faiss_IndexIDMap_sub_index(ptr)
			subIndex, err := castFromNewFaissIndex(subIndexPtr)
			if err != nil {
				return nil, err
			}
			return &IndexIDMap{*newBaseIndex(faissIndexPtr), subIndex}, nil
		}
		return &IndexIDMap{*newBaseIndex(faissIndexPtr), nil}, nil
	}

	// IndexIVF
	if ptr := C.faiss_IndexIVF_cast(faissIndexPtr); ptr != nil {
		// check if sub index is owned, should always true
		if C.faiss_IndexIVF_own_fields(ptr) == 1 {
			C.faiss_IndexIVF_set_own_fields(ptr, 0)
			quantizerPtr := C.faiss_IndexIVF_quantizer(ptr)
			quantizer, err := castFromNewFaissIndex(quantizerPtr)
			if err != nil {
				return nil, err
			}
			return &IndexIVF{*newBaseIndex(faissIndexPtr), quantizer}, nil
		}
		return &IndexIVF{*newBaseIndex(faissIndexPtr), nil}, nil
	}

	// IndexFlat, IndexFlatL2 or IndexFlatIP
	if ptr := C.faiss_IndexFlat_cast(faissIndexPtr); ptr != nil {
		metricType := C.faiss_Index_metric_type(faissIndexPtr)
		if metricType == C.METRIC_INNER_PRODUCT {
			return &IndexFlatIP{IndexFlat{*newBaseIndex(faissIndexPtr)}}, nil
		}
		if metricType == C.METRIC_L2 {
			return &IndexFlatL2{IndexFlat{*newBaseIndex(faissIndexPtr)}}, nil
		}
		return &IndexFlat{*newBaseIndex(faissIndexPtr)}, nil
	}
	return nil, errors.New("cast c index to index error")
}

// castFromFaissIndex cast to the final index class with prev index
//
// Parameters:
//   - faissIndexPtr, the just created new index
//   - fromIndex, the index copy/clone from
//
// Returns:
//   - Index, the casted index
//   - error, the failure reason, nil on success
func castFromFaissIndex(faissIndexPtr *C.FaissIndex, fromIndex Index) (Index, error) {
	if fromIndex == nil {
		return castFromNewFaissIndex(faissIndexPtr)
	}

	if _, ok := fromIndex.(*IndexFlatIP); ok {
		return &IndexFlatIP{IndexFlat{*newBaseIndex(faissIndexPtr)}}, nil
	}

	if _, ok := fromIndex.(*IndexFlatL2); ok {
		return &IndexFlatL2{IndexFlat{*newBaseIndex(faissIndexPtr)}}, nil
	}

	if _, ok := fromIndex.(*IndexFlat); ok {
		return &IndexFlat{*newBaseIndex(faissIndexPtr)}, nil
	}

	if index, ok := fromIndex.(*IndexIDMap); ok {
		clonedSubIndex, err := CloneIndex(index.subIndex)
		if err != nil {
			return nil, err
		}
		return &IndexIDMap{*newBaseIndex(faissIndexPtr), clonedSubIndex}, nil
	}

	return castFromNewFaissIndex(faissIndexPtr)
}

// NewIndex create index by metric with faiss's index_factory.
//
// Paramsters:
//   - d, for dimensions
//   - metric, for metric type
//
// Returns:
//   - Index, the created index
//   - error, the failure reason, nil on success
//
// More details, see: https://github.com/facebookresearch/faiss/wiki/The-index-factory
func NewIndex(d int, description string, metric MetricType) (Index, error) {

	desc := C.CString(description)
	defer C.free(unsafe.Pointer(desc))
	var faissIndexPtr *C.FaissIndex
	if ret := C.faiss_index_factory(
		&faissIndexPtr,
		C.int(d),
		desc,
		C.FaissMetricType(metric),
	); ret != 0 {
		return nil, GetLastError()
	}
	return castFromFaissIndex(faissIndexPtr, nil)
}
