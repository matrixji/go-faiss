package faiss

// #include <faiss/c_api/MetaIndexes_c.h>
import "C"

// IndexIDMap the index which could call AddWithIDs
type IndexIDMap struct {
	// base index
	baseIndex

	// sub index here, use for holding subIndex, and set own_fields to false
	subIndex Index
}

// NewIndexIDMap create IndexIDMap from sub index.
//
// Paramsters:
//   - index, the subindex for hold vectors
//
// Returns:
//   - *IndexIDMap, created index
//   - error, failure reason
func NewIndexIDMap(index Index) (*IndexIDMap, error) {
	var ptr *C.FaissIndexIDMap
	if ret := C.faiss_IndexIDMap_new(&ptr, index.Ptr()); ret != 0 {
		return nil, GetLastError()
	}
	myIndexIDMap := &IndexIDMap{*newBaseIndex(ptr), index}
	return myIndexIDMap, nil
}

// SubIndex sub index
func (index *IndexIDMap) SubIndex() Index {
	return index.subIndex
}

// TODO: IndexIDMap2
