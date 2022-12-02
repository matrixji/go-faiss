package faiss

// #include <faiss/c_api/MetaIndexes_c.h>
import "C"

type IndexIDMap struct {
	baseIndex

	// sub index here, use for holding subIndex, and set own_fields to false
	subIndex Index
}

// NewIndexIDMap create IndexIDMap from sub index.
// Returns the IndexIDMap and error
func NewIndexIDMap(index Index) (*IndexIDMap, error) {
	var ptr *C.FaissIndexIDMap
	if ret := C.faiss_IndexIDMap_new(&ptr, index.Ptr()); ret != 0 {
		return nil, GetLastError()
	}
	myIndexIDMap := &IndexIDMap{*NewBaseIndex(ptr), index}
	return myIndexIDMap, nil
}

// Return sub index
func (index *IndexIDMap) SubIndex() Index {
	return index.subIndex
}

// TODO: IndexIDMap2
