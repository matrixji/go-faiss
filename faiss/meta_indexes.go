package faiss

// #include <faiss/c_api/MetaIndexes_c.h>
import "C"

type IndexIDMap struct {
	baseIndex

	// sub index here, use for holding subIndex, and set own_fields to false
	subIndex *baseIndex
}

// NewIndexIDMap create IndexIDMap from sub index.
// Returns the IndexIDMap and error
func NewIndexIDMap(index Index) (*IndexIDMap, error) {
	var ptr *C.FaissIndexIDMap
	var subIndex, ok = index.(*baseIndex)
	if !ok {
		return nil, GetLastError()
	}
	if ret := C.faiss_IndexIDMap_new(&ptr, subIndex.Ptr()); ret != 0 {
		return nil, GetLastError()
	}

	myIndexIDMap := &IndexIDMap{*NewBaseIndex(ptr), subIndex}
	return myIndexIDMap, nil
}

// AsIndexIDMap casts index to id map index.
// Returns nil if not a id map index
func AsIndexIDMap(index Index) *IndexIDMap {
	myBaseIndex, ok := index.(*baseIndex)
	if !ok {
		return nil
	}

	// return if could cast in golang level
	myIndexIDMap, ok := index.(*IndexIDMap)
	if ok {
		return myIndexIDMap
	}

	// cast at c_api level
	ptr := C.faiss_IndexIDMap_cast(myBaseIndex.Ptr())
	if ptr == nil {
		return nil
	}

	myIndexIDMap = &IndexIDMap{baseIndex{ptr: nil, internalIndex: myBaseIndex}, nil}
	if C.faiss_IndexIDMap_own_fields(ptr) == 1 {
		// hold subIndex in golang level
		subIndex := baseIndex{C.faiss_IndexIDMap_sub_index(ptr), nil}
		C.faiss_IndexIDMap_set_own_fields(ptr, 0)
		myIndexIDMap.subIndex = &subIndex
	}
	return myIndexIDMap
}

// TODO: IndexIDMap2
