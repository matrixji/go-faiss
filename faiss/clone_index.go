package faiss

// #include <faiss/c_api/clone_index_c.h>
import "C"
import "errors"

// CloneIndex clone index to new index.
// Returns index and error.
func CloneIndex(index Index) (Index, error) {
	concreteIndex, ok := index.(*baseIndex)
	if !ok {
		return nil, errors.New("input index is not a baseIndex")
	}
	var newIndex *C.FaissIndex
	if ret := C.faiss_clone_index(concreteIndex.Ptr(), &newIndex); ret == 0 {
		return &baseIndex{newIndex, nil}, nil
	}
	return nil, GetLastError()
}
