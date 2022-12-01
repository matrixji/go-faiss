package faiss

// #include <faiss/c_api/clone_index_c.h>
import "C"
import "errors"

// CloneIndex clone index to new index.
// Returns index and error.
func CloneIndex(index Index) (Index, error) {
	if index == nil {
		return nil, errors.New("input index is nil")
	}
	var newIndex *C.FaissIndex
	if ret := C.faiss_clone_index(index.Ptr(), &newIndex); ret == 0 {
		return castFromFaissIndex(newIndex, index)
	}
	return nil, GetLastError()
}
