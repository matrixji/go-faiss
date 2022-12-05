package faiss

// #include <stdlib.h>
// #include <faiss/c_api/IndexIVF_c.h>
import "C"

// IndexIVF abstract ivf index
type IndexIVF struct {
	baseIndex       // index
	quantizer Index // subindex for quantizer
}

// NList number of possible key values
//
// Returns:
//   - uint64, the number of possible key values
func (index *IndexIVF) NList() uint64 {
	return uint64(C.faiss_IndexIVF_nlist(index.ptr))
}

// NProbe number of probes at query time
//
// Returns:
//   - uint64, number of probes at query time
func (index *IndexIVF) NProbe() uint64 {
	return uint64(C.faiss_IndexIVF_nprobe(index.ptr))
}

// SetNProbe set number of probes at query time
//
// Parameters:
//   - nprobe, the number of probes at query time
func (index *IndexIVF) SetNProbe(nprobe uint64) {
	C.faiss_IndexIVF_set_nprobe(index.ptr, C.size_t(nprobe))
}

// Quantizer that maps vectors to inverted lists
//
// Returns:
//   - uint64, the quantizer index
func (index *IndexIVF) Quantizer() Index {
	return index.quantizer
}

// QuantizerTrainsAlone
//
// Returns:
//   - int, flags
//
// Return Flags:
//   - 0: use the quantizer as index in a kmeans training
//   - 1: just pass on the training set to the train() of the quantizer
//   - 2: kmeans training on a flat index + add the centroids to the quantizer
func (index *IndexIVF) QuantizerTrainsAlone() int {
	return int(C.faiss_IndexIVF_quantizer_trains_alone(index.ptr))
}

// MergeFrom moves the entries from another dataset to self.
//
// Parameters:
//   - other, the other index, after merge it will empty
//   - addId, added to all moved ids
//
// Returns:
//   - error, failure reason
func (index *IndexIVF) MergeFrom(other *IndexIVF, addId int64) error {
	if ret := C.faiss_IndexIVF_merge_from(index.ptr, other.ptr, C.idx_t(addId)); ret != 0 {
		return GetLastError()
	}
	return nil
}

// TODO: faiss_IndexIVF_copy_subset_to ?
// TODO: faiss_IndexIVF_search_preassigned ?

// GetListSize get the nth list's size
//
// Paramsters:
//   - listNo: the number of nth list
//
// Returns:
//   - uint64: the number of vectors in that list
func (index *IndexIVF) GetListSize(listNo uint64) uint64 {
	return uint64(C.faiss_IndexIVF_get_list_size(index.ptr, C.size_t(listNo)))
}

// TODO: faiss_IndexIVF_make_direct_map ?

// ImbalanceFactor Check the inverted lists' imbalance factor.
//
// Returns:
//   - faloat64, factor =1 means perfectly balanced, >1 means imbalanced
func (index *IndexIVF) ImbalanceFactor() float64 {
	return float64(C.faiss_IndexIVF_imbalance_factor(index.ptr))
}

// PrintStats display some stats about the inverted lists of the index
func (index *IndexIVF) PrintStats() {
	C.faiss_IndexIVF_print_stats(index.ptr)
}

// InvlistsGetIds get the ids in n list
//
// Paramsters:
//   - listNo: the number of nth list
//
// Returns:
//   - []int64: the slice of ids for the nth list
func (index *IndexIVF) InvlistsGetIds(listNo uint64) []int64 {
	count := index.GetListSize(listNo)
	ret := make([]int64, count)
	if count > 0 {
		C.faiss_IndexIVF_invlists_get_ids(index.ptr, C.size_t(listNo), (*C.idx_t)(&ret[0]))
	}
	return ret
}

// IndexIVFStats shows the stats of IndexIVF
type IndexIVFStats struct {
	Nq               uint64  // nb of queries run
	Nlist            uint64  // nb of inverted lists scanned
	Ndis             uint64  // nb of distances computed
	NheapUpdates     uint64  // nb of times the heap was updated
	QuantizationTime float64 // time spent quantizing vectors (in ms)
	SearchTime       float64 // time spent searching lists (in ms)
}

// GetIndexIVFStats get the global stats for IndexIVF
//
// Returns:
//   - *IndexIVFStats, stats for IndexIVF
func GetIndexIVFStats() *IndexIVFStats {
	ret := IndexIVFStats{}
	stats := C.faiss_get_indexIVF_stats()
	ret.Nq = uint64(stats.nq)
	ret.Nlist = uint64(stats.nlist)
	ret.Ndis = uint64(stats.ndis)
	ret.NheapUpdates = uint64(stats.nheap_updates)
	ret.QuantizationTime = float64(stats.quantization_time)
	ret.SearchTime = float64(stats.search_time)

	return &ret
}
