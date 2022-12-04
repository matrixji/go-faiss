package faiss

// #cgo CFLAGS: -fopenmp
// #cgo LDFLAGS: -lfaiss_c -lfaiss -lgomp -lstdc++ -lm -lopenblas
import "C"
