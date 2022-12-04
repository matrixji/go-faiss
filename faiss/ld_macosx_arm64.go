package faiss

// TODO remove this hardcode flags
// #cgo CFLAGS: -I/Users/jibin/Working/faiss -I/Users/jibin/Working
// #cgo LDFLAGS: -L/Users/jibin/Working/faiss/build/c_api -L/Users/jibin/Working/faiss/build/faiss -L/opt/homebrew/lib -L/opt/homebrew/Cellar/openblas/0.3.21/lib  -lfaiss_c -lfaiss -lstdc++ -lm -lomp -lopenblas
import "C"
