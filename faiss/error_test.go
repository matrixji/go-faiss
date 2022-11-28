package faiss_test

import (
	"testing"

	"github.com/matrixji/go-faiss/faiss"
)

func TestGetLastError(t *testing.T) {
	err := faiss.GetLastError()
	if err != nil {
		t.Log(err)
		t.Error(err)
	}
}
