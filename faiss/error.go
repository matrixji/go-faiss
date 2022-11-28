package faiss

// #include <faiss/c_api/error_c.h>
import "C"
import "errors"

// GetLastError get last error from faiss_get_last_error
func GetLastError() error {
	if errText := C.faiss_get_last_error(); errText != nil {
		return errors.New(C.GoString(errText))
	}
	return nil
}
