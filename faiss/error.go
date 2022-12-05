package faiss

// #include <faiss/c_api/error_c.h>
import "C"
import "errors"

// GetLastError get last error from faiss_get_last_error
//
// Returns:
//   - error, the last error from faiss c api
func GetLastError() error {
	if errText := C.faiss_get_last_error(); errText != nil {
		return errors.New(C.GoString(errText))
	}
	return nil
}
