/*
 * Compatibility shim for aligned_alloc on Android API < 28
 * 
 * aligned_alloc is only available from Android API 28 (Android 9.0) onwards.
 * This file provides a compatibility implementation using posix_memalign for
 * older Android versions to support minSdk 26.
 * 
 * This symbol must be available globally so that TensorFlow Lite library
 * can resolve it when it loads.
 */

#include <cstdlib>
#include <cstring>
#include <malloc.h>

#if defined(__ANDROID__) && __ANDROID_API__ < 28

// Provide aligned_alloc for Android API < 28 using posix_memalign
// Use __attribute__((visibility("default"))) to ensure the symbol is exported
__attribute__((visibility("default")))
extern "C" void* aligned_alloc(size_t alignment, size_t size) {
    // aligned_alloc requires that size is a multiple of alignment
    if (size == 0) {
        return nullptr;
    }
    
    if (size % alignment != 0) {
        return nullptr;
    }
    
    // posix_memalign requires alignment to be a power of 2 and a multiple of sizeof(void*)
    if (alignment < sizeof(void*)) {
        alignment = sizeof(void*);
    }
    
    // Ensure alignment is a power of 2
    if ((alignment & (alignment - 1)) != 0) {
        return nullptr;
    }
    
    void* ptr = nullptr;
    int result = posix_memalign(&ptr, alignment, size);
    
    if (result != 0) {
        return nullptr;
    }
    
    return ptr;
}

#endif // __ANDROID__ && __ANDROID_API__ < 28

