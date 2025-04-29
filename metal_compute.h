/*
 MIT License

 Copyright (c) 2024 James Thompson

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 */

#ifndef metal_h
#define metal_h

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Error codes for various metal related failures.
typedef enum MetalErrorCode {
  MetalSuccess,
  MetalErrorMallocInvalidOutPtr,
  MetalErrorMallocFailed,
  MetalErrorDeviceNotFound,
  MetalErrorQueueCreationFailed,
  MetalErrorCommandBufferCreationFailed,
  MetalErrorEncoderCreationFailed,
  MetalErrorLibraryCreationFailed,
  MetalErrorPipelineStateCreationFailed,
  MetalErrorFailedToLocateBuffer,
  MetalErrorInvalidFunctionName,
  MetalErrorMPSMatrixCreationFailed,
  MetalErrorMPSMatrixMultiplicationCreationFailed,
  MetalErrorSetPipelineFailed,
  MetalErrorNotInitialized,
  MetalErrorBindArgFailed,
  MetalErrorInvalidArgument,
  MetalErrorCount // Should always be last
} MetalErrorCode;

// Tracks the last error code that was encountered.
static MetalErrorCode gLastError;

// A list of strings that map to the above error codes.
static const char *MetalErrorStrings[MetalErrorCount] = {
    "Success",
    "Invalid out ptr.",
    "Buffer creation failed.",
    "MTLDevice not found.",
    "Command Queue creation failed.",
    "Command Buffer creation failed.",
    "Encoder creation failed.",
    "Library creation failed.",
    "Pipeline state creation failed.",
    "Failed to locate buffer.",
    "Invalid function name.",
    "Failed to create MPSMatrix.",
    "Failed to create MPSMatrixMultiplication.",
    "Failed to set pipeline state on command encoder.",
    "Metal state is not yet initialized.",
    "Invalid argument provided.",
    "Failed to bind an argument to kernel function."};

void __metalCheck(MetalErrorCode error, const char *file, int line);
// Use this macro to wrap around an API that returns a MetalErrorCode to log
// source location, error string and exit.
#define metalCheck(err) (__metalCheck(err, __FILE__, __LINE__))

// The type of argument used in the launchKernel call.
typedef enum {
  Buffer,
  Scalar,
} MetalArgType;

/// Sets up the appropriate metal state and dispatches a given kernel.
/// - Parameters:
///   - name: The name of the kernel function as it was passed to initKernels
///   and as it appears in the kernel function.
///   - thread_count: The number of threads to dispatch. This does not need to
///   be a multiple of block size.
///   - block_size: The block size (threadgroup size) should be a multiple of 32
///   (i.e. thread execution width/simd size)
///   - shared_size: The shared memory size to allocate. Set to 0 for no shared
///   memory.
///   - num_args: The number of args to follow. The arguments must be passed in
///   the same order in which they are defined in the kernel in the shader. The
///   variadic list of arguments should be in the form of MetalArgType, void,
///   i.e. Buffer, myVoidptr, Scalar, &myScalar). The num_args param counts the
///   actual arguments not the argument types so the previous example num_args
///   would be 2. Buffer indicates a buffer allocated via metalMalloc and Scalar
///   should be used for small, scalar types or possibly statically sized
///   structs, i.e. data with sizes known at compile time. Anything large or
///   dynamic should just get passed in via a buffer.
void launchKernel(const char *name, size_t thread_count, size_t block_size,
                  size_t shared_size, int num_args, ...);

/// Allocates a buffer backed by Metal. As this repo only supports Unified
/// Memory devices i.e. M-series this memory is accessible from CPU and GPU
/// without copying. Any address that is valid within the range of  out to out +
/// sizeBytes - 1 is valid and may be passed to launchKernel or any other
/// function in  metal_compute.h that takes a \c void* parameter.
/// - Parameters:
///  - out: The pointer to the resulting buffer
///  - sizeBytes: The size of buffer to allocate.
MetalErrorCode metalMalloc(void **out, size_t sizeBytes);

/// Frees the backing buffer for the pointer
/// @param buf A pointer to a buffer allocated via metalMalloc. Can be any valid
/// pointer within the span of the buffer.
MetalErrorCode metalFree(void *buf);
MetalErrorCode __initKernels(int last, ...);
// A null terminated list of kernel names as they are declared within the
// compute shader source.
#define initKernels(...) __initKernels(0, __VA_ARGS__);
// Commits any pending commands. No actual computation happens until commands
// are commited. This version of this API will not wait for results to be
// produced. Call this periodically after launching kernels.
MetalErrorCode metalCommitCommands(void);
// Commits any pending commands and waits until the last command finishes to
// return control.
MetalErrorCode metalCommitCommandsAndWait(void);
// Gets the last error that was produced.
MetalErrorCode metalGetLastError(void);

/// Performs a matmul operation on two matrices.
/// - Parameters:
///   - leftT: Whether to transpose A before performing the operation.
///   - rightT: Whether ot transpose B before performing the operation.
///   - leftRows: The number of rows in A before transposing
///   - leftCols: The number of columns in A before transposing
///   - rightRows: The number of rows in B before transposing.
///   - rightCols: The number of columns in B before transposing.
///   - A: The buffer for A
///   - B: The buffer for B
///   - C: The result buffer. Its size should be consistent with the preceeding
///   parameters.
///   - batchCount: The number of batches to process. Does not support
///   broadcasting.
///   - alpha: The alpha parameter. Scales A and B.
///   - beta: The beta parameter. Scales C.
MetalErrorCode metalSgemmBatched(bool leftT, bool rightT, size_t leftRows,
                                 size_t leftCols, size_t rightRows,
                                 size_t rightCols, float *A, float *B, float *C,
                                 size_t batchCount, float alpha, float beta);
MetalErrorCode metalClearBuffer(void* buffer_base_ptr, size_t total_size_bytes);
#endif /* metal_h */
