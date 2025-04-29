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

#import "metal_compute.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <assert.h>
#define returnError(CODE) do{gLastError = CODE; return CODE;} while(0);

typedef struct {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputeCommandEncoder> encoder;
  id<MTLLibrary> library;
  NSMutableArray<id<MTLBuffer>> *buffers;
  NSMutableArray<id<MTLCommandBuffer>> *commandBuffers;
  NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelineStates;
} MetalState;
static MetalState gMetalState;

id<MTLCommandBuffer> createCommandBuffer(void) {
  id<MTLCommandBuffer> buf = [gMetalState.queue commandBuffer];
  [gMetalState.commandBuffers addObject:buf];
  return buf;
}

void __metalCheck(MetalErrorCode error, const char *file, int line) {
  if (error != 0) {
    printf("[METAL ERROR] at file %s:%d:\n%s\n", file, line, MetalErrorStrings[error]);
    exit(EXIT_FAILURE);
  }
}

MetalErrorCode initMetalState(void) {
  // Get the list of devices (GPUs)
  NSArray <id<MTLDevice>> *devices = MTLCopyAllDevices();
  // Just grabbing the first one. We're assuming Apple Silicon here so single GPU.
  gMetalState.device = [devices firstObject];
  if (!gMetalState.device) { returnError(MetalErrorDeviceNotFound); }
  // Create a command queue with 150 max buffer count. Default is 64 which should be sufficient.
  // This affords the opportunity to save up more command buffers before commiting them. It doesn't seem
  // affect perf at all. The command queue queues the execution of command buffers--the list of operations
  // sent to the GPU to perform.
  gMetalState.queue = [gMetalState.device newCommandQueueWithMaxCommandBufferCount:150];
  if (!gMetalState.queue) { returnError(MetalErrorQueueCreationFailed); }
  // The default library where all of the metal sources go that are compiled by the project.
  // Might be necessary to find the library by path at some point in lieu of xcode.
  gMetalState.library = [gMetalState.device newDefaultLibrary];
  if (!gMetalState.library) { returnError(MetalErrorLibraryCreationFailed); }
  // Create some dynamic arrays to hold some buffers.
  gMetalState.buffers = [[NSMutableArray alloc] initWithCapacity:64];
  gMetalState.commandBuffers = [[NSMutableArray alloc] initWithCapacity:16];
  // Set up a starting command buffer. The command buffer holds a sequence of encoded commands
  // and can be enqueued and committed at some point to execute the work on the GPU.
  id<MTLCommandBuffer> buf = createCommandBuffer();
  if (!buf) { returnError(MetalErrorCommandBufferCreationFailed); }
  // The encoder is what we use to write commands into the command buffer. Once a buffer is commited, the encoder
  // and buffer are discarded and new ones are created.
  gMetalState.encoder = [buf computeCommandEncoder];
  if (!gMetalState.encoder) { returnError(MetalErrorEncoderCreationFailed); }
  // Create dictionary (hash map) of pipeline states. The pipeline state controls the current context
  // for which the commands are being applied to. i.e. we set active our up our compute kernel function
  // and then subsequent calls to encoder apply to said function.
  gMetalState.pipelineStates = [NSMutableDictionary new];
  return MetalSuccess;
}

// Used for tracking buffer and an offset into it.
typedef struct {
  id<MTLBuffer> buffer;
  NSUInteger offset;
} BufferInfo;

// From an address, we need to determine:
// A) Is the address "within" a MTLBuffer we know about?
// B) And if so, what is the offset (in bytes) that addr is pointing to?
// From this we can bind the buffer to the argument table for the kernel.
// This allows us to use one contiguous buffer and offset into it instead
// of many smaller buffers.
BufferInfo findBufferInfo(void *addr) {
  @autoreleasepool {
    BufferInfo info = {NULL, 0};
    for (id<MTLBuffer> buffer in gMetalState.buffers) {
      if (buffer.contents == NULL) {
        continue;
      }
      intptr_t addr_ptr = (intptr_t)addr;
      intptr_t buf_ptr = (intptr_t)buffer.contents;
      if (addr_ptr >= buf_ptr && addr_ptr < buf_ptr + buffer.length) {
        NSUInteger relOffset = addr - buffer.contents;
        info.offset = relOffset; // If so get byte offset
        info.buffer = buffer; // and return buffer info
        return info;
      }
    }
    return info;
  }
}

// Binds a buffer to a given index in the argument table.
MetalErrorCode bindBuffer(void *buffer, NSUInteger index) {
  if (!gMetalState.encoder) { returnError(MetalErrorNotInitialized); }
  BufferInfo info = findBufferInfo(buffer);
  if (!info.buffer) {
    returnError(MetalErrorFailedToLocateBuffer);
  }
  [gMetalState.encoder setBuffer:info.buffer offset:info.offset atIndex:index];
  return MetalSuccess;
}

// Binds a scalar to a given index in the argument table.
MetalErrorCode bindScalar(void *arg, NSUInteger index) {
  if (!gMetalState.encoder) { returnError(MetalErrorNotInitialized); }
  [gMetalState.encoder setBytes:arg length:sizeof(int) atIndex:index];
  return MetalSuccess;
}

// A variable list of num_args MetalArgType and ptr pairs to bind to the current
// arg table.
MetalErrorCode metalBindArgs(int num_args, va_list args) {
  NSUInteger arg_idx = 0;
  for (int i = 0; i < num_args; i++) {
    MetalArgType ty = va_arg(args, MetalArgType);
    switch (ty) {
      case Buffer: {
        void *buf = va_arg(args, void*);
        MetalErrorCode c = bindBuffer(buf, arg_idx++);
        if (c) { returnError(c); }
      } break;
      case Scalar: {
        void *scalar = va_arg(args, void*);
        MetalErrorCode c = bindScalar(scalar, arg_idx++);
        if (c) { returnError(c); }
      } break;
      default:
        assert("Invalid arg type parameters should be type, arg, type, arg...");
    }
  }

  return MetalSuccess;
}

// Sets the pso for a given kernel function name.
MetalErrorCode metalPrepareFunction(const char *name) {
  if (!gMetalState.encoder) { returnError(MetalErrorNotInitialized); }
  id<MTLComputePipelineState> pso =
  gMetalState.pipelineStates[[NSString stringWithCString:name encoding:NSUTF8StringEncoding]];
  if (!pso) { returnError(MetalErrorSetPipelineFailed); }
  [gMetalState.encoder setComputePipelineState:pso];
  return MetalSuccess;
}

// Dispatches the active kernel with the provided grid and block size.
MetalErrorCode metalDispatch(size_t grid_size, size_t block_size) {
  if (!gMetalState.encoder) { returnError(MetalErrorNotInitialized); }
  MTLSize tSize = MTLSizeMake(grid_size, 1, 1);
  MTLSize tgSize = MTLSizeMake(block_size, 1, 1);
  [gMetalState.encoder dispatchThreads:tSize threadsPerThreadgroup:tgSize];
  return MetalSuccess;
}

// Convenience function to set up PSO, bind args and dispatch a kernel in one call.
// The order of the args must match the order in which args are defined in the shader code.
void launchKernel(const char *name, size_t thread_count, size_t block_size,
                  size_t shared_mem_size, int num_args, ...) {
  metalCheck(metalPrepareFunction(name));
  va_list args;
  va_start(args, num_args);
  metalCheck(metalBindArgs(num_args, args));
  va_end(args);

  if (shared_mem_size) { [gMetalState.encoder setThreadgroupMemoryLength:shared_mem_size atIndex:0]; }
  metalDispatch(thread_count, block_size);
}

// Creates a new buffer and sets out to the underlying pointer.
MetalErrorCode metalMalloc(void **out, size_t sizeBytes) {
  if (!gMetalState.buffers) {
    MetalErrorCode code = initMetalState();
    if (code != MetalSuccess) { returnError(code); }
  }
  if (!out) { returnError(MetalErrorMallocInvalidOutPtr); }
  id<MTLBuffer>buffer = [gMetalState.device newBufferWithLength:sizeBytes
                                                        options:MTLResourceStorageModeShared];
  if (!buffer) { returnError(MetalErrorMallocFailed); }
  [gMetalState.buffers addObject:buffer];
  *out = buffer.contents;
  return MetalSuccess;
}

// Finds the corresponding buffer and frees it (removes it from the array).
// Automatic reference counting will release the memory once retain count
// reaches zero. Might have to revisit memory management to get more explicit
// control
MetalErrorCode metalFree(void *buf) {
  // See metalMalloc comment
  BufferInfo info = findBufferInfo(buf);
  if (info.buffer) {
    [gMetalState.buffers removeObject:info.buffer];
  } else {
    returnError(MetalErrorFailedToLocateBuffer);
  }
  return MetalSuccess;
}

// Registers a variable list of kernel function names. last is just a dummy arg.
// The creation of PSOs is expensive(ish) and only has to happen once. They can
// be reused assuming we're not re-generating shaders dynamically.
MetalErrorCode __initKernels(int last, ...) {
  va_list args;
  va_start(args, last);

  if (!gMetalState.device) {
    MetalErrorCode c = initMetalState();
    if (c) { returnError(c); }
  }
  const char *name = NULL;
  while ((name = va_arg(args, const char *))) {
    NSString *nameString = [NSString stringWithCString:name encoding:NSUTF8StringEncoding];
    id<MTLFunction> fn = [gMetalState.library newFunctionWithName:nameString];
    if (!fn) { va_end(args); returnError(MetalErrorInvalidFunctionName); }
    NSError *error = nil;
    id<MTLComputePipelineState> pso =
    [gMetalState.device newComputePipelineStateWithFunction:fn error:&error];
    if (error) { va_end(args); returnError(MetalErrorPipelineStateCreationFailed); }
    gMetalState.pipelineStates[nameString] = pso;
  }
  va_end(args);
  return MetalSuccess;
}

// Commits all pending command buffers and sets lastBuf to the last buffer
// in the array. This can be used to create a barrier until the last kernel
// finishes. Buffers are executed serially in the order in which they are
// committed/enqueued.
MetalErrorCode metalCommitCommandsInternal(id<MTLCommandBuffer>* lastBuf) {
  if (!gMetalState.encoder) { returnError(MetalErrorNotInitialized); }
  [gMetalState.encoder endEncoding];
  gMetalState.encoder = nil;
  for (id<MTLCommandBuffer> buf in gMetalState.commandBuffers) {
    [buf commit];
  }
  if (lastBuf) {
    *lastBuf = [gMetalState.commandBuffers lastObject];
  }
  [gMetalState.commandBuffers removeAllObjects];
  // Refresh the command buffer/encoder for the next batch.
  id<MTLCommandBuffer> buf = createCommandBuffer();
  if (!gMetalState.commandBuffers.count) { returnError(MetalErrorCommandBufferCreationFailed); }
  gMetalState.encoder = [buf computeCommandEncoder];
  if (!gMetalState.encoder) { returnError(MetalErrorEncoderCreationFailed); }
  return MetalSuccess;
}

// Just same as above but with no arg.
MetalErrorCode metalCommitCommands(void) {
  return metalCommitCommandsInternal(NULL);
}

// This will block until all current command buffers are
// enqueued and finish running.
MetalErrorCode metalCommitCommandsAndWait(void) {
  id<MTLCommandBuffer> lastBuf = nil;
  MetalErrorCode code = metalCommitCommandsInternal(&lastBuf);
  if (code != MetalSuccess) {
    returnError(code);
  }
  [lastBuf waitUntilCompleted];
  return MetalSuccess;
}

MetalErrorCode metalGetLastError(void) {
  return gLastError;
}

// Uses MPS to perform matmul.
MetalErrorCode metalSgemmBatched(bool leftT, bool rightT,
                                 size_t leftRows, size_t leftCols,
                                 size_t rightRows, size_t rightCols,
                                 float* A, float* B, float* C,
                                 size_t batchCount, float alpha, float beta) {
  NSUInteger leftSize = leftRows * leftCols * sizeof(float);
  MPSMatrixDescriptor *aDesc =
  [MPSMatrixDescriptor matrixDescriptorWithRows:leftRows columns:leftCols matrices:batchCount
                       rowBytes:leftCols*sizeof(float) matrixBytes:leftSize dataType:MPSDataTypeFloat32];
  NSUInteger rightSize = rightRows * rightCols * sizeof(float);
  MPSMatrixDescriptor *bDesc =
  [MPSMatrixDescriptor matrixDescriptorWithRows:rightRows columns:rightCols matrices:batchCount
                       rowBytes:rightCols*sizeof(float) matrixBytes:rightSize dataType:MPSDataTypeFloat32];
  NSUInteger outRows = leftT ? leftCols : leftRows;
  NSUInteger outCols = rightT ? rightRows : rightCols;
  MPSMatrixDescriptor *cDesc =
  [MPSMatrixDescriptor matrixDescriptorWithRows:outRows columns:outCols matrices:batchCount
                       rowBytes:outCols*sizeof(float) matrixBytes:outRows*outCols*sizeof(float)
                                       dataType:MPSDataTypeFloat32];
  BufferInfo aInfo = findBufferInfo(A);
  BufferInfo bInfo = findBufferInfo(B);
  BufferInfo cInfo = findBufferInfo(C);
  MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:aInfo.buffer offset:aInfo.offset descriptor:aDesc];
  MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bInfo.buffer offset:bInfo.offset descriptor:bDesc];
  MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:cInfo.buffer offset:cInfo.offset descriptor:cDesc];
  NSUInteger interiorCols = leftT ? leftRows : leftCols;
  MPSMatrixMultiplication *matMul =
  [[MPSMatrixMultiplication alloc] initWithDevice:gMetalState.device transposeLeft:leftT transposeRight:rightT
                                       resultRows:outRows resultColumns:outCols interiorColumns:interiorCols
                                            alpha:alpha beta:beta];
  matMul.batchStart = 0;
  matMul.batchSize = batchCount;
  // MPS needs its own fresh buffer to encode into...
  [gMetalState.encoder endEncoding];
  id<MTLCommandBuffer> buf =  createCommandBuffer();
  // I wonder how much faster this would be if it didn't require a fresh cmd buffer?
  // Probably not a huge difference for large GPU bound workloads like this.
  [matMul encodeToCommandBuffer:buf leftMatrix:matrixA rightMatrix:matrixB resultMatrix:matrixC];
  // After encoding cycle the buffer and create a new command encoder for it.
  buf = createCommandBuffer();
  gMetalState.encoder = [buf computeCommandEncoder];
  return MetalSuccess;
}

/*

   Description: Clears (zeroes out) a specified portion of a metal buffer.

*/
MetalErrorCode metalClearBuffer(void* buffer_base_ptr, size_t total_size_bytes)
{
    if (!gMetalState.device) {
        returnError(MetalErrorFailedToLocateBuffer);
    }

    if (total_size_bytes == 0)
        return MetalSuccess;

    /*
         Find the actual metal Buffer Object (MTLBuffer) from the user-provided (buffer_base_ptr)
    */

    BufferInfo info = findBufferInfo(buffer_base_ptr);

    if (!info.buffer)
    {
        returnError(MetalErrorFailedToLocateBuffer);
    }

    size_t buffer_actual_length = info.buffer.length;

    if (total_size_bytes > buffer_actual_length){
        returnError(MetalErrorInvalidArgument);
    }

    /*
        If there are any previous compute commands --> we want to send them off to the GPU first
    */
    if (gMetalState.encoder)
    {
        MetalErrorCode commitCode = metalCommitCommands();
        if (commitCode != MetalSuccess)
            returnError(commitCode);

        [gMetalState.encoder endEncoding];
         gMetalState.encoder = nil;
    } else {
        if (!gMetalState.commandBuffers || gMetalState.commandBuffers.count == 0) {
            //create a new command buffer
            id<MTLCommandBuffer> first_ever_cmd_buffer = createCommandBuffer();
            if (!first_ever_cmd_buffer)
                returnError(MetalErrorCommandBufferCreationFailed);
        }
    }

    //get the command buffer and we will add Blit command to
    //blit --> Block image transfer (to fill block of texture or buffer memory)
    // the active commadn buffer is always the last one
    id<MTLCommandBuffer> currentCmdBuffer = [gMetalState.commandBuffers lastObject];
    if (!currentCmdBuffer){
       returnError(MetalErrorCommandBufferCreationFailed);
    }

    @autoreleasepool {
         id<MTLBlitCommandEncoder> blitEncoder = [currentCmdBuffer blitCommandEncoder];
         if (!blitEncoder){
             returnError(MetalErrorCommandBufferCreationFailed);
         }

         //fill user-specified range of the buffer with 0
         [blitEncoder fillBuffer:info.buffer
                        range:NSMakeRange(0, total_size_bytes)
                        value:0];

         [blitEncoder endEncoding];
    }

    //commit the command buffer
    [currentCmdBuffer commit];

    //remove the committed buffer
    [gMetalState.commandBuffers removeObject:currentCmdBuffer];

    //Create a new command buffer and comptue encoder for subsequent compute tasks
    id<MTLCommandBuffer> nextCmdBuffer = createCommandBuffer();
    if (!nextCmdBuffer) {
        returnError(MetalErrorCommandBufferCreationFailed);
    }

    gMetalState.encoder = [nextCmdBuffer computeCommandEncoder];
    if (!gMetalState.encoder) {
        returnError(MetalErrorCommandBufferCreationFailed);
    }

    return MetalSuccess;
}