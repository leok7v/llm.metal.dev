/*
 MIT License

 Copyright (c) 2024 Andrej Karpathy
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

#include "llm_cpu.h"
#include "metal_compute.h"
#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <float.h>

#define CHECK_TENSORS 0

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

//runtime scratch buffers for grad-clipping

//actual L2-norm
static float* grad_norm_sqrt_result_buffer = NULL;
/*
 * This will hold the sum of squares of all parameter-gradients produced in teh current step
 *  The reduction kernel sum_squares_kernel needs a place in global memory where every thread-group
 *  can automatically add its partial result.
 */
static float* grad_norm_sum_sq_buffer = NULL;
/*
 * The multiplicative factor that rescales every gradient when global-norm clipping is enabled.
 */
static float* clip_scale_factor_buffer = NULL;

void logFloats(float *a, size_t len) {
  for (int i = 0; i < len; i++) {
    if (i != len - 1) {
      // Only log to the hundredth place
      printf("%.03f, ", a[i]);
    } else {
      printf("%.03f\n", a[i]);
    }
  }
}

void logStridedFloats(float *a, size_t offset, size_t len, size_t stride) {
  for (size_t i = offset; i < offset + len; i += stride) {
    if (i != len - 1) {
      printf("%.03f, ", a[i]);
    } else {
      printf("%.03f\n", a[i]);
    }
  }
}

void logInts(int *a, size_t len) {
  for (int i = 0; i < len; i++) {
    if (i != len - 1) {
      printf("%d, ", a[i]);
    } else {
      printf("%d\n", a[i]);
    }
  }
}

// poor man's tensor checker
int check_tensor(float *a, float *b, size_t n, char *label) {
  int print_upto = 5;
  int ok = 1;
  printf("%s\n", label);
  for (int i = 0; i < n; i++) {
    if (fabsf(a[i] - b[i]) <= 1e-2 ||
        (a[i] == -INFINITY && b[i] == -INFINITY)) {
      if (i < print_upto) {
        printf("OK ");
      }
    } else {
      if (i < print_upto) {
        printf("NOT OK ");
      }
      ok = 0;
    }
    if (i < print_upto) {
      printf("%f %f\n", a[i], b[i]);
    }
  }
  // print the final result
  if (ok) {
    printf("TENSOR OK\n");
  } else {
    printf("TENSOR NOT OK\n");
  }
  return ok;
}

void tensor_stats(float *a, size_t n) {
  size_t count = 0;
  size_t infCount = 0;
  size_t nonZeroCount = 0;
  size_t firstZero = 0;
  float minVal = 100000.0;
  float maxVal = 0.0f;
  double meanVal = 0.0f;
  for (size_t i = 0; i < n; i++) {
    if (isnan(a[i])) {
      count++;
    }
    if (isinf(a[i])) {
      infCount++;
    }
    if (!firstZero && a[i] == 0.0) {
      firstZero = i;
    }
    if (a[i] != 0.0) {
      nonZeroCount++;
    }
    minVal = min(minVal, a[i]);
    maxVal = max(maxVal, a[i]);
    meanVal += a[i] / n;
  }
  printf("NaNs: %zu \n"
         "Non-zero count: %zu\n"
         "First zero idx: %zu\n"
         "Infinities: %zu\n"
         "Min: %f\n"
         "Max: %f\n"
         "Mean: %f\n\n",
         count, nonZeroCount, firstZero, infCount, minVal, maxVal, meanVal);
}

// ----------------------------------------------------------------------------
// kernel launchers
void encoder_forward_kernel2(int grid_size, int block_size, float *out,
                             int *inp, float *wte, float *wpe, int B, int T,
                             int C) {
  launchKernel(__FUNCTION__, grid_size, block_size, 0, 7, Buffer, out, Buffer,
               inp, Buffer, wte, Buffer, wpe, Scalar, &B, Scalar, &T, Scalar,
               &C);
}

void mean_kernel(int grid_size, int block_size, int shared_size, float *mean,
                 float *inp, int N, int C) {
  launchKernel(__FUNCTION__, grid_size, block_size, shared_size, 4, Buffer,
               mean, Buffer, inp, Scalar, &N, Scalar, &C);
}

void rstd_kernel(int grid_size, int block_size, int shared_size, float *rstd,
                 float *inp, float *mean, int N, int C) {
  launchKernel(__FUNCTION__, grid_size, block_size, shared_size, 5, Buffer,
               rstd, Buffer, inp, Buffer, mean, Scalar, &N, Scalar, &C);
}

void normalization_kernel(int grid_size, int block_size, float *out, float *inp,
                          float *mean, float *rstd, float *weight, float *bias,
                          int B, int T, int C) {
  launchKernel(__FUNCTION__, grid_size, block_size, 0, 9, Buffer, out, Buffer,
               inp, Buffer, mean, Buffer, rstd, Buffer, weight, Buffer, bias,
               Scalar, &B, Scalar, &T, Scalar, &C);
}

void add_bias_kernel(int grid_size, int block_size, float *out, float *bias,
                     int OC) {
  launchKernel(__FUNCTION__, grid_size, block_size, 0, 3, Buffer, out, Buffer,
               bias, Scalar, &OC);
}

void permute_kernel(int grid_size, int block_size, float *q, float *k, float *v,
                    float *inp, int B, int N, int NH, int d) {
  launchKernel(__FUNCTION__, grid_size, block_size, 0, 8, Buffer, q, Buffer, k,
               Buffer, v, Buffer, inp, Scalar, &B, Scalar, &N, Scalar, &NH,
               Scalar, &d);
}

void scale_kernel(int grid_size, int block_size, float *preatt, float scale,
                  int B, int NH, int T) {
  launchKernel(__FUNCTION__, grid_size, block_size, 0, 5, Buffer, preatt,
               Scalar, &scale, Scalar, &B, Scalar, &NH, Scalar, &T);
}

void softmax_forward_kernel1(int grid_size, int block_size, float *att,
                             float *preatt, int N, int C) {
  launchKernel(__FUNCTION__, grid_size, block_size, 0, 4, Buffer, att, Buffer,
               preatt, Scalar, &N, Scalar, &C);
}

void softmax_forward_kernel4(int grid_size, int block_size, size_t shared_size,
                             float *att, float *preatt, int N, int C) {
  launchKernel(__FUNCTION__, grid_size, block_size, shared_size, 4, Buffer, att,
               Buffer, preatt, Scalar, &N, Scalar, &C);
}

void unpermute_kernel(int grid_size, int block_size, float *inp, float *out,
                      int B, int T, int NH, int HS) {
  launchKernel(__FUNCTION__, grid_size, block_size, 0, 6, Buffer, inp, Buffer,
               out, Scalar, &B, Scalar, &T, Scalar, &NH, Scalar, &HS);
}

void residual_forward_kernel(int grid_size, int block_size, float *out,
                             float *inp1, float *inp2) {
  launchKernel(__FUNCTION__, grid_size, block_size, 0, 3, Buffer, out, Buffer,
               inp1, Buffer, inp2);
}

void gelu_kernel(int grid_size, int block_size, float *out, float *inp) {
  launchKernel(__FUNCTION__, grid_size, block_size, 0, 2, Buffer, out, Buffer,
               inp);
}

void crossentropy_forward_kernel1(int grid_size, int block_size, float *losses,
                                  float *probs, int *targets, int B, int T,
                                  int V) {
  launchKernel(__FUNCTION__, grid_size, block_size, 0, 5, Buffer, losses,
               Buffer, probs, Buffer, targets, Scalar, &T, Scalar, &V);
}

//backward pass

//perform backward pass for matrix muiltiplication Y = X * W^T +  B
//here we compute gradients for X, W and Bias
void matmul_backward(float* dinp, float* dweight, float* dbias,
                      float* dout, float* inp, float* weight,
                        int B, int T, int C, int OC)
{
  /*
   *  Dimensions
   *    dout (dY) --> (B*T, OC)
   *    inp (X) --> (B*T, C)
   *    weight (W) --> (OC, C)
   *    dinp (dX) : (B * T, C)
   *    dweight (dW) --> (OC, C)
   *    dbias (dB) --> OC
   */

  //calculate dX = dY * W
  //basically C = A * B where (C = dinp, A = dout, B = weight)
  const float alpha1 = 1.0f;
  const float beta1 = 1.0f;

  metalCheck(metalSgemmBatched(false, false, B*T, OC, OC, C, dout, weight, dinp, 1, alpha1, beta1));

  //now we calculate dW = dY^T * X
  //so basically C = A * B where (C = dweight, A = dout, B = inp)
  metalCheck(metalSgemmBatched(true, false, B * T, OC, B * T, C,
    dout, inp, dweight, 1, alpha1, beta1));

  if (dbias != NULL)
  {
    int grid_dim = OC; //set up one thread group per output channel
    int block_dim = 256;

    size_t shared_mem_size = block_dim * sizeof(float);
    //launch MSL kernel
    launchKernel("matmul_backward_bias_kernel",
                grid_dim * block_dim, //total threads = OC * block_dim
                block_dim, //threads per group
                shared_mem_size,
                5,
                Buffer, dbias,
                Buffer, dout,
                Buffer, dout,
                Scalar, &B,
                Scalar, &T,
                Scalar, &OC);
  }
}

void layernorm_backward(float* dinp, float* dweight, float* dbias,
            float* dout, float* inp, float* weight, float* mean, float* rstd,
            int B, int T, int C)
{
    int grid_size = B * T; //one threadgroup per (b, t) position
    int block_size = C;

    const int max_block_size = 512;// we can change this to 1024 or more
    if (block_size > max_block_size)
        block_size = max_block_size;

    //setup shared memory
    //2 floats per thread in shared memory
    size_t shared_mem_size = block_size * 2 * sizeof(float);

    //launch kernel
    // total threads = grid_size * block_size
    launchKernel("layernorm_backward_kernel", grid_size * block_size, block_size, shared_mem_size,
                  11,
                        Buffer, dinp,
                        Buffer, dweight,
                        Buffer, dbias,
                        Buffer, dout,
                        Buffer, inp,
                        Buffer, weight,
                        Buffer, mean,
                        Buffer, rstd,
                        Scalar, &B,
                        Scalar, &T,
                        Scalar, &C);
}


void gelu_backward(float* dinp, float* inp, float* dout, int N)
{
  // N is the total number of elements ( B * T * 4C for FC Expansion)
  int grid_size = N;
  int block_size = N;

  launchKernel("gelu_backward_kernel", grid_size, block_size, 0, 4,
    Buffer, dinp, Buffer, inp,  Buffer, dout, Scalar, &N);
}

void encoder_backward(float* dwte, float* dwpe, float* dout, int* inp, int B, int T, int C)
{
  int grid_size = C * B * T;
  if (grid_size == 0) return;

  int block_size = 256;

  block_size = (grid_size < block_size) ? grid_size : block_size;
  block_size = (block_size > 0) ? block_size : 1;

  launchKernel("encoder_backward_kernel", grid_size, block_size, 0,
    7,
    Buffer, dwte,
    Buffer, dwpe,
    Buffer, dout,
    Buffer, inp,
    Scalar, &B,
    Scalar, &T,
    Scalar, &C);
}

void residual_backward(float* dinp1, float* dinp2, float* dout, int N)
{
  int grid_size = N;
  int block_size = 256;
  block_size = grid_size < block_size ? grid_size : block_size;
  block_size = block_size > 0 ? block_size : 1;

  launchKernel("residual_backward_kernel", grid_size, block_size, 0,
              4,
              Buffer, dinp1,
              Buffer, dinp2,
              Buffer, dout,
              Scalar, &N);
}

void initialize_dlosses(float* dlosses_buffer, int B, int T)
{
  int N = B * T;
  float val = 1.0f / (float)(N);

  int grid_size = N;
  int block_size = 256;
  block_size = grid_size < block_size ? grid_size : block_size;

  launchKernel("initialize_dlosses_kernel", grid_size, block_size, 0,
              3,
              Buffer, dlosses_buffer,
              Scalar, &val,
              Scalar, &N);
}

void encoder_forward(float *out, int *inp, float *wte, float *wpe, int B, int T,
                     int C) {
  const int N = B * T * C;
  const int block_size = 512;
  const int grid_size = N;
  encoder_forward_kernel2(grid_size, block_size, out, inp, wte, wpe, B, T, C);
#if CHECK_TENSORS
  metalCommitCommandsAndWait();
  size_t gSz = B * T * C;
  float *g = malloc(sizeof(float) * gSz);
  memcpy(g, out, sizeof(float) * gSz);
  cpu_encoder_forward(out, inp, wte, wpe, B, T, C);
  check_tensor(out, g, gSz, "encoder_forward");
#endif
}

void layernorm_forward(float *out, float *mean, float *rstd, float *inp,
                       float *weight, float *bias, int B, int T, int C) {
  int N = B * T;
  const int block_size = 512;
  // in mean and rstd, threads cooperate within blocks via reductions
  mean_kernel(B * T * C, block_size, block_size * sizeof(float), mean, inp, N,
              C);
  rstd_kernel(B * T * C, block_size, block_size * sizeof(float), rstd, inp,
              mean, N, C);

  // in the normalization, everything just gets flattened out
  const int block_size2 = 128;
  const int grid_size = B * T * C;
  normalization_kernel(grid_size, block_size2, out, inp, mean, rstd, weight,
                       bias, B, T, C);
#if CHECK_TENSORS

  metalCommitCommandsAndWait();
  size_t gSz = B * T * C;
  size_t fSz = sizeof(float);
  float *g = malloc(fSz * gSz);
  memcpy(g, out, fSz * gSz);
  cpu_layernorm_forward(out, mean, rstd, inp, weight, bias, B, T, C);
  check_tensor(out, g, gSz, "layer_norm1");
#endif
}

// kernel 1 is the most naive matmul kernel
void matmul_forward(float *out, float *inp, float *weight, float *bias, int B,
                    int T, int C, int OC) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  // inp is (B,T,C), weight is (OC, C), bias is (OC)
  // out will be (B,T,OC)
  // [m k] by [k n]
  metalCheck(metalSgemmBatched(false, true, B * T, C, OC, C, inp, weight, out,
                               1, alpha, beta));

  // and now we still have to add the bias... (ew)
  if (bias != NULL) {
    int block_size = 128;
    int grid_size = OC * B * T;
    add_bias_kernel(grid_size, block_size, out, bias, OC);
  }
#if CHECK_TENSORS
  metalCommitCommandsAndWait();
  size_t gSz = B * T * OC;
  size_t fSz = sizeof(float);
  float *g = malloc(fSz * gSz);
  memcpy(g, out, fSz * gSz);
  cpu_matmul_forward(out, inp, weight, bias, B, T, C, OC);
  check_tensor(out, g, gSz, "matmul");
#endif
}

void attention_forward(float *out, float *vaccum, float *qkvr, float *preatt,
                       float *att, float *inp, int B, int T, int C, int NH) {
  const int block_size = 128;
  int HS = C / NH; // head size
  // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
  float *q, *k, *v;
  q = qkvr + 0 * B * T * C;
  k = qkvr + 1 * B * T * C;
  v = qkvr + 2 * B * T * C;
  int total_threads = B * NH * T * HS;
  permute_kernel(total_threads, block_size, q, k, v, inp, B, T, NH, HS);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  // (B, NH, T, HS) • (B, NH, T, HS)^T -> (B, NH, T, T)
  metalCheck(metalSgemmBatched(false, true, T, HS, T, HS, q, k, preatt, B * NH,
                               alpha, beta));
  // multiply all elements of preatt elementwise by scale
  float scale = 1.0 / sqrtf(HS);
  total_threads = B * NH * T * T;
  scale_kernel(total_threads, block_size, preatt, scale, B, NH, T);
  int softmax_block_size = 128;
  int grid_size = B * NH * T;
  // TODO: Implement better softmax
  //  size_t shared_mem_size = 2 * softmax_block_size / 32 * sizeof(float);
  //  softmax_forward_kernel4(grid_size, softmax_block_size,
  //                          shared_mem_size, att, preatt,
  //                          B * NH * T * T, T);
  // softmax. preatt is (B, NH, T, T) but we view it as (B * NH * T, T) and use
  // the softmax kernel
  softmax_forward_kernel1(grid_size, softmax_block_size, att, preatt,
                          B * NH * T, T);
  // v^T • att^T or (B, NH, T, HS)^T • (B, NH, T, T)^T -> (B, NH, HS, T)
  metalCheck(metalSgemmBatched(true, true, T, HS, T, T, v, att, vaccum, B * NH,
                               alpha, beta));
  // re-assemble all head outputs side by side
  grid_size = B * NH * HS * T;
  // permute B, NH, HS, T ->  B, T, NH, HS
  unpermute_kernel(grid_size, block_size, vaccum, out, B, T, NH, HS);
#if CHECK_TENSORS
  metalCommitCommandsAndWait();
  size_t gSz = B * T * NH * HS;
  size_t fSz = sizeof(float);
  float *g = malloc(fSz * gSz);
  memcpy(g, out, fSz * gSz);
  cpu_attention_forward(out, preatt, att, inp, B, T, C, NH);
  check_tensor(out, g, gSz, "attention");
#endif
}

void residual_forward(float *out, float *inp1, float *inp2, int N) {
  const int block_size = 128;
  residual_forward_kernel(N, block_size, out, inp1, inp2);
#if CHECK_TENSORS
  metalCommitCommandsAndWait();
  size_t gSz = N;
  size_t fSz = sizeof(float);
  float *g = malloc(fSz * gSz);
  memcpy(g, out, fSz * gSz);
  cpu_residual_forward(out, inp1, inp2, N);
  check_tensor(out, g, gSz, "residual");
#endif
}

void gelu_forward(float *out, float *inp, int N) {
  const int block_size = 256;
  gelu_kernel(N, block_size, out, inp);
#if CHECK_TENSORS
  metalCommitCommandsAndWait();
  size_t gSz = N;
  size_t fSz = sizeof(float);
  float *g = malloc(fSz * gSz);
  memcpy(g, out, fSz * gSz);
  cpu_gelu_forward(out, inp, N);
  check_tensor(out, g, gSz, "gelu");
#endif
}

void softmax_forward(float *out, float *inp, int B, int T, int V) {
  const int block_size = 128;
  int grid_size = B * T;
  //  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  softmax_forward_kernel1(grid_size, block_size, out, inp, B * T, V);
#if CHECK_TENSORS
  metalCommitCommandsAndWait();
  size_t gSz = B * T;
  size_t fSz = sizeof(float);
  float *g = malloc(fSz * gSz);
  memcpy(g, out, fSz * gSz);
  cpu_softmax_forward(out, inp, B, T, V);
  check_tensor(out, g, gSz, "softmax");
#endif
}

void crossentropy_forward(float *losses, float *probs, int *targets, int B,
                          int T, int V) {
  const int block_size = 256;
  crossentropy_forward_kernel1(B * T, block_size, losses, probs, targets, B, T,
                               V);
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

// the parameters of the model
#define NUM_PARAMETER_TENSORS 16
typedef struct {
  float *wte;      // (V, C)
  float *wpe;      // (maxT, C)
  float *ln1w;     // (L, C)
  float *ln1b;     // (L, C)
  float *qkvw;     // (L, 3*C, C)
  float *qkvb;     // (L, 3*C)
  float *attprojw; // (L, C, C)
  float *attprojb; // (L, C)
  float *ln2w;     // (L, C)
  float *ln2b;     // (L, C)
  float *fcw;      // (L, 4*C, C)
  float *fcb;      // (L, 4*C)
  float *fcprojw;  // (L, C, 4*C)
  float *fcprojb;  // (L, C)
  float *lnfw;     // (C)
  float *lnfb;     // (C)
} ParameterTensors;

// allocate memory for the parameters and point the individual tensors to the
// right places
float *malloc_and_point_parameters(ParameterTensors *params,
                                   size_t *param_sizes, int on_device) {
  // on_device: 0 = CPU, 1 = GPU
  // calculate the number of parameters
  size_t num_parameters = 0;
  for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    num_parameters += param_sizes[i];
  }
  // malloc all parameters all at once on the device
  float *params_memory = NULL;
  if (on_device) {
    metalCheck(
        metalMalloc((void **)&params_memory, num_parameters * sizeof(float)));
  } else {
    params_memory = (float *)malloc(num_parameters * sizeof(float));
  }
  // assign all the tensors their place in the array
  float **ptrs[] = {
      &params->wte,     &params->wpe,     &params->ln1w,     &params->ln1b,
      &params->qkvw,    &params->qkvb,    &params->attprojw, &params->attprojb,
      &params->ln2w,    &params->ln2b,    &params->fcw,      &params->fcb,
      &params->fcprojw, &params->fcprojb, &params->lnfw,     &params->lnfb};
  float *params_memory_iterator = params_memory;
  for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    *(ptrs[i]) = params_memory_iterator;
    params_memory_iterator += param_sizes[i];
  }
  return params_memory;
}

#define NUM_ACTIVATION_TENSORS 27
typedef struct {
  float *logits;    // (B, T, V)
  float *encoded;   // (B, T, C)
  float *ln1;       // (L, B, T, C)
  float *ln1_mean;  // (L, B, T)
  float *ln1_rstd;  // (L, B, T)
  float *qkv;       // (L, B, T, 3*C)
  float *atty;      // (L, B, T, C)
  float *preatt;    // (L, B, NH, T, T)
  float *att;       // (L, B, NH, T, T)
  float *attproj;   // (L, B, T, C)
  float *residual2; // (L, B, T, C)
  float *ln2;       // (L, B, T, C)
  float *ln2_mean;  // (L, B, T)
  float *ln2_rstd;  // (L, B, T)
  float *fch;       // (L, B, T, 4*C)
  float *fch_gelu;  // (L, B, T, 4*C)
  float *fcproj;    // (L, B, T, C)
  float *residual3; // (L, B, T, C)
  float *lnf;       // (B, T, C)
  float *lnf_mean;  // (B, T)
  float *lnf_rstd;  // (B, T)
  float *probs;     // (B, T, V)
  float *losses;    // (B, T)
  // adding these two compared to the CPU .c code, needed for attention kernel
  // as buffers
  float *qkvr;    // (L, B, T, 3*C)
  float *v_accum; // (L, B, T, C)
  float* dpreatt;   // (L, B, NH, T, T) - Intermediate gradient buffer
  float* datt;      // (L, B, NH, T, T) - Intermediate gradient buffer
} ActivationTensors;

float *malloc_and_point_activations(ActivationTensors *acts,
                                    size_t *act_sizes) {
  size_t num_activations = 0;
  for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
    num_activations += act_sizes[i];
  }
  float *acts_memory = NULL;
  metalCheck(
      metalMalloc((void **)&acts_memory, num_activations * sizeof(float)));
  float **ptrs[] = {
      &acts->logits,   &acts->encoded,   &acts->ln1,       &acts->ln1_mean,
      &acts->ln1_rstd, &acts->qkv,       &acts->atty,      &acts->preatt,
      &acts->att,      &acts->attproj,   &acts->residual2, &acts->ln2,
      &acts->ln2_mean, &acts->ln2_rstd,  &acts->fch,       &acts->fch_gelu,
      &acts->fcproj,   &acts->residual3, &acts->lnf,       &acts->lnf_mean,
      &acts->lnf_rstd, &acts->probs,     &acts->losses,    &acts->qkvr,
      &acts->v_accum};
  float *acts_memory_iterator = acts_memory;
  for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
    *(ptrs[i]) = acts_memory_iterator;
    acts_memory_iterator += act_sizes[i];
  }
  return acts_memory;
}

typedef struct {
  int max_seq_len; // max sequence length, e.g. 1024
  int vocab_size;  // vocab size, e.g. 50257
  int num_layers;  // number of layers, e.g. 12
  int num_heads;   // number of heads in attention, e.g. 12
  int channels;    // number of channels, e.g. 768
} GPT2Config;

typedef struct {
  GPT2Config config;
  // the weights of the model, and their sizes
  ParameterTensors params;
  size_t param_sizes[NUM_PARAMETER_TENSORS];
  float *params_memory;
  size_t num_parameters;
  // gradients of the weights
  ParameterTensors grads;
  float *grads_memory;
  // buffers for the AdamW optimizer
  float *m_memory;
  float *v_memory;
  // the activations of the model, and their sizes
  ActivationTensors acts;
  size_t act_sizes[NUM_ACTIVATION_TENSORS];
  float *acts_memory;
  size_t num_activations;
  // gradients of the activations
  ActivationTensors grads_acts;
  float *grads_acts_memory;
  // other run state configuration
  int batch_size;  // the batch size (B) of current forward pass
  int seq_len;     // the sequence length (T) of current forward pass
  int *inputs;     // the input tokens for the current forward pass
  int *targets;    // the target tokens for the current forward pass
  float mean_loss; // after a forward pass with targets, will be populated with
                   // the mean loss
} GPT2;

void crossentropy_softmax_backward(GPT2 * model)
{
    int B = model->batch_size;
    int T = model->seq_len;
    int V = model->config.vocab_size;

    float* dlogits = model->grads_acts.logits;
    float* dlosses = model->grads_acts.losses;
    float* probs = model->acts.probs;
    int* targets = model->targets;

    int grid_size = B * T * V;
    int block_size = 256;

    launchKernel("crossentropy_softmax_backward_kernel", grid_size, block_size, 0,
                7,
                Buffer, dlogits,
                Buffer, dlosses,
                Buffer, probs,
                Buffer, targets,
                Scalar, &B,
                Scalar, &T,
                Scalar &V);
}

void gpt2_build_from_checkpoint(GPT2 *model, char *checkpoint_path) {

  // read in model from a checkpoint file
  FILE *model_file = fopen(checkpoint_path, "rb");
  if (model_file == NULL) {
    printf("Error opening model file\n");
    exit(1);
  }
  int model_header[256];
  fread(model_header, sizeof(int), 256, model_file);
  if (model_header[0] != 20240326) {
    printf("Bad magic model file");
    exit(1);
  }
  if (model_header[1] != 1) {
    printf("Bad version in model file");
    exit(1);
  }

  // read in hyperparameters
  int maxT, V, L, NH, C;
  model->config.max_seq_len = maxT = model_header[2];
  model->config.vocab_size = V = model_header[3];
  model->config.num_layers = L = model_header[4];
  model->config.num_heads = NH = model_header[5];
  model->config.channels = C = model_header[6];
  printf("[GPT-2]\n");
  printf("max_seq_len: %d\n", maxT);
  printf("vocab_size: %d\n", V);
  printf("num_layers: %d\n", L);
  printf("num_heads: %d\n", NH);
  printf("channels: %d\n", C);

  // allocate space for all the parameters and read them in
  model->param_sizes[0] = V * C;
  model->param_sizes[1] = maxT * C;
  model->param_sizes[2] = L * C;
  model->param_sizes[3] = L * C;
  model->param_sizes[4] = L * (3 * C) * C;
  model->param_sizes[5] = L * (3 * C);
  model->param_sizes[6] = L * C * C;
  model->param_sizes[7] = L * C;
  model->param_sizes[8] = L * C;
  model->param_sizes[9] = L * C;
  model->param_sizes[10] = L * (4 * C) * C;
  model->param_sizes[11] = L * (4 * C);
  model->param_sizes[12] = L * C * (4 * C);
  model->param_sizes[13] = L * C;
  model->param_sizes[14] = C;
  model->param_sizes[15] = C;

  // cound the number of paramaters
  size_t num_parameters = 0;
  for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    num_parameters += model->param_sizes[i];
  }
  printf("num_parameters: %zu\n", num_parameters);
  model->num_parameters = num_parameters;

  // create memory for model parameters on the device
  model->params_memory =
      malloc_and_point_parameters(&model->params, model->param_sizes, 1);

  // read in all the parameters from file and copy them to device
  //  float* params_memory_cpu = (float*)malloc(num_parameters * sizeof(float));
  //  metalCheck(metalMalloc((void**)&model->params_memory, num_parameters *
  //  sizeof(float)));
  fread(model->params_memory, sizeof(float), num_parameters, model_file);
  //  metalCheck(cudaMemcpy(model->params_memory, params_memory_cpu,
  //  num_parameters * sizeof(float), cudaMemcpyHostToDevice));
  //  free(params_memory_cpu);
  fclose(model_file);

  // other inits
  model->acts_memory = NULL;
  model->grads_memory = NULL;
  model->m_memory = NULL;
  model->v_memory = NULL;
  model->grads_acts_memory = NULL;
  model->inputs = NULL;
  model->targets = NULL;
  model->batch_size = 0;
  model->seq_len = 0;
  model->mean_loss = -1.0f; // -1.0f will designate no loss
}

void gpt2_forward(GPT2 *model, int *inputs, int *targets, int B, int T) {
  // targets are optional and could be NULL

  // ensure the model was initialized or error out
  if (model->params_memory == NULL) {
    printf("Error: model was not initialized properly.\n");
    exit(1);
  }

  // convenience parameters
  int V = model->config.vocab_size;
  int L = model->config.num_layers;
  int NH = model->config.num_heads;
  int C = model->config.channels;

  // allocate space for all the activations if needed (done here, lazily)
  if (model->acts_memory == NULL) {
    // record the current B,T as well
    model->batch_size = B;
    model->seq_len = T;
    // and now allocate the space
    size_t act_offset = 0;
    model->act_sizes[act_offset++] = B * T * V;
    model->act_sizes[act_offset++] = B * T * C;
    model->act_sizes[act_offset++] = L * B * T * C;
    model->act_sizes[act_offset++] = L * B * T;
    model->act_sizes[act_offset++] = L * B * T;
    model->act_sizes[act_offset++] = L * B * T * 3 * C;
    model->act_sizes[act_offset++] = L * B * T * C;
    model->act_sizes[act_offset++] = L * B * NH * T * T;
    model->act_sizes[act_offset++] = L * B * NH * T * T;
    model->act_sizes[act_offset++] = L * B * T * C;
    model->act_sizes[act_offset++] = L * B * T * C;
    model->act_sizes[act_offset++] = L * B * T * C;
    model->act_sizes[act_offset++] = L * B * T;
    model->act_sizes[act_offset++] = L * B * T;
    model->act_sizes[act_offset++] = L * B * T * 4 * C;
    model->act_sizes[act_offset++] = L * B * T * 4 * C;
    model->act_sizes[act_offset++] = L * B * T * C;
    model->act_sizes[act_offset++] = L * B * T * C;
    model->act_sizes[act_offset++] = B * T * C;
    model->act_sizes[act_offset++] = B * T;
    model->act_sizes[act_offset++] = B * T;
    model->act_sizes[act_offset++] = B * T * V;
    model->act_sizes[act_offset++] = B * T;
    model->act_sizes[act_offset++] = L * B * T * 3 * C; // qkvr
    model->act_sizes[act_offset++] = L * B * T * C;     // v_accum
    model->act_sizes[act_offset++] = L * B * NH * T * T;// dpreatt
    model->act_sizes[act_offset++] = L * B * NH * T * T;// datt
    // Verify offset matches NUM_ACTIVATION_TENSORS
    assert(act_offset == NUM_ACTIVATION_TENSORS);

    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
      num_activations += model->act_sizes[i];
    }
    printf("num_activations: %zu\n", num_activations);
    model->num_activations = num_activations;
    model->acts_memory =
        malloc_and_point_activations(&model->acts, model->act_sizes);
  } else {
    // validate B,T is no larger than what was previously allocated
    // in principle, we could re-allocate a larger chunk of memory, for now we
    // just error out
    if (B > model->batch_size || T > model->seq_len) {
      printf("Error: batch size or sequence length is inadequately large\n");
      printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size,
             model->seq_len, B, T);
      exit(1);
    }
  }

  // Store pointers to the current batch's GPU
  // We just need to save these pointers in the model struct
  // so the backward pass functions (like crossentropy_softmax_backward and encoder_backward)
  // can access the correct data for the current batch.
  model->inputs = inputs;
  model->targets = targets; // This pointer is NULL if targets are not provided (during inference)

  // forward pass
  ParameterTensors params = model->params; // for brevity
  ActivationTensors acts = model->acts;
  float *residual;

  // Use model->inputs which now points to the correct GPU buffer from DataLoader
  encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C);
  for (int l = 0; l < L; l++) {
    residual = l == 0 ? acts.encoded : acts.residual3 + (l - 1) * B * T * C;
    // get the pointers of the weights for this layer
    float *l_ln1w = params.ln1w + l * C;
    float *l_ln1b = params.ln1b + l * C;
    float *l_qkvw = params.qkvw + l * 3 * C * C;
    float *l_qkvb = params.qkvb + l * 3 * C;
    float *l_attprojw = params.attprojw + l * C * C;
    float *l_attprojb = params.attprojb + l * C;
    float *l_ln2w = params.ln2w + l * C;
    float *l_ln2b = params.ln2b + l * C;
    float *l_fcw = params.fcw + l * 4 * C * C;
    float *l_fcb = params.fcb + l * 4 * C;
    float *l_fcprojw = params.fcprojw + l * C * 4 * C;
    float *l_fcprojb = params.fcprojb + l * C;

    // get the pointers of the activations for this layer
    float *l_ln1 = acts.ln1 + l * B * T * C;
    float *l_ln1_mean = acts.ln1_mean + l * B * T;
    float *l_ln1_rstd = acts.ln1_rstd + l * B * T;
    float *l_qkv = acts.qkv + l * B * T * 3 * C;
    float *l_qkvr = acts.qkvr + l * B * T * 3 * C;
    float *l_atty = acts.atty + l * B * T * C;
    float *l_preatt = acts.preatt + l * B * NH * T * T;
    float *l_att = acts.att + l * B * NH * T * T;
    float *l_v_accum = acts.v_accum + l * B * T * C;
    float *l_attproj = acts.attproj + l * B * T * C;
    float *l_residual2 = acts.residual2 + l * B * T * C;
    float *l_ln2 = acts.ln2 + l * B * T * C;
    float *l_ln2_mean = acts.ln2_mean + l * B * T;
    float *l_ln2_rstd = acts.ln2_rstd + l * B * T;
    float *l_fch = acts.fch + l * B * T * 4 * C;
    float *l_fch_gelu = acts.fch_gelu + l * B * T * 4 * C;
    float *l_fcproj = acts.fcproj + l * B * T * C;
    float *l_residual3 = acts.residual3 + l * B * T * C;

    // now do the forward pass
    layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b,
                      B, T, C);
    matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
    attention_forward(l_atty, l_v_accum, l_qkvr, l_preatt, l_att, l_qkv, B, T,
                      C, NH);
    matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
    residual_forward(l_residual2, residual, l_attproj, B * T * C);
    layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w,
                      l_ln2b, B, T, C);
    matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
    gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C);
    matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
    residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);

    metalCommitCommands();
  }

  residual =
      acts.residual3 + (L - 1) * B * T * C; // last residual is in residual3
  layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual,
                    params.lnfw, params.lnfb, B, T, C);

  matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, V);
  softmax_forward(acts.probs, acts.logits, B, T, V);
  // also forward the cross-entropy loss function if we have the targets
  if (targets != NULL) {
    crossentropy_forward(acts.losses, acts.probs, targets, B, T, V);
    metalCommitCommandsAndWait();
    float mean_loss = 0.0f;
    for (int i = 0; i < B * T; i++) {
      mean_loss += acts.losses[i];
    }
    mean_loss /= B * T;
    model->mean_loss = mean_loss;

  } else {
    // if we don't have targets, we don't have a loss
    model->mean_loss = -1.0f;
    metalCommitCommandsAndWait();
  }
}

void point_parameters_metal(ParameterTensors* params, float* params_memory, size_t* param_sizes) {
  float** ptrs[] = {
    &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
    &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
    &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
};
  float* params_memory_iterator = params_memory;
  for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    *(ptrs[i]) = params_memory_iterator;
    params_memory_iterator += param_sizes[i];
  }
}

void point_activations_metal(ActivationTensors* acts, float* acts_memory, size_t* act_sizes) {
  float** ptrs[] = {
    &acts->logits,   &acts->encoded,   &acts->ln1,       &acts->ln1_mean,
    &acts->ln1_rstd, &acts->qkv,       &acts->atty,      &acts->preatt, // Forward pass intermediates
    &acts->att,      &acts->attproj,   &acts->residual2, &acts->ln2,
    &acts->ln2_mean, &acts->ln2_rstd,  &acts->fch,       &acts->fch_gelu,
    &acts->fcproj,   &acts->residual3, &acts->lnf,       &acts->lnf_mean,
    &acts->lnf_rstd, &acts->probs,     &acts->losses,    &acts->qkvr,
    &acts->v_accum,
    &acts->dpreatt, &acts->datt
  };
  float* acts_memory_iterator = acts_memory;
  size_t num_ptrs = sizeof(ptrs)/sizeof(ptrs[0]);
  size_t loop_limit = (num_ptrs < NUM_ACTIVATION_TENSORS) ? num_ptrs : NUM_ACTIVATION_TENSORS;

  for (size_t i = 0; i < loop_limit; i++) {
    *(ptrs[i]) = acts_memory_iterator;
    acts_memory_iterator += act_sizes[i];
  }
}

void gpt2_backward(GPT2 *model) {

  //  Lazy Allocation of Gradient Buffers
  if (model->grads_memory == NULL) {
    printf("Allocating gradient and optimizer state memory\n");

    // Two contiguous chunks, one for parameter gradients and one for activation (inter-layer) gradients
    const size_t grads_size      = model->num_parameters  * sizeof(float);
    const size_t grads_acts_size = model->num_activations * sizeof(float);

    metalCheck(metalMalloc((void **)&model->grads_memory, grads_size));

    //point pointers to pieces of these chunks to appropriate tensors
    point_parameters_metal(&model->grads,
                           model->grads_memory,
                           model->param_sizes);

    metalCheck(metalClearBuffer(model->grads_memory, grads_size));
    metalCheck(metalMalloc((void **)&model->grads_acts_memory,
                           grads_acts_size));
    point_activations_metal(&model->grads_acts,
                            model->grads_acts_memory,
                            model->act_sizes);

    metalCheck(metalClearBuffer(model->grads_acts_memory,
                                grads_acts_size));

    //AdamW moment buffers
    metalCheck(metalMalloc((void **)&model->m_memory, grads_size));
    metalCheck(metalMalloc((void **)&model->v_memory, grads_size));
    metalCheck(metalClearBuffer(model->m_memory, grads_size));
    metalCheck(metalClearBuffer(model->v_memory, grads_size));

    printf("Gradient, activations, M and V buffers allocated and zero-initialised.\n");
  }

    int B = model->batch_size;
    int T = model->seq_len;
    int V = model->config.vocab_size;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;

    //pointers to weights
    ParameterTensors params = model->params;
    //Pointers to parameter gradient buffers
    ParameterTensors grads = model->grads;
    //pointers to forward activation buffers
    ActivationTensors acts = model->acts;
    //pointers to activation's gradient buffers
    ActivationTensors grads_acts = model->grads_acts;


    // Step 1 --> Initialize dlosses gradient buffer
    initialize_dlosses(grads_acts.losses, B, T);

    // Step 2 --> Backprop through CrossEntropy and Softmax
    crossentropy_softmax_backward(model);

    // Step 3 --> Backprop through final MatMul (classifier) -> dLNF, dWTE
    matmul_backward(grads_acts.lnf, grads.wte, NULL,
                    grads_acts.logits, acts.lnf, params.wte,
                    B, T, C, V);

    // Step 4 --> Backprop through final LayerNorm
    float* final_residual = acts.residual3 + (L - 1) * B * T * C; // Input to lnf
    float* dgrad_residual3_last = grads_acts.residual3 + (L - 1) * B * T * C; // Gradient flowing OUT of lnf_backward
    layernorm_backward(dgrad_residual3_last, grads.lnfw, grads.lnfb, // Gradients for lnf weights/bias
                       grads_acts.lnf,      // Gradient flowing INTO lnf_backward
                       final_residual,      // Forward input to lnf
                       params.lnfw,         // Forward lnf weight
                       acts.lnf_mean,       // Forward lnf mean
                       acts.lnf_rstd,       // Forward lnf rstd
                       B, T, C);            // Dimensions

    // Loop through L layers in reverse
    for (int l = L - 1; l >= 0; l--) {
        float* l_ln1w = params.ln1w + l * C; float* dl_ln1w = grads.ln1w + l * C;
        float* l_ln1b = params.ln1b + l * C; float* dl_ln1b = grads.ln1b + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C; float* dl_qkvw = grads.qkvw + l * 3*C * C;
        float* l_qkvb = params.qkvb + l * 3*C; float* dl_qkvb = grads.qkvb + l * 3*C;
        float* l_attprojw = params.attprojw + l * C * C; float* dl_attprojw = grads.attprojw + l * C * C;
        float* l_attprojb = params.attprojb + l * C; float* dl_attprojb = grads.attprojb + l * C;
        float* l_ln2w = params.ln2w + l * C; float* dl_ln2w = grads.ln2w + l * C;
        float* l_ln2b = params.ln2b + l * C; float* dl_ln2b = grads.ln2b + l * C;
        float* l_fcw = params.fcw + l * 4*C * C; float* dl_fcw = grads.fcw + l * 4*C * C;
        float* l_fcb = params.fcb + l * 4*C; float* dl_fcb = grads.fcb + l * 4*C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C; float* dl_fcprojw = grads.fcprojw + l * C * 4*C;
        float* l_fcprojb = params.fcprojb + l * C; float* dl_fcprojb = grads.fcprojb + l * C;

        // Get pointers to activations and their gradients for layer l
        float* l_residual_in = (l == 0) ? acts.encoded : acts.residual3 + (l - 1) * B * T * C;
        float* dl_residual_in_grad = (l == 0) ? grads_acts.encoded : grads_acts.residual3 + (l - 1) * B * T * C;
        float* l_ln1 = acts.ln1 + l*B*T*C; float* dl_ln1 = grads_acts.ln1 + l*B*T*C;
        float* l_ln1_mean = acts.ln1_mean + l*B*T; float* l_ln1_rstd = acts.ln1_rstd + l*B*T;
        float* l_qkv = acts.qkv + l*B*T*3*C; float* dl_qkv = grads_acts.qkv + l*B*T*3*C;
        float* l_atty = acts.atty + l*B*T*C; float* dl_atty = grads_acts.atty + l*B*T*C;
        float* l_attproj = acts.attproj + l*B*T*C; float* dl_attproj = grads_acts.attproj + l*B*T*C;
        float* l_residual2 = acts.residual2 + l*B*T*C; float* dl_residual2 = grads_acts.residual2 + l*B*T*C;
        float* l_ln2 = acts.ln2 + l*B*T*C; float* dl_ln2 = grads_acts.ln2 + l*B*T*C;
        float* l_ln2_mean = acts.ln2_mean + l*B*T; float* l_ln2_rstd = acts.ln2_rstd + l*B*T;
        float* l_fch = acts.fch + l*B*T*4*C; float* dl_fch = grads_acts.fch + l*B*T*4*C;
        float* l_fch_gelu = acts.fch_gelu + l*B*T*4*C; float* dl_fch_gelu = grads_acts.fch_gelu + l*B*T*4*C;
        float* l_fcproj = acts.fcproj + l*B*T*C; float* dl_fcproj = grads_acts.fcproj + l*B*T*C;
        float* l_residual3 = acts.residual3 + l*B*T*C; float* dl_residual3 = grads_acts.residual3 + l*B*T*C;


        // Backprop through Residual 3 addition
        residual_backward(dl_residual2, dl_fcproj, dl_residual3, B*T*C);
        matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb,
                        dl_fcproj, l_fch_gelu, l_fcprojw,
                        B, T, 4*C, C);

        int N_gelu = B * T * 4 * C;
        gelu_backward(dl_fch, l_fch, dl_fch_gelu, N_gelu);

        matmul_backward(dl_ln2, dl_fcw, dl_fcb,
                        dl_fch, l_ln2, l_fcw,
                        B, T, C, 4*C);

        layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b,
                           dl_ln2, l_residual2, l_ln2w,
                           l_ln2_mean, l_ln2_rstd,
                           B, T, C);

        // Backprop through Residual 2 addition
        residual_backward(dl_residual_in_grad, dl_attproj, dl_residual2, B*T*C);

        // Backprop through Attention projection matmul -> computes dl_atty, dl_attprojw, dl_attprojb
        matmul_backward(dl_atty, dl_attprojw, dl_attprojb,
                        dl_attproj, l_atty, l_attprojw,
                        B, T, C, C);

        // Backprop through QKV matmul
        matmul_backward(dl_ln1, dl_qkvw, dl_qkvb,
                        dl_qkv, l_ln1, l_qkvw,
                        B, T, C, 3*C);

        //Backprop through LayerNorm 1 (ln1) -> computes dl_residual_in_grad (accumulates), dl_ln1w, dl_ln1b
        layernorm_backward(dl_residual_in_grad, dl_ln1w, dl_ln1b,
                           dl_ln1, l_residual_in, l_ln1w,
                           l_ln1_mean, l_ln1_rstd,
                           B, T, C);
    }

  //Backprop through Encoder
  encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, model->inputs, B, T, C);
  metalCommitCommandsAndWait();
}

void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2,
                 float eps, float weight_decay, int t)
{
    uint32_t num_parameters = (uint32_t)model->num_parameters;
    if (model->m_memory == NULL || model->v_memory == NULL || grad_norm_sum_sq_buffer == NULL || clip_scale_factor_buffer == NULL) {
        printf("Allocating gradient, optimizer state, and clipping buffers...\n");
        size_t grads_size = model->num_parameters * sizeof(float);
        if (!model->grads_memory) { metalCheck(metalMalloc((void**)&model->grads_memory, grads_size)); point_parameters_metal(&model->grads, model->grads_memory, model->param_sizes); }
        if (!model->m_memory) { metalCheck(metalMalloc((void**)&model->m_memory, grads_size)); metalCheck(metalClearBuffer(model->m_memory, grads_size)); }
        if (!model->v_memory) { metalCheck(metalMalloc((void**)&model->v_memory, grads_size)); metalCheck(metalClearBuffer(model->v_memory, grads_size)); }
        if (!grad_norm_sum_sq_buffer) { metalCheck(metalMalloc((void**)&grad_norm_sum_sq_buffer, sizeof(float))); }
        if (!clip_scale_factor_buffer) { metalCheck(metalMalloc((void**)&clip_scale_factor_buffer, sizeof(float))); }
    }

    //adamw update
    int adamw_grid_size = num_parameters;
    int adamw_block_size = 256;
    adamw_block_size = adamw_grid_size < adamw_block_size ? adamw_grid_size : adamw_block_size;
    adamw_block_size = adamw_block_size > 0 ? adamw_block_size : 1;
      float m_corr = 1.0f / (1.0f - powf(beta1, (float)(t)));
      float v_corr = 1.0f / (1.0f - powf(beta2, (float)(t)));

    launchKernel("adamw_kernel",
                   adamw_grid_size, adamw_block_size, 0,
                   12,
                   Buffer, model->params_memory,
                   Buffer, model->grads_memory,
                   Buffer, model->m_memory,
                   Buffer, model->v_memory,
                   Scalar, &learning_rate,
                   Scalar, &beta1,
                   Scalar, &beta2,
                   Scalar, &eps,
                   Scalar, &weight_decay,
                   Scalar, &num_parameters,
                   Scalar, &m_corr,
                   Scalar, &v_corr);

    metalCommitCommandsAndWait();
}

void gpt2_free(GPT2 *model) {
  metalCheck(metalFree(model->params_memory));
   metalCheck(metalFree(model->grads_memory));
  metalCheck(metalFree(model->grads_acts_memory));
  metalCheck(metalFree(model->m_memory));
  metalCheck(metalFree(model->v_memory));
  metalCheck(metalFree(model->inputs));
  metalCheck(metalFree(model->targets));
  metalCheck(metalFree(model->acts_memory));
}

#ifndef TESTING
// if we are TESTING (see test_gpt2.cu), we'll skip the int main below

// ----------------------------------------------------------------------------
// data loader lite
// returns random batches of data from a file of integers

typedef struct {
  // hyperparameters
  int B;
  int T;
  // input handling and its state
  FILE *tokens_file;
  long file_size;
  long current_position;
  // output memory
  int *batch;
  int *inputs;
  int *targets;
  // convenience variables
  int num_batches;
} DataLoader;

void dataloader_init(DataLoader *loader, char *filename, int B, int T) {
  loader->B = B;
  loader->T = T;

  // open the input file for reading
  loader->tokens_file = fopen(filename, "rb");
  if (loader->tokens_file == NULL) {
    printf("Error opening tokens file\n");
    exit(1);
  }

  // determine the file size
  fseek(loader->tokens_file, 0, SEEK_END);
  loader->file_size = ftell(loader->tokens_file);
  fseek(loader->tokens_file, 0, SEEK_SET);
  if (loader->file_size < (B * T + 1) * sizeof(int)) {
    printf("Error: file size is too small for the batch size and sequence "
           "length\n");
    exit(1);
  }
  loader->current_position = 0; // start at the beginning

  // allocate space for B*T + 1 integers to store the inputs and targets
  metalCheck(metalMalloc((void **)&loader->batch, (B * T + 1) * sizeof(int)));
  //  loader->batch = (int*) malloc((B * T + 1) * sizeof(int));
  loader->inputs = loader->batch;
  loader->targets = loader->batch + 1; // targets are shifted by one
  loader->num_batches = loader->file_size / (B * T * sizeof(int));
}

void dataloader_reset(DataLoader *loader) { loader->current_position = 0; }

void dataloader_next_batch(DataLoader *loader) {
  int B = loader->B;
  int T = loader->T;
  // if we are at the end of the file, loop back to the beginning
  if (loader->current_position + (B * T + 1) * sizeof(int) >
      loader->file_size) {
    loader->current_position = 0;
  }
  // read the B*T+1 integers from the file into batch
  fseek(loader->tokens_file, loader->current_position, SEEK_SET);
  fread(loader->batch, sizeof(int), B * T + 1, loader->tokens_file);
  // advance the current position by B*T integers
  loader->current_position += B * T * sizeof(int);
}

void dataloader_free(DataLoader *loader) {
  fclose(loader->tokens_file);
  metalFree(loader->batch);
}

// ----------------------------------------------------------------------------
// sampler

#define GPT2_EOT 50256

unsigned int random_u32(unsigned long long *state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
  return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float *probabilities, int n, float coin) {
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1; // in case of rounding errors
}
// ----------------------------------------------------------------------------
// Tokenizer (only supports decoding)

typedef struct {
  uint32_t vocab_size;
  char **token_table;
  int init_ok;
} Tokenizer;

void safe_printf(const char *piece) {
  // the tokens are raw bytes, and we we only want to print the printable ones
  // many bytes can be various control codes, backspace, etc.
  if (piece == NULL) {
    return;
  }
  if (piece[0] == '\0') {
    return;
  }
  // handle individual byte tokens
  // every token is asserted to be at least one byte so doing piece[1] is ok
  if (piece[1] == '\0') {
    unsigned char byte_val = piece[0];
    if (!(isprint(byte_val) || isspace(byte_val))) {
      return; // weird byte, don't print it
    }
  }
  printf("%s", piece);
}

void tokenizer_init(Tokenizer *tokenizer, const char *filename) {
  FILE *file = fopen(filename, "rb");
  if (file == NULL) {
    // try to be more helpful as we just added this feature, erase later
    printf("---\n");
    printf("WARNING: Failed to open the tokenizer file %s\n", filename);
    printf("The Tokenizer is a new feature added April 14 2024.\n");
    printf("Re-run `python train_gpt2.py` to write it\n");
    printf("---\n");
    tokenizer->init_ok = 0;
    return;
  }
  // read in the header
  uint32_t header[256];
  fread(header, sizeof(uint32_t), 256, file);
  assert(header[0] == 20240328);
  assert(header[1] == 1);
  tokenizer->vocab_size = header[2];
  // read in all the tokens
  unsigned char length;
  tokenizer->token_table =
      (char **)malloc(tokenizer->vocab_size * sizeof(char *));
  for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
    fread(&length, sizeof(unsigned char), 1, file);
    assert(length > 0); // every token should be at least one character
    char *token_bytes = (char *)malloc(length + 1);
    fread(token_bytes, sizeof(char), length, file);
    token_bytes[length] = '\0'; // Add null terminator for printing
    tokenizer->token_table[i] = token_bytes;
  }
  // cleanups
  fclose(file);
  tokenizer->init_ok = 1;
}

const char *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
  if (tokenizer->init_ok == 0) {
    return NULL;
  }
  if (token_id < tokenizer->vocab_size) {
    return tokenizer->token_table[token_id];
  } else {
    printf("invalid token id %d!\n", token_id);
    return NULL;
  }
}

void tokenizer_free(Tokenizer *tokenizer) {
  if (tokenizer->init_ok) {
    for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
      free(tokenizer->token_table[i]);
    }
    free(tokenizer->token_table);
  }
}

void gpt2_zero_grad(GPT2* model)
{
    if (model->grads_memory == NULL || model->grads_acts_memory == NULL)
      return;

    size_t grads_size = model->num_parameters*sizeof(float);
    metalCheck(metalClearBuffer(model->grads_memory, grads_size));

    size_t grads_acts_size = model->num_activations*sizeof(float);
    metalCheck(metalClearBuffer(model->grads_acts_memory, grads_acts_size));

}
// ----------------------------------------------------------------------------
// main training loop
int main(void) {
  GPT2 model;
  gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

  initKernels("encoder_forward_kernel2", "mean_kernel", "rstd_kernel",
              "normalization_kernel", "add_bias_kernel", "permute_kernel",
              "unpermute_kernel", "softmax_forward_kernel4",
              "residual_forward_kernel", "gelu_kernel",
              "crossentropy_forward_kernel1", "scale_kernel",
              "softmax_forward_kernel1",
              "crossentropy_softmax_backward_kernel",
              "softmax_backward_kernel",
              "matmul_backward_bias_kernel_fixed",
              "layernorm_backward_kernel",
              "residual_backward_kernel",
              "gelu_backward_kernel",
              "softmax_backward_attn_kernel",
              "encoder_backward_kernel",
              "sum_squares_kernel",
              "adamw_kernel",
              "initialize_dlosses_kernel",
              NULL);

  // build the DataLoaders from tokens files. for now use tiny_shakespeare if
  // available, else tiny_stories
  char *tiny_stories_train = "data/TinyStories_train.bin";
  char *tiny_stories_val = "data/TinyStories_val.bin";
  char *tiny_shakespeare_train = "data/tiny_shakespeare_train.bin";
  char *tiny_shakespeare_val = "data/tiny_shakespeare_val.bin";
  char *train_tokens = access(tiny_shakespeare_train, F_OK) != -1
                           ? tiny_shakespeare_train
                           : tiny_stories_train;
  char *val_tokens = access(tiny_shakespeare_val, F_OK) != -1
                         ? tiny_shakespeare_val
                         : tiny_stories_val;
  int B = 4;
  int T = 64;
  DataLoader train_loader;
  dataloader_init(&train_loader, train_tokens, B, T);
  printf("train dataset num_batches: %d\n", train_loader.num_batches);
  DataLoader val_loader;
  dataloader_init(&val_loader, val_tokens, B, T);
  printf("val dataset num_batches: %d\n", val_loader.num_batches);
  int val_num_batches = 5;

  // build the Tokenizer
  Tokenizer tokenizer;
  tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

  printf("batch size: %d\n", B);
  printf("sequence length: %d\n", T);
  printf("val_num_batches: %d\n", val_num_batches);

  // some memory for generating samples from the model
  unsigned long long rng_state = 1337;
  const int genT = 64;
  int *gen_tokens = NULL;
  metalMalloc((void **)&gen_tokens, sizeof(int) * B * T);

  // train
  struct timespec start, end;
  for (int step = 0; step <= 40; step++) {
    // once in a while estimate the validation loss
    if (step % 10 == 0) {
      float val_loss = 0.0f;
      dataloader_reset(&val_loader);
      for (int i = 0; i < val_num_batches; i++) {
        dataloader_next_batch(&val_loader);
        gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
        val_loss += model.mean_loss;
      }
      val_loss /= val_num_batches;
      printf("val loss %f\n", val_loss);
    }

    // once in a while do model inference to print generated text
    if (step > 0 && step % 20 == 0) {
      for (int i = 0; i < B * T; ++i) {
        gen_tokens[i] = GPT2_EOT;
      }
      printf("generating:\n---\n");
      for (int t = 1; t < genT; t++) {
        // note that inference is wasteful here because
        // for each t, we re-compute all activations between 0 and t
        // leaving this alone because you want separate code for inference
        // anyway the inference here is just for sanity checking purposes
        gpt2_forward(&model, gen_tokens, NULL, B, T);
        float *probs = model.acts.probs + (t - 1) * model.config.vocab_size;
        float coin = random_f32(&rng_state);
        // move probs back to CPU and sample
        int next_token = sample_mult(probs, model.config.vocab_size, coin);
        gen_tokens[t] = next_token;
        if (tokenizer.init_ok) {
          const char *token_str = tokenizer_decode(&tokenizer, next_token);
          safe_printf(token_str);
        } else {
          // fall back to printing the token id
          printf("%d ", next_token);
        }
        fflush(stdout);
      }
      printf("\n---\n");
    }

    // do a training step
    clock_gettime(CLOCK_MONOTONIC, &start);
    dataloader_next_batch(&train_loader);
    gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
    // these are still TODO
    gpt2_zero_grad(&model);
    gpt2_backward(&model);
    gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_elapsed_s =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("step %d: train loss %f (took %f ms)\n", step, model.mean_loss,
           time_elapsed_s * 1000);
  }

  // free
  dataloader_free(&train_loader);
  dataloader_free(&val_loader);
  tokenizer_free(&tokenizer);
  gpt2_free(&model);
  metalFree(gen_tokens);
  return 0;
}
#endif
