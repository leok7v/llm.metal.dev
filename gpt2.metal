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

#include <metal_stdlib>
using namespace metal;

#define MAX_RANK 5
#define SIMD_GROUP_SIZE 32 // TODO: Shouldn't hardcode this. Can get it from host API (threadExecutionWidth).
#define M_PI 3.14159265358979323846264338327950288

inline void flat_to_nd(int index,
                       int inShape[MAX_RANK], // The shape of the tensor
                       int inStrides[MAX_RANK], // The strides for the tensor
                       int outCoords[MAX_RANK]) { // The resulting output.
  int flatIdx = index;
  // Start at the most significant rank and work our way down.
  for (int i = 0; i < MAX_RANK; i++) {
    int coord = flatIdx / inStrides[i];
    outCoords[i] = min(coord, inShape[i] - 1);
    // Take what's left over to the next rank...
    flatIdx -= coord * inStrides[i];
  }
}

inline int nd_to_flat(int inCoords[MAX_RANK],
                      int inStrides[MAX_RANK]) {
  int flatIdx = 0;
  for (int i = 0; i < MAX_RANK; i++) {
    flatIdx += inCoords[i] * inStrides[i];
  }
  return flatIdx;
}

inline void calc_strides(const int srcShape[MAX_RANK],
                         int outStrides[MAX_RANK]) {
  for (int i = 0; i < MAX_RANK; i++) {
    int prod = 1;
    for (int j = i + 1; j < MAX_RANK; j++) {
      prod *= srcShape[j];
    }
    outStrides[i] = prod;
  }
}

inline int calc_perm_idx(const int srcShape[MAX_RANK],
                         const int permOrder[MAX_RANK],
                         const uint index) {
  int strides[MAX_RANK];
  calc_strides(srcShape, strides);

  int permuted_shape[MAX_RANK];
  int permuted_strides[MAX_RANK];
  for (int i = 0; i < MAX_RANK; i++) {
    int axis = permOrder[i];
    permuted_strides[i] = strides[axis];
    permuted_shape[i] = srcShape[axis];
  }

  int strides_T[MAX_RANK];
  calc_strides(permuted_shape, strides_T);
  int coords[MAX_RANK];
  flat_to_nd(index, permuted_shape, strides_T, coords);
  int permIdx = nd_to_flat(coords, permuted_strides);

  return permIdx;
}

template<typename T>
static inline T simdGroupReduceMax(T val, uint simd_size) {
  for (int offset = simd_size / 2; offset > 0; offset >>= 1) {
    T other = simd_shuffle_down(val, offset);
    val = max(val, other);
  }
  return val;
}

template<typename T>
static inline T simdGroupReduceSum(T val, uint simd_size) {
  for (int offset = simd_size / 2; offset > 0; offset >>= 1) {
    T other = simd_shuffle_down(val, offset);
    val += other;
  }
  return val;
}

kernel void encoder_forward_kernel2(device float* out [[buffer(0)]],
                                    device int* inp [[buffer(1)]],
                                    device float* wte [[buffer(2)]],
                                    device float* wpe [[buffer(3)]],
                                    constant uint& B [[ buffer(4) ]],
                                    constant uint& T [[ buffer(5) ]],
                                    constant uint& C [[ buffer(6) ]],
                                    uint tid [[thread_position_in_grid]]) {
  uint N = B * T * C;

  if (tid < N) {
    int bt = tid / C;
    int b = bt / T;
    int t = bt % T;
    int c = tid % C;

    int ix = inp[b * T + t];
    device float* out_btc = out + b * T * C + t * C + c;
    device float* wte_ix = wte + ix * C + c;
    device float* wpe_tc = wpe + t * C + c;
    *out_btc = *wte_ix + *wpe_tc;
  }
}

kernel void mean_kernel(device float* mean [[buffer(0)]],
                        device float* inp [[buffer(1)]],
                        constant int& N [[buffer(2)]],
                        constant int& C [[buffer(3)]],
                        uint block_size [[threads_per_threadgroup]],
                        uint idx [[threadgroup_position_in_grid]],
                        uint tgid [[thread_position_in_threadgroup]],
                        threadgroup float* shared [[threadgroup(0)]]) {
  int index = idx; // range [0, B*T)
  int thread_id = tgid; // range [0, block_size)
  device float* x = inp + index * C;
  // thread coarsening
  float sum = 0.0f;
  for (int i = thread_id; i < C; i += block_size) {
    sum += x[i];
  }
  shared[thread_id] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // reductions
  for (int stride = block_size / 2; stride >= 1; stride /= 2) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_id < stride) {
      shared[thread_id] += shared[thread_id + stride];
    }
  }
  // write the final result (at thread 0) to global memory
  if (thread_id == 0) {
    mean[index] = shared[0] / C;
  }
}

kernel void rstd_kernel(device float* rstd [[buffer(0)]],
                        device float* inp [[buffer(1)]],
                        device float* mean [[buffer(2)]],
                        constant uint& N [[ buffer(3) ]],
                        constant uint& C [[ buffer(4) ]],
                        uint idx [[threadgroup_position_in_grid]],
                        uint tgid [[thread_position_in_threadgroup]],
                        uint bsize [[threads_per_threadgroup]],
                        threadgroup float* shared [[threadgroup(0)]]) {
//  if (idx >= N) return; // Guard against out-of-bounds work items

  device float* x = inp + idx * C;
  float m = mean[idx];
  // thread coarsening
  float sum = 0.0f;
  for (uint i = tgid; i < C; i += bsize) {
    float diff = x[i] - m;
    sum += diff * diff;
  }
  shared[tgid] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // reductions
  for (uint stride = bsize / 2; stride >= 1; stride /= 2) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tgid < stride) {
      shared[tgid] += shared[tgid + stride];
    }
  }

  // write the final result (at thread 0) to global memory
  if (tgid == 0) {
    rstd[idx] = 1.0f / precise::sqrt(shared[0] / C + 1e-5f);
  }
}

kernel void normalization_kernel(device float* out [[buffer(0)]],
                                 device float* inp [[buffer(1)]],
                                 device float* mean [[buffer(2)]],
                                 device float* rstd [[buffer(3)]],
                                 device float* weight [[buffer(4)]],
                                 device float* bias [[buffer(5)]],
                                 constant uint& B [[ buffer(6) ]],
                                 constant uint& T [[ buffer(7) ]],
                                 constant uint& C [[ buffer(8) ]],
                                 uint tid [[thread_position_in_grid]]) {
  uint bt = tid / C;
  uint c = tid % C;
  float m = mean[bt];
  float s = rstd[bt];
  float xi = inp[tid];
  float n = s * (xi - m);
  float o = bias[c] + n * weight[c];
  out[tid] = o;
}

kernel void permute_kernel(device float* q [[ buffer(0) ]],
                           device float* k [[ buffer(1) ]],
                           device float* v [[ buffer(2) ]],
                           const device float* inp [[ buffer(3) ]],
                           constant uint& B [[ buffer(4) ]],
                           constant uint& N [[ buffer(5) ]],
                           constant uint& NH [[ buffer(6) ]],
                           constant uint& d [[ buffer(7) ]],
                           uint tid [[ thread_position_in_grid ]]) {
  if (tid < B * NH * N * d) {
    uint b = tid / (NH * N * d);
    uint rest = tid % (NH * N * d);
    uint nh_ = rest / (N * d);
    rest = rest % (N * d);
    uint n = rest / d;
    uint d_ = rest % d;

    uint inp_idx = \
    (b * N * 3 * NH * d)
    +   (n * 3 * NH * d)
    +       (0 * NH * d)
    +          (nh_ * d)
    +                d_;

    q[tid] = inp[inp_idx];
    k[tid] = inp[inp_idx + NH * d];
    v[tid] = inp[inp_idx + 2 * NH * d];
  }
}

kernel void unpermute_kernel(const device float* inp [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             constant uint& B [[buffer(2)]],
                             constant uint& T [[buffer(3)]],
                             constant uint& NH [[buffer(4)]],
                             constant uint& HS [[buffer(5)]],
                             uint tid [[thread_position_in_grid]]) {

  // B, NH, HS, T ->  B, T, NH, HS
  // 0   1   2  3     0  3  1   2
  const int src_shape[5] = {(int)B, (int)NH, (int)HS, (int)T, 1};
  const int perm_order[5] = {0, 3, 1, 2, 4};
  int permIdx = calc_perm_idx(src_shape, perm_order, tid);
  out[tid] = inp[permIdx];
}

kernel void add_bias_kernel(device float* out [[ buffer(0) ]],
                            device float* bias [[ buffer(1) ]],
                            constant uint& OC [[ buffer(2) ]],
                            uint tid [[ thread_position_in_grid ]]) {
  out[tid] = out[tid] + bias[tid % OC];
}

kernel void scale_kernel(device float* inout [[buffer(0)]],
                         constant float& scale [[buffer(1)]],
                         constant uint& B [[buffer(2)]],
                         constant uint& NH [[buffer(3)]],
                         constant uint& T [[ buffer(4) ]],
                         uint tid [[thread_position_in_grid]])
{
  int rest = tid % (NH * T * T);
  rest = rest % (T * T);
  int t2 = rest / T;
  int t = rest % T;
  if (t > t2) {
    inout[tid] = -INFINITY;
  } else {
    inout[tid] *= scale;
  }
}

kernel void softmax_forward_kernel1(device float* out [[buffer(0)]],
                                    device float* inp [[buffer(1)]],
                                    constant int& N,
                                    constant int& C,
                                    uint tid [[thread_position_in_grid]]) {
  device float* inp_row = inp + tid * C;
  device float* out_row = out + tid * C;

  float maxval = -INFINITY;
  for (int j = 0; j < C; j++) {
    if (inp_row[j] > maxval) {
      maxval = inp_row[j];
    }
  }

  float sum = 0.0f;
  for (int j = 0; j < C; j++) {
    out_row[j] = exp(inp_row[j] - maxval);
    sum += out_row[j];
  }
  for (int j = 0; j < C; j++) {
    out_row[j] /= sum;
  }
}

kernel void softmax_forward_kernel4(device float* out [[buffer(0)]],
                                    device float* inp [[buffer(1)]],
                                    constant uint& N [[ buffer(2) ]],
                                    constant uint& C [[ buffer(3) ]],
                                    uint simdSize [[ thread_execution_width ]],
                                    uint laneID [[ thread_index_in_simdgroup ]],
                                    uint tgIdx [[thread_position_in_threadgroup]],
                                    uint tid [[thread_position_in_grid]],
                                    uint idx [[threadgroup_position_in_grid]],
                                    uint bsize [[threads_per_grid]],
                                    uint simdGroupID [[simdgroup_index_in_threadgroup]],
                                    uint simdGroupsPerBlock [[simdgroups_per_threadgroup]],
                                    threadgroup float* shared [[threadgroup(0)]]) {
  // out is (N, C) just like inp. Each row of inp will get softmaxed.
  // each row of C elements is handled by bsize threads
  // furthermore, each bsize threads get executed in simd groups of SIMD_GROUP_SIZE threads

  // shared[] must be allocated to have 2 * simdGroupsPerBlock elements
  // first half for max values, the second half for sum values
  threadgroup float* maxvals = &shared[0];
  threadgroup float* sumvals = &shared[simdGroupsPerBlock];

  // one row of inp, i.e. inp[idx, :] of shape (C,)
  device float* x = inp + idx * C;

  // first, thread coarsening by directly accessing global memory in series
  float maxval = -INFINITY;
  for (uint i = tgIdx; i < C; i += bsize) {
    maxval = fmax(maxval, x[i]);
  }
  // now within-simd group reductions for maxval
  maxval = simdGroupReduceMax(maxval, simdSize);

  // the 0th thread of each simd group writes the maxval of that simd group to shared memory
  if (laneID == 0) maxvals[simdGroupID] = maxval;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // now the 0th thread reduces the maxvals in shared memory, i.e. across simd groups
  if (tgIdx == 0) {
    float val = maxvals[0];
    for (uint i = 1; i < simdGroupsPerBlock; i++) {
      val = fmax(val, maxvals[i]);
    }
    // store the final max in the first position
    maxvals[0] = val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // broadcast the max to all threads
  float offset = maxvals[0];

  // compute exp and write the result to global memory
  for (uint i = tgIdx; i < C; i += bsize) {
    out[idx * C + i] = exp(x[i] - offset);
  }

  // okay now we calculated exp(x - max(x))
  // step 2: sum all the values and divide by the sum

  // thread coarsening for sum
  x = out + idx * C;
  float sumval = 0.0f;
  for (uint i = tgIdx; i < C; i += bsize) {
    sumval += x[i];
  }
  // within-simd group reduction for sumval
  sumval = simdGroupReduceSum(sumval, simdSize);

  // write sumval to shared memory
  if (laneID == 0) sumvals[simdGroupID] = sumval;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // inter-thread reduction of sum
  if (tgIdx == 0) {
    float val = sumvals[0];
    for (uint i = 1; i < simdGroupsPerBlock; ++i) {
      val += sumvals[i];
    }
    sumvals[0] = val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // broadcast the sum to all threads
  float sum = sumvals[0];

  // divide the whole row by the sum
  for (uint i = tgIdx; i < C; i += bsize) {
    out[idx * C + i] = x[i] / sum;
  }
}

kernel void residual_forward_kernel(device float* out [[ buffer(0) ]],
                                    device float* inp1 [[ buffer(1) ]],
                                    device float* inp2 [[ buffer(2) ]],
                                    uint tid [[ thread_position_in_grid ]]) {
  out[tid] = inp1[tid] + inp2[tid];
}

kernel void gelu_kernel(device float* out [[ buffer(0) ]],
                        device float* inp [[ buffer(1) ]],
                        uint tid [[ thread_position_in_grid ]]) {
  float xi = inp[tid];
  float s = sqrt(2.0f / M_PI);
  float cube = 0.044715f * xi * xi * xi;
  // Use precise variant for tanh since fast-math mode is on.
  out[tid] = 0.5f * xi * (1.0f + precise::tanh(s * (xi + cube)));
}

kernel void crossentropy_forward_kernel1(device float* losses [[ buffer(0) ]],
                                         device float* probs [[ buffer(1) ]],
                                         device int* targets [[ buffer(2) ]],
                                         constant uint& T [[ buffer(3) ]],
                                         constant uint& V [[ buffer(4) ]],
                                         uint tid [[ thread_position_in_grid ]]) {
  uint b = tid / T;
  uint t = tid % T;
  device float* probs_bt = probs + b * T * V + t * V;
  int ix = targets[b * T + t];
  losses[b * T + t] = -log(probs_bt[ix]);
}

/*
    this kernel calculates the gradient of the cross entropy loss function with respect
    to the inputs of the softmax layer (logits).

    Take the output probabilities from the forward pass's softmax, the true target labels
    and the incoming gradients signal (dlosses).

    Idea is to compute how much each logit must change to reduce the loss

*/
kernel void crossentropy_softmax_backward_kernel(
    // output gradients
    device float * dlogits [[buffer(0)]], // (B.T.V)

    //inputs
    device const float *dlosses [[buffer(1)]], // (B.T)
    device const float *probs [[buffer(2)]], // (B.T.V) from softmax
    device const int *targets [[buffer(3)]], // (B.T)

    //scalars
    constant int& B [[buffer(4)]],
    constant int& T [[buffer(5)]],
    constant int& V [[buffer(6)]],

    //launch info
    uint gid [[thread_position_in_grid]]
){

    //guard threads outside the tensor
    const uint N = static_cast<uint>(B) * static_cast<uint>(T) * static_cast<uint>(V);

    if (gid >= N)   return;


    //decode the flattened index
    const uint i = gid % V; //slot in vocabulary
    const uint t = (gid/V) % T; //time step
    const uint b = gid / (V * T); // batch element
    const uint bt = b * T + t;

    //inputs
    const int target = targets[bt];
    const float p_raw = probs[gid];

    //clamp p to avoid gradient blow ups when p is close to 0
    constexpr float k_eps = 1e-30f;
    const float p = fmax(p_raw, k_eps);

    const float indicator = (i == static_cast<uint>(target))? 1.0f : 0.0f;
    const float gradient = (p - indicator) * dlosses[bt];

    // write back
    dlogits[gid] += gradient;
}

/*
    Big picture: FC layer has a bias vector b (Len = OC)
            Here we accumulate dbias across {b,t} (for all output channels j)
    One thread-group processes VEC_SIZE(4) consecutive channels
    Strategy:
        Each thread accumulates its own partial sums for those 4 channels
        SIMD-group (warp) wide reduction with simd_sum();
        Cross-SIMD reduction through threadgroup memory
        Thread-0 atomically adds results into global dbias
*/

kernel void matmul_backward_bias_kernel(
    //output bias gradient vector (length OC) --> uses atomic adds
    device atomic_float *dbias [[buffer(0)]],

    //upstream gradient tensor flattened as (B*T*OC)
    device const float *dout [[buffer(1)]],

    constant int& B [[buffer(2)]],
    constant int& T [[buffer(3)]],
    constant int& OC [[buffer(4)]],

    //threadgroup 0's scratchpad memory (c uda smem per block) bug enough for VEC_SIZE * simd_groups_per_block floats
    threadgroup float* local_sums [[threadgroup(0)]],

    //thread, warp, group identifiers
    uint tix [[thread_index_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    ushort block_size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_groups_per_block [[simdgroups_per_threadgroup]],
    uint simd_size [[thread_execution_width]]
){
    constexpr ushort VEC_SIZE = 4;

    //which 4 channel slice this thread-group owns
    uint o_base = group_id * VEC_SIZE;
    if (o_base >= OC) return;

    int total_elems = B * T;
    if (total_elems == 0) return;

    //per-thread partial sums intialized to zero
    float thread_sums[VEC_SIZE] = {0.0f, 0.0f, 0.0f, 0.0f};

    //grid-stride loop over B * T elements
    // each iteration handles one (b,t) for this thread
    for (int idx = int(tix); idx < total_elems; idx += int(block_size))
    {
        //pointer to the start of row (b, t) inside dout
        const device float *row_ptr = dout + idx * OC;

        //accumulate all of 4 channels
        #pragma unroll
        for (ushort k = 0; k < VEC_SIZE; ++k)
        {
            uint ch = o_base + k; //channel idx
            if (ch < OC)
            {
                float val = row_ptr[ch];
                if (!isnan(val) && !isinf(val))
                    thread_sums[k] += val;
            }
        }
    }

    //first reduction stage inside each SIMD warp
    //simd_sum() returns the sum of the value across all lanes in the warp
    float warp_sums[VEC_SIZE];
    for (ushort k = 0; k < VEC_SIZE; k++)
    {
        warp_sums[k] = simd_sum(thread_sums[k]);
    }

    //lane 0 writes the warp subttotal into threadgroup memory
    if (simd_lane_id == 0)
    {
        #pragma unroll
        for (ushort k = 0; k < VEC_SIZE; k++)
        {
            uint offset = k * simd_groups_per_block + simd_group_id;
            local_sums[offset] = warp_sums[k];
        }
    }

    //synchronize
    threadgroup_barrier(mem_flags::mem_threadgroup);

    //second reduction --> sum across warps
    float final_sums[VEC_SIZE] = {0.0f, 0.0f, 0.0f, 0.0f};

    //only first warp will do this
    if (simd_group_id == 0)
    {
        #pragma unroll
        for (ushort k = 0; k < VEC_SIZE; k++)
        {
            //each lane i loads the subtotal written by warp i
            float subtotal = (tix < simd_groups_per_block) ? local_sums[k*simd_groups_per_block + tix] : 0.0f;

        //reduce subtotals inside this warp again
        subtotal = simd_sum(subtotal);

        //lane - now has block-wide sum for channel k
        if (simd_lane_id == 0)
            final_sums[k] = subtotal;
        }
    }

    //now thread 0 of entire block performs atomic adds

    if (tix == 0)
    {
        #pragma unroll
        for (ushort k = 0; k < VEC_SIZE; k++)
        {
            uint ch = o_base + k;
            if (ch < OC)
            {
                float v = final_sums[k];
                if (!isnan(v) && !isinf(v))
                    atomic_fetch_add_explicit(&dbias[ch], v, memory_order_relaxed);
            }
        }
    }
}

/*
    Layer Normalization kernel

    What it computes:
        For one specific (b,t) sample, i.e. one row of the tensor with length C
        inp_row = inp [b, t, :] normalized values
        dout_row = dout [b, t, :] (gradients coming in from next layer)
        mean_bt = mean[b, t] (saved in forward pass)
        rstd_bt = rstd [b, t] (saved in forward pass)

    It produces
        dinp_row (gradient with respect to each element
        dweights
        dbias

    Parallel Strategy
        Grid Dim.x = B * T -> one thread-group per (b, t) row
        each thread strides over the channel dimension C
            i = threadIdx; i += blockDim until i >=C
            accumulating the two partial sums
        First reduction -> inside each SIMD-group -> gens 1 float per warp
        Second reduction -> across warps via shared memory -> gens 1 float for block

        Both these floats get divided by C to form the means above

        Every thread re-scans its portion of channels and writes dinp

        dweight/dbias requires atomic adds as many thread groups update these concurrently
*/
kernel void layernorm_backward_kernel(
   device atomic_float * dinp [[buffer(0)]],
   device atomic_float* dweight [[buffer(1)]],
   device atomic_float* dbias [[buffer(2)]],
   device const float * dout [[buffer(3)]],
   device const float *inp [[buffer(4)]],
   device const float *weight [[buffer(5)]],
   device const float *mean [[buffer(6)]],
   device const float* rstd [[buffer(7)]],
   //usual
   constant const int &B [[buffer(8)]],
   constant const int &T [[buffer(9)]],
   constant const int &C [[buffer(10)]],
   threadgroup float * shared_mem [[threadgroup(0)]],

   uint tix [[thread_index_in_threadgroup]],
   uint bid [[threadgroup_position_in_grid]],
   ushort block_size [[threads_per_threadgroup]],
   uint simd_lane_id [[thread_index_in_simdgroup]],
   uint simd_group_id [[simdgroup_index_in_threadgroup]],
   uint simd_groups_per_block [[simdgroups_per_threadgroup]],
   uint simd_size [[thread_execution_width]])
{
    const int bt_index = int(bid);
    const int bt_offset = bt_index * C;

    float mean_bt = mean [bt_index];
    float rstd_bt = rstd [bt_index];

    if (isnan(mean_bt) || isnan(rstd_bt) || isinf(mean_bt) || isinf(rstd_bt) ||
            rstd_bt <= 1e-10f || C == 0)    return;

    //scan over all channels and accumulate warp-local dnorm
    float dnorm_sum = 0.0f;
    float dnorm_xhat_sum = 0.0f;

    for (int i = int(tix); i < C; i += int(block_size))
    {
        int idx = bt_offset + i;
        float x = inp [idx];
        float dy = dout [idx];
        float gamma = weight [i];

        if (isnan(dy) || isnan(gamma) || isinf(dy) || isinf(gamma)) continue;

        float xhat = (x - mean_bt) * rstd_bt; //normalized activation
        float dnorm = gamma * dy;

        dnorm_sum += dnorm;
        dnorm_xhat_sum += dnorm * xhat;

        atomic_fetch_add_explicit(&dbias [i], dy, memory_order_relaxed);
        atomic_fetch_add_explicit(&dweight[i], dy * xhat, memory_order_relaxed);
    }

    //warp level reduction
    float warp_sum = simd_sum(dnorm_sum);
    float warp_xhat_sum = simd_sum(dnorm_xhat_sum);

    //lane 0 of each warp stores its subttoal into shared memory
    threadgroup float *buf_sum = shared_mem;
    threadgroup float *buf_xhat_sum = shared_mem + simd_groups_per_block;

    if (simd_lane_id == 0)
    {
        buf_sum [simd_group_id] = warp_sum;
        buf_xhat_sum[simd_group_id] = warp_xhat_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    //cross warp reduction
    /*
    float block_sum = (tix < simd_groups_per_block) ? buf_sum [tix] : 0.0f;
    float block_xhat_sum = (tix < simd_groups_per_block) ? buf_xhat_sum[tix] : 0.0f;
    */

    uint lanes_active = (simd_groups_per_block < simd_size) ? simd_groups_per_block : simd_size;

    float block_sum = (tix < lanes_active) ? buf_sum [tix] : 0.0f;
    float block_xhat_sum = (tix < lanes_active) ? buf_xhat_sum[tix] : 0.0f;

    if (simd_group_id == 0)
    {
        block_sum = simd_sum(block_sum);
        block_xhat_sum = simd_sum(block_xhat_sum);
    }

    // thread 0 computes the two means and broadcasts them
    if (tix == 0)
    {
        buf_sum [0] = block_sum / float(C);
        buf_xhat_sum [0] = block_xhat_sum / float (C);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean_dnorm = buf_sum[0];
    float mean_dnorm_xhat = buf_xhat_sum[0];

    //sanity check
    if (isnan(mean_dnorm) || isnan(mean_dnorm_xhat) ||
        isinf(mean_dnorm) || isinf(mean_dnorm_xhat)) return;

    for (int i = int(tix); i < C; i+= int(block_size))
    {
        int idx = bt_offset + i;
        float x = inp [idx];
        float dy = dout[idx];
        float gamma = weight[i];

        if (isnan(dy) || isnan(gamma) || isinf(dy) || isinf(gamma)) continue;

        float xhat = (x - mean_bt) * rstd_bt;
        float dnorm = gamma * dy;

        float dx = (dnorm - mean_dnorm - xhat * mean_dnorm_xhat) * rstd_bt;

        if (!isnan(dx) && !isinf(dx))
            atomic_fetch_add_explicit(&dinp[idx], dx, memory_order_relaxed);
    }
}

/*
    GELU backward computes dinp[i] += GELU'(x[i] * dout[i])

    Threading model 1D grid, gid ranges from[0, N)
    Inputs: inp -original forward-pass activations (x)
            dout - incoming gradient from next layer (dy)

    Output: dinp - gradient wrt x
*/

#define GELU_SCALING_FACTOR sqrt(2.0f/M_PI_F)

kernel void gelu_backward_kernel(
    device float *dinp,
    device const float* inp,
    device const float* dout,
    constant const int &N,
    uint gid [[thread_position_in_grid]])
{

    if (gid >= N) return;

    float x = inp[gid]; //forward activation
    float dy = dout[gid]; //upstream gradient

    // now evaluate GELU'

    float x2 = x * x;
    float x3_term = 0.044715f * x2 * x;
    float s = GELU_SCALING_FACTOR * (x + x3_term);

    float tanh_s = tanh(s);
    float cosh_s = cosh(s);
    float sech2_s = 1.0f / (cosh_s * cosh_s);

    float gelu_grad = 0.5f * (1.0f + tanh_s)
                + 0.5f * x * sech2_s * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715 * x2);

    //accumulate into dinp
    dinp[gid] += gelu_grad * dy;
}

/*
    softmax backward for attention scores
    each thread --> exactly one element dprpeatt[row, i]

*/

kernel void softmax_backward_attn_kernel(
    device float *dpreatt,
    device const float *datt,
    device const float * att,
    constant int  &N,
    constant int  &T_dim,
    uint gid [[thread_position_in_grid]])
{
    int row = gid / T_dim;
    int i = gid % T_dim;

    if (row >= N) return;

    const device float * att_row = att + row * T_dim;
    const device float * datt_row = datt + row * T_dim;
    device float * dp_row = dpreatt + row * T_dim;

    float acc = 0.0f;
    float att_i = att_row[i];
    for (int j = 0; j < T_dim; j++)
    {
        float indicator = (j == i)? 1.0f: 0.0f;
        acc += att_row[j] * (indicator - att_i) * datt_row[j];
    }

    dp_row[i] = acc;
}

/*
    encoder backward: accumuate dWTE  and DWPE
    Each thread -_>> compute one element of dout (dt, ch)
    Grid.x = C, Grid.y = B * T
*/

kernel void encoder_backward_kernel(
    device atomic_float *dwte, // [V, C]
    device atomic_float *dwpe, // [T, C]
    device const float * dout, // [B*T, C]
    device const int *inp, // token IDs [B*T]
    constant const int &B,
    constant const int &T,
    constant const int &C,
    uint2 gid [[thread_position_in_grid]])
{
    uint i = gid.x; //channel (0 .. C-1)
    long bt = gid.y; // flattened (b, t) --> index (0...B*T - 1)
    if (i >= C || bt >= B*T) return;

    float d = dout[bt * C + i]; //upstream gradient

    //token embedding gradient
    int token_id = inp[bt]; //vocabulary index
    size_t wte_index = size_t(token_id) * C + i;
    atomic_fetch_add_explicit(&dwte[wte_index], d, memory_order_relaxed);

    // positional embedding gradient
    uint pos = bt % T; // 0 .. T -1
    size_t wpe_index = size_t(pos) * C + i;
    atomic_fetch_add_explicit(&dwpe[wpe_index], d, memory_order_relaxed);
}

kernel void adamw_kernel(
    device float *params,
    device const float *grads,
    device float *m_memory,
    device float *v_memory,
    constant const float &lr,
    constant const float &beta1,
    constant const float &beta2,
    constant const float &eps,
    constant const float &wd,
    constant const uint &num_params,
    constant const float &m_corr,
    constant const float &v_corr,
    uint gid [[thread_position_in_grid]])
{
    if (gid >= num_params) return;

    float p = params[gid];
    float g = grads [gid];
    float m = m_memory[gid];
    float v = v_memory[gid];

    if (!isfinite(p) || !isfinite(g) || !isfinite(m) || !isfinite(v)) return;

    m = fma(beta1, m, (1.0f - beta1) * g);
    v = fma(beta2, v, (1.0f - beta2) * (g * g));

    m_memory[gid] = m;
    v_memory[gid] = v;

    float m_hat = m * m_corr;
    float v_hat = v * v_corr;

    if (!isfinite(m_hat) || !isfinite(v_hat) || v_hat < 0.0f) return;

    float denom = sqrt(v_hat) + eps;
    if (!isfinite(denom) || denom < 1e-10f) return;

    float update = fma(wd, p, m_hat/denom);
    params[gid] = fma(-lr, update, p);
}

kernel void residual_backward_kernel(
    device float *dinp1,
    device float *dinp2,
    device const float *dout,
    constant const int &N,
    uint gid [[thread_position_in_grid]])
{
    if (gid >= N) return;

    float dy = dout [gid]; //upstream gradient
    dinp1[gid] += dy;
    dinp2[gid] += dy;
}

kernel void initialize_dlosses_kernel(
    device float * dlosses,
    constant const float &val,
    constant const int &N,
    uint gid [[thread_position_in_grid]])
{
    if (gid < N) dlosses[gid] = val;
}

kernel void sum_squares_kernel(
    device  const float*  grads                  [[buffer(0)]],
    device  atomic_float* result_sum_sq         [[buffer(1)]], // single float
    constant const uint&  N                     [[buffer(2)]], // total elements

    uint           tix                [[thread_index_in_threadgroup]],
    uint           gid                [[thread_position_in_grid]],
    ushort         block_size         [[threads_per_threadgroup]],
    uint           grid_size          [[threads_per_grid]],

    uint           simd_lane_id       [[thread_index_in_simdgroup]],
    uint           simd_group_id      [[simdgroup_index_in_threadgroup]],
    uint           simd_groups_per_block [[simdgroups_per_threadgroup]],

    threadgroup float* shared_mem     [[threadgroup(0)]])
{
       float thread_sum_sq = 0.0f;
    for (uint i = gid; i < N; i += grid_size) {
        float g = grads[i];
        if (isfinite(g)) thread_sum_sq = fma(g, g, thread_sum_sq);
    }

    float simd_group_sum = simd_sum(thread_sum_sq);        // built-in

    if (simd_lane_id == 0)
        shared_mem[simd_group_id] = simd_group_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float block_sum_sq = (tix < simd_groups_per_block) ? shared_mem[tix] : 0.0f;

    if (simd_group_id == 0)
        block_sum_sq = simd_sum(block_sum_sq);             // now lane 0 holds sum

    if (tix == 0 && isfinite(block_sum_sq))
        atomic_fetch_add_explicit(result_sum_sq,
                                  block_sum_sq,
                                  memory_order_relaxed);
}
