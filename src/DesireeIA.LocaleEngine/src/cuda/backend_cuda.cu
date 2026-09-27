// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

// CUDA backend - see docs/CUDAPiano.md.
//
// One kernel, for the Q8_0 format (the simplest: no nibble to unpack, the
// weights are already int8). Ports 1:1 the same math already validated on
// CPU in matmul_q8_0 (src/core/matmul.cpp): for every row, sum over every
// 32-element sub-block of
// (weight_scale[row,block] * activation_scale[block] * dot_int8(weights, activation)).
// No new algorithm, no invented format choice — just a device port of a
// kernel already correct and measured.
//
// HISTORY (2026-09-14): the first version of this file (matmul_q8_0_cuda,
// below) uploaded the ENTIRE weight matrix on every single call
// (cudaMalloc + memcpy H2D + kernel + memcpy D2H + cudaFree per matvec).
// Measured end to end against the CPU path (desireeia-cli bench, Qwen2.5-
// Coder-3B Q8_0): 28x SLOWER than CPU (0.49 vs 13.90 tok/s decode), the
// cause isolated with the profiler (almost all the time in
// "seriale_fra_dispatch", not the kernel itself: PCIe transfer overhead
// plus synchronous malloc/free repeated on every token, for every weight
// matrix). Added here are the "resident" functions (upload the weights
// ONCE, reuse them for the whole session): same principle as the CPU-side
// weight cache (LayerWeights/layer_cache_ in dense_forward.cpp), the
// missing piece already planned in docs/CUDAPiano.md section 4.

#include "../core/engine.h"
#include "../core/profile.h"
#include "../quant/quant.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <algorithm>

namespace desireeia {

namespace {

// Q8_0 mat-vec kernel. Input layout, prepared host-side by the upload
// functions below:
//   w_qs:     rows * nb * 32 int8   (quantized weights, row-major)
//   w_scale:  rows * nb float       (per row/block scale, fp16->fp32 already)
//   x_qs:     nb * 32 int8          (quantized activation, shared by all rows)
//   x_scale:  nb float              (per-block activation scale)
//
// Techniques ported from the external reference implementation and
// reimplemented here on our own layout — not copied. The first version of
// this kernel used scalar byte-by-byte loads and a shared-memory
// reduction, measured at ~56 GB/s asymptotic against the ~256 GB/s the
// card can do. The three things that matter:
//   1. weights are read as `int` (4 packed int8), not one byte at a time;
//   2. the dot product uses __dp4a (4 int8 MACs in one instruction)
//      instead of 4 separate multiplies;
//   3. the final reduction is per warp via __shfl_xor_sync, with no
//      shared memory and no __syncthreads.
// The scale is still applied ONCE per block of 32: same maths as the
// already validated CPU kernel, just executed better.
#define DESIREEIA_CUDA_WARP 32

static __device__ __forceinline__ int desireeia_dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    // Fallback for architectures without DP4A: same arithmetic, slower.
    const int8_t* pa = reinterpret_cast<const int8_t*>(&a);
    const int8_t* pb = reinterpret_cast<const int8_t*>(&b);
    #pragma unroll
    for (int i = 0; i < 4; ++i) c += (int) pa[i] * (int) pb[i];
    return c;
#endif
}

// One warp per pair of output rows, nwarps warps per block. Each thread
// of the warp takes blocks of 32 weights with a warp-sized stride, so the
// 32 threads read 1024 contiguous bytes per iteration (coalesced), each
// with 8 32-bit loads instead of 32 8-bit ones.
//
// TWO rows per warp: the activation is read once and serves both rows,
// and the final reductions are halved for the same useful work. An odd
// trailing row falls back to the single-row path.
template <int nwarps>
__global__ void matmul_q8_0_kernel(const int8_t* __restrict__ w_qs,
                                    const __half* __restrict__ w_scale,
                                    const int8_t* __restrict__ x_qs,
                                    const float* __restrict__ x_scale,
                                    size_t nb, size_t rows, float* __restrict__ y,
                                    const float* __restrict__ bias) {
    const size_t row0 = ((size_t) blockIdx.x * nwarps + threadIdx.y) * 2;
    if (row0 >= rows) return;
    const bool has_second = (row0 + 1) < rows;

    const int* w_row0 = reinterpret_cast<const int*>(w_qs + row0 * nb * 32);
    const int* w_row1 = has_second ? w_row0 + nb * 8 : w_row0;
    const __half* scale0 = w_scale + row0 * nb;
    const __half* scale1 = has_second ? scale0 + nb : scale0;
    const int* x_row = reinterpret_cast<const int*>(x_qs);

    float acc0 = 0.0f, acc1 = 0.0f;
    for (size_t b = threadIdx.x; b < nb; b += DESIREEIA_CUDA_WARP) {
        // 128-bit loads: a Q8_0 block is 32 int8 = 8 int = two int4, so the
        // whole block moves in two transactions instead of eight. Every
        // buffer comes from cudaMalloc and block offsets are multiples of
        // 32 bytes, so the wider load stays aligned.
        const int4* xv4 = reinterpret_cast<const int4*>(x_row + b * 8);
        const int4 x0 = xv4[0];
        const int4 x1 = xv4[1];
        const float xsc = x_scale[b];

        const int4* w04 = reinterpret_cast<const int4*>(w_row0 + b * 8);
        const int4 a0 = w04[0];
        const int4 a1 = w04[1];
        int sumi0 = desireeia_dp4a(a0.x, x0.x, 0);
        sumi0 = desireeia_dp4a(a0.y, x0.y, sumi0);
        sumi0 = desireeia_dp4a(a0.z, x0.z, sumi0);
        sumi0 = desireeia_dp4a(a0.w, x0.w, sumi0);
        sumi0 = desireeia_dp4a(a1.x, x1.x, sumi0);
        sumi0 = desireeia_dp4a(a1.y, x1.y, sumi0);
        sumi0 = desireeia_dp4a(a1.z, x1.z, sumi0);
        sumi0 = desireeia_dp4a(a1.w, x1.w, sumi0);
        acc0 += __half2float(scale0[b]) * xsc * (float) sumi0;

        if (has_second) {
            const int4* w14 = reinterpret_cast<const int4*>(w_row1 + b * 8);
            const int4 c0 = w14[0];
            const int4 c1 = w14[1];
            int sumi1 = desireeia_dp4a(c0.x, x0.x, 0);
            sumi1 = desireeia_dp4a(c0.y, x0.y, sumi1);
            sumi1 = desireeia_dp4a(c0.z, x0.z, sumi1);
            sumi1 = desireeia_dp4a(c0.w, x0.w, sumi1);
            sumi1 = desireeia_dp4a(c1.x, x1.x, sumi1);
            sumi1 = desireeia_dp4a(c1.y, x1.y, sumi1);
            sumi1 = desireeia_dp4a(c1.z, x1.z, sumi1);
            sumi1 = desireeia_dp4a(c1.w, x1.w, sumi1);
            acc1 += __half2float(scale1[b]) * xsc * (float) sumi1;
        }
    }

    #pragma unroll
    for (int off = DESIREEIA_CUDA_WARP / 2; off > 0; off >>= 1) {
        acc0 += __shfl_xor_sync(0xffffffff, acc0, off, DESIREEIA_CUDA_WARP);
        acc1 += __shfl_xor_sync(0xffffffff, acc1, off, DESIREEIA_CUDA_WARP);
    }
    if (threadIdx.x == 0) {
        // Bias folded into the epilogue instead of a separate kernel: on
        // this workload a launch costs more than the handful of adds it
        // would perform.
        y[row0] = bias ? acc0 + bias[row0] : acc0;
        if (has_second) y[row0 + 1] = bias ? acc1 + bias[row0 + 1] : acc1;
    }
}

// Gated FFN activation, done on device so gate and up never have to come
// back to the host. Replicates EXACTLY the two CPU variants in
// dense_forward.cpp: `ffn[i] = silu(gate[i]) * ffn[i]` and geglu_inplace
// (tanh-approximated GELU on the gate branch, then the product). The
// result ends up in `up_inout`, like the CPU version that writes in place
// onto `ffn`.
__global__ void ffn_act_kernel(float* __restrict__ up_inout, const float* __restrict__ gate,
                                size_t n, int act_gelu) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    float a;
    if (act_gelu) {
        // 0.5*g*(1+tanh(sqrt(2/pi)*g*(1+0.044715*g^2)))
        const float u = 0.7978845608028654f * g * (1.0f + 0.044715f * g * g);
        a = 0.5f * g * (1.0f + tanhf(u));
    } else {
        a = g / (1.0f + __expf(-g));  // silu
    }
    up_inout[i] = a * up_inout[i];
}

// Device-side Q8_0 quantization of the activation: same formula as
// quantize_q8_0 (core/quant.cpp) — per 32-block, d = max|v|/127 (0 if
// all-zero), q = clamp(rint(v/d), -127, 127), tail zeroed. One warp per
// block, absmax via shuffle. Needed because the FFN intermediate is
// already on device: bringing it back to the host just to quantize it
// would undo the whole gain.
__global__ void quantize_q8_0_kernel(const float* __restrict__ src, size_t n,
                                      int8_t* __restrict__ q, float* __restrict__ scales,
                                      int32_t* __restrict__ sums = nullptr) {
    const size_t b = blockIdx.x;
    const size_t idx = b * 32 + threadIdx.x;
    const float v = idx < n ? src[idx] : 0.0f;
    float m = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off, 32));
    }
    const float d = m > 0.0f ? m / 127.0f : 0.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    if (threadIdx.x == 0) scales[b] = d;
    int qi = idx < n ? (int) rintf(v * id) : 0;
    qi = min(127, max(-127, qi));
    q[idx < n ? idx : b * 32 + threadIdx.x] = (int8_t) qi;

    // Per-sub-block sum of the quantized activation. Q4_K needs it for the
    // minimum term, and it is the same for every output row, so computing
    // it once here beats recomputing it per row inside the mat-vec.
    if (sums) {
        int sv = qi;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) sv += __shfl_xor_sync(0xffffffff, sv, off, 32);
        if (threadIdx.x == 0) sums[b] = sv;
    }
}

// Causal attention with GQA, one block per head.
//
// Replicates exactly the float branch (non kv_quantized_, no ALiBi) of
// dense_forward.cpp: for every head h, hkv = h / (n_head/n_head_kv);
//   scores[cc] = dot(q_h, k[cc][hkv]) * inv_d   for cc in [cc_start, pos]
//   stable softmax over just the [cc_start, pos] window
//   out[d]    = sum_cc scores[cc] * v[cc][hkv][d]
// The KV cache layout is the same as the host's:
//   k[(layer*capacity + cc)*kv_dim + hkv*head_dim + d]
// so the device copy is a bit-for-bit mirror of the host one and the two
// can never diverge in layout.
//
// One kernel instead of three (scores / softmax / V accumulation) to
// avoid paying for three launches: the phases are separated by
// __syncthreads.
__global__ void attention_kernel(const float* __restrict__ q,
                                  const float* __restrict__ kcache,
                                  const float* __restrict__ vcache,
                                  float* __restrict__ scores,
                                  float* __restrict__ out,
                                  uint32_t n_head_kv, uint32_t head_dim,
                                  uint32_t heads_per_kv, uint32_t kv_dim,
                                  size_t layer_off, uint32_t cc_start, uint32_t pos,
                                  float inv_d) {
    extern __shared__ float smem[];
    const uint32_t h = blockIdx.x;
    const uint32_t hkv = h / heads_per_kv;
    const uint32_t n_cc = pos + 1 - cc_start;

    const float* qh = q + (size_t) h * head_dim;
    const float* kb = kcache + layer_off + (size_t) hkv * head_dim;
    const float* vb = vcache + layer_off + (size_t) hkv * head_dim;
    float* sc = scores + (size_t) h * n_cc;

    // Fase 1: punteggi, e massimo per il softmax stabile.
    float local_max = -INFINITY;
    for (uint32_t i = threadIdx.x; i < n_cc; i += blockDim.x) {
        const uint32_t cc = cc_start + i;
        const float* krow = kb + (size_t) cc * kv_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; ++d) dot += qh[d] * krow[d];
        const float s = dot * inv_d;
        sc[i] = s;
        local_max = fmaxf(local_max, s);
    }
    smem[threadIdx.x] = local_max;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        __syncthreads();
    }
    const float m = smem[0];
    __syncthreads();

    // Fase 2: exp e somma.
    float local_sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_cc; i += blockDim.x) {
        const float e = __expf(sc[i] - m);
        sc[i] = e;
        local_sum += e;
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv_sum = 1.0f / smem[0];
    __syncthreads();

    // Fase 3: combinazione di V. Ogni thread possiede un sottoinsieme
    // delle dimensioni d, cosi' l'accumulo non ha bisogno di riduzioni.
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t i = 0; i < n_cc; ++i) {
            acc += sc[i] * vb[(size_t) (cc_start + i) * kv_dim + d];
        }
        out[(size_t) h * head_dim + d] = acc * inv_sum;
    }
}

// Attention over the Q8_0-quantized KV cache (the default layout when
// plan.kv_compression is on). Host layout, mirrored bit-for-bit:
//   row(cc, hkv) = base + cc*pos_bytes + hkv*row_bytes
// each row holding head_dim/32 block_q8_0 (fp16 scale + 32 int8), with the
// query kept in float — the same convention as kv_dot_q8_0 / kv_axpy_q8_0
// in kv/kv_quant.cpp, which define the reference behaviour on CPU.
//
// Structure ported from the external reference implementation's
// decode-oriented attention kernel and reimplemented here on our own
// layout. The first version of this kernel was
// written without looking at it and was SLOWER than the CPU attention it
// replaced (3.4 ms/token vs 1.7); the three things it got wrong, and that
// the reference gets right, are:
//   1. the V accumulator lives in REGISTERS, one slice of head_dim per
//      thread, instead of being recomputed out of a global scores array;
//   2. the softmax is ONLINE (running max and sum, rescaling the
//      accumulator when the max grows), so no scores array is ever
//      materialised in global memory;
//   3. the KV positions are the outer loop and, in the V phase, adjacent
//      threads read adjacent head dimensions of the SAME row, which is
//      coalesced — the first version had each thread walk one column with
//      a pos_bytes stride, so every single read hit a different cache line.
// A thread owns head dimensions d = tid, tid+nthreads, ... hence the small
// fixed-size accumulator array below.
#define DESIREEIA_ATTN_MAX_ACC 4

// Prefill attention: one block per (token, head) pair, the whole batch in
// a single launch.
//
// Same arithmetic as attention_kernel_q8 below (online softmax over the
// Q8_0 KV cache), with two differences that come from processing a batch
// rather than one position:
//   - `pos` is derived from the block's token index instead of being read
//     from device memory, since every token of the batch has a different
//     one;
//   - the sliding window is applied by passing n_swa = 0 on layers that
//     don't use it, so the caller doesn't need a second kernel.
//
// Why it exists: profiled on a 601-token prompt, prefill spent 3.9 s in
// GPU kernels and ~13 s on the CPU, nearly all of it this attention. It
// is also the only O(n^2) part of prefill, so it is what makes a long
// prompt feel like a freeze.
__global__ void attention_batch_kernel_q8(const float* __restrict__ q_all,
                                           const uint8_t* __restrict__ kcache,
                                           const uint8_t* __restrict__ vcache,
                                           float* __restrict__ out_all,
                                           uint32_t head_dim, uint32_t heads_per_kv,
                                           uint32_t q_dim,
                                           size_t row_bytes, size_t pos_bytes,
                                           size_t layer_off,
                                           uint32_t col0, uint32_t n_swa,
                                           float inv_d) {
    const uint32_t h = blockIdx.x;
    const uint32_t p = blockIdx.y;
    const uint32_t pos = col0 + p;
    const uint32_t cc_start = (n_swa > 0 && pos + 1 > n_swa) ? (pos + 1 - n_swa) : 0;

    extern __shared__ float bsmem[];
    float* sh_score = bsmem;
    float* sh_red   = bsmem + blockDim.x;

    const uint32_t hkv = h / heads_per_kv;
    const uint32_t tid = threadIdx.x;
    const uint32_t nthreads = blockDim.x;
    const uint32_t nblk = head_dim / 32;

    const float* qh = q_all + (size_t) p * q_dim + (size_t) h * head_dim;
    const uint8_t* kb = kcache + layer_off + (size_t) hkv * row_bytes;
    const uint8_t* vb = vcache + layer_off + (size_t) hkv * row_bytes;

    float acc[DESIREEIA_ATTN_MAX_ACC];
    #pragma unroll
    for (int a = 0; a < DESIREEIA_ATTN_MAX_ACC; ++a) acc[a] = 0.0f;

    float run_max = -INFINITY;
    float run_sum = 0.0f;

    for (uint32_t base = cc_start; base <= pos; base += nthreads) {
        const uint32_t cc = base + tid;
        const bool active = cc <= pos;

        float sc = -INFINITY;
        if (active) {
            const uint8_t* row = kb + (size_t) cc * pos_bytes;
            float dot = 0.0f;
            for (uint32_t b = 0; b < nblk; ++b) {
                const uint8_t* blk = row + (size_t) b * 34;
                const float d = __half2float(*reinterpret_cast<const __half*>(blk));
                const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
                float bd = 0.0f;
                #pragma unroll
                for (int j = 0; j < 32; ++j) bd += qh[b * 32 + j] * (float) qs[j];
                dot += d * bd;
            }
            sc = dot * inv_d;
        }
        sh_red[tid] = sc;
        __syncthreads();
        for (unsigned stride = nthreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) sh_red[tid] = fmaxf(sh_red[tid], sh_red[tid + stride]);
            __syncthreads();
        }
        const float chunk_max = sh_red[0];
        __syncthreads();

        const float new_max = fmaxf(run_max, chunk_max);
        const float rescale = (run_max == -INFINITY) ? 0.0f : __expf(run_max - new_max);
        const float e = active ? __expf(sc - new_max) : 0.0f;
        sh_score[tid] = e;
        sh_red[tid] = e;
        __syncthreads();
        for (unsigned stride = nthreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) sh_red[tid] += sh_red[tid + stride];
            __syncthreads();
        }
        const float chunk_sum = sh_red[0];
        run_sum = run_sum * rescale + chunk_sum;
        run_max = new_max;

        const uint32_t n_in_chunk = min(nthreads, pos + 1 - base);
        for (uint32_t d = tid, a = 0; d < head_dim; d += nthreads, ++a) {
            const uint32_t b = d / 32;
            const uint32_t j = d % 32;
            float sum = 0.0f;
            for (uint32_t i = 0; i < n_in_chunk; ++i) {
                const uint8_t* blk = vb + (size_t) (base + i) * pos_bytes + (size_t) b * 34;
                const float dq = __half2float(*reinterpret_cast<const __half*>(blk));
                const int8_t qv = reinterpret_cast<const int8_t*>(blk + 2)[j];
                sum += sh_score[i] * dq * (float) qv;
            }
            acc[a] = acc[a] * rescale + sum;
        }
        __syncthreads();
    }

    const float inv_sum = 1.0f / run_sum;
    float* out = out_all + (size_t) p * q_dim + (size_t) h * head_dim;
    for (uint32_t d = tid, a = 0; d < head_dim; d += nthreads, ++a) {
        out[d] = acc[a] * inv_sum;
    }
}

__global__ void attention_kernel_q8(const float* __restrict__ q,
                                     const uint8_t* __restrict__ kcache,
                                     const uint8_t* __restrict__ vcache,
                                     float* __restrict__ out,
                                     uint32_t head_dim, uint32_t heads_per_kv,
                                     size_t row_bytes, size_t pos_bytes,
                                     size_t layer_off, const uint32_t* __restrict__ dyn,
                                     float inv_d) {
    // pos and cc_start live in device memory, not in kernel arguments, so
    // the enclosing graph stays static across tokens and never needs to be
    // re-captured or updated.
    const uint32_t pos = dyn[0];
    const uint32_t cc_start = dyn[1];
    extern __shared__ float smem[];
    float* sh_score = smem;                    // one score per thread of the chunk
    float* sh_red   = smem + blockDim.x;       // scratch for the two reductions

    const uint32_t h = blockIdx.x;
    const uint32_t hkv = h / heads_per_kv;
    const uint32_t tid = threadIdx.x;
    const uint32_t nthreads = blockDim.x;
    const uint32_t nblk = head_dim / 32;

    const float* qh = q + (size_t) h * head_dim;
    const uint8_t* kb = kcache + layer_off + (size_t) hkv * row_bytes;
    const uint8_t* vb = vcache + layer_off + (size_t) hkv * row_bytes;

    float acc[DESIREEIA_ATTN_MAX_ACC];
    #pragma unroll
    for (int a = 0; a < DESIREEIA_ATTN_MAX_ACC; ++a) acc[a] = 0.0f;

    float run_max = -INFINITY;
    float run_sum = 0.0f;

    for (uint32_t base = cc_start; base <= pos; base += nthreads) {
        const uint32_t cc = base + tid;
        const bool active = cc <= pos;

        // Stage A: one KV position per thread -> its own score.
        float sc = -INFINITY;
        if (active) {
            const uint8_t* row = kb + (size_t) cc * pos_bytes;
            float dot = 0.0f;
            for (uint32_t b = 0; b < nblk; ++b) {
                const uint8_t* blk = row + (size_t) b * 34;
                const float d = __half2float(*reinterpret_cast<const __half*>(blk));
                const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
                float bd = 0.0f;
                #pragma unroll
                for (int j = 0; j < 32; ++j) bd += qh[b * 32 + j] * (float) qs[j];
                dot += d * bd;
            }
            sc = dot * inv_d;
        }
        sh_red[tid] = sc;
        __syncthreads();
        for (unsigned stride = nthreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) sh_red[tid] = fmaxf(sh_red[tid], sh_red[tid + stride]);
            __syncthreads();
        }
        const float chunk_max = sh_red[0];
        __syncthreads();

        // Stage B: online softmax. Growing the running max rescales what
        // has already been accumulated, instead of keeping every score.
        const float new_max = fmaxf(run_max, chunk_max);
        const float rescale = (run_max == -INFINITY) ? 0.0f : __expf(run_max - new_max);
        const float e = active ? __expf(sc - new_max) : 0.0f;
        sh_score[tid] = e;
        sh_red[tid] = e;
        __syncthreads();
        for (unsigned stride = nthreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) sh_red[tid] += sh_red[tid + stride];
            __syncthreads();
        }
        const float chunk_sum = sh_red[0];
        run_sum = run_sum * rescale + chunk_sum;
        run_max = new_max;

        // Stage C: accumulate V. Adjacent threads read adjacent head
        // dimensions of the same row -> coalesced.
        const uint32_t n_in_chunk = min(nthreads, pos + 1 - base);
        for (uint32_t d = tid, a = 0; d < head_dim; d += nthreads, ++a) {
            const uint32_t b = d / 32;
            const uint32_t j = d % 32;
            float sum = 0.0f;
            for (uint32_t i = 0; i < n_in_chunk; ++i) {
                const uint8_t* blk = vb + (size_t) (base + i) * pos_bytes + (size_t) b * 34;
                const float dq = __half2float(*reinterpret_cast<const __half*>(blk));
                const int8_t qv = reinterpret_cast<const int8_t*>(blk + 2)[j];
                sum += sh_score[i] * dq * (float) qv;
            }
            acc[a] = acc[a] * rescale + sum;
        }
        __syncthreads();
    }

    const float inv_sum = 1.0f / run_sum;
    for (uint32_t d = tid, a = 0; d < head_dim; d += nthreads, ++a) {
        out[(size_t) h * head_dim + d] = acc[a] * inv_sum;
    }
}


// ---------------------------------------------------------------------
// Decode attention, split over the keys (sm_70+ shuffles, any GPU).
//
// attention_kernel_q8 gives each thread one cache position: the thread
// walks all head_dim values of that key alone (uncoalesced byte loads),
// then every thread walks every key of the chunk again for V, and there
// is one block per head - 8 blocks on a 20-SM GPU for a 4B model. Here a
// WARP handles a key: lane l reads byte l of each 32-block (32 adjacent
// bytes, one transaction), a shuffle reduction gives the score, and the
// same lanes accumulate V for dimensions l, l+32, ... Each warp keeps its
// own online softmax; the 8 warps of a block are merged in shared memory,
// and the key range is split over `nsplit` blocks per head (merged by
// attn_dec_merge_kernel) so every SM has work.
constexpr int kAdecWarps = 8;
constexpr int kAdecMaxSplit = 16;
constexpr int kAdecMaxBlk = 8;                // head_dim up to 256

__global__ void __launch_bounds__(kAdecWarps * 32) attn_dec_part_kernel(
        const float* __restrict__ q, const uint8_t* __restrict__ kcache, const uint8_t* __restrict__ vcache,
        float* __restrict__ part, uint32_t head_dim, uint32_t heads_per_kv, size_t row_bytes, size_t pos_bytes,
        size_t layer_off, const uint32_t* __restrict__ dyn, float inv_d, uint32_t nsplit) {
    const uint32_t pos = dyn[0], cc_start = dyn[1];
    const uint32_t h = blockIdx.x, sp = blockIdx.y;
    const uint32_t hkv = h / heads_per_kv;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const uint32_t nblk = head_dim / 32;
    const uint32_t total = pos + 1 - cc_start;
    const uint32_t per = (total + nsplit - 1) / nsplit;
    const uint32_t beg = cc_start + sp * per;
    const uint32_t end = min(pos + 1, beg + per);

    const uint8_t* kb = kcache + layer_off + (size_t) hkv * row_bytes;
    const uint8_t* vb = vcache + layer_off + (size_t) hkv * row_bytes;
    float qv[kAdecMaxBlk], acc[kAdecMaxBlk];
    #pragma unroll
    for (int i = 0; i < kAdecMaxBlk; ++i) {
        qv[i] = i < (int) nblk ? q[(size_t) h * head_dim + i * 32 + lane] * inv_d : 0.0f;
        acc[i] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;
    for (uint32_t key = beg + warp; key < end; key += kAdecWarps) {
        const uint8_t* kr = kb + (size_t) key * pos_bytes;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < kAdecMaxBlk; ++i) {
            if (i < (int) nblk) {
                const uint8_t* blk = kr + i * 34;
                dot += __half2float(*reinterpret_cast<const __half*>(blk)) * qv[i] *
                       (float) reinterpret_cast<const int8_t*>(blk + 2)[lane];
            }
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, o);
        const float mn = fmaxf(m, dot);
        const float c = __expf(m - mn), pr = __expf(dot - mn);
        l = l * c + pr;
        m = mn;
        const uint8_t* vr = vb + (size_t) key * pos_bytes;
        #pragma unroll
        for (int i = 0; i < kAdecMaxBlk; ++i) {
            if (i < (int) nblk) {
                const uint8_t* blk = vr + i * 34;
                acc[i] = acc[i] * c + pr * __half2float(*reinterpret_cast<const __half*>(blk)) *
                         (float) reinterpret_cast<const int8_t*>(blk + 2)[lane];
            }
        }
    }

    // merge the warps: [warp] m, l and [warp][head_dim] acc
    __shared__ float sm_m[kAdecWarps], sm_l[kAdecWarps];
    __shared__ float sm_acc[kAdecWarps][kAdecMaxBlk * 32];
    if (lane == 0) { sm_m[warp] = m; sm_l[warp] = l; }
    #pragma unroll
    for (int i = 0; i < kAdecMaxBlk; ++i) if (i < (int) nblk) sm_acc[warp][i * 32 + lane] = acc[i];
    __syncthreads();
    float M = -INFINITY;
    #pragma unroll
    for (int w = 0; w < kAdecWarps; ++w) M = fmaxf(M, sm_m[w]);
    float* out = part + ((size_t) h * nsplit + sp) * (head_dim + 2);
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float a = 0.0f;
        #pragma unroll
        for (int w = 0; w < kAdecWarps; ++w) {
            if (sm_m[w] != -INFINITY) a += sm_acc[w][d] * __expf(sm_m[w] - M);
        }
        out[2 + d] = a;
    }
    if (threadIdx.x == 0) {
        float L = 0.0f;
        for (int w = 0; w < kAdecWarps; ++w) if (sm_m[w] != -INFINITY) L += sm_l[w] * __expf(sm_m[w] - M);
        out[0] = M;
        out[1] = L;
    }
}

__global__ void attn_dec_merge_kernel(const float* __restrict__ part, float* __restrict__ out,
                                      uint32_t head_dim, uint32_t nsplit) {
    const uint32_t h = blockIdx.x;
    const float* p = part + (size_t) h * nsplit * (head_dim + 2);
    float M = -INFINITY;
    for (uint32_t s = 0; s < nsplit; ++s) M = fmaxf(M, p[s * (head_dim + 2)]);
    float L = 0.0f;
    for (uint32_t s = 0; s < nsplit; ++s) {
        const float ms = p[s * (head_dim + 2)];
        if (ms != -INFINITY) L += p[s * (head_dim + 2) + 1] * __expf(ms - M);
    }
    const float inv = L > 0.0f ? 1.0f / L : 0.0f;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float a = 0.0f;
        for (uint32_t s = 0; s < nsplit; ++s) {
            const float ms = p[s * (head_dim + 2)];
            if (ms != -INFINITY) a += p[s * (head_dim + 2) + 2 + d] * __expf(ms - M);
        }
        out[(size_t) h * head_dim + d] = a * inv;
    }
}


// Decode attention on the Q8_0 cache: the split kernel when its scratch is
// there and the head fits, otherwise attention_kernel_q8. Graph-safe: the
// split count depends only on the shape, positions come from `dyn`.
static void attention_decode_q8(const float* q, const uint8_t* kc, const uint8_t* vc, float* out,
                                uint32_t n_head, uint32_t head_dim, uint32_t heads_per_kv, size_t row_bytes,
                                size_t pos_bytes, size_t layer_off, const uint32_t* dyn, float inv_d,
                                cudaStream_t stream);
// ---------------------------------------------------------------------
// K-quant kernels (Q4_K, Q6_K).
//
// Unlike Q8_0 these keep their NATIVE on-disk layout in VRAM and are
// decoded by the kernel. That is the point: Q4_K is 144 bytes per 256
// weights (4.5 bits/weight), so widening it to Q8_0 at upload time would
// double both the VRAM footprint and — decode being bandwidth-bound —
// the time per token. Block sizes (144 and 210 bytes) are multiples of
// 16, so rows stay aligned.
//
// The arithmetic mirrors the already validated CPU kernels and the
// dequantisers in quant/quant.cpp, whose exact index mapping was read
// rather than reconstructed: a first attempt at Q6_K written from memory
// had the interleaving wrong in three separate places.


// Reads a 32-bit word from data that is only guaranteed 2-byte aligned,
// as two 16-bit loads. Needed because a Q6_K block is 210 bytes, so every
// other block starts at an odd multiple of 2 and a plain int load would be
// undefined behaviour. Technique taken from the external reference.
static __device__ __forceinline__ int load_int_b2(const void* x, int i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    return ((int) x16[2 * i32 + 0]) | (((int) x16[2 * i32 + 1]) << 16);
}

static __device__ __forceinline__ void get_scale_min_k4_dev(int j, const uint8_t* q,
                                                             uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}


// Unpacks the eight 6-bit scales and eight 6-bit minimums of a Q4_K/Q5_K
// super-block, two at a time, out of 16-bit words.
//
// The straightforward per-index unpacking (get_scale_min_k4_dev) has a
// branch and several shifts, and running it eight times per super-block
// left the kernel limited by scalar work rather than by memory — the same
// effect our CPU kernel documents. Reading the packed scales as uint16
// pairs, as the external reference does, produces two scales and two
// minimums with a couple of masks. Layout, from the packing routine:
//   bytes 0-3   low 6 bits: scale[i]   top 2 bits: scale[i+4] high bits
//   bytes 4-7   low 6 bits: min[i]     top 2 bits: min[i+4]   high bits
//   bytes 8-11  low nibble: scale[i+4] high nibble: min[i+4]
static __device__ __forceinline__ void get_scales_pair_k4(int pair, const uint8_t* q,
                                                           uint8_t* sc, uint8_t* mn) {
    const uint16_t* q16 = (const uint16_t*) q;   // scales are 2-byte aligned
    uint16_t a0, a1;
    if (pair < 2) {
        a0 = q16[pair + 0] & 0x3f3f;
        a1 = q16[pair + 2] & 0x3f3f;
    } else {
        a0 = ((q16[pair + 2] >> 0) & 0x0f0f) | ((q16[pair - 2] & 0xc0c0) >> 2);
        a1 = ((q16[pair + 2] >> 4) & 0x0f0f) | ((q16[pair - 0] & 0xc0c0) >> 2);
    }
    sc[0] = (uint8_t) (a0 & 0xff);
    sc[1] = (uint8_t) (a0 >> 8);
    mn[0] = (uint8_t) (a1 & 0xff);
    mn[1] = (uint8_t) (a1 >> 8);
}

// y = W * x with W in Q4_K.
//
// Per super-block of 256: eight sub-blocks of 32, each with a 6-bit scale
// and a 6-bit minimum. The packed nibbles hold two sub-blocks per 32
// bytes — low nibbles are sub-block 2g, high nibbles 2g+1 — which matches
// dequantize_row_q4_K exactly.
//
//   value = d*sc_j*q - dmin*m_j
// so the row dot splits into
//   d * sum_j sc_j*<q_j, a_j>  -  dmin * sum_j m_j*sum(a_j)
// The second term only needs the per-sub-block sum of the activation,
// which is identical for every row and so is computed once (x_sum), the
// same trick the CPU kernel uses.
// The per-row-group body of the Q4_K mat-vec, shared by the single-matrix
// kernel and the grouped one below so the arithmetic has exactly one
// definition.
//
// Work is split per SUB-BLOCK of 32 weights, not per super-block of 256:
// a super-block covers 256 columns, so a 2560-wide matrix has only ten
// per row, and one-per-thread left 22 of 32 threads idle. That single
// change took this kernel from a third of the bandwidth the Q8_0 one
// reaches to parity of occupancy.
//
// FOUR rows per warp on top of that: the activation word and its
// scale/sum are read once and serve all four rows, and the final
// reductions are paid once per four rows instead of once per row.
static __device__ __forceinline__ void q4k_row_group(
        const uint8_t* __restrict__ w,
        const int8_t* __restrict__ x_qs,
        const float* __restrict__ x_scale,
        const int32_t* __restrict__ x_sum,
        size_t n_super, size_t rows,
        float* __restrict__ y,
        const float* __restrict__ bias,
        size_t row0) {
    constexpr int R = 4;
    if (row0 >= rows) return;
    const int nrow = (int) min((size_t) R, rows - row0);

    const size_t row_bytes = n_super * 144;
    const uint8_t* wr[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        wr[r] = w + (row0 + (size_t) (r < nrow ? r : 0)) * row_bytes;
    }

    const size_t n_sub = n_super * 8;
    float acc[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;

    for (size_t sub = threadIdx.x; sub < n_sub; sub += DESIREEIA_CUDA_WARP) {
        const size_t sblk = sub >> 3;
        const int j = (int) (sub & 7);
        const int shift = (j & 1) ? 4 : 0;
        const int qoff = (j / 2) * 8;
        const int* a = reinterpret_cast<const int*>(x_qs + sub * 32);
        const float xs = x_scale[sub];
        const float xsum = (float) x_sum[sub];

        int av[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) av[k] = a[k];

        #pragma unroll
        for (int r = 0; r < R; ++r) {
            if (r >= nrow) continue;
            // 144-byte blocks are 16-byte aligned: the header (d, dmin and
            // the 12 scale bytes) in one 16-byte load, the 32 bytes of this
            // pair of sub-blocks in two - same values as byte-wise loads.
            const uint8_t* b = wr[r] + sblk * 144;
            const int4 hdr = *reinterpret_cast<const int4*>(b);
            const int4* q4 = reinterpret_cast<const int4*>(b + 16) + (qoff >> 2);
            const int4 qa = q4[0], qb = q4[1];
            const int qs[8] = { qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w };
            // q16[i] of get_scales_pair_k4: the i-th 16-bit word of bytes 4..15
            auto q16 = [&](int i) -> uint32_t {
                const uint32_t w = (uint32_t) (i < 2 ? hdr.y : (i < 4 ? hdr.z : hdr.w));
                return (i & 1) ? (w >> 16) : (w & 0xffffu);
            };
            const int pair = j / 2;
            uint32_t a0, a1;
            if (pair < 2) {
                a0 = q16(pair + 0) & 0x3f3fu;
                a1 = q16(pair + 2) & 0x3f3fu;
            } else {
                a0 = ((q16(pair + 2) >> 0) & 0x0f0fu) | ((q16(pair - 2) & 0xc0c0u) >> 2);
                a1 = ((q16(pair + 2) >> 4) & 0x0f0fu) | ((q16(pair - 0) & 0xc0c0u) >> 2);
            }
            const uint32_t scj = (j & 1) ? ((a0 >> 8) & 0xffu) : (a0 & 0xffu);
            const uint32_t mnj = (j & 1) ? ((a1 >> 8) & 0xffu) : (a1 & 0xffu);
            int dot = 0;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                dot = desireeia_dp4a((qs[k] >> shift) & 0x0F0F0F0F, av[k], dot);
            }
            const float d    = __half2float(__ushort_as_half((unsigned short) ((uint32_t) hdr.x & 0xffffu)));
            const float dmin = __half2float(__ushort_as_half((unsigned short) ((uint32_t) hdr.x >> 16)));
            acc[r] += xs * (d * (float) scj * (float) dot -
                            dmin * (float) mnj * xsum);
        }
    }

    #pragma unroll
    for (int r = 0; r < R; ++r) {
        #pragma unroll
        for (int off = DESIREEIA_CUDA_WARP / 2; off > 0; off >>= 1) {
            acc[r] += __shfl_xor_sync(0xffffffff, acc[r], off, DESIREEIA_CUDA_WARP);
        }
    }
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            if (r < nrow) {
                const size_t rr = row0 + (size_t) r;
                y[rr] = bias ? acc[r] + bias[rr] : acc[r];
            }
        }
    }
}

template <int nwarps>
__global__ void matmul_q4_k_kernel(const uint8_t* __restrict__ w,
                                    const int8_t* __restrict__ x_qs,
                                    const float* __restrict__ x_scale,
                                    const int32_t* __restrict__ x_sum,
                                    size_t n_super, size_t rows,
                                    float* __restrict__ y,
                                    const float* __restrict__ bias) {
    q4k_row_group(w, x_qs, x_scale, x_sum, n_super, rows, y, bias,
                  ((size_t) blockIdx.x * nwarps + threadIdx.y) * 4);
}

// Several mat-vecs that share the activation, the format and the column
// count, in ONE launch over one grid.
//
// Why: a decode step is a chain of small mat-vecs, and a small one does
// not fill the device. The K/V projections of this model are 1024 rows,
// which is 64 blocks over 20 SMs, and they were three separate launches
// that could not overlap because the graph serialises them. Merged, the
// grid is the sum and the device stays busy. The arithmetic per row is
// byte-for-byte the one above: this is a scheduling change, not a
// numerical one.
struct KQuantGroupDesc {
    const uint8_t* w[4];
    float* y[4];
    const float* bias[4];
    uint32_t rows[4];
    uint32_t blk0[4];   // first grid block that serves matrix i
    int n;
    int fmt[4];         // per matrix, for the mixed-format group only
};

static __device__ __forceinline__ int kquant_group_pick(const KQuantGroupDesc& g) {
    int m = 0;
    for (int i = 1; i < g.n; ++i) {
        if (blockIdx.x >= g.blk0[i]) m = i;
    }
    return m;
}

template <int nwarps>
__global__ void matmul_q4_k_group_kernel(KQuantGroupDesc g,
                                          const int8_t* __restrict__ x_qs,
                                          const float* __restrict__ x_scale,
                                          const int32_t* __restrict__ x_sum,
                                          size_t n_super) {
    const int m = kquant_group_pick(g);
    const size_t lb = blockIdx.x - g.blk0[m];
    q4k_row_group(g.w[m], x_qs, x_scale, x_sum, n_super, g.rows[m],
                  g.y[m], g.bias[m], (lb * nwarps + threadIdx.y) * 4);
}

// Prefill: one weight read serving MANY activation columns.
//
// The decode kernels above are mat-VEC: one column, so every weight byte
// is read once and used once, and the whole thing is bandwidth-bound.
// Prefill has hundreds of columns available at once, and running it as
// hundreds of independent mat-vecs re-reads the entire weight matrix once
// per token — which is exactly why prefill measured SLOWER per token than
// decode (30 vs 58 tok/s on gemma-2b) despite being the part that can be
// batched.
//
// Here each warp decodes a weight sub-block ONCE into registers and then
// multiplies it against TOK activation columns, cutting weight traffic by
// a factor of TOK. Columns beyond n_tok in the last tile are simply
// skipped, so no padding of the activation buffer is needed.
//
// Layouts: x_qs/x_scale/x_sum are [token][sub-block] (exactly what
// quantize_q8_0_kernel produces over a flat n_tok*cols activation), and y
// is [token][row], which is the layout matvec_batch's callers expect.
template <int nwarps, int TOK>
__global__ void matmul_q4_k_batch_kernel(const uint8_t* __restrict__ w,
                                          const int8_t* __restrict__ x_qs,
                                          const float* __restrict__ x_scale,
                                          const int32_t* __restrict__ x_sum,
                                          size_t n_super, size_t rows,
                                          uint32_t n_tok,
                                          float* __restrict__ y) {
    const size_t row = (size_t) blockIdx.x * nwarps + threadIdx.y;
    if (row >= rows) return;
    const uint32_t t0 = blockIdx.y * TOK;
    const size_t n_sub = n_super * 8;
    const uint8_t* wr = w + row * n_super * 144;

    float acc[TOK];
    #pragma unroll
    for (int t = 0; t < TOK; ++t) acc[t] = 0.0f;

    for (size_t sub = threadIdx.x; sub < n_sub; sub += DESIREEIA_CUDA_WARP) {
        const size_t sblk = sub >> 3;
        const int j = (int) (sub & 7);
        const int shift = (j & 1) ? 4 : 0;
        const int qoff = (j / 2) * 8;

        const uint8_t* b = wr + sblk * 144;
        uint8_t scv[2], mnv[2];
        get_scales_pair_k4(j / 2, b + 4, scv, mnv);
        const int* qs = reinterpret_cast<const int*>(b + 16) + qoff;

        // Decoded once, reused by every column in this tile.
        int wv[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) wv[k] = (qs[k] >> shift) & 0x0F0F0F0F;
        const float d    = __half2float(*reinterpret_cast<const __half*>(b));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(b + 2));
        const float wsc = d * (float) scv[j & 1];
        const float wmn = dmin * (float) mnv[j & 1];

        #pragma unroll
        for (int t = 0; t < TOK; ++t) {
            const uint32_t tk = t0 + (uint32_t) t;
            if (tk >= n_tok) continue;
            const size_t idx = (size_t) tk * n_sub + sub;
            const int* a = reinterpret_cast<const int*>(x_qs + idx * 32);
            int dot = 0;
            #pragma unroll
            for (int k = 0; k < 8; ++k) dot = desireeia_dp4a(wv[k], a[k], dot);
            acc[t] += x_scale[idx] * (wsc * (float) dot - wmn * (float) x_sum[idx]);
        }
    }

    #pragma unroll
    for (int t = 0; t < TOK; ++t) {
        #pragma unroll
        for (int off = DESIREEIA_CUDA_WARP / 2; off > 0; off >>= 1) {
            acc[t] += __shfl_xor_sync(0xffffffff, acc[t], off, DESIREEIA_CUDA_WARP);
        }
    }
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int t = 0; t < TOK; ++t) {
            const uint32_t tk = t0 + (uint32_t) t;
            if (tk < n_tok) y[(size_t) tk * rows + row] = acc[t];
        }
    }
}

// y = W * x with W in Q6_K.
//
// Index mapping taken from dequantize_row_q6_K: per half of 128 weights
// and for l in [0,32),
//   out[l +  0] = (ql[l]    & 0xF) | ((qh[l] >> 0) & 3) << 4, scale sc[l/16 + 0]
//   out[l + 32] = (ql[l+32] & 0xF) | ((qh[l] >> 2) & 3) << 4, scale sc[l/16 + 2]
//   out[l + 64] = (ql[l]    >> 4)  | ((qh[l] >> 4) & 3) << 4, scale sc[l/16 + 4]
//   out[l + 96] = (ql[l+32] >> 4)  | ((qh[l] >> 6) & 3) << 4, scale sc[l/16 + 6]
// each minus 32, then scaled by d. No minimum term.
//
// Four weights at a time: the low nibbles, the two high bits shifted into
// place and the -32 offset are all applied to a packed word (__vsubss4),
// then one dp4a against the activation. A first scalar version of this
// kernel measured ~18 GB/s and made the output head of a Q4_K_M model
// cost 21 ms per token on its own.
// Eight consecutive 32-bit words from a 2-byte-aligned address (a Q6_K
// block is 210 bytes, so half of them start 2 bytes past a word): nine
// aligned loads and a funnel shift instead of sixteen 16-bit loads. The
// ninth word stays inside the block - ql/qh are always followed by more
// of it (qh, the scales) - so nothing past the tensor is read.
static __device__ __forceinline__ void load8_b2(const uint8_t* p, int (&v)[8]) {
    const uint32_t* base = reinterpret_cast<const uint32_t*>(reinterpret_cast<uintptr_t>(p) & ~(uintptr_t) 3);
    const uint32_t sh = (uint32_t) (reinterpret_cast<uintptr_t>(p) & 2) * 8;
    uint32_t w[9];
    #pragma unroll
    for (int i = 0; i < 9; ++i) w[i] = base[i];
    #pragma unroll
    for (int k = 0; k < 8; ++k) v[k] = (int) __funnelshift_r(w[k], w[k + 1], sh);
}

// Shared body of the Q6_K mat-vec, for the same reason as q4k_row_group.
//
// Same two changes as the Q4_K kernel: work split per sub-block of 32
// weights rather than per super-block of 256 (a 2560-wide matrix has
// only ten super-blocks per row, which left most of the warp idle),
// and two rows per warp so each activation word serves both.
//
// The sub-block index maps onto the format as half = sub/4, group =
// sub%4, which is the traversal dequantize_row_q6_K defines.
static __device__ __forceinline__ void q6k_row_group(
        const uint8_t* __restrict__ w,
        const int8_t* __restrict__ x_qs,
        const float* __restrict__ x_scale,
        size_t n_super, size_t rows,
        float* __restrict__ y,
        const float* __restrict__ bias,
        size_t row0) {
    if (row0 >= rows) return;
    const bool has_second = (row0 + 1) < rows;

    const uint8_t* w_row0 = w + row0 * n_super * 210;
    const uint8_t* w_row1 = has_second ? w_row0 + n_super * 210 : w_row0;

    const size_t n_sub = n_super * 8;
    float acc0 = 0.0f, acc1 = 0.0f;

    for (size_t sub = threadIdx.x; sub < n_sub; sub += DESIREEIA_CUDA_WARP) {
        const size_t s = sub >> 3;
        const int idx = (int) (sub & 7);
        const int half = idx >> 2;
        const int g = idx & 3;

        const uint8_t* b0 = w_row0 + s * 210;
        const uint8_t* b1 = w_row1 + s * 210;
        const int* a = reinterpret_cast<const int*>(x_qs + sub * 32);
        const int nib_shift = (g < 2) ? 0 : 4;
        const int h_shift = 2 * g;

        int d0lo = 0, d0hi = 0, d1lo = 0, d1hi = 0;
        {
            const uint8_t* ql = b0 + half * 64 + ((g & 1) ? 32 : 0);
            const uint8_t* qh = b0 + 128 + half * 32;
            int vls[8], vhs[8];
            load8_b2(ql, vls);
            load8_b2(qh, vhs);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int vl = vls[k];
                const int vh = vhs[k];
                const int vi = __vsubss4((((vl >> nib_shift) & 0x0F0F0F0F) |
                                          ((((vh >> h_shift) & 0x03030303)) << 4)),
                                         0x20202020);
                if (k < 4) d0lo = desireeia_dp4a(vi, a[k], d0lo);
                else       d0hi = desireeia_dp4a(vi, a[k], d0hi);
            }
        }
        if (has_second) {
            const uint8_t* ql = b1 + half * 64 + ((g & 1) ? 32 : 0);
            const uint8_t* qh = b1 + 128 + half * 32;
            int vls[8], vhs[8];
            load8_b2(ql, vls);
            load8_b2(qh, vhs);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int vl = vls[k];
                const int vh = vhs[k];
                const int vi = __vsubss4((((vl >> nib_shift) & 0x0F0F0F0F) |
                                          ((((vh >> h_shift) & 0x03030303)) << 4)),
                                         0x20202020);
                if (k < 4) d1lo = desireeia_dp4a(vi, a[k], d1lo);
                else       d1hi = desireeia_dp4a(vi, a[k], d1hi);
            }
        }

        const float xs = x_scale[sub];
        const int8_t* sc0 = reinterpret_cast<const int8_t*>(b0 + 192) + half * 8;
        const float dd0 = __half2float(*reinterpret_cast<const __half*>(b0 + 208));
        acc0 += xs * dd0 * ((float) sc0[2 * g + 0] * (float) d0lo +
                            (float) sc0[2 * g + 1] * (float) d0hi);
        if (has_second) {
            const int8_t* sc1 = reinterpret_cast<const int8_t*>(b1 + 192) + half * 8;
            const float dd1 = __half2float(*reinterpret_cast<const __half*>(b1 + 208));
            acc1 += xs * dd1 * ((float) sc1[2 * g + 0] * (float) d1lo +
                                (float) sc1[2 * g + 1] * (float) d1hi);
        }
    }

    #pragma unroll
    for (int off = DESIREEIA_CUDA_WARP / 2; off > 0; off >>= 1) {
        acc0 += __shfl_xor_sync(0xffffffff, acc0, off, DESIREEIA_CUDA_WARP);
        acc1 += __shfl_xor_sync(0xffffffff, acc1, off, DESIREEIA_CUDA_WARP);
    }
    if (threadIdx.x == 0) {
        y[row0] = bias ? acc0 + bias[row0] : acc0;
        if (has_second) y[row0 + 1] = bias ? acc1 + bias[row0 + 1] : acc1;
    }
}

// Q6_K prefill, same idea as matmul_q4_k_batch_kernel: the six-bit
// weights of a sub-block are unpacked ONCE into registers and then reused
// across TOK activation columns, instead of being unpacked again for
// every column.
template <int nwarps, int TOK>
__global__ void matmul_q6_k_batch_kernel(const uint8_t* __restrict__ w,
                                          const int8_t* __restrict__ x_qs,
                                          const float* __restrict__ x_scale,
                                          size_t n_super, size_t rows,
                                          uint32_t n_tok,
                                          float* __restrict__ y) {
    const size_t row = (size_t) blockIdx.x * nwarps + threadIdx.y;
    if (row >= rows) return;
    const uint32_t t0 = blockIdx.y * TOK;
    const size_t n_sub = n_super * 8;
    const uint8_t* wr = w + row * n_super * 210;

    float acc[TOK];
    #pragma unroll
    for (int t = 0; t < TOK; ++t) acc[t] = 0.0f;

    for (size_t sub = threadIdx.x; sub < n_sub; sub += DESIREEIA_CUDA_WARP) {
        const size_t s = sub >> 3;
        const int idx = (int) (sub & 7);
        const int half = idx >> 2;
        const int g = idx & 3;

        const uint8_t* b = wr + s * 210;
        const int nib_shift = (g < 2) ? 0 : 4;
        const int h_shift = 2 * g;
        const uint8_t* ql = b + half * 64 + ((g & 1) ? 32 : 0);
        const uint8_t* qh = b + 128 + half * 32;

        // Unpacked once, reused by every column in this tile.
        int vi[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int vl = load_int_b2(ql, k);
            const int vh = load_int_b2(qh, k);
            vi[k] = __vsubss4((((vl >> nib_shift) & 0x0F0F0F0F) |
                               ((((vh >> h_shift) & 0x03030303)) << 4)),
                              0x20202020);
        }
        const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192) + half * 8;
        const float dd = __half2float(*reinterpret_cast<const __half*>(b + 208));
        const float sc_lo = (float) sc[2 * g + 0];
        const float sc_hi = (float) sc[2 * g + 1];

        #pragma unroll
        for (int t = 0; t < TOK; ++t) {
            const uint32_t tk = t0 + (uint32_t) t;
            if (tk >= n_tok) continue;
            const size_t ai = (size_t) tk * n_sub + sub;
            const int* a = reinterpret_cast<const int*>(x_qs + ai * 32);
            int dlo = 0, dhi = 0;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                if (k < 4) dlo = desireeia_dp4a(vi[k], a[k], dlo);
                else       dhi = desireeia_dp4a(vi[k], a[k], dhi);
            }
            acc[t] += x_scale[ai] * dd * (sc_lo * (float) dlo + sc_hi * (float) dhi);
        }
    }

    #pragma unroll
    for (int t = 0; t < TOK; ++t) {
        #pragma unroll
        for (int off = DESIREEIA_CUDA_WARP / 2; off > 0; off >>= 1) {
            acc[t] += __shfl_xor_sync(0xffffffff, acc[t], off, DESIREEIA_CUDA_WARP);
        }
    }
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int t = 0; t < TOK; ++t) {
            const uint32_t tk = t0 + (uint32_t) t;
            if (tk < n_tok) y[(size_t) tk * rows + row] = acc[t];
        }
    }
}

template <int nwarps>
__global__ void matmul_q6_k_kernel(const uint8_t* __restrict__ w,
                                    const int8_t* __restrict__ x_qs,
                                    const float* __restrict__ x_scale,
                                    size_t n_super, size_t rows,
                                    float* __restrict__ y,
                                    const float* __restrict__ bias) {
    q6k_row_group(w, x_qs, x_scale, n_super, rows, y, bias,
                  ((size_t) blockIdx.x * nwarps + threadIdx.y) * 2);
}

template <int nwarps>
__global__ void matmul_q6_k_group_kernel(KQuantGroupDesc g,
                                          const int8_t* __restrict__ x_qs,
                                          const float* __restrict__ x_scale,
                                          size_t n_super) {
    const int m = kquant_group_pick(g);
    const size_t lb = blockIdx.x - g.blk0[m];
    q6k_row_group(g.w[m], x_qs, x_scale, n_super, g.rows[m],
                  g.y[m], g.bias[m], (lb * nwarps + threadIdx.y) * 2);
}

// ---------------------------------------------------------------------
// Remaining quantized formats.
//
// Index mappings all taken from the dequantisers in quant/quant.cpp,
// which are the validated definition, rather than reconstructed.
//
// A note on alignment: the legacy 32-weight blocks are 18/20/22/24 bytes,
// so a block's payload is NOT 4-byte aligned in general and reading it
// through an int* would be undefined behaviour on the device. Those
// kernels therefore read bytes. The K-quant blocks (144/176/210) are
// 16-byte multiples, so their int-sized reads are safe.
//
// Every one of these formats stores the weight as (quant + offset) * d
// (plus a per-block minimum where present), so each needs the sum of the
// activation over the block — x_sum, computed once by the quantisation
// kernel and shared by all rows.

// Q4_0: w[j] = (qs[j] & 0xF) - 8, w[j+16] = (qs[j] >> 4) - 8, times d.
template <int nwarps>
__global__ void matmul_q4_0_kernel(const uint8_t* __restrict__ w,
                                    const int8_t* __restrict__ x_qs,
                                    const float* __restrict__ x_scale,
                                    const int32_t* __restrict__ x_sum,
                                    size_t nb, size_t rows,
                                    float* __restrict__ y,
                                    const float* __restrict__ bias) {
    const size_t row = (size_t) blockIdx.x * nwarps + threadIdx.y;
    if (row >= rows) return;
    const uint8_t* w_row = w + row * nb * 18;

    float acc = 0.0f;
    for (size_t b = threadIdx.x; b < nb; b += DESIREEIA_CUDA_WARP) {
        const uint8_t* blk = w_row + b * 18;
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint8_t* qs = blk + 2;
        const int8_t* a = x_qs + b * 32;
        int dot = 0;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const uint8_t q = qs[j];
            dot += (int) (q & 0x0F) * (int) a[j];
            dot += (int) (q >> 4)   * (int) a[j + 16];
        }
        // value = (q - 8)*d  ->  the -8 factors out over the block sum.
        acc += d * x_scale[b] * (float) (dot - 8 * x_sum[b]);
    }
    #pragma unroll
    for (int off = DESIREEIA_CUDA_WARP / 2; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, off, DESIREEIA_CUDA_WARP);
    }
    if (threadIdx.x == 0) y[row] = bias ? acc + bias[row] : acc;
}

// Q4_1: w = q*d + m, with q the raw nibble (no offset).
template <int nwarps>
__global__ void matmul_q4_1_kernel(const uint8_t* __restrict__ w,
                                    const int8_t* __restrict__ x_qs,
                                    const float* __restrict__ x_scale,
                                    const int32_t* __restrict__ x_sum,
                                    size_t nb, size_t rows,
                                    float* __restrict__ y,
                                    const float* __restrict__ bias) {
    const size_t row = (size_t) blockIdx.x * nwarps + threadIdx.y;
    if (row >= rows) return;
    const uint8_t* w_row = w + row * nb * 20;

    float acc = 0.0f;
    for (size_t b = threadIdx.x; b < nb; b += DESIREEIA_CUDA_WARP) {
        const uint8_t* blk = w_row + b * 20;
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const float m = __half2float(*reinterpret_cast<const __half*>(blk + 2));
        const uint8_t* qs = blk + 4;
        const int8_t* a = x_qs + b * 32;
        int dot = 0;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const uint8_t q = qs[j];
            dot += (int) (q & 0x0F) * (int) a[j];
            dot += (int) (q >> 4)   * (int) a[j + 16];
        }
        acc += x_scale[b] * (d * (float) dot + m * (float) x_sum[b]);
    }
    #pragma unroll
    for (int off = DESIREEIA_CUDA_WARP / 2; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, off, DESIREEIA_CUDA_WARP);
    }
    if (threadIdx.x == 0) y[row] = bias ? acc + bias[row] : acc;
}

// Q5_0: the fifth bit comes from a 32-bit qh field, bit j for the low
// half and bit j+12 shifted for the high half, exactly as
// dequantize_row_q5_0 reads it. w = ((nibble | bit4) - 16) * d.
template <int nwarps>
__global__ void matmul_q5_0_kernel(const uint8_t* __restrict__ w,
                                    const int8_t* __restrict__ x_qs,
                                    const float* __restrict__ x_scale,
                                    const int32_t* __restrict__ x_sum,
                                    size_t nb, size_t rows,
                                    float* __restrict__ y,
                                    const float* __restrict__ bias) {
    const size_t row = (size_t) blockIdx.x * nwarps + threadIdx.y;
    if (row >= rows) return;
    const uint8_t* w_row = w + row * nb * 22;

    float acc = 0.0f;
    for (size_t b = threadIdx.x; b < nb; b += DESIREEIA_CUDA_WARP) {
        const uint8_t* blk = w_row + b * 22;
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        uint32_t qh = (uint32_t) blk[2] | ((uint32_t) blk[3] << 8) |
                      ((uint32_t) blk[4] << 16) | ((uint32_t) blk[5] << 24);
        const uint8_t* qs = blk + 6;
        const int8_t* a = x_qs + b * 32;
        int dot = 0;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const uint8_t q = qs[j];
            const int xh0 = (int) (((qh >> j) << 4) & 0x10);
            const int xh1 = (int) ((qh >> (j + 12)) & 0x10);
            dot += (int) ((q & 0x0F) | xh0) * (int) a[j];
            dot += (int) ((q >> 4)   | xh1) * (int) a[j + 16];
        }
        acc += d * x_scale[b] * (float) (dot - 16 * x_sum[b]);
    }
    #pragma unroll
    for (int off = DESIREEIA_CUDA_WARP / 2; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, off, DESIREEIA_CUDA_WARP);
    }
    if (threadIdx.x == 0) y[row] = bias ? acc + bias[row] : acc;
}

// Q5_1: same five-bit packing as Q5_0 but with a minimum instead of the
// -16 offset.
template <int nwarps>
__global__ void matmul_q5_1_kernel(const uint8_t* __restrict__ w,
                                    const int8_t* __restrict__ x_qs,
                                    const float* __restrict__ x_scale,
                                    const int32_t* __restrict__ x_sum,
                                    size_t nb, size_t rows,
                                    float* __restrict__ y,
                                    const float* __restrict__ bias) {
    const size_t row = (size_t) blockIdx.x * nwarps + threadIdx.y;
    if (row >= rows) return;
    const uint8_t* w_row = w + row * nb * 24;

    float acc = 0.0f;
    for (size_t b = threadIdx.x; b < nb; b += DESIREEIA_CUDA_WARP) {
        const uint8_t* blk = w_row + b * 24;
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const float m = __half2float(*reinterpret_cast<const __half*>(blk + 2));
        uint32_t qh = (uint32_t) blk[4] | ((uint32_t) blk[5] << 8) |
                      ((uint32_t) blk[6] << 16) | ((uint32_t) blk[7] << 24);
        const uint8_t* qs = blk + 8;
        const int8_t* a = x_qs + b * 32;
        int dot = 0;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const uint8_t q = qs[j];
            const int xh0 = (int) (((qh >> j) << 4) & 0x10);
            const int xh1 = (int) ((qh >> (j + 12)) & 0x10);
            dot += (int) ((q & 0x0F) | xh0) * (int) a[j];
            dot += (int) ((q >> 4)   | xh1) * (int) a[j + 16];
        }
        acc += x_scale[b] * (d * (float) dot + m * (float) x_sum[b]);
    }
    #pragma unroll
    for (int off = DESIREEIA_CUDA_WARP / 2; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, off, DESIREEIA_CUDA_WARP);
    }
    if (threadIdx.x == 0) y[row] = bias ? acc + bias[row] : acc;
}

// Q5_K: like Q4_K plus a fifth bit per weight taken from qh. Sub-block sb
// uses group g = sb/2 of the packed nibbles and bit (2g + (sb&1)) of qh,
// matching dequantize_row_q5_K.
template <int nwarps>
__global__ void matmul_q5_k_kernel(const uint8_t* __restrict__ w,
                                    const int8_t* __restrict__ x_qs,
                                    const float* __restrict__ x_scale,
                                    const int32_t* __restrict__ x_sum,
                                    size_t n_super, size_t rows,
                                    float* __restrict__ y,
                                    const float* __restrict__ bias) {
    const size_t row = (size_t) blockIdx.x * nwarps + threadIdx.y;
    if (row >= rows) return;
    const uint8_t* w_row = w + row * n_super * 176;

    float acc = 0.0f;
    for (size_t s = threadIdx.x; s < n_super; s += DESIREEIA_CUDA_WARP) {
        const uint8_t* blk = w_row + s * 176;
        const float d    = __half2float(*reinterpret_cast<const __half*>(blk));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(blk + 2));
        const uint8_t* scales = blk + 4;
        const uint8_t* qh = blk + 16;
        const uint8_t* qs = blk + 48;

        const size_t sub0 = s * 8;
        float sum_d = 0.0f;
        float sum_m = 0.0f;
        #pragma unroll
        for (int sb = 0; sb < 8; ++sb) {
            uint8_t sc, mn;
            get_scale_min_k4_dev(sb, scales, &sc, &mn);
            const uint8_t* ql = qs + (sb / 2) * 32;
            const int shift = (sb & 1) ? 4 : 0;
            const uint8_t hmask = (uint8_t) (1u << (2 * (sb / 2) + (sb & 1)));
            const int8_t* a = x_qs + (sub0 + sb) * 32;
            int dot = 0;
            #pragma unroll
            for (int l = 0; l < 32; ++l) {
                const int q = ((ql[l] >> shift) & 0x0F) + ((qh[l] & hmask) ? 16 : 0);
                dot += q * (int) a[l];
            }
            const float xs = x_scale[sub0 + sb];
            sum_d += xs * (float) sc * (float) dot;
            sum_m += xs * (float) mn * (float) x_sum[sub0 + sb];
        }
        acc += d * sum_d - dmin * sum_m;
    }
    #pragma unroll
    for (int off = DESIREEIA_CUDA_WARP / 2; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, off, DESIREEIA_CUDA_WARP);
    }
    if (threadIdx.x == 0) y[row] = bias ? acc + bias[row] : acc;
}

// RMS norm, matching rms_norm_vec in dense_forward.cpp: the sum of squares
// is accumulated in double (the CPU version does, and this is a reduction
// over n_embd values where it matters), then
// y[i] = x[i] * 1/sqrt(mean + eps) * w[i]. One block, strided over n.
__global__ void rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                 float* __restrict__ y, uint32_t n, float eps) {
    // Float accumulation, not double. The CPU version accumulates in
    // double for stability, but on a consumer GPU FP64 runs at a small
    // fraction of FP32 throughput, and the tree reduction below already
    // keeps the error low (it is a pairwise sum, not a serial chain).
    //
    // One block per ROW: launched with a grid of 1 (decode) it is the
    // single-vector norm it always was; the batched prefill launches one
    // block per token over a [token][n] buffer.
    x += (size_t) blockIdx.x * n;
    y += (size_t) blockIdx.x * n;
    extern __shared__ float fsmem[];
    float local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[i];
        local += v * v;
    }
    fsmem[threadIdx.x] = local;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) fsmem[threadIdx.x] += fsmem[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = 1.0f / sqrtf(fsmem[0] / (float) n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        y[i] = x[i] * scale * w[i];
    }
}

// Optional residual add, RMS norm, and Q8_0 quantisation of the result, in
// ONE launch instead of the two or three it used to take.
//
// Why fuse these specifically: the norm needs a reduction over the whole
// vector, so it was always a single block, and the quantisation that
// followed was a separate launch whose blocks re-read the very values the
// norm had just written. Chained inside the graph they cannot overlap, so
// the second launch bought nothing and cost a launch. The residual add in
// front of the FFN norm is in the same position for the same reason.
//
// Requires n % 32 == 0 and a power-of-two block size, both guaranteed by
// the caller. kHasResid folds `x += resid` in first, writing the sum back
// to x because the layer still needs it as the FFN residual.
// RMS norm whose consumers all run on the FP16 tensor cores: the float
// result (kept for a fallback) and the FP16 input of the next products in
// the act_to_f16_kernel format (one block per token of m_pad, rows past m
// zero, row_scale = the factor the row was divided by) - no int8 copy and
// no separate conversion launch. kHasResid as in rms_norm_quant_kernel.
template <bool kHasResid>
__global__ void rms_norm_f16_kernel(float* __restrict__ x, const float* __restrict__ resid,
                                    const float* __restrict__ w, float* __restrict__ y,
                                    __half* __restrict__ out, float* __restrict__ row_scale,
                                    uint32_t n, float eps, uint32_t m) {
    const uint32_t t = blockIdx.x;
    __half* dst = out + (size_t) t * n;
    if (t >= m) {
        for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) dst[i] = __float2half(0.0f);
        if (threadIdx.x == 0) row_scale[t] = 1.0f;
        return;
    }
    x += (size_t) t * n;
    y += (size_t) t * n;
    if (kHasResid) resid += (size_t) t * n;
    __shared__ float red[32];
    auto block_sum = [&](float v, bool is_max) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float u = __shfl_xor_sync(0xffffffff, v, o);
            v = is_max ? fmaxf(v, u) : v + u;
        }
        __syncthreads();
        if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = v;
        __syncthreads();
        if (threadIdx.x < 32) {
            float r = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.0f;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float u = __shfl_xor_sync(0xffffffff, r, o);
                r = is_max ? fmaxf(r, u) : r + u;
            }
            if (threadIdx.x == 0) red[0] = r;
        }
        __syncthreads();
        return red[0];
    };
    float local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[i];
        if (kHasResid) {
            v += resid[i];
            x[i] = v;
        }
        local += v * v;
    }
    // Same reduction order as rms_norm_quant_kernel (a tree over the block
    // in shared memory): the two paths give bit-identical norms.
    extern __shared__ float nsum[];
    nsum[threadIdx.x] = local;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) nsum[threadIdx.x] += nsum[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = 1.0f / sqrtf(nsum[0] / (float) n + eps);
    float amax = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[i] * scale * w[i];
        y[i] = v;
        amax = fmaxf(amax, fabsf(v));
    }
    amax = block_sum(amax, true);
    const float rs = amax > 1024.0f ? amax / 1024.0f : 1.0f;     // as act_to_f16_kernel
    const float inv = 1.0f / rs;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) dst[i] = __float2half(y[i] * inv);
    if (threadIdx.x == 0) row_scale[t] = rs;
}

template <bool kHasResid>
__global__ void rms_norm_quant_kernel(float* __restrict__ x,
                                       const float* __restrict__ resid,
                                       const float* __restrict__ w,
                                       float* __restrict__ y,
                                       int8_t* __restrict__ q,
                                       float* __restrict__ scales,
                                       int32_t* __restrict__ sums,
                                       uint32_t n, float eps) {
    // Float accumulation, for the same reason as rms_norm_kernel.
    //
    // One block per ROW, like rms_norm_kernel: grid 1 is the decode case,
    // the batched prefill launches one block per token. The quantised
    // output follows the same [token][sub-block] layout the batched mat-mul
    // kernels read.
    {
        const size_t row = blockIdx.x;
        x += row * n;
        if (kHasResid) resid += row * n;
        y += row * n;
        q += row * n;
        scales += row * (n / 32);
        if (sums) sums += row * (n / 32);
    }
    extern __shared__ float rqsmem[];
    float local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[i];
        if (kHasResid) {
            v += resid[i];
            x[i] = v;
        }
        local += v * v;
    }
    rqsmem[threadIdx.x] = local;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) rqsmem[threadIdx.x] += rqsmem[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = 1.0f / sqrtf(rqsmem[0] / (float) n + eps);

    // One WARP per 32-wide sub-block, strided over the vector, so the
    // absmax and sum reductions stay shuffles inside a warp — exactly what
    // quantize_q8_0_kernel did with one block per sub-block.
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t nwarps = blockDim.x >> 5;
    const uint32_t nb = n / 32;
    for (uint32_t b = warp; b < nb; b += nwarps) {
        const uint32_t i = b * 32 + lane;
        const float v = x[i] * scale * w[i];
        y[i] = v;
        float m = fabsf(v);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off, 32));
        }
        const float d = m > 0.0f ? m / 127.0f : 0.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        if (lane == 0) scales[b] = d;
        int qi = min(127, max(-127, (int) rintf(v * id)));
        q[i] = (int8_t) qi;
        if (sums) {
            int sv = qi;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) sv += __shfl_xor_sync(0xffffffff, sv, off, 32);
            if (lane == 0) sums[b] = sv;
        }
    }
}

// The gated FFN activation and the Q8_0 quantisation of its result, in one
// launch. Unlike the norm above this needs no cross-block reduction, so it
// keeps one warp per sub-block and simply does the activation first.
__global__ void ffn_act_quant_kernel(float* __restrict__ up_inout,
                                      const float* __restrict__ gate,
                                      int8_t* __restrict__ q,
                                      float* __restrict__ scales,
                                      int32_t* __restrict__ sums,
                                      size_t n, int act_gelu) {
    const size_t b = blockIdx.x;
    const size_t i = b * 32 + threadIdx.x;
    float v = 0.0f;
    if (i < n) {
        const float g = gate[i];
        float a;
        if (act_gelu) {
            const float u = 0.7978845608028654f * g * (1.0f + 0.044715f * g * g);
            a = 0.5f * g * (1.0f + tanhf(u));
        } else {
            a = g / (1.0f + __expf(-g));  // silu
        }
        v = a * up_inout[i];
        up_inout[i] = v;
    }
    float m = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off, 32));
    }
    const float d = m > 0.0f ? m / 127.0f : 0.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    if (threadIdx.x == 0) scales[b] = d;
    int qi = min(127, max(-127, (int) rintf(v * id)));
    q[b * 32 + threadIdx.x] = (int8_t) qi;
    if (sums) {
        int sv = qi;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) sv += __shfl_xor_sync(0xffffffff, sv, off, 32);
        if (threadIdx.x == 0) sums[b] = sv;
    }
}

// The gated FFN activation straight into the FP16 input of the next
// tensor-core product (same format as act_to_f16_kernel: one block per
// token, rows past m zero, row_scale = the factor the row was divided by).
// Two passes over up/gate - the maximum first - instead of a float result
// written and read back: the pair of rows stays in L2 between them.
__device__ __forceinline__ float ffn_act(float g, int act_gelu) {
    if (act_gelu) {
        const float u = 0.7978845608028654f * g * (1.0f + 0.044715f * g * g);
        return 0.5f * g * (1.0f + tanhf(u));
    }
    return g / (1.0f + __expf(-g));           // silu
}
__global__ void ffn_act_f16_kernel(const float* __restrict__ up, const float* __restrict__ gate, size_t n_ff,
                                   uint32_t m, int act_gelu, __half* __restrict__ out, float* __restrict__ row_scale) {
    const uint32_t t = blockIdx.x;
    __half2* dst = reinterpret_cast<__half2*>(out + (size_t) t * n_ff);
    const size_t n2 = n_ff / 2;
    if (t >= m) {
        for (size_t i = threadIdx.x; i < n2; i += blockDim.x) dst[i] = __float2half2_rn(0.0f);
        if (threadIdx.x == 0) row_scale[t] = 1.0f;
        return;
    }
    const float2* u2 = reinterpret_cast<const float2*>(up + (size_t) t * n_ff);
    const float2* g2 = reinterpret_cast<const float2*>(gate + (size_t) t * n_ff);
    float amax = 0.0f;
    for (size_t i = threadIdx.x; i < n2; i += blockDim.x) {
        const float2 u = u2[i], g = g2[i];
        amax = fmaxf(amax, fmaxf(fabsf(ffn_act(g.x, act_gelu) * u.x), fabsf(ffn_act(g.y, act_gelu) * u.y)));
    }
    __shared__ float red[32];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = amax;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    const float scale = red[0] > 1024.0f ? red[0] / 1024.0f : 1.0f;   // as act_to_f16_kernel
    const float inv = 1.0f / scale;
    for (size_t i = threadIdx.x; i < n2; i += blockDim.x) {
        const float2 u = u2[i], g = g2[i];
        dst[i] = __floats2half2_rn(ffn_act(g.x, act_gelu) * u.x * inv, ffn_act(g.y, act_gelu) * u.y * inv);
    }
    if (threadIdx.x == 0) row_scale[t] = scale;
}

// Completes an FP16 product input written by its producer for the first m
// rows: rows m..gridDim.x-1 zero, every row scale 1.
__global__ void f16_rows_finish_kernel(__half* __restrict__ a, float* __restrict__ row_scale, size_t cols, uint32_t m) {
    const uint32_t t = blockIdx.x;
    if (threadIdx.x == 0) row_scale[t] = 1.0f;
    if (t < m) return;
    for (size_t i = threadIdx.x; i < cols; i += blockDim.x) a[(size_t) t * cols + i] = __float2half(0.0f);
}

// Batched: rms_norm_kernel(out, w) then add2_kernel(x, out, y), one block per
// token row - the same arithmetic in one launch (the sandwich-norm layers).
__global__ void post_norm_add_rows_kernel(float* __restrict__ out, const float* __restrict__ w,
                                          float* __restrict__ x, const float* __restrict__ y, uint32_t n, float eps) {
    extern __shared__ float pn_red[];
    const size_t row = (size_t) blockIdx.x * n;
    out += row; x += row; y += row;
    float local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) { const float v = out[i]; local += v * v; }
    pn_red[threadIdx.x] = local;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) pn_red[threadIdx.x] += pn_red[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = 1.0f / sqrtf(pn_red[0] / (float) n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = __fmul_rn(__fmul_rn(out[i], scale), w[i]);
        out[i] = v;
        x[i] = __fadd_rn(v, y[i]);
    }
}

// y += a, and out = a + b: the two residual shapes the dense layer needs.
__global__ void add_inplace_kernel(float* __restrict__ y, const float* __restrict__ a, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a[i];
}
__global__ void add2_kernel(float* __restrict__ out, const float* __restrict__ a,
                             const float* __restrict__ b, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}


// Per-head RMS norm, used by architectures that normalise Q and K before
// RoPE (gemma3 and siblings). One block per head; the same weight vector
// applies to every head, exactly as the CPU path does it.
__global__ void rms_norm_heads_kernel(float* __restrict__ v, const float* __restrict__ w,
                                       uint32_t head_dim, float eps) {
    // Float accumulation for the same reason as rms_norm_kernel: FP64
    // runs at a small fraction of FP32 throughput on a consumer GPU, and
    // the tree reduction below is pairwise, not a serial chain.
    extern __shared__ float fsmem2[];
    float* vh = v + (size_t) blockIdx.x * head_dim;
    float local = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float t = vh[i];
        local += t * t;
    }
    fsmem2[threadIdx.x] = local;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) fsmem2[threadIdx.x] += fsmem2[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = 1.0f / sqrtf(fsmem2[0] / (float) head_dim + eps);
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        vh[i] = vh[i] * scale * w[i];
    }
}

// Adds an optional bias vector in place. Mirrors add_bias in
// dense_forward.cpp; a null pointer means "no bias", same as there.
// Per-head output gate: one block per head, scaling that head's whole
// slice of the attention output by sigmoid of its single gate value.
// The gate vector is the output of a matvec whose weight has one ROW per
// head, so it is n_head long, not n_embd.
__global__ void attn_gate_apply_kernel(float* __restrict__ attn,
                                        const float* __restrict__ gate,
                                        uint32_t head_dim) {
    const float g = 1.0f / (1.0f + __expf(-gate[blockIdx.x]));
    float* ah = attn + (size_t) blockIdx.x * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        ah[d] *= g;
    }
}

__global__ void add_bias_kernel(float* __restrict__ y, const float* __restrict__ b, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && b) y[i] += b[i];
}

// NeoX-style RoPE, applied per head over the first n_rot dimensions, with
// the cos/sin table built host-side once per layer and uploaded. Same
// rotation as rope_neox_cached in dense_forward.cpp:
//   v[j]      = x0*cos - x1*sin
//   v[j+half] = x0*sin + x1*cos
// One block per head, threads over the j pairs.
__global__ void rope_neox_kernel(float* __restrict__ q, float* __restrict__ k,
                                  const float* __restrict__ cache,
                                  uint32_t n_rot, uint32_t head_dim, uint32_t n_head) {
    // Blocks [0, n_head) rotate Q, the rest rotate K: one launch instead of
    // two, for the same reason the bias is folded above.
    const uint32_t b = blockIdx.x;
    float* vh = (b < n_head ? q + (size_t) b * head_dim
                             : k + (size_t) (b - n_head) * head_dim);
    const uint32_t half = n_rot / 2;
    for (uint32_t j = threadIdx.x; j < half; j += blockDim.x) {
        const float c = cache[2 * j + 0];
        const float s = cache[2 * j + 1];
        const float x0 = vh[j];
        const float x1 = vh[j + half];
        vh[j]        = x0 * c - x1 * s;
        vh[j + half] = x0 * s + x1 * c;
    }
}


// ---------------------------------------------------------------------
// Decode: chains of small kernels merged into one launch each. Inside a
// layer graph every node costs a launch slot whatever its size, and these
// ran one after the other on a single block. Each merged kernel performs
// exactly the operations, in the same order and with the same reductions,
// of the kernels it replaces, so the results are identical bit for bit.

// Tree sum over the block in shared memory, as rms_norm_kernel does it.
__device__ __forceinline__ float dec_block_sum(float local, float* red) {
    red[threadIdx.x] = local;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();                               // red is reused by the caller
    return r;
}

// rms_norm_kernel(y, w_post) in place, then rms_norm_quant_kernel<true>
// (y += resid; norm with w_ffn; Q8_0 of the result). One block.
__global__ void dec_post_norm_resid_quant_kernel(float* __restrict__ y, const float* __restrict__ resid,
                                                 const float* __restrict__ w_post, const float* __restrict__ w_ffn,
                                                 float* __restrict__ xn, int8_t* __restrict__ q,
                                                 float* __restrict__ scales, int32_t* __restrict__ sums,
                                                 uint32_t n, float eps) {
    extern __shared__ float dsm[];
    float local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) { const float v = y[i]; local += v * v; }
    const float s1 = 1.0f / sqrtf(dec_block_sum(local, dsm) / (float) n + eps);
    local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        // no contraction into an fma: the separate kernels rounded the
        // normalised value before the residual add
        float v = __fadd_rn(__fmul_rn(__fmul_rn(y[i], s1), w_post[i]), resid[i]);
        y[i] = v;
        local += v * v;
    }
    const float s2 = 1.0f / sqrtf(dec_block_sum(local, dsm) / (float) n + eps);
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    const uint32_t nb = n / 32;
    for (uint32_t b = warp; b < nb; b += nwarps) {
        const uint32_t i = b * 32 + lane;
        const float v = y[i] * s2 * w_ffn[i];
        xn[i] = v;
        float m = fabsf(v);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off, 32));
        const float d = m > 0.0f ? m / 127.0f : 0.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        if (lane == 0) scales[b] = d;
        const int qi = min(127, max(-127, (int) rintf(v * id)));
        q[i] = (int8_t) qi;
        if (sums) {
            int sv = qi;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) sv += __shfl_xor_sync(0xffffffff, sv, off, 32);
            if (lane == 0) sums[b] = sv;
        }
    }
}

// rms_norm_kernel(out, w_post) in place, then add2_kernel(x, out, y). One block.
__global__ void dec_post_norm_add_kernel(float* __restrict__ out, const float* __restrict__ w_post,
                                         float* __restrict__ x, const float* __restrict__ y,
                                         uint32_t n, float eps) {
    extern __shared__ float dsm[];
    float local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) { const float v = out[i]; local += v * v; }
    const float s1 = 1.0f / sqrtf(dec_block_sum(local, dsm) / (float) n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = __fmul_rn(__fmul_rn(out[i], s1), w_post[i]);
        out[i] = v;
        x[i] = __fadd_rn(v, y[i]);
    }
}

// rms_norm_heads_kernel on Q (blocks [0, n_head)) and K (the rest), then
// rope_neox_kernel on the same head when a RoPE table is given. 64 threads.
__global__ void dec_qk_norm_rope_kernel(float* __restrict__ q, float* __restrict__ k,
                                        const float* __restrict__ qw, const float* __restrict__ kw,
                                        const float* __restrict__ rope, uint32_t n_rot,
                                        uint32_t head_dim, uint32_t n_head, float eps) {
    extern __shared__ float dsm[];
    const uint32_t b = blockIdx.x;
    const bool is_q = b < n_head;
    float* vh = is_q ? q + (size_t) b * head_dim : k + (size_t) (b - n_head) * head_dim;
    const float* w = is_q ? qw : kw;
    float local = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) { const float t = vh[i]; local += t * t; }
    const float scale = 1.0f / sqrtf(dec_block_sum(local, dsm) / (float) head_dim + eps);
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) vh[i] = vh[i] * scale * w[i];
    if (!rope) return;
    __syncthreads();
    const uint32_t half = n_rot / 2;
    for (uint32_t j = threadIdx.x; j < half; j += blockDim.x) {
        const float c = rope[2 * j + 0];
        const float sn = rope[2 * j + 1];
        const float x0 = vh[j];
        const float x1 = vh[j + half];
        vh[j]        = x0 * c - x1 * sn;
        vh[j + half] = x0 * sn + x1 * c;
    }
}


// Decode norms built for latency. The vector (a few thousand floats) is
// read ONCE into registers by 1024 threads, the sums use warp shuffles and
// a single shared-memory step instead of a tree of barriers, and the Q8_0
// form comes straight from the registers: with 1024 threads, the values
// thread t holds at step j are element t + 1024*j, so each warp holds
// exactly one 32-block per step. About 4x shorter than the tree kernels,
// which is what decode pays for, one block on one SM, several times per layer.
constexpr int kDecNormThreads = 1024;
constexpr int kDecNormMaxV = 8;                  // n up to 8192

__device__ __forceinline__ float dec_fast_sum(float v, float* red) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = lane < (int) (blockDim.x >> 5) ? red[lane] : 0.0f;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffff, t, o);
    __syncthreads();                               // red is reused by the next call
    return t;
}

// v_io [n]: optional sandwich norm (w_post), optional residual add (resid),
// the result written back to v_io when either applied; then the norm with
// w, its float form in xn and its Q8_0 form in q/scales/sums.
__global__ void __launch_bounds__(kDecNormThreads) dec_norm_fast_kernel(
        float* __restrict__ v_io, const float* __restrict__ resid, const float* __restrict__ w_post,
        const float* __restrict__ w, float* __restrict__ xn, int8_t* __restrict__ q,
        float* __restrict__ scales, int32_t* __restrict__ sums, uint32_t n, float eps) {
    __shared__ float red[32];
    const uint32_t t = threadIdx.x;
    float v[kDecNormMaxV];
    float local = 0.0f;
    #pragma unroll
    for (int j = 0; j < kDecNormMaxV; ++j) {
        const uint32_t i = t + j * kDecNormThreads;
        v[j] = i < n ? v_io[i] : 0.0f;
        local += v[j] * v[j];
    }
    if (w_post) {
        const float s1 = 1.0f / sqrtf(dec_fast_sum(local, red) / (float) n + eps);
        #pragma unroll
        for (int j = 0; j < kDecNormMaxV; ++j) {
            const uint32_t i = t + j * kDecNormThreads;
            if (i < n) v[j] = v[j] * s1 * w_post[i];
        }
    }
    if (resid || w_post) {
        local = 0.0f;
        #pragma unroll
        for (int j = 0; j < kDecNormMaxV; ++j) {
            const uint32_t i = t + j * kDecNormThreads;
            if (i < n) {
                if (resid) v[j] += resid[i];
                v_io[i] = v[j];
            }
            local += v[j] * v[j];
        }
    }
    const float s2 = 1.0f / sqrtf(dec_fast_sum(local, red) / (float) n + eps);
    const int lane = t & 31;
    #pragma unroll
    for (int j = 0; j < kDecNormMaxV; ++j) {
        const uint32_t i = t + j * kDecNormThreads;
        if (j * kDecNormThreads >= (int) n) break;         // uniform over the block
        const bool in = i < n;
        const float o = in ? v[j] * s2 * w[i] : 0.0f;
        if (in) xn[i] = o;
        float m = fabsf(o);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
        const float d = m > 0.0f ? m / 127.0f : 0.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        const int qi = min(127, max(-127, (int) rintf(o * id)));
        const uint32_t b = i / 32;
        if (in) {
            q[i] = (int8_t) qi;
            if (lane == 0) scales[b] = d;
        }
        if (sums) {
            int sv = qi;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) sv += __shfl_xor_sync(0xffffffff, sv, off);
            if (in && lane == 0) sums[b] = sv;
        }
    }
}

// out = norm(out) * w_post; x = out + y. One block, registers as above.
__global__ void __launch_bounds__(kDecNormThreads) dec_post_add_fast_kernel(
        float* __restrict__ out, const float* __restrict__ w_post, float* __restrict__ x,
        const float* __restrict__ y, uint32_t n, float eps) {
    __shared__ float red[32];
    const uint32_t t = threadIdx.x;
    float v[kDecNormMaxV];
    float local = 0.0f;
    #pragma unroll
    for (int j = 0; j < kDecNormMaxV; ++j) {
        const uint32_t i = t + j * kDecNormThreads;
        v[j] = i < n ? out[i] : 0.0f;
        local += v[j] * v[j];
    }
    const float s1 = 1.0f / sqrtf(dec_fast_sum(local, red) / (float) n + eps);
    #pragma unroll
    for (int j = 0; j < kDecNormMaxV; ++j) {
        const uint32_t i = t + j * kDecNormThreads;
        if (i < n) {
            const float o = v[j] * s1 * w_post[i];
            out[i] = o;
            x[i] = o + y[i];
        }
    }
}

// Quantizes k (or v) into the Q8_0 KV-cache row layout, one row per kv
// head: [fp16 scale][32 int8] repeated head_dim/32 times. Same recipe as
// kv_quantize_row in kv/kv_quant.cpp (scale = amax/127, round to nearest,
// clamp to +-127), which defines the reference behaviour on CPU.
// One block per (kv head, sub-block of 32), 32 threads.
__global__ void kv_quantize_kernel(const float* __restrict__ ksrc, uint8_t* __restrict__ kbase,
                                    const float* __restrict__ vsrc, uint8_t* __restrict__ vbase,
                                    uint32_t head_dim, size_t row_bytes, uint32_t blocks_per_side,
                                    size_t pos_bytes, const uint32_t* __restrict__ dyn) {
    // Destination row derived from the device-side position, so the graph
    // that contains this kernel stays valid for every token.
    uint8_t* kdst = kbase + (size_t) dyn[0] * pos_bytes;
    uint8_t* vdst = vbase + (size_t) dyn[0] * pos_bytes;
    // First half of the grid quantizes K, second half V: one launch.
    const bool is_v = blockIdx.x >= blocks_per_side;
    const uint32_t bid = is_v ? blockIdx.x - blocks_per_side : blockIdx.x;
    const float* src = is_v ? vsrc : ksrc;
    uint8_t* dst = is_v ? vdst : kdst;

    const uint32_t nblk = head_dim / 32;
    const uint32_t h = bid / nblk;
    const uint32_t b = bid % nblk;
    const uint32_t j = threadIdx.x;

    const float x = src[(size_t) h * head_dim + b * 32 + j];
    float m = fabsf(x);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off, 32));
    }
    const float d = m > 0.0f ? m / 127.0f : 0.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;

    uint8_t* blk = dst + (size_t) h * row_bytes + (size_t) b * 34;
    if (j == 0) *reinterpret_cast<__half*>(blk) = __float2half(d);
    int qi = (int) rintf(x * id);
    qi = min(127, max(-127, qi));
    reinterpret_cast<int8_t*>(blk + 2)[j] = (int8_t) qi;
}

// Several mat-vecs that share the same activation, in ONE launch.
//
// Measured: what dominates decode on this workload is the NUMBER of kernel
// launches (~32 us of overhead each), not the number of synchronisations
// nor the kernels' own speed. Q/K/V read the same activation and only
// differ in weights and output buffer, so they are one grid: each block
// looks up which matrix it belongs to and offsets accordingly.
struct Q80GroupDesc {
    const int8_t* qs[4];
    const __half* scale[4];
    float* y[4];
    const float* bias[4];
    uint32_t rows[4];
    uint32_t block0[4];   // first grid block belonging to this matrix
    uint32_t n;
};

// Launch shared by the two entry points (resident and per-call): same
// geometry, so a grid change applies to both.
constexpr int kMatmulQ80Warps = 4;

template <int nwarps>
__global__ void matmul_q8_0_group_kernel(Q80GroupDesc g,
                                          const int8_t* __restrict__ x_qs,
                                          const float* __restrict__ x_scale,
                                          size_t nb) {
    uint32_t m = 0;
    #pragma unroll
    for (uint32_t i = 1; i < 4; ++i) {
        if (i < g.n && blockIdx.x >= g.block0[i]) m = i;
    }
    const uint32_t local_block = blockIdx.x - g.block0[m];
    const size_t rows = g.rows[m];
    const size_t row0 = ((size_t) local_block * nwarps + threadIdx.y) * 2;
    if (row0 >= rows) return;
    const bool has_second = (row0 + 1) < rows;

    const int* w_row0 = reinterpret_cast<const int*>(g.qs[m] + row0 * nb * 32);
    const int* w_row1 = has_second ? w_row0 + nb * 8 : w_row0;
    const __half* scale0 = g.scale[m] + row0 * nb;
    const __half* scale1 = has_second ? scale0 + nb : scale0;
    const int* x_row = reinterpret_cast<const int*>(x_qs);

    float acc0 = 0.0f, acc1 = 0.0f;
    for (size_t b = threadIdx.x; b < nb; b += DESIREEIA_CUDA_WARP) {
        // 128-bit loads: a Q8_0 block is 32 int8 = 8 int = two int4, so the
        // whole block moves in two transactions instead of eight. Every
        // buffer comes from cudaMalloc and block offsets are multiples of
        // 32 bytes, so the wider load stays aligned.
        const int4* xv4 = reinterpret_cast<const int4*>(x_row + b * 8);
        const int4 x0 = xv4[0];
        const int4 x1 = xv4[1];
        const float xsc = x_scale[b];

        const int4* w04 = reinterpret_cast<const int4*>(w_row0 + b * 8);
        const int4 a0 = w04[0];
        const int4 a1 = w04[1];
        int sumi0 = desireeia_dp4a(a0.x, x0.x, 0);
        sumi0 = desireeia_dp4a(a0.y, x0.y, sumi0);
        sumi0 = desireeia_dp4a(a0.z, x0.z, sumi0);
        sumi0 = desireeia_dp4a(a0.w, x0.w, sumi0);
        sumi0 = desireeia_dp4a(a1.x, x1.x, sumi0);
        sumi0 = desireeia_dp4a(a1.y, x1.y, sumi0);
        sumi0 = desireeia_dp4a(a1.z, x1.z, sumi0);
        sumi0 = desireeia_dp4a(a1.w, x1.w, sumi0);
        acc0 += __half2float(scale0[b]) * xsc * (float) sumi0;

        if (has_second) {
            const int4* w14 = reinterpret_cast<const int4*>(w_row1 + b * 8);
            const int4 c0 = w14[0];
            const int4 c1 = w14[1];
            int sumi1 = desireeia_dp4a(c0.x, x0.x, 0);
            sumi1 = desireeia_dp4a(c0.y, x0.y, sumi1);
            sumi1 = desireeia_dp4a(c0.z, x0.z, sumi1);
            sumi1 = desireeia_dp4a(c0.w, x0.w, sumi1);
            sumi1 = desireeia_dp4a(c1.x, x1.x, sumi1);
            sumi1 = desireeia_dp4a(c1.y, x1.y, sumi1);
            sumi1 = desireeia_dp4a(c1.z, x1.z, sumi1);
            sumi1 = desireeia_dp4a(c1.w, x1.w, sumi1);
            acc1 += __half2float(scale1[b]) * xsc * (float) sumi1;
        }
    }

    #pragma unroll
    for (int off = DESIREEIA_CUDA_WARP / 2; off > 0; off >>= 1) {
        acc0 += __shfl_xor_sync(0xffffffff, acc0, off, DESIREEIA_CUDA_WARP);
        acc1 += __shfl_xor_sync(0xffffffff, acc1, off, DESIREEIA_CUDA_WARP);
    }
    if (threadIdx.x == 0) {
        const float* bias = g.bias[m];
        g.y[m][row0] = bias ? acc0 + bias[row0] : acc0;
        if (has_second) g.y[m][row0 + 1] = bias ? acc1 + bias[row0 + 1] : acc1;
    }
}

void launch_matmul_q8_0_group(Q80GroupDesc& g, const int8_t* d_x_qs,
                               const float* d_x_scale, size_t nb, cudaStream_t stream) {
    const size_t rows_per_block = (size_t) kMatmulQ80Warps * 2;
    unsigned total = 0;
    for (uint32_t i = 0; i < g.n; ++i) {
        g.block0[i] = total;
        total += (unsigned) ((g.rows[i] + rows_per_block - 1) / rows_per_block);
    }
    const dim3 block(DESIREEIA_CUDA_WARP, kMatmulQ80Warps);
    matmul_q8_0_group_kernel<kMatmulQ80Warps><<<total, block, 0, stream>>>(
        g, d_x_qs, d_x_scale, nb);
}

void launch_matmul_q8_0(const int8_t* d_w_qs, const __half* d_w_scale,
                         const int8_t* d_x_qs, const float* d_x_scale,
                         size_t nb, size_t rows, float* d_y, cudaStream_t stream,
                         const float* d_bias = nullptr) {
    // Each warp covers 2 rows (see the kernel), so a block covers nwarps*2.
    const dim3 block(DESIREEIA_CUDA_WARP, kMatmulQ80Warps);
    const size_t rows_per_block = (size_t) kMatmulQ80Warps * 2;
    const unsigned grid = (unsigned) ((rows + rows_per_block - 1) / rows_per_block);
    matmul_q8_0_kernel<kMatmulQ80Warps><<<grid, block, 0, stream>>>(
        d_w_qs, d_w_scale, d_x_qs, d_x_scale, nb, rows, d_y, d_bias);
}

// Persistent CUDA stream + device scratch buffers reused across calls.
//
// Why: with the weights already resident in VRAM, the measurement (see
// CUDAPiano.md) showed decode time was still dominated NOT by the kernel
// (in_dispatch=38ms over 576 calls) but by the overhead around it: a
// cudaMalloc + cudaFree for the activation/output buffers on EVERY matvec
// (~36 per token) plus the submission cost on the default stream.
// Allocating once and reusing removes the malloc/free per call; an
// explicit stream avoids the null stream's implicit synchronization and
// allows async memcpys ordered with the kernel, with a single
// synchronization point at the end of the call (the caller expects y
// already ready on return).
//
// Thread safety: the same assumption already documented for
// g_active_backend in dense_forward.cpp — the forward path calls the
// matvecs in sequence from a single thread (CPU parallelism lives INSIDE
// the CPU kernels, not around these dispatches). If multiple threads ever
// need to launch CUDA kernels together, this state has to become
// per-thread or be protected.
struct CudaScratch {
    cudaStream_t stream = nullptr;
    bool stream_ready = false;
    // [nb*32 bytes of xq][nb floats of xscale], a single allocation.
    // WARNING: the two views inside d_stage are NOT stored here. The
    // buffers only ever grow (nb_cap >= nb), so the scale offset depends
    // on the nb OF THIS CALL, not on the capacity: storing it at
    // allocation time made the kernel read the scales at the wrong offset
    // for every matrix smaller than the largest one seen so far — output
    // "!!!!" instead of text, with the self-test (a single shape) still
    // passing.
    uint8_t* d_stage = nullptr;
    float* d_y = nullptr;
    size_t nb_cap = 0;    // capacity in 32-blocks of the activation
    size_t rows_cap = 0;  // capacity in output rows
    // Host buffers for quantization: reused across calls, so quantize_q8_0
    // doesn't reallocate three vectors on every matvec (252 per token).
    std::vector<int8_t> xq;
    std::vector<float> xscale;
    std::vector<uint8_t> signs;
    // Fused-FFN buffers: gate, up/h, quantized h and the output all stay
    // on device for the whole FFN block (see matmul_q8_0_cuda_ffn_gated).
    float* d_gate = nullptr;
    float* d_up = nullptr;
    int8_t* d_hq = nullptr;
    float* d_hscale = nullptr;
    float* d_out = nullptr;
    size_t ff_cap = 0;    // capacity in elements of d_gate/d_up/d_hq
    size_t out_cap = 0;   // capacity in elements of d_out
    // Device KV cache: bit-for-bit MIRROR of the host k_cache_/v_cache_,
    // same layout [layer][pos][kv_head][head_dim]. Being a mirror written
    // at the single existing write point (write_kv_cache), it cannot
    // diverge from the original: every path (prefill, decode, CPU
    // fallback) goes through there regardless.
    uint8_t* d_kcache = nullptr;
    uint8_t* d_vcache = nullptr;
    size_t kv_bytes = 0;   // bytes allocated for each of the two
    float* d_q = nullptr;
    size_t q_cap = 0;
    float* d_scores = nullptr;
    size_t scores_cap = 0;
    float* d_attn = nullptr;
    size_t attn_cap = 0;
    // Buffers for the fused attention episode (see cuda_qkv_attention_out).
    float* d_xin = nullptr;    size_t xin_cap = 0;
    float* d_qbuf = nullptr;   size_t qbuf_cap = 0;
    float* d_kbuf = nullptr;
    float* d_vbuf = nullptr;   size_t kvbuf_cap = 0;
    float* d_rope = nullptr;   size_t rope_cap = 0;
    float* d_bq = nullptr;     size_t bq_cap = 0;
    float* d_bk = nullptr;
    float* d_bv = nullptr;     size_t bkv_cap = 0;
    uint8_t* d_stage2 = nullptr; size_t stage2_cap = 0;
    uint32_t* d_dyn = nullptr;   // [pos, cc_start], updated before each graph launch
    // Per-layer copies of the above for models whose layers differ (see
    // CudaLayerArgs::dyn_all): [slot][2] and [slot][rope_stride].
    uint32_t* d_dyn_all = nullptr; size_t dyn_all_cap = 0;
    float* d_rope_all = nullptr;   size_t rope_all_cap = 0;
    float* d_apart = nullptr;    size_t apart_cap = 0;   // split decode attention partials
    int32_t* d_xsum = nullptr;   // per-sub-block activation sums (Q4_K min term)
    size_t xsum_cap = 0;
    // Prefill (batched) buffers, kept SEPARATE from everything above on
    // purpose: none of these is ever captured into a graph, so growing
    // them costs nothing and cannot invalidate the decode path's graphs.
    float*   d_bat_x = nullptr;   size_t bat_x_cap = 0;    // n_tok*cols floats
    uint8_t* d_bat_q = nullptr;   size_t bat_q_cap = 0;    // [int8 qs][float scales]
    int32_t* d_bat_sum = nullptr; size_t bat_sum_cap = 0;  // n_tok*nb sums
    float*   d_bat_y = nullptr;   size_t bat_y_cap = 0;    // n_tok*rows floats
    float*   d_bat_qa = nullptr;  size_t bat_qa_cap = 0;   // n_tok*q_dim floats (prefill attention Q)
    float*   d_bat_o = nullptr;   size_t bat_o_cap = 0;    // n_tok*q_dim floats (prefill attention out)
    // Whole-layer episode buffers.
    float* d_xn = nullptr;
    float* d_xn2 = nullptr;
    float* d_xnew = nullptr;
    size_t xn_cap = 0;
};

CudaScratch g_scratch;

// Per-layer captured CUDA graphs.
//
// Why: measured on this workload, what dominates decode is the NUMBER of
// kernel launches — ~500 per token at roughly 32 us of driver overhead
// each, about 16 ms of the 30 ms a token takes, against a pure-bandwidth
// floor of ~15 ms. Fusing kernels and cutting synchronisations both
// failed to move it (each was measured, each is documented in
// docs/CUDAPiano.md). A captured graph turns a whole episode into ONE
// submission, which is the only thing that attacks that cost directly.
//
// The FFN episode is captured as-is because nothing in it changes between
// tokens: weights, buffers and shapes are all fixed per layer, and the
// only varying input (the activation) already arrives through a fixed
// device buffer written before the launch. The graph is therefore static
// and needs no per-token update.
struct CapturedGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
};
std::unordered_map<const void*, CapturedGraph> g_ffn_graphs;

// Attention episode: same idea, plus per-layer bias copies. The biases are
// layer constants, so they are uploaded once at capture time instead of
// being re-sent every token into shared scratch (which a captured graph
// could not reference safely anyway, since another layer would overwrite
// it before the graph ran).
struct AttnGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    float* d_bq = nullptr;
    float* d_bk = nullptr;
    float* d_bv = nullptr;
};
std::unordered_map<const void*, AttnGraph> g_attn_graphs;

// Whole-layer episode: attention block AND feed-forward in a single graph,
// with the norms and residuals on device too. This is what removes the
// per-episode synchronisation: one graph launch and one sync per layer
// instead of two of each, and the layer's intermediate values never touch
// the host.
struct LayerGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    float* d_bq = nullptr;
    float* d_bk = nullptr;
    float* d_bv = nullptr;
    float* d_attn_norm = nullptr;
    float* d_ffn_norm = nullptr;
    float* d_q_norm = nullptr;
    float* d_k_norm = nullptr;
    float* d_post_attn_norm = nullptr;
    float* d_post_ffn_norm = nullptr;
};
std::unordered_map<const void*, LayerGraph> g_layer_graphs;

// Any buffer reallocation invalidates every captured graph, since the
// captured nodes hold the old addresses.
// Opt-in diagnostics: DESIREEIA_CUDA_DEBUG=1 reports graph invalidation and
// any runtime CUDA error. Kept because the failure mode these were written
// to chase is SILENT - wrong output, no error - so the next time something
// similar happens there is already a way to see what the backend is doing.
static bool cuda_debug_enabled() {
    static const bool on = std::getenv("DESIREEIA_CUDA_DEBUG") != nullptr;
    return on;
}

// Defined with the batched prefill layer at the end of this file.
void layer_batch_release(bool free_buffers);

void invalidate_graphs() {
    if (cuda_debug_enabled() && !g_layer_graphs.empty()) {
        std::fprintf(stderr, "[cuda] invalidate_graphs: dropping %zu layer graphs\n", g_layer_graphs.size());
    }
    for (auto& kv : g_ffn_graphs) {
        if (kv.second.exec) cudaGraphExecDestroy(kv.second.exec);
        if (kv.second.graph) cudaGraphDestroy(kv.second.graph);
    }
    g_ffn_graphs.clear();
    for (auto& kv : g_attn_graphs) {
        if (kv.second.exec) cudaGraphExecDestroy(kv.second.exec);
        if (kv.second.graph) cudaGraphDestroy(kv.second.graph);
        if (kv.second.d_bq) cudaFree(kv.second.d_bq);
        if (kv.second.d_bk) cudaFree(kv.second.d_bk);
        if (kv.second.d_bv) cudaFree(kv.second.d_bv);
    }
    g_attn_graphs.clear();
    for (auto& kv : g_layer_graphs) {
        if (kv.second.exec) cudaGraphExecDestroy(kv.second.exec);
        if (kv.second.graph) cudaGraphDestroy(kv.second.graph);
        if (kv.second.d_bq) cudaFree(kv.second.d_bq);
        if (kv.second.d_bk) cudaFree(kv.second.d_bk);
        if (kv.second.d_bv) cudaFree(kv.second.d_bv);
        if (kv.second.d_attn_norm) cudaFree(kv.second.d_attn_norm);
        if (kv.second.d_ffn_norm) cudaFree(kv.second.d_ffn_norm);
        if (kv.second.d_q_norm) cudaFree(kv.second.d_q_norm);
        if (kv.second.d_k_norm) cudaFree(kv.second.d_k_norm);
        if (kv.second.d_post_attn_norm) cudaFree(kv.second.d_post_attn_norm);
        if (kv.second.d_post_ffn_norm) cudaFree(kv.second.d_post_ffn_norm);
    }
    g_layer_graphs.clear();
    // The batched prefill layer keys its per-layer norm/bias copies the
    // same way (device pointer of Wq): whenever the decode graphs are
    // dropped - weights re-laid out, model unloaded - those go too, or a
    // reloaded model could hit a reused pointer and read the previous
    // model's norms.
    layer_batch_release(false);
}

cudaStream_t scratch_stream() {
    if (!g_scratch.stream_ready) {
        // Active spin instead of the default blocking wait: decode does
        // ~108 synchronizations per token (one per matvec group), each on
        // a few tens of microseconds of GPU work. With blocking scheduling
        // the cost of putting the thread to sleep and waking it up is the
        // same order of magnitude as the work being waited on; with a spin
        // the thread stays on the core and per-synchronization latency
        // collapses. The trade-off (one core busy waiting) is acceptable
        // here: that thread would have nothing else to do until the
        // result anyway.
        cudaSetDeviceFlags(cudaDeviceScheduleSpin);
        if (cudaStreamCreate(&g_scratch.stream) != cudaSuccess) {
            g_scratch.stream = nullptr; // falls back to the null stream: correct, just slower
        }
        g_scratch.stream_ready = true;
    }
    return g_scratch.stream;
}

// Grows the scratch buffers only when a larger one is needed (shapes are
// stable per model: after the first few tokens every matrix has already
// found its capacity and never reallocates again).
bool scratch_reserve(size_t nb, size_t rows) {
    if (nb > g_scratch.nb_cap || rows > g_scratch.rows_cap) invalidate_graphs();
    if (nb > g_scratch.nb_cap) {
        if (g_scratch.d_stage) cudaFree(g_scratch.d_stage);
        g_scratch.d_stage = nullptr;
        g_scratch.nb_cap = 0;
        // A single device allocation with the same layout as the host
        // staging ([xq][xscale]), so the H2D is one contiguous copy.
        // nb*32 is a multiple of 4, so the float part stays aligned.
        if (cudaMalloc(&g_scratch.d_stage, nb * 32 + nb * sizeof(float)) != cudaSuccess) return false;
        g_scratch.nb_cap = nb;
    }
    if (rows > g_scratch.rows_cap) {
        if (g_scratch.d_y) cudaFree(g_scratch.d_y);
        g_scratch.d_y = nullptr;
        g_scratch.rows_cap = 0;
        if (cudaMalloc(&g_scratch.d_y, rows * sizeof(float)) != cudaSuccess) return false;
        g_scratch.rows_cap = rows;
    }
    return true;
}

// Grow-only buffers for the fused attention episode.
// EVERY buffer this reserves is baked into the captured per-layer graphs
// as a fixed device address, so any reallocation here has to drop those
// graphs first — otherwise the next replay reads freed VRAM. This is the
// same failure that made the KV cache corrupt output past its first grow
// (see cuda_kv_cache_reserve); it is latent here rather than routine only
// because these shapes usually settle on the first token. One case where
// it does not settle: an architecture whose n_rot differs per layer (a
// hybrid sliding-window model rotating 64 dims on full layers and 256 on
// local ones) grows d_rope on a LATER layer than the one that captured
// its graph first.
static void attention_decode_q8(const float* q, const uint8_t* kc, const uint8_t* vc, float* out,
                                uint32_t n_head, uint32_t head_dim, uint32_t heads_per_kv, size_t row_bytes,
                                size_t pos_bytes, size_t layer_off, const uint32_t* dyn, float inv_d,
                                cudaStream_t stream) {
    static const bool split_on = std::getenv("DESIREEIA_CUDA_NO_ATTN_SPLIT") == nullptr;
    const size_t q_dim = (size_t) n_head * head_dim;
    if (split_on && head_dim % 32 == 0 && head_dim <= 32 * kAdecMaxBlk && g_scratch.d_apart &&
        g_scratch.apart_cap >= (size_t) kAdecMaxSplit * 2 * q_dim && head_dim >= 2) {
        static const uint32_t sms = [] {
            int dev = 0, n = 0;
            if (cudaGetDevice(&dev) != cudaSuccess ||
                cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || n <= 0) return 1u;
            return (uint32_t) n;
        }();
        uint32_t nsplit = (2 * sms + n_head - 1) / n_head;
        nsplit = nsplit < 1 ? 1 : (nsplit > (uint32_t) kAdecMaxSplit ? (uint32_t) kAdecMaxSplit : nsplit);
        attn_dec_part_kernel<<<dim3(n_head, nsplit), kAdecWarps * 32, 0, stream>>>(
            q, kc, vc, g_scratch.d_apart, head_dim, heads_per_kv, row_bytes, pos_bytes, layer_off, dyn, inv_d, nsplit);
        attn_dec_merge_kernel<<<n_head, 128, 0, stream>>>(g_scratch.d_apart, out, head_dim, nsplit);
        return;
    }
    attention_kernel_q8<<<n_head, 128, 2 * 128 * sizeof(float), stream>>>(
        q, kc, vc, out, head_dim, heads_per_kv, row_bytes, pos_bytes, layer_off, dyn, inv_d);
}

bool scratch_reserve_attn(size_t q_dim, size_t kv_dim, size_t n_rot, size_t pos_bytes) {
    (void) pos_bytes;
    auto grow_f = [](float*& p, size_t& cap, size_t need) {
        if (need <= cap && p) return true;
        invalidate_graphs();
        if (p) cudaFree(p);
        p = nullptr; cap = 0;
        if (cudaMalloc(&p, need * sizeof(float)) != cudaSuccess) return false;
        cap = need;
        return true;
    };
    if (!grow_f(g_scratch.d_xin, g_scratch.xin_cap, q_dim + kv_dim)) return false;
    if (!grow_f(g_scratch.d_qbuf, g_scratch.qbuf_cap, q_dim)) return false;
    // d_attn holds the attention output before the `o` projection. It used
    // to be allocated only by cuda_attention_out; the fused episode
    // replaces that call, so without this it stayed null and the kernel
    // wrote to address 0 — a sticky CUDA error that then made every later
    // launch fail silently.
    if (!grow_f(g_scratch.d_attn, g_scratch.attn_cap, q_dim)) return false;
    // partials of the split decode attention: up to kAdecMaxSplit x (q_dim + 2 per head)
    if (!grow_f(g_scratch.d_apart, g_scratch.apart_cap, (size_t) kAdecMaxSplit * 2 * q_dim)) return false;
    if (g_scratch.kvbuf_cap < kv_dim) {
        invalidate_graphs();
        if (g_scratch.d_kbuf) cudaFree(g_scratch.d_kbuf);
        if (g_scratch.d_vbuf) cudaFree(g_scratch.d_vbuf);
        g_scratch.d_kbuf = nullptr; g_scratch.d_vbuf = nullptr; g_scratch.kvbuf_cap = 0;
        if (cudaMalloc(&g_scratch.d_kbuf, kv_dim * sizeof(float)) != cudaSuccess) return false;
        if (cudaMalloc(&g_scratch.d_vbuf, kv_dim * sizeof(float)) != cudaSuccess) return false;
        g_scratch.kvbuf_cap = kv_dim;
    }
    if (!grow_f(g_scratch.d_rope, g_scratch.rope_cap, n_rot)) return false;
    if (!grow_f(g_scratch.d_bq, g_scratch.bq_cap, q_dim)) return false;
    if (g_scratch.bkv_cap < kv_dim) {
        invalidate_graphs();
        if (g_scratch.d_bk) cudaFree(g_scratch.d_bk);
        if (g_scratch.d_bv) cudaFree(g_scratch.d_bv);
        g_scratch.d_bk = nullptr; g_scratch.d_bv = nullptr; g_scratch.bkv_cap = 0;
        if (cudaMalloc(&g_scratch.d_bk, kv_dim * sizeof(float)) != cudaSuccess) return false;
        if (cudaMalloc(&g_scratch.d_bv, kv_dim * sizeof(float)) != cudaSuccess) return false;
        g_scratch.bkv_cap = kv_dim;
    }
    const size_t stage2_need = (q_dim / 32) * 32 + (q_dim / 32) * sizeof(float);
    if (stage2_need > g_scratch.stage2_cap) {
        invalidate_graphs();
        if (g_scratch.d_stage2) cudaFree(g_scratch.d_stage2);
        g_scratch.d_stage2 = nullptr; g_scratch.stage2_cap = 0;
        if (cudaMalloc(&g_scratch.d_stage2, stage2_need) != cudaSuccess) return false;
        g_scratch.stage2_cap = stage2_need;
    }
    return true;
}

bool scratch_reserve_sums(size_t nb) {
    if (nb <= g_scratch.xsum_cap && g_scratch.d_xsum) return true;
    if (g_scratch.d_xsum) cudaFree(g_scratch.d_xsum);
    g_scratch.d_xsum = nullptr;
    g_scratch.xsum_cap = 0;
    invalidate_graphs();
    if (cudaMalloc(&g_scratch.d_xsum, nb * sizeof(int32_t)) != cudaSuccess) return false;
    g_scratch.xsum_cap = nb;
    return true;
}

// Fused-FFN buffers. Same grow-only policy as the others.
bool scratch_reserve_ffn(size_t n_ff, size_t n_embd) {
    if (n_ff > g_scratch.ff_cap || n_embd > g_scratch.out_cap) invalidate_graphs();
    if (n_ff > g_scratch.ff_cap) {
        if (g_scratch.d_gate) cudaFree(g_scratch.d_gate);
        if (g_scratch.d_up) cudaFree(g_scratch.d_up);
        if (g_scratch.d_hq) cudaFree(g_scratch.d_hq);
        if (g_scratch.d_hscale) cudaFree(g_scratch.d_hscale);
        g_scratch.d_gate = nullptr; g_scratch.d_up = nullptr;
        g_scratch.d_hq = nullptr; g_scratch.d_hscale = nullptr;
        g_scratch.ff_cap = 0;
        const size_t nblk = (n_ff + 31) / 32;
        if (cudaMalloc(&g_scratch.d_gate, n_ff * sizeof(float)) != cudaSuccess) return false;
        if (cudaMalloc(&g_scratch.d_up, n_ff * sizeof(float)) != cudaSuccess) return false;
        if (cudaMalloc(&g_scratch.d_hq, nblk * 32) != cudaSuccess) return false;
        if (cudaMalloc(&g_scratch.d_hscale, nblk * sizeof(float)) != cudaSuccess) return false;
        g_scratch.ff_cap = n_ff;
    }
    if (n_embd > g_scratch.out_cap) {
        if (g_scratch.d_out) cudaFree(g_scratch.d_out);
        g_scratch.d_out = nullptr;
        g_scratch.out_cap = 0;
        if (cudaMalloc(&g_scratch.d_out, n_embd * sizeof(float)) != cudaSuccess) return false;
        g_scratch.out_cap = n_embd;
    }
    return true;
}

// Unpacks the native block_q8_0 bytes (fp16 scale + 32 int8 qs, as read
// from the GGUF) into two flat device-friendly arrays: contiguous int8 qs
// and scales. Same logic for both the one-shot (resident) upload and the
// per-call (fallback) path, factored out here to avoid duplicating it.
// Weight scales are kept in fp16, exactly as they are on disk, rather
// than widened to float. They are read once per (row, block), i.e. one
// value per 32 weights: at float that is 385 MB of the 3.47 GB a token
// reads on the test model, and decode here is bandwidth-bound, so halving
// them is a straight 5.6% cut of everything that has to move. No
// precision is lost either: the value came from a 16-bit field to begin
// with.
void unpack_q8_0(const uint8_t* q8_data, size_t rows, size_t nb,
                  std::vector<int8_t>& w_qs, std::vector<uint16_t>& w_scale) {
    const size_t row_bytes = nb * (sizeof(desireeia_half) + 32);
    w_qs.resize(rows * nb * 32);
    w_scale.resize(rows * nb);
    for (size_t r = 0; r < rows; ++r) {
        const uint8_t* row_ptr = q8_data + r * row_bytes;
        for (size_t b = 0; b < nb; ++b) {
            const auto* blk = reinterpret_cast<const block_q8_0*>(row_ptr + b * sizeof(block_q8_0));
            w_scale[r * nb + b] = (uint16_t) blk->d;   // already fp16 on disk
            std::memcpy(w_qs.data() + (r * nb + b) * 32, blk->qs, 32);
        }
    }
}

} // namespace

// Frees a device buffer allocated by any function below. Generic (void*)
// signature so it can be used as the deleter of a std::shared_ptr<void>
// on the C++ side (dense_forward.h/.cpp) without having to include
// cuda_runtime.h there.
// Per-matrix preparation of the FP16 product on the stored blocks (half2
// scales, repacked Q6_K): it depends only on the weights, so after the
// first prefill it is kept on the device - while there is room - instead
// of being rebuilt for every layer of every prompt (~7% of a 512-token
// prefill). Keyed by the weights' device address; the entry goes when those
// weights are freed, so another model reusing the address never sees it.
struct HkqPrep { void* scales = nullptr; void* repack = nullptr; size_t bytes = 0; };
std::unordered_map<const void*, HkqPrep> g_hkq_prep;

static void hkq_prep_erase(const void* w) {
    auto it = g_hkq_prep.find(w);
    if (it == g_hkq_prep.end()) return;
    if (it->second.scales) cudaFree(it->second.scales);
    if (it->second.repack) cudaFree(it->second.repack);
    g_hkq_prep.erase(it);
}
static void hkq_prep_clear() {
    for (auto& kv : g_hkq_prep) {
        if (kv.second.scales) cudaFree(kv.second.scales);
        if (kv.second.repack) cudaFree(kv.second.repack);
    }
    g_hkq_prep.clear();
}

void cuda_free_device(void* p) {
    if (!p) return;
    hkq_prep_erase(p);
    cudaFree(p);
}

// Releases the persistent stream and scratch buffers. Called from
// DenseForward's destructor: without this, loading multiple models in the
// same process (the CLI and the tests both do this) would leave a
// dangling stream and the previous model's buffers, sized for shapes no
// longer needed.
// State of cuda_head_forward, released by cuda_backend_shutdown.
struct HeadNormCache { float* d = nullptr; std::vector<float> host; };
std::unordered_map<const float*, HeadNormCache> g_head_norms;
float* g_head_pinned = nullptr;
size_t g_head_pinned_cap = 0;

void cuda_backend_shutdown() {
    invalidate_graphs();
    hkq_prep_clear();
    for (auto& kv : g_head_norms) if (kv.second.d) cudaFree(kv.second.d);
    g_head_norms.clear();
    if (g_head_pinned) { cudaFreeHost(g_head_pinned); g_head_pinned = nullptr; g_head_pinned_cap = 0; }
    if (g_scratch.d_dyn_all) { cudaFree(g_scratch.d_dyn_all); g_scratch.d_dyn_all = nullptr; g_scratch.dyn_all_cap = 0; }
    if (g_scratch.d_rope_all) { cudaFree(g_scratch.d_rope_all); g_scratch.d_rope_all = nullptr; g_scratch.rope_all_cap = 0; }
    layer_batch_release(true);
    if (g_scratch.d_stage) { cudaFree(g_scratch.d_stage); g_scratch.d_stage = nullptr; }
    if (g_scratch.d_y) { cudaFree(g_scratch.d_y); g_scratch.d_y = nullptr; }
    if (g_scratch.d_gate) { cudaFree(g_scratch.d_gate); g_scratch.d_gate = nullptr; }
    if (g_scratch.d_up) { cudaFree(g_scratch.d_up); g_scratch.d_up = nullptr; }
    if (g_scratch.d_hq) { cudaFree(g_scratch.d_hq); g_scratch.d_hq = nullptr; }
    if (g_scratch.d_hscale) { cudaFree(g_scratch.d_hscale); g_scratch.d_hscale = nullptr; }
    if (g_scratch.d_out) { cudaFree(g_scratch.d_out); g_scratch.d_out = nullptr; }
    if (g_scratch.d_kcache) { cudaFree(g_scratch.d_kcache); g_scratch.d_kcache = nullptr; }
    if (g_scratch.d_vcache) { cudaFree(g_scratch.d_vcache); g_scratch.d_vcache = nullptr; }
    if (g_scratch.d_q) { cudaFree(g_scratch.d_q); g_scratch.d_q = nullptr; }
    if (g_scratch.d_scores) { cudaFree(g_scratch.d_scores); g_scratch.d_scores = nullptr; }
    if (g_scratch.d_attn) { cudaFree(g_scratch.d_attn); g_scratch.d_attn = nullptr; }
    if (g_scratch.d_apart) { cudaFree(g_scratch.d_apart); g_scratch.d_apart = nullptr; g_scratch.apart_cap = 0; }
    if (g_scratch.d_xin) { cudaFree(g_scratch.d_xin); g_scratch.d_xin = nullptr; }
    if (g_scratch.d_qbuf) { cudaFree(g_scratch.d_qbuf); g_scratch.d_qbuf = nullptr; }
    if (g_scratch.d_kbuf) { cudaFree(g_scratch.d_kbuf); g_scratch.d_kbuf = nullptr; }
    if (g_scratch.d_vbuf) { cudaFree(g_scratch.d_vbuf); g_scratch.d_vbuf = nullptr; }
    if (g_scratch.d_rope) { cudaFree(g_scratch.d_rope); g_scratch.d_rope = nullptr; }
    if (g_scratch.d_bq) { cudaFree(g_scratch.d_bq); g_scratch.d_bq = nullptr; }
    if (g_scratch.d_bk) { cudaFree(g_scratch.d_bk); g_scratch.d_bk = nullptr; }
    if (g_scratch.d_bv) { cudaFree(g_scratch.d_bv); g_scratch.d_bv = nullptr; }
    if (g_scratch.d_stage2) { cudaFree(g_scratch.d_stage2); g_scratch.d_stage2 = nullptr; }
    if (g_scratch.d_xsum) { cudaFree(g_scratch.d_xsum); g_scratch.d_xsum = nullptr; }
    if (g_scratch.d_bat_x) { cudaFree(g_scratch.d_bat_x); g_scratch.d_bat_x = nullptr; }
    if (g_scratch.d_bat_q) { cudaFree(g_scratch.d_bat_q); g_scratch.d_bat_q = nullptr; }
    if (g_scratch.d_bat_sum) { cudaFree(g_scratch.d_bat_sum); g_scratch.d_bat_sum = nullptr; }
    if (g_scratch.d_bat_y) { cudaFree(g_scratch.d_bat_y); g_scratch.d_bat_y = nullptr; }
    if (g_scratch.d_bat_qa) { cudaFree(g_scratch.d_bat_qa); g_scratch.d_bat_qa = nullptr; }
    if (g_scratch.d_bat_o) { cudaFree(g_scratch.d_bat_o); g_scratch.d_bat_o = nullptr; }
    g_scratch.xsum_cap = 0;
    g_scratch.xin_cap = 0; g_scratch.qbuf_cap = 0; g_scratch.kvbuf_cap = 0;
    g_scratch.rope_cap = 0; g_scratch.bq_cap = 0; g_scratch.bkv_cap = 0;
    g_scratch.stage2_cap = 0;
    g_scratch.kv_bytes = 0; g_scratch.q_cap = 0;
    g_scratch.scores_cap = 0; g_scratch.attn_cap = 0;
    g_scratch.ff_cap = 0;
    g_scratch.out_cap = 0;
    g_scratch.nb_cap = 0;
    g_scratch.rows_cap = 0;
    if (g_scratch.stream) { cudaStreamDestroy(g_scratch.stream); g_scratch.stream = nullptr; }
    g_scratch.stream_ready = false;
}

// Uploads a weight matrix to VRAM in the representation its format wants.
//
// Q8_0 is split into a contiguous int8 array plus an fp16 scale array,
// because that layout lets the kernel use aligned 128-bit loads. The
// K-quants are uploaded byte-for-byte as they are on disk: their blocks
// are already compact (144 bytes per 256 weights for Q4_K, 210 for Q6_K)
// and widening them would cost exactly the bandwidth decode is limited
// by. out_d_scale stays null for those — the scales live inside the
// blocks.
// Bytes per block and weights per block for each format we decode on
// device. Zero means "not handled here".
size_t cuda_block_bytes(int format) {
    switch (format) {
        case DESIREEIA_CUDA_FMT_Q4_0: return 18;
        case DESIREEIA_CUDA_FMT_Q4_1: return 20;
        case DESIREEIA_CUDA_FMT_Q5_0: return 22;
        case DESIREEIA_CUDA_FMT_Q5_1: return 24;
        case DESIREEIA_CUDA_FMT_Q4_K: return 144;
        case DESIREEIA_CUDA_FMT_Q5_K: return 176;
        case DESIREEIA_CUDA_FMT_Q6_K: return 210;
        default: return 0;
    }
}
size_t cuda_block_weights(int format) {
    switch (format) {
        case DESIREEIA_CUDA_FMT_Q4_0:
        case DESIREEIA_CUDA_FMT_Q4_1:
        case DESIREEIA_CUDA_FMT_Q5_0:
        case DESIREEIA_CUDA_FMT_Q5_1: return 32;
        case DESIREEIA_CUDA_FMT_Q4_K:
        case DESIREEIA_CUDA_FMT_Q5_K:
        case DESIREEIA_CUDA_FMT_Q6_K: return 256;
        default: return 0;
    }
}

bool cuda_upload_weights(int format, const uint8_t* data, size_t rows, size_t cols,
                          void** out_d_qs, void** out_d_scale) {
    *out_d_qs = nullptr;
    *out_d_scale = nullptr;
    if (rows == 0 || cols == 0) return false;

    const size_t blk = cuda_block_bytes(format);
    const size_t per_block = cuda_block_weights(format);
    if (blk && per_block) {
        if (cols % per_block != 0) return false;
        const size_t bytes = rows * (cols / per_block) * blk;
        void* d = nullptr;
        if (cudaMalloc(&d, bytes) != cudaSuccess) return false;
        if (cudaMemcpy(d, data, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(d);
            return false;
        }
        *out_d_qs = d;
        return true;
    }
    return false;
}

bool matmul_q8_0_cuda_upload_weights(const uint8_t* q8_data, size_t rows, size_t cols,
                                      void** out_d_qs, void** out_d_scale) {
    *out_d_qs = nullptr;
    *out_d_scale = nullptr;
    if (cols == 0 || cols % 32 != 0 || rows == 0) return false;
    const size_t nb = cols / 32;

    std::vector<int8_t> w_qs;
    std::vector<uint16_t> w_scale;
    unpack_q8_0(q8_data, rows, nb, w_qs, w_scale);

    int8_t* d_qs = nullptr;
    uint16_t* d_scale = nullptr;
    if (cudaMalloc(&d_qs, w_qs.size()) != cudaSuccess) return false;
    if (cudaMalloc(&d_scale, w_scale.size() * sizeof(uint16_t)) != cudaSuccess) {
        cudaFree(d_qs);
        return false;
    }
    cudaMemcpy(d_qs, w_qs.data(), w_qs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale, w_scale.data(), w_scale.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

    *out_d_qs = d_qs;
    *out_d_scale = d_scale;
    return true;
}

// Like matmul_q8_0_cuda (below) but with the weights ALREADY on device
// (from an earlier matmul_q8_0_cuda_upload_weights): every call quantizes
// and uploads only the activation (a few tens of KB, not the whole weight
// matrix) and downloads only the result — the dominant cost measured in
// the earlier version (weight upload repeated every token) disappears.
int matmul_q8_0_cuda_resident(const void* d_qs, const void* d_scale, size_t rows, size_t cols,
                               const float* x, float* y) {
    if (!d_qs || !d_scale || cols == 0 || cols % 32 != 0 || rows == 0) {
        return DESIREEIA_ERR_NOT_SUPPORTED;
    }
    ScopedTimer prof_t(profile_counters().ns_cuda_resident);
    profile_counters().calls_cuda_resident.fetch_add(1, std::memory_order_relaxed);
    const size_t nb = cols / 32;

    // Buffers reused across calls (no cudaMalloc/cudaFree here) and a
    // persistent stream: see the note on CudaScratch at the top of the file.
    if (!scratch_reserve(nb, rows)) return DESIREEIA_ERR_IO;

    // The quantization vectors live in scratch: quantize_q8_0 only resizes
    // them the first time, then reuses the capacity already there.
    quantize_q8_0(x, cols, g_scratch.xq, g_scratch.xscale, g_scratch.signs);
    if (g_scratch.xq.size() != nb * 32 || g_scratch.xscale.size() != nb) {
        return DESIREEIA_ERR_NOT_SUPPORTED;
    }

    cudaStream_t stream = scratch_stream();

    // Views computed on THIS call's nb (see the note on d_stage). Two H2D
    // copies straight from the vectors, at the right offsets inside the
    // single allocation: measured faster than staging through pinned
    // memory (one copy, but preceded by two host memcpys), which at these
    // volumes cost more than it saved.
    int8_t* d_xq = reinterpret_cast<int8_t*>(g_scratch.d_stage);
    float* d_xscale = reinterpret_cast<float*>(g_scratch.d_stage + nb * 32);
    cudaMemcpyAsync(d_xq, g_scratch.xq.data(), nb * 32, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_xscale, g_scratch.xscale.data(), nb * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    launch_matmul_q8_0(static_cast<const int8_t*>(d_qs), static_cast<const __half*>(d_scale),
                        d_xq, d_xscale, nb, rows, g_scratch.d_y, stream);
    if (cudaGetLastError() != cudaSuccess) return DESIREEIA_ERR_IO;

    cudaMemcpyAsync(y, g_scratch.d_y, rows * sizeof(float), cudaMemcpyDeviceToHost, stream);
    // Single synchronization point: the operations above are already
    // ordered relative to each other by the stream, the caller expects y ready.
    if (cudaStreamSynchronize(stream) != cudaSuccess) return DESIREEIA_ERR_IO;
    return DESIREEIA_OK;
}

// One launch point for every format decoded natively on device, so the
// callers stay format-agnostic.
void launch_matmul_kquant(int format, const uint8_t* d_w,
                           const int8_t* d_xq, const float* d_xscale,
                           const int32_t* d_xsum, size_t cols, size_t rows,
                           float* d_y, const float* d_bias, cudaStream_t stream) {
    const dim3 block(DESIREEIA_CUDA_WARP, kMatmulQ80Warps);
    const unsigned grid = (unsigned) ((rows + kMatmulQ80Warps - 1) / kMatmulQ80Warps);
    // Q6_K covers two rows per warp and Q4_K four, so both use smaller
    // grids than the one-row formats.
    const size_t rows_pair = (size_t) kMatmulQ80Warps * 2;
    const unsigned grid2 = (unsigned) ((rows + rows_pair - 1) / rows_pair);
    const size_t rows_quad = (size_t) kMatmulQ80Warps * 4;
    const unsigned grid4 = (unsigned) ((rows + rows_quad - 1) / rows_quad);
    const size_t nb = cols / 32;
    const size_t n_super = cols / 256;
    switch (format) {
        case DESIREEIA_CUDA_FMT_Q4_0:
            matmul_q4_0_kernel<kMatmulQ80Warps><<<grid, block, 0, stream>>>(
                d_w, d_xq, d_xscale, d_xsum, nb, rows, d_y, d_bias);
            break;
        case DESIREEIA_CUDA_FMT_Q4_1:
            matmul_q4_1_kernel<kMatmulQ80Warps><<<grid, block, 0, stream>>>(
                d_w, d_xq, d_xscale, d_xsum, nb, rows, d_y, d_bias);
            break;
        case DESIREEIA_CUDA_FMT_Q5_0:
            matmul_q5_0_kernel<kMatmulQ80Warps><<<grid, block, 0, stream>>>(
                d_w, d_xq, d_xscale, d_xsum, nb, rows, d_y, d_bias);
            break;
        case DESIREEIA_CUDA_FMT_Q5_1:
            matmul_q5_1_kernel<kMatmulQ80Warps><<<grid, block, 0, stream>>>(
                d_w, d_xq, d_xscale, d_xsum, nb, rows, d_y, d_bias);
            break;
        case DESIREEIA_CUDA_FMT_Q4_K:
            matmul_q4_k_kernel<kMatmulQ80Warps><<<grid4, block, 0, stream>>>(
                d_w, d_xq, d_xscale, d_xsum, n_super, rows, d_y, d_bias);
            break;
        case DESIREEIA_CUDA_FMT_Q5_K:
            matmul_q5_k_kernel<kMatmulQ80Warps><<<grid, block, 0, stream>>>(
                d_w, d_xq, d_xscale, d_xsum, n_super, rows, d_y, d_bias);
            break;
        case DESIREEIA_CUDA_FMT_Q6_K:
            matmul_q6_k_kernel<kMatmulQ80Warps><<<grid2, block, 0, stream>>>(
                d_w, d_xq, d_xscale, n_super, rows, d_y, d_bias);
            break;
        default:
            break;
    }
}

// Several K-quant mat-vecs that share the activation, the format and the
// column count, launched over one grid. Returns false when the format has
// no grouped kernel, so the caller keeps its per-matrix launches.
//
// The grid is the sum of what the members would have used on their own,
// and each block maps itself back to its matrix through blk0. Rows per
// warp differ by format (four for Q4_K, two for Q6_K), so the block
// counts are computed with the same divisor the single-matrix path uses.
// Q4_K and Q6_K matrices in one grid (a Q4_K_M layer often has Q and K in
// Q4_K and V in Q6_K): each block runs the row routine of its matrix's
// format - a uniform branch - with that format's rows per warp.
template <int nwarps>
__global__ void matmul_kmixed_group_kernel(KQuantGroupDesc g,
                                           const int8_t* __restrict__ x_qs,
                                           const float* __restrict__ x_scale,
                                           const int32_t* __restrict__ x_sum,
                                           size_t n_super) {
    const int m = kquant_group_pick(g);
    const size_t lb = blockIdx.x - g.blk0[m];
    if (g.fmt[m] == DESIREEIA_CUDA_FMT_Q4_K) {
        q4k_row_group(g.w[m], x_qs, x_scale, x_sum, n_super, g.rows[m], g.y[m], g.bias[m],
                      (lb * nwarps + threadIdx.y) * 4);
    } else {
        q6k_row_group(g.w[m], x_qs, x_scale, n_super, g.rows[m], g.y[m], g.bias[m],
                      (lb * nwarps + threadIdx.y) * 2);
    }
}

// Mixed Q4_K/Q6_K group (g.fmt set per matrix). False when a matrix has
// another format.
bool launch_matmul_kquant_mixed_group(KQuantGroupDesc& g, const int8_t* d_xq, const float* d_xscale,
                                      const int32_t* d_xsum, size_t cols, cudaStream_t stream) {
    static const bool on = std::getenv("DESIREEIA_CUDA_NO_GROUP") == nullptr &&
                           std::getenv("DESIREEIA_CUDA_NO_MIXED_GROUP") == nullptr;
    if (!on || g.n <= 1) return false;
    unsigned total = 0;
    for (int i = 0; i < g.n; ++i) {
        size_t rpb;
        if (g.fmt[i] == DESIREEIA_CUDA_FMT_Q4_K)      rpb = (size_t) kMatmulQ80Warps * 4;
        else if (g.fmt[i] == DESIREEIA_CUDA_FMT_Q6_K) rpb = (size_t) kMatmulQ80Warps * 2;
        else return false;
        g.blk0[i] = total;
        total += (unsigned) ((g.rows[i] + rpb - 1) / rpb);
    }
    if (total == 0) return false;
    matmul_kmixed_group_kernel<kMatmulQ80Warps><<<total, dim3(DESIREEIA_CUDA_WARP, kMatmulQ80Warps), 0, stream>>>(
        g, d_xq, d_xscale, d_xsum, cols / 256);
    return true;
}

bool launch_matmul_kquant_group(int format, KQuantGroupDesc& g,
                                 const int8_t* d_xq, const float* d_xscale,
                                 const int32_t* d_xsum, size_t cols,
                                 cudaStream_t stream) {
    // Escape hatch, same shape as the other CUDA ones: grouping must be a
    // pure scheduling change, so the two paths have to be comparable
    // bit-for-bit on demand rather than on the strength of the argument
    // that they obviously are.
    static const bool group_enabled = std::getenv("DESIREEIA_CUDA_NO_GROUP") == nullptr;
    if (!group_enabled) return false;
    if (g.n <= 1) return false;
    size_t rows_per_block;
    if (format == DESIREEIA_CUDA_FMT_Q4_K)      rows_per_block = (size_t) kMatmulQ80Warps * 4;
    else if (format == DESIREEIA_CUDA_FMT_Q6_K) rows_per_block = (size_t) kMatmulQ80Warps * 2;
    else return false;

    unsigned total = 0;
    for (int i = 0; i < g.n; ++i) {
        g.blk0[i] = total;
        total += (unsigned) ((g.rows[i] + rows_per_block - 1) / rows_per_block);
    }
    if (total == 0) return false;

    const dim3 block(DESIREEIA_CUDA_WARP, kMatmulQ80Warps);
    const size_t n_super = cols / 256;
    if (format == DESIREEIA_CUDA_FMT_Q4_K) {
        matmul_q4_k_group_kernel<kMatmulQ80Warps><<<total, block, 0, stream>>>(
            g, d_xq, d_xscale, d_xsum, n_super);
    } else {
        matmul_q6_k_group_kernel<kMatmulQ80Warps><<<total, block, 0, stream>>>(
            g, d_xq, d_xscale, n_super);
    }
    return true;
}

// Resident mat-vec for the K-quant formats: activation quantized on
// device, weights read in their native layout, one synchronisation.
int matmul_kquant_cuda_resident(int format, const void* d_w, size_t rows, size_t cols,
                                 const float* x, float* y) {
    const size_t per_block = cuda_block_weights(format);
    if (!d_w || rows == 0 || per_block == 0 || cols % per_block != 0 || cols % 32 != 0) {
        return DESIREEIA_ERR_NOT_SUPPORTED;
    }

    ScopedTimer prof_t(profile_counters().ns_cuda_resident);
    profile_counters().calls_cuda_resident.fetch_add(1, std::memory_order_relaxed);

    const size_t nb = cols / 32;
    if (!scratch_reserve(nb, rows)) return DESIREEIA_ERR_IO;
    if (!scratch_reserve_sums(nb)) return DESIREEIA_ERR_IO;
    if (cols > g_scratch.xin_cap) {
        if (g_scratch.d_xin) cudaFree(g_scratch.d_xin);
        g_scratch.d_xin = nullptr;
        g_scratch.xin_cap = 0;
        invalidate_graphs();
        if (cudaMalloc(&g_scratch.d_xin, cols * sizeof(float)) != cudaSuccess) return DESIREEIA_ERR_IO;
        g_scratch.xin_cap = cols;
    }

    cudaStream_t stream = scratch_stream();
    cudaMemcpyAsync(g_scratch.d_xin, x, cols * sizeof(float), cudaMemcpyHostToDevice, stream);

    int8_t* d_xq = reinterpret_cast<int8_t*>(g_scratch.d_stage);
    float* d_xscale = reinterpret_cast<float*>(g_scratch.d_stage + nb * 32);
    quantize_q8_0_kernel<<<(unsigned) nb, 32, 0, stream>>>(g_scratch.d_xin, cols,
                                                            d_xq, d_xscale, g_scratch.d_xsum);

    launch_matmul_kquant(format, static_cast<const uint8_t*>(d_w), d_xq, d_xscale,
                          g_scratch.d_xsum, cols, rows, g_scratch.d_y, nullptr, stream);
    if (cudaGetLastError() != cudaSuccess) return DESIREEIA_ERR_IO;

    cudaMemcpyAsync(y, g_scratch.d_y, rows * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) return DESIREEIA_ERR_IO;
    return DESIREEIA_OK;
}

// Output head of a decoded token without leaving the device: the last
// layer left x in d_xin, so the final RMS norm (in place), the Q8_0 form of
// the result and the head product run right behind it, and only the logits
// come back. Replaces download x -> norm on the host -> upload -> product,
// which kept the GPU idle between tokens.
// Final norm, Q8_0 form and head product of a decoded token, into
// g_scratch.d_y (see cuda_head_forward).
static int head_forward_launch(int format, const void* d_qs, const void* d_scale, const float* norm_w,
                               size_t n_embd, float eps, size_t rows) {
    const size_t per_block = format == 0 ? 32 : cuda_block_weights(format);
    if (!d_qs || !norm_w || rows == 0 || per_block == 0 || n_embd % per_block != 0 || n_embd % 32 != 0 ||
        (format == 0 && !d_scale) || !g_scratch.d_xin || g_scratch.xin_cap < n_embd) {
        return DESIREEIA_ERR_NOT_SUPPORTED;
    }
    const size_t nb = n_embd / 32;
    if (!scratch_reserve(nb, rows)) return DESIREEIA_ERR_IO;
    if (!scratch_reserve_sums(nb)) return DESIREEIA_ERR_IO;
    // The model's output norm, uploaded once. The host address alone does
    // not identify it (a later model can reuse the allocation), so the
    // cached copy is also compared by content.
    HeadNormCache& hc = g_head_norms[norm_w];
    if (!hc.d || hc.host.size() != n_embd || std::memcmp(hc.host.data(), norm_w, n_embd * sizeof(float)) != 0) {
        if (hc.d) cudaFree(hc.d);
        hc.d = nullptr;
        hc.host.assign(norm_w, norm_w + n_embd);
        if (cudaMalloc(&hc.d, n_embd * sizeof(float)) != cudaSuccess) { hc.d = nullptr; return DESIREEIA_ERR_IO; }
        if (cudaMemcpy(hc.d, norm_w, n_embd * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(hc.d); hc.d = nullptr; return DESIREEIA_ERR_IO;
        }
    }
    float* const d_norm = hc.d;
    cudaStream_t stream = scratch_stream();
    constexpr int nthr = 256;
    rms_norm_kernel<<<1, nthr, nthr * sizeof(float), stream>>>(g_scratch.d_xin, d_norm, g_scratch.d_xin,
                                                               (uint32_t) n_embd, eps);
    int8_t* d_xq = reinterpret_cast<int8_t*>(g_scratch.d_stage);
    float* d_xscale = reinterpret_cast<float*>(g_scratch.d_stage + nb * 32);
    quantize_q8_0_kernel<<<(unsigned) nb, 32, 0, stream>>>(g_scratch.d_xin, n_embd, d_xq, d_xscale, g_scratch.d_xsum);
    if (format == 0) {
        launch_matmul_q8_0(static_cast<const int8_t*>(d_qs), static_cast<const __half*>(d_scale),
                           d_xq, d_xscale, nb, rows, g_scratch.d_y, stream);
    } else {
        launch_matmul_kquant(format, static_cast<const uint8_t*>(d_qs), d_xq, d_xscale,
                             g_scratch.d_xsum, n_embd, rows, g_scratch.d_y, nullptr, stream);
    }
    if (cudaGetLastError() != cudaSuccess) return DESIREEIA_ERR_IO;
    return DESIREEIA_OK;
}

// Top-k of the logits on the device. One pass: each block sorts 1024
// candidates (bitonic, larger value first, lower id first on ties - the
// sampler's order) and keeps its kk best; passes repeat until one block
// holds them all. 262144 logits with kk = 40 take three passes.
constexpr int kTopkBlock = 1024;
__global__ void __launch_bounds__(kTopkBlock / 2) topk_pass_kernel(
        const float* __restrict__ vin, const int32_t* __restrict__ iin, uint32_t n, uint32_t kk,
        float* __restrict__ vout, int32_t* __restrict__ iout) {
    __shared__ float sv[kTopkBlock];
    __shared__ int32_t si[kTopkBlock];
    const uint32_t base = blockIdx.x * kTopkBlock;
    for (int j = threadIdx.x; j < kTopkBlock; j += blockDim.x) {
        const uint32_t gidx = base + (uint32_t) j;
        if (gidx < n) { sv[j] = vin[gidx]; si[j] = iin ? iin[gidx] : (int32_t) gidx; }
        else          { sv[j] = -INFINITY; si[j] = INT32_MAX; }
    }
    __syncthreads();
    auto better = [&](int a, int b) { return sv[a] > sv[b] || (sv[a] == sv[b] && si[a] < si[b]); };
    for (int size = 2; size <= kTopkBlock; size <<= 1) {
        for (int stride = size / 2; stride > 0; stride >>= 1) {
            const int t = threadIdx.x;
            const int i = (t / stride) * 2 * stride + (t % stride), j = i + stride;
            const bool first_desc = (i & size) == 0;               // this run ends up largest-first
            if (first_desc ? better(j, i) : better(i, j)) {
                const float fv = sv[i]; sv[i] = sv[j]; sv[j] = fv;
                const int32_t fi = si[i]; si[i] = si[j]; si[j] = fi;
            }
            __syncthreads();
        }
    }
    for (uint32_t j = threadIdx.x; j < kk; j += blockDim.x) {
        vout[(size_t) blockIdx.x * kk + j] = sv[j];
        iout[(size_t) blockIdx.x * kk + j] = si[j];
    }
}

// kk = 1 (greedy): the same selection - the largest value, the lowest id
// among equals - as a reduction instead of a 1024-wide sort per block.
__global__ void __launch_bounds__(256) argmax_pass_kernel(
        const float* __restrict__ vin, const int32_t* __restrict__ iin, uint32_t n,
        float* __restrict__ vout, int32_t* __restrict__ iout) {
    const uint32_t base = blockIdx.x * kTopkBlock;
    float bv = -INFINITY;
    int32_t bi = INT32_MAX;
    for (uint32_t j = threadIdx.x; j < (uint32_t) kTopkBlock; j += blockDim.x) {
        const uint32_t g = base + j;
        if (g >= n) break;
        const float v = vin[g];
        const int32_t id = iin ? iin[g] : (int32_t) g;
        if (v > bv || (v == bv && id < bi)) { bv = v; bi = id; }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffff, bv, o);
        const int32_t oi = __shfl_xor_sync(0xffffffff, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    __shared__ float wv[8];
    __shared__ int32_t wi[8];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) { wv[warp] = bv; wi[warp] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (int) (blockDim.x / 32); ++w) {
            if (wv[w] > bv || (wv[w] == bv && wi[w] < bi)) { bv = wv[w]; bi = wi[w]; }
        }
        vout[blockIdx.x] = bv;
        iout[blockIdx.x] = bi;
    }
}

float* g_topk_v[2] = {};
int32_t* g_topk_i[2] = {};
size_t g_topk_cap = 0;
void* g_topk_pinned = nullptr;

// ---------------------------------------------------------------------
// Pipelined decode: the token on the device, the recent history the
// penalties read (a ring), and four pinned readback slots with an event
// each (a token and the one drawn speculatively after it are in flight).
constexpr int kHistRing = 256;
int32_t* g_dtok = nullptr;          // [1] the last drawn token
int32_t* g_dtok_ring = nullptr;     // [4] each draw, kept for its readback
int32_t* g_hist = nullptr;          // [kHistRing]
uint32_t* g_hist_meta = nullptr;    // [0] tokens in the history, [1] next ring position
int32_t* g_tok_pinned = nullptr;    // [4]
cudaEvent_t g_tok_ev[4] = {};
// The readback copy runs on its own stream: on the decode stream the next
// step waited for the device-to-host copy (tens of microseconds on WDDM).
cudaStream_t g_tok_stream = nullptr;
cudaEvent_t g_drawn_ev[4] = {};

static bool sample_state_ready() {
    if (g_dtok) return true;
    if (cudaMalloc(&g_dtok, sizeof(int32_t)) != cudaSuccess) { g_dtok = nullptr; return false; }
    if (cudaMalloc(&g_dtok_ring, 4 * sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc(&g_hist, kHistRing * sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc(&g_hist_meta, 2 * sizeof(uint32_t)) != cudaSuccess ||
        cudaHostAlloc(&g_tok_pinned, 4 * sizeof(int32_t), cudaHostAllocDefault) != cudaSuccess) return false;
    for (int i = 0; i < 4; ++i) {
        if (cudaEventCreateWithFlags(&g_tok_ev[i], cudaEventDisableTiming) != cudaSuccess) return false;
        if (cudaEventCreateWithFlags(&g_drawn_ev[i], cudaEventDisableTiming) != cudaSuccess) return false;
    }
    if (cudaStreamCreateWithFlags(&g_tok_stream, cudaStreamNonBlocking) != cudaSuccess) return false;
    cudaMemset(g_hist_meta, 0, 2 * sizeof(uint32_t));
    return true;
}

// x = the Q8_0 embedding row of the device token (Q8_0 dequantization as
// on the host: fp16 scale times the int8 value, in float), times mul.
__global__ void embed_q8_0_token_kernel(const int8_t* __restrict__ qs, const __half* __restrict__ sc,
                                        const int32_t* __restrict__ tok, uint32_t n_embd, float mul,
                                        float* __restrict__ out) {
    const size_t t = (size_t) *tok;
    const size_t nb = n_embd / 32;
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n_embd; i += blockDim.x * gridDim.x) {
        float v = __half2float(sc[t * nb + i / 32]) * (float) qs[t * n_embd + i];
        if (mul != 0.0f) v *= mul;
        out[i] = v;
    }
}

// Sampler::sample_candidates on the device, one block of 256 threads: the
// penalties on the candidates that carry them, the order (larger logit
// first, lower id on ties), greedy or top-k + temperature + top-p + draw.
// The sums run in the host's order on one thread.
__global__ void __launch_bounds__(256) sample_candidates_kernel(
        const float* __restrict__ vin, const int32_t* __restrict__ iin, uint32_t n, CudaSampleArgs a,
        int32_t* __restrict__ tok, int32_t* __restrict__ hist, uint32_t* __restrict__ meta,
        int32_t* __restrict__ tok_copy) {
    __shared__ float sv[256];
    __shared__ int32_t si[256];
    __shared__ float probs[256];
    const int t = threadIdx.x;
    if (a.append_input && t == 0) {                  // the step's input token enters the history
        const uint32_t pos = meta[1];
        hist[pos] = *tok;
        meta[1] = (pos + 1) % kHistRing;
        meta[0] = meta[0] + 1;
    }
    __syncthreads();
    const bool pen = a.penalty_last_n > 0 &&
        (a.penalty_repeat != 1.0f || a.penalty_freq != 0.0f || a.penalty_present != 0.0f);
    const uint32_t len = meta[0], wpos = meta[1];
    const uint32_t look = min((uint32_t) a.penalty_last_n, min(len, (uint32_t) kHistRing));
    if (t < (int) n) {
        float lg = vin[t];
        const int32_t id = iin[t];
        if (a.logit_scale != 0.0f) lg *= a.logit_scale;
        if (pen) {
            int32_t cnt = 0;
            for (uint32_t j = 0; j < look; ++j) cnt += hist[(wpos + kHistRing - 1 - j) % kHistRing] == id;
            if (cnt > 0) {
                if (lg <= 0.0f) lg *= a.penalty_repeat;
                else            lg /= a.penalty_repeat;
                lg -= (float) cnt * a.penalty_freq + a.penalty_present;
            }
        }
        sv[t] = lg; si[t] = id;
    } else {
        sv[t] = -INFINITY; si[t] = INT32_MAX;
    }
    __syncthreads();
    // bitonic sort of the 256 entries, larger value first, lower id on ties
    auto better = [&](int x, int y) { return sv[x] > sv[y] || (sv[x] == sv[y] && si[x] < si[y]); };
    for (int size = 2; size <= 256; size <<= 1) {
        for (int stride = size / 2; stride > 0; stride >>= 1) {
            if (t < 128) {
                const int i = (t / stride) * 2 * stride + (t % stride), j = i + stride;
                const bool first_desc = (i & size) == 0;
                if (first_desc ? better(j, i) : better(i, j)) {
                    const float fv = sv[i]; sv[i] = sv[j]; sv[j] = fv;
                    const int32_t fi = si[i]; si[i] = si[j]; si[j] = fi;
                }
            }
            __syncthreads();
        }
    }
    if (t != 0) return;
    int32_t pick = si[0];
    if (a.temperature > 0.0f) {
        const uint32_t k = min((uint32_t) a.top_k, n);
        const float inv_t = 1.0f / a.temperature;
        const float max_l = sv[0] * inv_t;
        float sum = 0.0f;
        for (uint32_t i = 0; i < k; ++i) {
            const float p = expf(sv[i] * inv_t - max_l);
            probs[i] = p;
            sum += p;
        }
        if (sum > 0.0f) {
            for (uint32_t i = 0; i < k; ++i) probs[i] /= sum;
            uint32_t keep = k;
            if (a.top_p < 1.0f) {
                float cum = 0.0f;
                for (uint32_t i = 0; i < k; ++i) {
                    cum += probs[i];
                    if (cum >= a.top_p) { keep = i + 1; break; }
                }
                if (keep < 1) keep = 1;
            }
            float cum = 0.0f;
            for (uint32_t i = 0; i < keep; ++i) cum += probs[i];
            if (cum > 0.0f) {
                const float r = __fadd_rn(__fmul_rn(a.u, cum), 0.0f);
                float acc = 0.0f;
                pick = si[keep - 1];
                for (uint32_t i = 0; i < keep; ++i) {
                    acc += probs[i];
                    if (r <= acc) { pick = si[i]; break; }
                }
            }
        }
    }
    *tok = pick;
    *tok_copy = pick;                                // the readback slot of this draw
}

int cuda_head_forward_topk(int format, const void* d_qs, const void* d_scale, const float* norm_w,
                           size_t n_embd, float eps, size_t rows, uint32_t kk, int32_t* ids, float* vals) {
    if (kk == 0 || kk > 256 || rows < kk) return DESIREEIA_ERR_NOT_SUPPORTED;
    const size_t cap = ((rows + kTopkBlock - 1) / kTopkBlock) * kk;
    if (cap > g_topk_cap || !g_topk_pinned) {
        for (int b = 0; b < 2; ++b) {
            if (g_topk_v[b]) cudaFree(g_topk_v[b]);
            if (g_topk_i[b]) cudaFree(g_topk_i[b]);
            g_topk_v[b] = nullptr; g_topk_i[b] = nullptr;
        }
        if (g_topk_pinned) cudaFreeHost(g_topk_pinned);
        g_topk_pinned = nullptr; g_topk_cap = 0;
        for (int b = 0; b < 2; ++b) {
            if (cudaMalloc(&g_topk_v[b], cap * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&g_topk_i[b], cap * sizeof(int32_t)) != cudaSuccess) return DESIREEIA_ERR_NOT_SUPPORTED;
        }
        if (cudaHostAlloc(&g_topk_pinned, 256 * 8, cudaHostAllocDefault) != cudaSuccess) {
            g_topk_pinned = nullptr;
            return DESIREEIA_ERR_NOT_SUPPORTED;
        }
        g_topk_cap = cap;
    }
    const int rc = head_forward_launch(format, d_qs, d_scale, norm_w, n_embd, eps, rows);
    if (rc != DESIREEIA_OK) return rc;
    cudaStream_t stream = scratch_stream();
    const float* vin = g_scratch.d_y;
    const int32_t* iin = nullptr;
    uint32_t n = (uint32_t) rows;
    int cur = 0;
    for (;;) {
        const uint32_t blocks = (n + kTopkBlock - 1) / kTopkBlock;
        if (kk == 1) argmax_pass_kernel<<<blocks, 256, 0, stream>>>(vin, iin, n, g_topk_v[cur], g_topk_i[cur]);
        else topk_pass_kernel<<<blocks, kTopkBlock / 2, 0, stream>>>(vin, iin, n, kk, g_topk_v[cur], g_topk_i[cur]);
        vin = g_topk_v[cur]; iin = g_topk_i[cur];
        n = blocks * kk;
        cur ^= 1;
        if (blocks == 1) break;
    }
    if (cudaGetLastError() != cudaSuccess) return DESIREEIA_ERR_IO;
    float* pv = static_cast<float*>(g_topk_pinned);
    int32_t* pi = reinterpret_cast<int32_t*>(pv + 256);
    cudaMemcpyAsync(pv, vin, kk * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(pi, iin, kk * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) return DESIREEIA_ERR_IO;
    std::memcpy(vals, pv, kk * sizeof(float));
    std::memcpy(ids, pi, kk * sizeof(int32_t));
    return DESIREEIA_OK;
}

int cuda_head_forward_sample(int format, const void* d_qs, const void* d_scale, const float* norm_w,
                             size_t n_embd, float eps, size_t rows, const CudaSampleArgs& s) {
    const uint32_t kk = s.kk;
    if (kk == 0 || kk > 256 || rows < kk || s.slot > 3 || !sample_state_ready()) return DESIREEIA_ERR_NOT_SUPPORTED;
    const size_t cap = ((rows + kTopkBlock - 1) / kTopkBlock) * kk;
    if (cap > g_topk_cap || !g_topk_v[0]) {
        for (int b = 0; b < 2; ++b) {
            if (g_topk_v[b]) cudaFree(g_topk_v[b]);
            if (g_topk_i[b]) cudaFree(g_topk_i[b]);
            g_topk_v[b] = nullptr; g_topk_i[b] = nullptr;
        }
        g_topk_cap = 0;
        for (int b = 0; b < 2; ++b) {
            if (cudaMalloc(&g_topk_v[b], cap * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&g_topk_i[b], cap * sizeof(int32_t)) != cudaSuccess) return DESIREEIA_ERR_NOT_SUPPORTED;
        }
        g_topk_cap = cap;
    }
    const int rc = head_forward_launch(format, d_qs, d_scale, norm_w, n_embd, eps, rows);
    if (rc != DESIREEIA_OK) return rc;
    cudaStream_t stream = scratch_stream();
    const float* vin = g_scratch.d_y;
    const int32_t* iin = nullptr;
    uint32_t n = (uint32_t) rows;
    int cur = 0;
    for (;;) {
        const uint32_t blocks = (n + kTopkBlock - 1) / kTopkBlock;
        if (kk == 1) argmax_pass_kernel<<<blocks, 256, 0, stream>>>(vin, iin, n, g_topk_v[cur], g_topk_i[cur]);
        else topk_pass_kernel<<<blocks, kTopkBlock / 2, 0, stream>>>(vin, iin, n, kk, g_topk_v[cur], g_topk_i[cur]);
        vin = g_topk_v[cur]; iin = g_topk_i[cur];
        n = blocks * kk;
        cur ^= 1;
        if (blocks == 1) break;
    }
    // The draw also lands in its own ring slot, read back from there: the
    // next step overwrites g_dtok while the copy may still be pending.
    sample_candidates_kernel<<<1, 256, 0, stream>>>(vin, iin, kk, s, g_dtok, g_hist, g_hist_meta, g_dtok_ring + s.slot);
    cudaEventRecord(g_drawn_ev[s.slot], stream);
    cudaStreamWaitEvent(g_tok_stream, g_drawn_ev[s.slot], 0);
    cudaMemcpyAsync(g_tok_pinned + s.slot, g_dtok_ring + s.slot, sizeof(int32_t), cudaMemcpyDeviceToHost, g_tok_stream);
    cudaEventRecord(g_tok_ev[s.slot], g_tok_stream);
    return cudaGetLastError() == cudaSuccess ? DESIREEIA_OK : DESIREEIA_ERR_IO;
}

int cuda_sample_wait(uint32_t slot, int32_t& token) {
    if (slot > 3 || !g_tok_ev[slot]) return DESIREEIA_ERR_IO;
    if (cudaEventSynchronize(g_tok_ev[slot]) != cudaSuccess) return DESIREEIA_ERR_IO;
    token = g_tok_pinned[slot];
    return DESIREEIA_OK;
}

int cuda_sample_history(const int32_t* hist, size_t n) {
    if (!sample_state_ready()) return DESIREEIA_ERR_NOT_SUPPORTED;
    const size_t m = n < (size_t) kHistRing ? n : (size_t) kHistRing;
    std::vector<int32_t> ring(kHistRing, -1);
    for (size_t i = 0; i < m; ++i) ring[i] = hist[n - m + i];
    const uint32_t meta[2] = { (uint32_t) m, (uint32_t) (m % kHistRing) };
    cudaStream_t stream = scratch_stream();
    cudaMemcpyAsync(g_hist, ring.data(), kHistRing * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(g_hist_meta, meta, sizeof(meta), cudaMemcpyHostToDevice, stream);
    return cudaStreamSynchronize(stream) == cudaSuccess ? DESIREEIA_OK : DESIREEIA_ERR_IO;
}

int cuda_head_forward(int format, const void* d_qs, const void* d_scale, const float* norm_w,
                      size_t n_embd, float eps, size_t rows, float* y) {
    const int rc = head_forward_launch(format, d_qs, d_scale, norm_w, n_embd, eps, rows);
    if (rc != DESIREEIA_OK) return rc;
    cudaStream_t stream = scratch_stream();
    // The logits (a whole vocabulary, ~1 MB) come back through a page-locked
    // buffer: a copy to pageable memory is staged by the driver, starts
    // later and runs slower, and it is on the critical path of every token.
    if (rows > g_head_pinned_cap) {
        if (g_head_pinned) cudaFreeHost(g_head_pinned);
        g_head_pinned = nullptr; g_head_pinned_cap = 0;
        if (cudaHostAlloc(reinterpret_cast<void**>(&g_head_pinned), rows * sizeof(float), cudaHostAllocDefault) == cudaSuccess) {
            g_head_pinned_cap = rows;
        }
    }
    if (g_head_pinned) {
        cudaMemcpyAsync(g_head_pinned, g_scratch.d_y, rows * sizeof(float), cudaMemcpyDeviceToHost, stream);
        if (cudaStreamSynchronize(stream) != cudaSuccess) return DESIREEIA_ERR_IO;
        std::memcpy(y, g_head_pinned, rows * sizeof(float));
        return DESIREEIA_OK;
    }
    cudaMemcpyAsync(y, g_scratch.d_y, rows * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) return DESIREEIA_ERR_IO;
    return DESIREEIA_OK;
}

// The x the last device layer left in d_xin, for a caller that planned to
// run the head on device and then could not.
int cuda_fetch_x(float* dst, size_t n) {
    if (!g_scratch.d_xin || g_scratch.xin_cap < n) return DESIREEIA_ERR_NOT_SUPPORTED;
    cudaStream_t stream = scratch_stream();
    cudaMemcpyAsync(dst, g_scratch.d_xin, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
    return cudaStreamSynchronize(stream) == cudaSuccess ? DESIREEIA_OK : DESIREEIA_ERR_IO;
}

// Prefill attention for a whole batch of tokens in one episode.
//
// The KV cache on device is already current: write_kv_cache mirrors every
// position as it is produced, and the caller runs this only after the
// per-token preparation loop (bias, QK-norm, RoPE, KV write) has finished
// for the entire batch.
//
// Returns NOT_SUPPORTED when the shape doesn't fit the kernel's register
// budget or the KV cache isn't on device, in which case the caller keeps
// its CPU path — which stays the definition of record.
int cuda_attention_batch(const float* q_all, float* out_all,
                          uint32_t n_tokens, uint32_t n_head, uint32_t heads_per_kv,
                          uint32_t head_dim, uint32_t q_dim,
                          size_t row_bytes, size_t pos_bytes, size_t layer_off,
                          uint32_t col0, uint32_t n_swa) {
    static const bool attn_batch_enabled = std::getenv("DESIREEIA_CUDA_NO_ATTN_BATCH") == nullptr;
    if (!attn_batch_enabled) return DESIREEIA_ERR_NOT_SUPPORTED;
    if (!g_scratch.d_kcache || n_tokens == 0 || n_head == 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    constexpr int athr = 128;
    // Same register-budget limit as the decode kernel: the V accumulator
    // holds one entry per head dimension a thread owns.
    if (head_dim > athr * DESIREEIA_ATTN_MAX_ACC || head_dim % 32 != 0) {
        return DESIREEIA_ERR_NOT_SUPPORTED;
    }

    ScopedTimer prof_t(profile_counters().ns_cuda_attn);
    profile_counters().calls_cuda_attn.fetch_add(n_tokens, std::memory_order_relaxed);

    const size_t n_elem = (size_t) n_tokens * q_dim;
    auto grow = [](void** p, size_t& cap, size_t need_bytes) {
        if (need_bytes <= cap && *p) return true;
        if (*p) cudaFree(*p);
        *p = nullptr; cap = 0;
        if (cudaMalloc(p, need_bytes) != cudaSuccess) return false;
        cap = need_bytes;
        return true;
    };
    if (!grow((void**) &g_scratch.d_bat_qa, g_scratch.bat_qa_cap, n_elem * sizeof(float))) return DESIREEIA_ERR_IO;
    if (!grow((void**) &g_scratch.d_bat_o, g_scratch.bat_o_cap, n_elem * sizeof(float))) return DESIREEIA_ERR_IO;

    cudaStream_t stream = scratch_stream();
    cudaMemcpyAsync(g_scratch.d_bat_qa, q_all, n_elem * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    const dim3 grid(n_head, n_tokens);
    attention_batch_kernel_q8<<<grid, athr, 2 * athr * sizeof(float), stream>>>(
        g_scratch.d_bat_qa, g_scratch.d_kcache, g_scratch.d_vcache, g_scratch.d_bat_o,
        head_dim, heads_per_kv, q_dim, row_bytes, pos_bytes, layer_off,
        col0, n_swa, 1.0f / sqrtf((float) head_dim));
    if (cudaGetLastError() != cudaSuccess) return DESIREEIA_ERR_IO;

    cudaMemcpyAsync(out_all, g_scratch.d_bat_o, n_elem * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) return DESIREEIA_ERR_IO;
    return DESIREEIA_OK;
}

// Batched prefill mat-mul: every activation column uploaded once, one
// quantization pass over all of them, ONE kernel, one download, one
// synchronization — against the per-token loop's n of each.
//
// Only Q4_K is batched here. Other formats return NOT_SUPPORTED and the
// caller keeps its per-column path, which stays correct; extending this
// is a matter of writing the equivalent kernel for them, not of changing
// anything around it.
int matmul_kquant_cuda_batch(int format, const void* d_w, size_t rows, size_t cols,
                              const float* x, size_t n_tok, float* y) {
    // Escape hatch, same purpose as DESIREEIA_CUDA_NO_GROUP: batching must
    // be a pure scheduling change, and that has to be checkable against
    // the per-column path rather than argued.
    static const bool batch_enabled = std::getenv("DESIREEIA_CUDA_NO_BATCH") == nullptr;
    if (!batch_enabled) return DESIREEIA_ERR_NOT_SUPPORTED;
    if (format != DESIREEIA_CUDA_FMT_Q4_K && format != DESIREEIA_CUDA_FMT_Q6_K) {
        return DESIREEIA_ERR_NOT_SUPPORTED;
    }
    if (!d_w || rows == 0 || n_tok == 0 || cols == 0 || cols % 256 != 0) {
        return DESIREEIA_ERR_NOT_SUPPORTED;
    }

    ScopedTimer prof_t(profile_counters().ns_cuda_resident);
    profile_counters().calls_cuda_resident.fetch_add((int64_t) n_tok, std::memory_order_relaxed);

    const size_t nb = cols / 32;
    const size_t n_elem = n_tok * cols;
    const size_t n_blk = n_tok * nb;
    const size_t n_out = n_tok * rows;

    auto grow = [](void** p, size_t& cap, size_t need_bytes) {
        if (need_bytes <= cap && *p) return true;
        if (*p) cudaFree(*p);
        *p = nullptr; cap = 0;
        if (cudaMalloc(p, need_bytes) != cudaSuccess) return false;
        cap = need_bytes;
        return true;
    };
    if (!grow((void**) &g_scratch.d_bat_x, g_scratch.bat_x_cap, n_elem * sizeof(float))) return DESIREEIA_ERR_IO;
    if (!grow((void**) &g_scratch.d_bat_q, g_scratch.bat_q_cap, n_blk * 32 + n_blk * sizeof(float))) return DESIREEIA_ERR_IO;
    if (!grow((void**) &g_scratch.d_bat_sum, g_scratch.bat_sum_cap, n_blk * sizeof(int32_t))) return DESIREEIA_ERR_IO;
    if (!grow((void**) &g_scratch.d_bat_y, g_scratch.bat_y_cap, n_out * sizeof(float))) return DESIREEIA_ERR_IO;

    cudaStream_t stream = scratch_stream();
    cudaMemcpyAsync(g_scratch.d_bat_x, x, n_elem * sizeof(float), cudaMemcpyHostToDevice, stream);

    int8_t* d_xq = reinterpret_cast<int8_t*>(g_scratch.d_bat_q);
    float* d_xscale = reinterpret_cast<float*>(g_scratch.d_bat_q + n_blk * 32);
    // One quantization over the whole flat activation: the kernel is
    // already per-32-block, so a flat n_tok*cols input yields exactly the
    // [token][sub-block] layout the batched kernel indexes.
    quantize_q8_0_kernel<<<(unsigned) n_blk, 32, 0, stream>>>(g_scratch.d_bat_x, n_elem,
                                                               d_xq, d_xscale, g_scratch.d_bat_sum);

    constexpr int kTok = 8;
    const dim3 block(DESIREEIA_CUDA_WARP, kMatmulQ80Warps);
    const dim3 grid((unsigned) ((rows + kMatmulQ80Warps - 1) / kMatmulQ80Warps),
                    (unsigned) ((n_tok + kTok - 1) / kTok));
    if (format == DESIREEIA_CUDA_FMT_Q4_K) {
        matmul_q4_k_batch_kernel<kMatmulQ80Warps, kTok><<<grid, block, 0, stream>>>(
            static_cast<const uint8_t*>(d_w), d_xq, d_xscale, g_scratch.d_bat_sum,
            cols / 256, rows, (uint32_t) n_tok, g_scratch.d_bat_y);
    } else {
        matmul_q6_k_batch_kernel<kMatmulQ80Warps, kTok><<<grid, block, 0, stream>>>(
            static_cast<const uint8_t*>(d_w), d_xq, d_xscale,
            cols / 256, rows, (uint32_t) n_tok, g_scratch.d_bat_y);
    }
    if (cudaGetLastError() != cudaSuccess) return DESIREEIA_ERR_IO;

    cudaMemcpyAsync(y, g_scratch.d_bat_y, n_out * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) return DESIREEIA_ERR_IO;
    return DESIREEIA_OK;
}

// Group of matvecs over the same activation: one quantization, one H2D,
// n kernels, n async D2H, ONE synchronization. See the note on CudaQ80Job
// in engine.h for why.
int matmul_q8_0_cuda_resident_group(const CudaQ80Job* jobs, size_t n, size_t cols, const float* x) {
    if (!jobs || n == 0 || cols == 0 || cols % 32 != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    size_t rows_total = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!jobs[i].d_qs || !jobs[i].d_scale || jobs[i].rows == 0) return DESIREEIA_ERR_NOT_SUPPORTED;
        rows_total += jobs[i].rows;
    }

    ScopedTimer prof_t(profile_counters().ns_cuda_resident);
    // Counted as n calls, not one: this keeps the comparison with earlier
    // measurements (us per matvec) readable.
    profile_counters().calls_cuda_resident.fetch_add((int64_t) n, std::memory_order_relaxed);

    const size_t nb = cols / 32;
    // d_y holds the outputs of ALL the jobs, one after another.
    if (!scratch_reserve(nb, rows_total)) return DESIREEIA_ERR_IO;

    quantize_q8_0(x, cols, g_scratch.xq, g_scratch.xscale, g_scratch.signs);
    if (g_scratch.xq.size() != nb * 32 || g_scratch.xscale.size() != nb) {
        return DESIREEIA_ERR_NOT_SUPPORTED;
    }

    cudaStream_t stream = scratch_stream();
    int8_t* d_xq = reinterpret_cast<int8_t*>(g_scratch.d_stage);
    float* d_xscale = reinterpret_cast<float*>(g_scratch.d_stage + nb * 32);
    cudaMemcpyAsync(d_xq, g_scratch.xq.data(), nb * 32, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_xscale, g_scratch.xscale.data(), nb * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    size_t off = 0;
    for (size_t i = 0; i < n; ++i) {
        launch_matmul_q8_0(static_cast<const int8_t*>(jobs[i].d_qs),
                            static_cast<const __half*>(jobs[i].d_scale),
                            d_xq, d_xscale, nb, jobs[i].rows, g_scratch.d_y + off, stream);
        off += jobs[i].rows;
    }
    if (cudaGetLastError() != cudaSuccess) return DESIREEIA_ERR_IO;

    off = 0;
    for (size_t i = 0; i < n; ++i) {
        cudaMemcpyAsync(jobs[i].y, g_scratch.d_y + off, jobs[i].rows * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        off += jobs[i].rows;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) return DESIREEIA_ERR_IO;
    return DESIREEIA_OK;
}

// --- Device KV cache (mirror of the host one) ---

// (Re)allocates the device KV cache. Called when the host capacity
// changes (grow_cache): the valid content is reloaded right after with
// cuda_kv_cache_upload, so the mirror starts back aligned.
bool cuda_kv_cache_reserve(size_t total_bytes) {
    if (total_bytes <= g_scratch.kv_bytes && g_scratch.d_kcache) return true;
    // The captured per-layer graphs bake in d_kcache/d_vcache as fixed
    // addresses. Reallocating underneath them without dropping the graphs
    // leaves every replay reading FREED VRAM from that point on — output
    // stays perfect until the cache first grows (the host cache starts at
    // 512 positions and doubles), then collapses into token garbage and
    // never recovers, which is exactly how this was found. Every other
    // grow path in this file already invalidates; this one did not.
    invalidate_graphs();
    if (g_scratch.d_kcache) cudaFree(g_scratch.d_kcache);
    if (g_scratch.d_vcache) cudaFree(g_scratch.d_vcache);
    g_scratch.d_kcache = nullptr;
    g_scratch.d_vcache = nullptr;
    g_scratch.kv_bytes = 0;
    if (total_bytes == 0) return true;
    if (cudaMalloc(&g_scratch.d_kcache, total_bytes) != cudaSuccess) return false;
    if (cudaMalloc(&g_scratch.d_vcache, total_bytes) != cudaSuccess) {
        cudaFree(g_scratch.d_kcache);
        g_scratch.d_kcache = nullptr;
        return false;
    }
    g_scratch.kv_bytes = total_bytes;
    return true;
}

void cuda_kv_cache_release() {
    invalidate_graphs();
    if (g_scratch.d_kcache) cudaFree(g_scratch.d_kcache);
    if (g_scratch.d_vcache) cudaFree(g_scratch.d_vcache);
    g_scratch.d_kcache = nullptr;
    g_scratch.d_vcache = nullptr;
    g_scratch.kv_bytes = 0;
}

int32_t cuda_gpu_count() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) {
        cudaGetLastError();                  // no driver / no device: not an error for the caller
        return 0;
    }
    return n;
}

bool cuda_probe_gpu(int32_t index, char* name, size_t name_len, uint64_t& total, uint64_t& free_bytes,
                    int32_t& cc_major, int32_t& cc_minor, int32_t& sms) {
    if (index < 0 || index >= cuda_gpu_count()) return false;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, index) != cudaSuccess) return false;
    std::snprintf(name, name_len, "%s", prop.name);
    cc_major = prop.major;
    cc_minor = prop.minor;
    sms = prop.multiProcessorCount;
    total = prop.totalGlobalMem;
    free_bytes = 0;
    // Free memory is per device: switch to it just for the query and back.
    int prev = 0;
    if (cudaGetDevice(&prev) != cudaSuccess) prev = 0;
    if (cudaSetDevice(index) == cudaSuccess) {
        size_t f = 0, t = 0;
        if (cudaMemGetInfo(&f, &t) == cudaSuccess) { free_bytes = f; total = t; }
        cudaSetDevice(prev);
    }
    cudaGetLastError();
    return true;
}

// Reads the whole device cache back into the host buffers.
//
// Needed because the whole-layer fused path writes the KV cache ONLY on
// device (kv_quantize_kernel inside the graph) and then skips the host
// write_kv_cache entirely, so after any decoding through that path the
// device copy is the authoritative one and the host copy is stale. Any
// code that re-lays-out the cache from the host side (grow_cache) has to
// pull the device copy down FIRST, or it re-uploads stale data over the
// real thing.
bool cuda_kv_cache_download(void* k_host, void* v_host, size_t total_bytes) {
    if (!g_scratch.d_kcache || total_bytes > g_scratch.kv_bytes) return false;
    cudaStream_t stream = scratch_stream();
    cudaMemcpyAsync(k_host, g_scratch.d_kcache, total_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(v_host, g_scratch.d_vcache, total_bytes, cudaMemcpyDeviceToHost, stream);
    return cudaStreamSynchronize(stream) == cudaSuccess;
}

// Reloads the whole host cache onto device (after a grow/reset).
bool cuda_kv_cache_upload(const void* k_host, const void* v_host, size_t total_bytes) {
    if (!g_scratch.d_kcache || total_bytes > g_scratch.kv_bytes) return false;
    cudaStream_t stream = scratch_stream();
    cudaMemcpyAsync(g_scratch.d_kcache, k_host, total_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(g_scratch.d_vcache, v_host, total_bytes, cudaMemcpyHostToDevice, stream);
    return cudaStreamSynchronize(stream) == cudaSuccess;
}

// Mirrors a single position (k and v of a layer). Async on the shared
// stream: ordering with the attention kernel that reads it is guaranteed
// by the stream, so no synchronization is needed here.
bool cuda_kv_cache_write(size_t byte_off, const void* k, const void* v, size_t bytes) {
    if (!g_scratch.d_kcache || byte_off + bytes > g_scratch.kv_bytes) return false;
    cudaStream_t stream = scratch_stream();
    cudaMemcpyAsync(g_scratch.d_kcache + byte_off, k, bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(g_scratch.d_vcache + byte_off, v, bytes, cudaMemcpyHostToDevice, stream);
    return true;
}

// Attention + output projection in a single episode: q goes up once,
// attention reads the KV cache already on device, and the attention
// output feeds `wo`'s matvec directly without ever going back to the
// host. Only `proj` comes down.
int cuda_attention_out(const void* d_wo_qs, const void* d_wo_scale,
                        const float* q, uint32_t n_head, uint32_t heads_per_kv,
                        uint32_t head_dim, uint32_t kv_dim,
                        size_t layer_off, uint32_t cc_start, uint32_t pos,
                        size_t n_embd, size_t q_dim, float* proj,
                        int kv_quantized, size_t row_bytes) {
    if (!g_scratch.d_kcache || !d_wo_qs || !d_wo_scale) return DESIREEIA_ERR_NOT_SUPPORTED;
    if (q_dim % 32 != 0) return DESIREEIA_ERR_NOT_SUPPORTED;

    ScopedTimer prof_t(profile_counters().ns_cuda_attn);
    profile_counters().calls_cuda_attn.fetch_add(1, std::memory_order_relaxed);

    const uint32_t n_cc = pos + 1 - cc_start;
    const size_t scores_need = (size_t) n_head * n_cc;
    if (scores_need > g_scratch.scores_cap) {
        if (g_scratch.d_scores) cudaFree(g_scratch.d_scores);
        g_scratch.d_scores = nullptr;
        g_scratch.scores_cap = 0;
        if (cudaMalloc(&g_scratch.d_scores, scores_need * sizeof(float)) != cudaSuccess) {
            return DESIREEIA_ERR_IO;
        }
        g_scratch.scores_cap = scores_need;
    }
    if (q_dim > g_scratch.q_cap) {
        if (g_scratch.d_q) cudaFree(g_scratch.d_q);
        g_scratch.d_q = nullptr;
        g_scratch.q_cap = 0;
        if (cudaMalloc(&g_scratch.d_q, q_dim * sizeof(float)) != cudaSuccess) return DESIREEIA_ERR_IO;
        g_scratch.q_cap = q_dim;
    }
    if (q_dim > g_scratch.attn_cap) {
        // d_attn is shared with the whole-layer graphs, so growing it here
        // has to drop them too (same reason as scratch_reserve_attn).
        invalidate_graphs();
        if (g_scratch.d_attn) cudaFree(g_scratch.d_attn);
        g_scratch.d_attn = nullptr;
        g_scratch.attn_cap = 0;
        if (cudaMalloc(&g_scratch.d_attn, q_dim * sizeof(float)) != cudaSuccess) return DESIREEIA_ERR_IO;
        g_scratch.attn_cap = q_dim;
    }
    const size_t nb_attn = q_dim / 32;
    if (!scratch_reserve(nb_attn, n_embd)) return DESIREEIA_ERR_IO;

    cudaStream_t stream = scratch_stream();
    cudaMemcpyAsync(g_scratch.d_q, q, q_dim * sizeof(float), cudaMemcpyHostToDevice, stream);

    const int threads = 128;
    const float inv_d = 1.0f / sqrtf((float) head_dim);
    if (kv_quantized) {
        // The quantized kernel now reads pos/cc_start from device memory
        // (so the fused path can capture it in a graph); this non-graph
        // caller has to publish them the same way.
        if (!g_scratch.d_dyn) {
            if (cudaMalloc(&g_scratch.d_dyn, 4 * sizeof(uint32_t)) != cudaSuccess) {
                return DESIREEIA_ERR_IO;
            }
        }
        const uint32_t dyn[2] = { pos, cc_start };
        cudaMemcpyAsync(g_scratch.d_dyn, dyn, sizeof(dyn), cudaMemcpyHostToDevice, stream);
        attention_decode_q8(g_scratch.d_q, g_scratch.d_kcache, g_scratch.d_vcache, g_scratch.d_attn,
                            n_head, head_dim, heads_per_kv, row_bytes, row_bytes * (kv_dim / head_dim),
                            layer_off, g_scratch.d_dyn, inv_d, stream);
    } else {
        // layer_off arrives in bytes (uniform contract with the quantized
        // layout); the float kernel indexes a float*, so convert here.
        attention_kernel<<<n_head, threads, threads * sizeof(float), stream>>>(
            g_scratch.d_q, reinterpret_cast<const float*>(g_scratch.d_kcache),
            reinterpret_cast<const float*>(g_scratch.d_vcache),
            g_scratch.d_scores, g_scratch.d_attn,
            0, head_dim, heads_per_kv, kv_dim, layer_off / sizeof(float),
            cc_start, pos, inv_d);
    }

    // The attention output is already on device: it gets quantized there
    // and passed straight to wo's matvec, with no round trip to the host.
    int8_t* d_xq = reinterpret_cast<int8_t*>(g_scratch.d_stage);
    float* d_xscale = reinterpret_cast<float*>(g_scratch.d_stage + nb_attn * 32);
    quantize_q8_0_kernel<<<(unsigned) nb_attn, 32, 0, stream>>>(g_scratch.d_attn, q_dim,
                                                                 d_xq, d_xscale);
    launch_matmul_q8_0(static_cast<const int8_t*>(d_wo_qs), static_cast<const __half*>(d_wo_scale),
                        d_xq, d_xscale, nb_attn, n_embd, g_scratch.d_y, stream);
    if (cudaGetLastError() != cudaSuccess) return DESIREEIA_ERR_IO;

    cudaMemcpyAsync(proj, g_scratch.d_y, n_embd * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) return DESIREEIA_ERR_IO;
    return DESIREEIA_OK;
}


// Emits one mat-vec inside the layer graph, picking the kernel for that
// matrix format. Inside a captured graph the number of kernels costs
// almost nothing (the graph is a single submission), so K-quant matrices
// are launched individually instead of needing grouped variants.
static void emit_matvec(int fmt, const void* w, const void* scale,
                         const int8_t* d_xq, const float* d_xscale,
                         const int32_t* d_xsum, size_t cols, size_t rows,
                         float* d_y, const float* d_bias, cudaStream_t stream) {
    if (fmt == 0) {
        launch_matmul_q8_0(static_cast<const int8_t*>(w), static_cast<const __half*>(scale),
                            d_xq, d_xscale, cols / 32, rows, d_y, stream, d_bias);
    } else {
        launch_matmul_kquant(fmt, static_cast<const uint8_t*>(w), d_xq, d_xscale,
                              d_xsum, cols, rows, d_y, d_bias, stream);
    }
}

// One whole dense layer in a single captured graph.
//
// Sequence, identical in maths to the CPU path it replaces:
//   xn    = rms_norm(x, attn_norm)
//   q,k,v = Wq/Wk/Wv * xn  (+bias), RoPE on q and k, KV written at pos
//   attn  = causal GQA attention over the device KV cache
//   proj  = Wo * attn,  then proj += x            (attention residual)
//   xn2   = rms_norm(proj, ffn_norm)
//   h     = act(gate(xn2)) * up(xn2), requantized on device
//   fout  = Wdown * h
//   xnew  = fout + proj                           (FFN residual)
//
// Only x goes up and only xnew comes down: everything in between stays on
// the GPU, and the whole layer is one submission and one synchronisation
// instead of two of each.
int cuda_layer_forward(const CudaLayerArgs& a) {
    if (!g_scratch.d_kcache) return DESIREEIA_ERR_NOT_SUPPORTED;
    if (a.n_embd % 32 != 0 || a.q_dim % 32 != 0 || a.n_ff % 32 != 0) {
        return DESIREEIA_ERR_NOT_SUPPORTED;
    }
    // The attention kernel keeps its V accumulator in registers, one entry
    // per head dimension a thread owns. A head wider than the block can
    // cover would run off the end of that array, so it is rejected here
    // rather than corrupting memory silently.
    if (a.head_dim > 128 * DESIREEIA_ATTN_MAX_ACC) return DESIREEIA_ERR_NOT_SUPPORTED;

    ScopedTimer prof_t(profile_counters().ns_cuda_attn);
    profile_counters().calls_cuda_attn.fetch_add(1, std::memory_order_relaxed);

    const size_t nb_x = a.n_embd / 32;
    const size_t nb_h = a.n_ff / 32;
    const size_t nb_attn = a.q_dim / 32;
    const uint32_t nblk = a.head_dim / 32;
    const size_t kv_row_bytes = (size_t) nblk * 34;
    const size_t pos_bytes = (size_t) a.n_head_kv * kv_row_bytes;

    if (!scratch_reserve(nb_x > nb_h ? nb_x : nb_h, a.n_ff)) return DESIREEIA_ERR_IO;
    if (!scratch_reserve_ffn(a.n_ff, a.n_embd)) return DESIREEIA_ERR_IO;
    if (!scratch_reserve_attn(a.q_dim, a.kv_dim, a.n_rot, pos_bytes)) return DESIREEIA_ERR_IO;
    // One sums buffer serves all three quantisations in the layer (input,
    // attention output, FFN intermediate): inside the graph they run in
    // sequence and each is consumed before the next overwrites it.
    {
        size_t need = nb_x > nb_h ? nb_x : nb_h;
        if (nb_attn > need) need = nb_attn;
        if (!scratch_reserve_sums(need)) return DESIREEIA_ERR_IO;
    }
    if (a.n_embd > g_scratch.xn_cap) {
        if (g_scratch.d_xn) cudaFree(g_scratch.d_xn);
        if (g_scratch.d_xn2) cudaFree(g_scratch.d_xn2);
        if (g_scratch.d_xnew) cudaFree(g_scratch.d_xnew);
        g_scratch.d_xn = nullptr; g_scratch.d_xn2 = nullptr; g_scratch.d_xnew = nullptr;
        g_scratch.xn_cap = 0;
        invalidate_graphs();
        if (cudaMalloc(&g_scratch.d_xn, a.n_embd * sizeof(float)) != cudaSuccess) return DESIREEIA_ERR_IO;
        if (cudaMalloc(&g_scratch.d_xn2, a.n_embd * sizeof(float)) != cudaSuccess) return DESIREEIA_ERR_IO;
        if (cudaMalloc(&g_scratch.d_xnew, a.n_embd * sizeof(float)) != cudaSuccess) return DESIREEIA_ERR_IO;
        g_scratch.xn_cap = a.n_embd;
    }
    if (!g_scratch.d_dyn) {
        if (cudaMalloc(&g_scratch.d_dyn, 4 * sizeof(uint32_t)) != cudaSuccess) return DESIREEIA_ERR_IO;
    }

    cudaStream_t stream = scratch_stream();

    // pos/cc_start and the RoPE table are the same for every layer of a
    // token unless the model alternates sliding-window layers, so the
    // caller asks for them to be sent once instead of once per layer.
    // Alternating models send the values of every layer at once instead
    // (dyn_all): the per-layer copies were a stall between layer graphs.
    const bool slots = a.dyn_all != nullptr && a.layer_slot < a.n_slots;
    if (slots && a.upload_x) {
        const size_t nd = (size_t) 2 * a.n_slots, nr = (size_t) a.n_slots * a.rope_stride;
        if (nd > g_scratch.dyn_all_cap || nr > g_scratch.rope_all_cap || !g_scratch.d_dyn_all) {
            invalidate_graphs();                       // the graphs hold the slot addresses
            if (g_scratch.d_dyn_all) cudaFree(g_scratch.d_dyn_all);
            if (g_scratch.d_rope_all) cudaFree(g_scratch.d_rope_all);
            g_scratch.d_dyn_all = nullptr; g_scratch.d_rope_all = nullptr;
            g_scratch.dyn_all_cap = g_scratch.rope_all_cap = 0;
            if (cudaMalloc(&g_scratch.d_dyn_all, nd * sizeof(uint32_t)) != cudaSuccess) return DESIREEIA_ERR_IO;
            if (cudaMalloc(&g_scratch.d_rope_all, (nr ? nr : 1) * sizeof(float)) != cudaSuccess) return DESIREEIA_ERR_IO;
            g_scratch.dyn_all_cap = nd; g_scratch.rope_all_cap = nr;
        }
        cudaMemcpyAsync(g_scratch.d_dyn_all, a.dyn_all, nd * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
        if (nr) cudaMemcpyAsync(g_scratch.d_rope_all, a.rope_all, nr * sizeof(float), cudaMemcpyHostToDevice, stream);
    }
    if (!slots && a.upload_dyn) {
        const uint32_t dyn[2] = { a.pos, a.cc_start };
        cudaMemcpyAsync(g_scratch.d_dyn, dyn, sizeof(dyn), cudaMemcpyHostToDevice, stream);
    }
    if (a.upload_x && a.x_from_token) {
        if (!g_dtok || !a.embd_qs || !a.embd_scale || a.n_embd % 32 != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
        embed_q8_0_token_kernel<<<(unsigned) ((a.n_embd + 255) / 256), 256, 0, stream>>>(
            static_cast<const int8_t*>(a.embd_qs), static_cast<const __half*>(a.embd_scale), g_dtok,
            (uint32_t) a.n_embd, a.embd_mul, g_scratch.d_xin);
    } else if (a.upload_x) {
        cudaMemcpyAsync(g_scratch.d_xin, a.x, a.n_embd * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    }
    if (!slots && a.rope_cache && a.upload_dyn) {
        cudaMemcpyAsync(g_scratch.d_rope, a.rope_cache, a.n_rot * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    }
    // This layer's device copies of pos/cc_start and of its RoPE table.
    const uint32_t* const d_dynl = slots ? g_scratch.d_dyn_all + 2 * (size_t) a.layer_slot : g_scratch.d_dyn;
    const float* const d_ropel = slots ? g_scratch.d_rope_all + (size_t) a.layer_slot * a.rope_stride : g_scratch.d_rope;

    LayerGraph& lg = g_layer_graphs[a.wq_qs];
    if (!lg.exec) {
        auto upload = [&](const float* src, size_t n, float** dst) {
            if (!src) return true;
            if (cudaMalloc(dst, n * sizeof(float)) != cudaSuccess) return false;
            return cudaMemcpy(*dst, src, n * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
        };
        if (!upload(a.bq, a.q_dim, &lg.d_bq)) return DESIREEIA_ERR_IO;
        if (!upload(a.bk, a.kv_dim, &lg.d_bk)) return DESIREEIA_ERR_IO;
        if (!upload(a.bv, a.kv_dim, &lg.d_bv)) return DESIREEIA_ERR_IO;
        if (!upload(a.attn_norm_w, a.n_embd, &lg.d_attn_norm)) return DESIREEIA_ERR_IO;
        if (!upload(a.ffn_norm_w, a.n_embd, &lg.d_ffn_norm)) return DESIREEIA_ERR_IO;
        if (!upload(a.q_norm_w, a.head_dim, &lg.d_q_norm)) return DESIREEIA_ERR_IO;
        if (!upload(a.k_norm_w, a.head_dim, &lg.d_k_norm)) return DESIREEIA_ERR_IO;
        if (!upload(a.post_attn_norm_w, a.n_embd, &lg.d_post_attn_norm)) return DESIREEIA_ERR_IO;
        if (!upload(a.post_ffn_norm_w, a.n_embd, &lg.d_post_ffn_norm)) return DESIREEIA_ERR_IO;

        if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
            return DESIREEIA_ERR_IO;
        }

        const int nthr = 256;
        // The norm needs a reduction over the whole vector, so it runs in
        // ONE block, i.e. on one SM out of twenty. Profiled, the two of
        // them cost 0.83 ms per token (4% of the decode) for 10 KB of
        // arithmetic, so the width of that single block looked like free
        // money. It is not: 1024 threads measured better on one model and
        // 4% WORSE on another, and 512 was worse than 256 on both, which
        // is not a shape any real effect has. The spread is the noise
        // floor of this machine, so the original width stays until there
        // is a measurement that separates the two.
        const int norm_thr = nthr;
        const unsigned embd_blocks = (unsigned) ((a.n_embd + nthr - 1) / nthr);
        int8_t* d_xq = reinterpret_cast<int8_t*>(g_scratch.d_stage);
        float* d_xscale = reinterpret_cast<float*>(g_scratch.d_stage + nb_x * 32);

        static const bool dec_fast = std::getenv("DESIREEIA_CUDA_NO_DEC_FAST") == nullptr;
        const bool fast_norm = dec_fast && a.n_embd <= (uint32_t) (kDecNormThreads * kDecNormMaxV);
        if (fast_norm) {
            dec_norm_fast_kernel<<<1, kDecNormThreads, 0, stream>>>(
                g_scratch.d_xin, nullptr, nullptr, lg.d_attn_norm, g_scratch.d_xn,
                d_xq, d_xscale, g_scratch.d_xsum, (uint32_t) a.n_embd, a.rms_eps);
        } else {
            rms_norm_quant_kernel<false><<<1, norm_thr, norm_thr * sizeof(float), stream>>>(
                g_scratch.d_xin, nullptr, lg.d_attn_norm, g_scratch.d_xn,
                d_xq, d_xscale, g_scratch.d_xsum, (uint32_t) a.n_embd, a.rms_eps);
        }
        // Set when the per-head gate was folded into the Q/K/V group below.
        bool gate_grouped = false;
        if (a.fmt_q == 0 && a.fmt_k == 0 && a.fmt_v == 0) {
            Q80GroupDesc qkv{};
            qkv.n = 3;
            qkv.qs[0] = static_cast<const int8_t*>(a.wq_qs); qkv.scale[0] = static_cast<const __half*>(a.wq_scale);
            qkv.y[0] = g_scratch.d_qbuf; qkv.bias[0] = lg.d_bq; qkv.rows[0] = (uint32_t) a.q_dim;
            qkv.qs[1] = static_cast<const int8_t*>(a.wk_qs); qkv.scale[1] = static_cast<const __half*>(a.wk_scale);
            qkv.y[1] = g_scratch.d_kbuf; qkv.bias[1] = lg.d_bk; qkv.rows[1] = (uint32_t) a.kv_dim;
            qkv.qs[2] = static_cast<const int8_t*>(a.wv_qs); qkv.scale[2] = static_cast<const __half*>(a.wv_scale);
            qkv.y[2] = g_scratch.d_vbuf; qkv.bias[2] = lg.d_bv; qkv.rows[2] = (uint32_t) a.kv_dim;
            launch_matmul_q8_0_group(qkv, d_xq, d_xscale, nb_x, stream);
        } else {
            // One launch when the three share a format — which they do
            // whenever they come from one fused QKV tensor, and usually
            // otherwise too. K and V are 1024 rows here, 64 blocks over
            // twenty SMs: far too small to fill the device on their own.
            KQuantGroupDesc qkv{};
            bool grouped = false;
            if (a.fmt_q == a.fmt_k && a.fmt_k == a.fmt_v) {
                qkv.n = 3;
                // The per-head gate reads the same activation as Q/K/V, so
                // it can ride along instead of being its own launch. On its
                // own it is ONE block (one output row per head) — the whole
                // device running a single block, serialised between two
                // real kernels.
                gate_grouped = a.wag_qs != nullptr && a.fmt_ag == a.fmt_q;
                if (gate_grouped) {
                    qkv.n = 4;
                    qkv.w[3] = static_cast<const uint8_t*>(a.wag_qs);
                    qkv.y[3] = g_scratch.d_gate; qkv.bias[3] = nullptr;
                    qkv.rows[3] = a.n_head;
                }
                qkv.w[0] = static_cast<const uint8_t*>(a.wq_qs);
                qkv.y[0] = g_scratch.d_qbuf; qkv.bias[0] = lg.d_bq; qkv.rows[0] = (uint32_t) a.q_dim;
                qkv.w[1] = static_cast<const uint8_t*>(a.wk_qs);
                qkv.y[1] = g_scratch.d_kbuf; qkv.bias[1] = lg.d_bk; qkv.rows[1] = (uint32_t) a.kv_dim;
                qkv.w[2] = static_cast<const uint8_t*>(a.wv_qs);
                qkv.y[2] = g_scratch.d_vbuf; qkv.bias[2] = lg.d_bv; qkv.rows[2] = (uint32_t) a.kv_dim;
                grouped = launch_matmul_kquant_group(a.fmt_q, qkv, d_xq, d_xscale,
                                                      g_scratch.d_xsum, a.n_embd, stream);
                gate_grouped = gate_grouped && grouped;
            } else if (!a.wag_qs) {
                // Formats differ (typically V in Q6_K): one mixed launch.
                qkv.n = 3;
                qkv.w[0] = static_cast<const uint8_t*>(a.wq_qs); qkv.fmt[0] = a.fmt_q;
                qkv.y[0] = g_scratch.d_qbuf; qkv.bias[0] = lg.d_bq; qkv.rows[0] = (uint32_t) a.q_dim;
                qkv.w[1] = static_cast<const uint8_t*>(a.wk_qs); qkv.fmt[1] = a.fmt_k;
                qkv.y[1] = g_scratch.d_kbuf; qkv.bias[1] = lg.d_bk; qkv.rows[1] = (uint32_t) a.kv_dim;
                qkv.w[2] = static_cast<const uint8_t*>(a.wv_qs); qkv.fmt[2] = a.fmt_v;
                qkv.y[2] = g_scratch.d_vbuf; qkv.bias[2] = lg.d_bv; qkv.rows[2] = (uint32_t) a.kv_dim;
                grouped = launch_matmul_kquant_mixed_group(qkv, d_xq, d_xscale, g_scratch.d_xsum, a.n_embd, stream);
            }
            if (!grouped) {
                emit_matvec(a.fmt_q, a.wq_qs, a.wq_scale, d_xq, d_xscale, g_scratch.d_xsum,
                             a.n_embd, a.q_dim, g_scratch.d_qbuf, lg.d_bq, stream);
                emit_matvec(a.fmt_k, a.wk_qs, a.wk_scale, d_xq, d_xscale, g_scratch.d_xsum,
                             a.n_embd, a.kv_dim, g_scratch.d_kbuf, lg.d_bk, stream);
                emit_matvec(a.fmt_v, a.wv_qs, a.wv_scale, d_xq, d_xscale, g_scratch.d_xsum,
                             a.n_embd, a.kv_dim, g_scratch.d_vbuf, lg.d_bv, stream);
            }
        }

        // QK-norm, before RoPE, same order as the CPU path. With both norms
        // present the two norms and the rotation are one launch.
        static const bool dec_fuse = std::getenv("DESIREEIA_CUDA_NO_DEC_FUSE") == nullptr;
        if (dec_fuse && lg.d_q_norm && lg.d_k_norm) {
            dec_qk_norm_rope_kernel<<<a.n_head + a.n_head_kv, 64, 64 * sizeof(float), stream>>>(
                g_scratch.d_qbuf, g_scratch.d_kbuf, lg.d_q_norm, lg.d_k_norm,
                a.rope_cache ? d_ropel : nullptr, a.n_rot, a.head_dim, a.n_head, a.rms_eps);
        } else {
            if (lg.d_q_norm) {
                rms_norm_heads_kernel<<<a.n_head, 64, 64 * sizeof(float), stream>>>(
                    g_scratch.d_qbuf, lg.d_q_norm, a.head_dim, a.rms_eps);
            }
            if (lg.d_k_norm) {
                rms_norm_heads_kernel<<<a.n_head_kv, 64, 64 * sizeof(float), stream>>>(
                    g_scratch.d_kbuf, lg.d_k_norm, a.head_dim, a.rms_eps);
            }
            if (a.rope_cache) {
                rope_neox_kernel<<<a.n_head + a.n_head_kv, 64, 0, stream>>>(
                    g_scratch.d_qbuf, g_scratch.d_kbuf, d_ropel,
                    a.n_rot, a.head_dim, a.n_head);
            }
        }

        const unsigned kvgrid = (unsigned) (a.n_head_kv * nblk);
        kv_quantize_kernel<<<kvgrid * 2, 32, 0, stream>>>(
            g_scratch.d_kbuf, g_scratch.d_kcache + a.kv_layer_off,
            g_scratch.d_vbuf, g_scratch.d_vcache + a.kv_layer_off,
            a.head_dim, kv_row_bytes, kvgrid, pos_bytes, d_dynl);

        attention_decode_q8(g_scratch.d_qbuf, g_scratch.d_kcache, g_scratch.d_vcache, g_scratch.d_attn,
                            a.n_head, a.head_dim, a.heads_per_kv, kv_row_bytes, pos_bytes,
                            a.kv_layer_off, d_dynl, 1.0f / sqrtf((float) a.head_dim), stream);

        // Per-head output gate, between the attention and its projection.
        // It has to be emitted HERE, before the attention output is
        // quantized below: the gate matvec reads the quantized normalized
        // input (d_xq/d_xscale/d_xsum) that fed Q/K/V, and that next
        // quantisation overwrites the sums buffer.
        //
        // d_gate is borrowed for the n_head gate values. It belongs to the
        // FFN, which runs later in this same graph and rewrites it before
        // reading it, so the two uses never overlap.
        if (a.wag_qs) {
            // Already computed alongside Q/K/V when the formats allowed it;
            // d_gate has held the values since, untouched (the FFN rewrites
            // it only further down).
            if (!gate_grouped) {
                emit_matvec(a.fmt_ag, a.wag_qs, a.wag_scale, d_xq, d_xscale, g_scratch.d_xsum,
                             a.n_embd, a.n_head, g_scratch.d_gate, nullptr, stream);
            }
            attn_gate_apply_kernel<<<a.n_head, 64, 0, stream>>>(
                g_scratch.d_attn, g_scratch.d_gate, a.head_dim);
        }

        int8_t* d_aq = reinterpret_cast<int8_t*>(g_scratch.d_stage2);
        float* d_ascale = reinterpret_cast<float*>(g_scratch.d_stage2 + nb_attn * 32);
        quantize_q8_0_kernel<<<(unsigned) nb_attn, 32, 0, stream>>>(g_scratch.d_attn, a.q_dim,
                                                                     d_aq, d_ascale, g_scratch.d_xsum);
        emit_matvec(a.fmt_o, a.wo_qs, a.wo_scale, d_aq, d_ascale, g_scratch.d_xsum,
                     a.q_dim, a.n_embd, g_scratch.d_y, nullptr, stream);
        // Sandwich norm: the projection is normalised before the residual.
        // Then the attention residual, FFN norm and its quantisation: d_y
        // takes the residual sum (the layer still needs it) while d_xn2 and
        // d_xq take the normalised and quantised form. One launch for all.
        if (fast_norm) {
            dec_norm_fast_kernel<<<1, kDecNormThreads, 0, stream>>>(
                g_scratch.d_y, g_scratch.d_xin, lg.d_post_attn_norm, lg.d_ffn_norm, g_scratch.d_xn2,
                d_xq, d_xscale, g_scratch.d_xsum, (uint32_t) a.n_embd, a.rms_eps);
        } else if (dec_fuse && lg.d_post_attn_norm) {
            dec_post_norm_resid_quant_kernel<<<1, norm_thr, norm_thr * sizeof(float), stream>>>(
                g_scratch.d_y, g_scratch.d_xin, lg.d_post_attn_norm, lg.d_ffn_norm, g_scratch.d_xn2,
                d_xq, d_xscale, g_scratch.d_xsum, (uint32_t) a.n_embd, a.rms_eps);
        } else {
            if (lg.d_post_attn_norm) {
                rms_norm_kernel<<<1, nthr, nthr * sizeof(float), stream>>>(
                    g_scratch.d_y, lg.d_post_attn_norm, g_scratch.d_y,
                    (uint32_t) a.n_embd, a.rms_eps);
            }
            rms_norm_quant_kernel<true><<<1, norm_thr, norm_thr * sizeof(float), stream>>>(
                g_scratch.d_y, g_scratch.d_xin, lg.d_ffn_norm, g_scratch.d_xn2,
                d_xq, d_xscale, g_scratch.d_xsum, (uint32_t) a.n_embd, a.rms_eps);
        }
        if (a.fmt_up == 0 && a.fmt_gate == 0) {
            Q80GroupDesc gu{};
            gu.n = 2;
            gu.qs[0] = static_cast<const int8_t*>(a.wup_qs); gu.scale[0] = static_cast<const __half*>(a.wup_scale);
            gu.y[0] = g_scratch.d_up; gu.bias[0] = nullptr; gu.rows[0] = (uint32_t) a.n_ff;
            gu.qs[1] = static_cast<const int8_t*>(a.wgate_qs); gu.scale[1] = static_cast<const __half*>(a.wgate_scale);
            gu.y[1] = g_scratch.d_gate; gu.bias[1] = nullptr; gu.rows[1] = (uint32_t) a.n_ff;
            launch_matmul_q8_0_group(gu, d_xq, d_xscale, nb_x, stream);
        } else {
            KQuantGroupDesc gu{};
            bool grouped = false;
            if (a.fmt_up == a.fmt_gate) {
                gu.n = 2;
                gu.w[0] = static_cast<const uint8_t*>(a.wup_qs);
                gu.y[0] = g_scratch.d_up;   gu.bias[0] = nullptr; gu.rows[0] = (uint32_t) a.n_ff;
                gu.w[1] = static_cast<const uint8_t*>(a.wgate_qs);
                gu.y[1] = g_scratch.d_gate; gu.bias[1] = nullptr; gu.rows[1] = (uint32_t) a.n_ff;
                grouped = launch_matmul_kquant_group(a.fmt_up, gu, d_xq, d_xscale,
                                                      g_scratch.d_xsum, a.n_embd, stream);
            }
            if (!grouped) {
                emit_matvec(a.fmt_up, a.wup_qs, a.wup_scale, d_xq, d_xscale, g_scratch.d_xsum,
                             a.n_embd, a.n_ff, g_scratch.d_up, nullptr, stream);
                emit_matvec(a.fmt_gate, a.wgate_qs, a.wgate_scale, d_xq, d_xscale, g_scratch.d_xsum,
                             a.n_embd, a.n_ff, g_scratch.d_gate, nullptr, stream);
            }
        }

        ffn_act_quant_kernel<<<(unsigned) nb_h, 32, 0, stream>>>(
            g_scratch.d_up, g_scratch.d_gate, g_scratch.d_hq, g_scratch.d_hscale,
            g_scratch.d_xsum, a.n_ff, a.act_gelu);
        emit_matvec(a.fmt_down, a.wdown_qs, a.wdown_scale, g_scratch.d_hq, g_scratch.d_hscale,
                     g_scratch.d_xsum, a.n_ff, a.n_embd, g_scratch.d_out, nullptr, stream);
        // The new x is written back into the SAME buffer the layer read
        // from, so consecutive layers chain on device with no transfer in
        // between. proj (d_y) and fout (d_out) are distinct buffers, so
        // there is no aliasing hazard here.
        if (fast_norm && lg.d_post_ffn_norm) {
            dec_post_add_fast_kernel<<<1, kDecNormThreads, 0, stream>>>(
                g_scratch.d_out, lg.d_post_ffn_norm, g_scratch.d_xin, g_scratch.d_y, (uint32_t) a.n_embd, a.rms_eps);
        } else if (dec_fuse && lg.d_post_ffn_norm) {
            dec_post_norm_add_kernel<<<1, nthr, nthr * sizeof(float), stream>>>(
                g_scratch.d_out, lg.d_post_ffn_norm, g_scratch.d_xin, g_scratch.d_y, (uint32_t) a.n_embd, a.rms_eps);
        } else {
            if (lg.d_post_ffn_norm) {
                rms_norm_kernel<<<1, nthr, nthr * sizeof(float), stream>>>(
                    g_scratch.d_out, lg.d_post_ffn_norm, g_scratch.d_out,
                    (uint32_t) a.n_embd, a.rms_eps);
            }
            add2_kernel<<<embd_blocks, nthr, 0, stream>>>(g_scratch.d_xin, g_scratch.d_out,
                                                           g_scratch.d_y, (uint32_t) a.n_embd);
        }

        if (cudaStreamEndCapture(stream, &lg.graph) != cudaSuccess) return DESIREEIA_ERR_IO;
        if (cudaGraphInstantiate(&lg.exec, lg.graph, nullptr, nullptr, 0) != cudaSuccess) {
            cudaGraphDestroy(lg.graph);
            lg.graph = nullptr;
            return DESIREEIA_ERR_IO;
        }
    }

    if (cudaGraphLaunch(lg.exec, stream) != cudaSuccess) {
        if (std::getenv("DESIREEIA_CUDA_DEBUG")) std::fprintf(stderr, "[cuda] graph launch failed pos=%u\n", a.pos);
        return DESIREEIA_ERR_IO;
    }
    {
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            if (std::getenv("DESIREEIA_CUDA_DEBUG")) {
                std::fprintf(stderr, "[cuda] error after launch pos=%u: %s\n", a.pos, cudaGetErrorString(e));
            }
            return DESIREEIA_ERR_IO;
        }
    }
    // Only the last layer of the token brings x back and waits: the
    // intermediate layers just queue behind each other on the stream.
    if (a.download_x) {
        cudaMemcpyAsync(a.x_out, g_scratch.d_xin, a.n_embd * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        if (cudaStreamSynchronize(stream) != cudaSuccess) return DESIREEIA_ERR_IO;
    }
    return DESIREEIA_OK;
}

// Whole attention block in ONE device episode: Q/K/V projections, biases,
// RoPE, the KV-cache write, the attention itself and the output
// projection. One H2D (the normalised input plus the small RoPE table),
// one D2H (proj, plus the quantized KV row mirrored back to the host
// cache), one synchronisation.
//
// Why: measured, the cost of an episode is ~124 us on top of the pure
// bandwidth term, and the decode was doing 109 of them per token. Every
// optimisation that cut the NUMBER of episodes paid off; the ones that
// only made kernels faster did not. This merges three episodes per layer
// into two.
//
// Only the plain path is handled here. Anything else (ALiBi, qk-norm,
// clamp, MLA, non-Q8_0 weights, LoRA) is rejected up front by the caller,
// which then runs the CPU path that remains the definition of record.
int cuda_qkv_attention_out(const CudaQkvAttnArgs& a) {
    if (!g_scratch.d_kcache) return DESIREEIA_ERR_NOT_SUPPORTED;
    if (a.n_embd % 32 != 0 || a.q_dim % 32 != 0) return DESIREEIA_ERR_NOT_SUPPORTED;

    ScopedTimer prof_t(profile_counters().ns_cuda_attn);
    profile_counters().calls_cuda_attn.fetch_add(1, std::memory_order_relaxed);

    const size_t nb_x = a.n_embd / 32;
    const uint32_t nblk = a.head_dim / 32;
    const size_t kv_row_bytes = (size_t) nblk * 34;
    const size_t pos_bytes = (size_t) a.n_head_kv * kv_row_bytes;

    if (!scratch_reserve(nb_x, a.q_dim)) return DESIREEIA_ERR_IO;
    if (!scratch_reserve_attn(a.q_dim, a.kv_dim, a.n_rot, pos_bytes)) return DESIREEIA_ERR_IO;

    cudaStream_t stream = scratch_stream();

    if (!g_scratch.d_dyn) {
        if (cudaMalloc(&g_scratch.d_dyn, 4 * sizeof(uint32_t)) != cudaSuccess) return DESIREEIA_ERR_IO;
    }

    // Everything that varies per token goes up before the graph launch,
    // into buffers whose addresses the graph already holds.
    const uint32_t dyn[2] = { a.pos, a.cc_start };
    cudaMemcpyAsync(g_scratch.d_dyn, dyn, sizeof(dyn), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(g_scratch.d_xin, a.attn_in, a.n_embd * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    if (a.rope_cache) {
        cudaMemcpyAsync(g_scratch.d_rope, a.rope_cache, a.n_rot * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    }

    AttnGraph& ag = g_attn_graphs[a.wq_qs];
    if (!ag.exec) {
        // Biases are layer constants: copied once into per-layer buffers
        // the graph can safely reference.
        auto upload_bias = [&](const float* src, size_t n, float** dst) {
            if (!src) return true;
            if (cudaMalloc(dst, n * sizeof(float)) != cudaSuccess) return false;
            return cudaMemcpy(*dst, src, n * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
        };
        if (!upload_bias(a.bq, a.q_dim, &ag.d_bq)) return DESIREEIA_ERR_IO;
        if (!upload_bias(a.bk, a.kv_dim, &ag.d_bk)) return DESIREEIA_ERR_IO;
        if (!upload_bias(a.bv, a.kv_dim, &ag.d_bv)) return DESIREEIA_ERR_IO;

        if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
            return DESIREEIA_ERR_IO;
        }

        int8_t* d_xq = reinterpret_cast<int8_t*>(g_scratch.d_stage);
        float* d_xscale = reinterpret_cast<float*>(g_scratch.d_stage + nb_x * 32);
        quantize_q8_0_kernel<<<(unsigned) nb_x, 32, 0, stream>>>(g_scratch.d_xin, a.n_embd,
                                                                  d_xq, d_xscale);

        Q80GroupDesc qkv{};
        qkv.n = 3;
        qkv.qs[0] = static_cast<const int8_t*>(a.wq_qs); qkv.scale[0] = static_cast<const __half*>(a.wq_scale);
        qkv.y[0] = g_scratch.d_qbuf; qkv.bias[0] = ag.d_bq; qkv.rows[0] = (uint32_t) a.q_dim;
        qkv.qs[1] = static_cast<const int8_t*>(a.wk_qs); qkv.scale[1] = static_cast<const __half*>(a.wk_scale);
        qkv.y[1] = g_scratch.d_kbuf; qkv.bias[1] = ag.d_bk; qkv.rows[1] = (uint32_t) a.kv_dim;
        qkv.qs[2] = static_cast<const int8_t*>(a.wv_qs); qkv.scale[2] = static_cast<const __half*>(a.wv_scale);
        qkv.y[2] = g_scratch.d_vbuf; qkv.bias[2] = ag.d_bv; qkv.rows[2] = (uint32_t) a.kv_dim;
        launch_matmul_q8_0_group(qkv, d_xq, d_xscale, nb_x, stream);

        if (a.rope_cache) {
            rope_neox_kernel<<<a.n_head + a.n_head_kv, 64, 0, stream>>>(
                g_scratch.d_qbuf, g_scratch.d_kbuf, g_scratch.d_rope,
                a.n_rot, a.head_dim, a.n_head);
        }

        const unsigned kvgrid = (unsigned) (a.n_head_kv * nblk);
        kv_quantize_kernel<<<kvgrid * 2, 32, 0, stream>>>(
            g_scratch.d_kbuf, g_scratch.d_kcache + a.kv_layer_off,
            g_scratch.d_vbuf, g_scratch.d_vcache + a.kv_layer_off,
            a.head_dim, kv_row_bytes, kvgrid, pos_bytes, g_scratch.d_dyn);

        attention_decode_q8(g_scratch.d_qbuf, g_scratch.d_kcache, g_scratch.d_vcache, g_scratch.d_attn,
                            a.n_head, a.head_dim, a.heads_per_kv, kv_row_bytes, pos_bytes,
                            a.kv_layer_off, g_scratch.d_dyn, 1.0f / sqrtf((float) a.head_dim), stream);

        const size_t nb_attn = a.q_dim / 32;
        int8_t* d_aq = reinterpret_cast<int8_t*>(g_scratch.d_stage2);
        float* d_ascale = reinterpret_cast<float*>(g_scratch.d_stage2 + nb_attn * 32);
        quantize_q8_0_kernel<<<(unsigned) nb_attn, 32, 0, stream>>>(g_scratch.d_attn, a.q_dim,
                                                                     d_aq, d_ascale);
        launch_matmul_q8_0(static_cast<const int8_t*>(a.wo_qs), static_cast<const __half*>(a.wo_scale),
                            d_aq, d_ascale, nb_attn, a.n_embd, g_scratch.d_y, stream);

        if (cudaStreamEndCapture(stream, &ag.graph) != cudaSuccess) return DESIREEIA_ERR_IO;
        if (cudaGraphInstantiate(&ag.exec, ag.graph, nullptr, nullptr, 0) != cudaSuccess) {
            cudaGraphDestroy(ag.graph);
            ag.graph = nullptr;
            return DESIREEIA_ERR_IO;
        }
    }

    if (cudaGraphLaunch(ag.exec, stream) != cudaSuccess) return DESIREEIA_ERR_IO;
    if (cudaGetLastError() != cudaSuccess) return DESIREEIA_ERR_IO;

    cudaMemcpyAsync(a.proj, g_scratch.d_y, a.n_embd * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) return DESIREEIA_ERR_IO;
    return DESIREEIA_OK;
}

// FFN gated interamente su device: gate, up, attivazione, riquantizzazione
// dell'intermedio e proiezione down, con UNA sola H2D (l'attivazione in
// ingresso), UNA D2H (il risultato) e UNA sincronizzazione.
//
// Before this, the block cost, per layer: 2 D2H of n_ff floats (gate and
// up), the activation and requantization of the intermediate on CPU, one
// H2D of n_ff, and 2 synchronizations. The intermediate (n_ff = 11008 on
// the test model) is never needed by the host: it's born and dies inside
// the FFN.
int matmul_q8_0_cuda_ffn_gated(const void* d_up_qs, const void* d_up_scale,
                                const void* d_gate_qs, const void* d_gate_scale,
                                const void* d_down_qs, const void* d_down_scale,
                                size_t n_ff, size_t n_embd,
                                const float* x, float* out, int act_gelu) {
    if (!d_up_qs || !d_up_scale || !d_gate_qs || !d_gate_scale || !d_down_qs || !d_down_scale) {
        return DESIREEIA_ERR_NOT_SUPPORTED;
    }
    if (n_ff == 0 || n_embd == 0 || n_embd % 32 != 0 || n_ff % 32 != 0) {
        return DESIREEIA_ERR_NOT_SUPPORTED;
    }

    ScopedTimer prof_t(profile_counters().ns_cuda_resident);
    profile_counters().calls_cuda_resident.fetch_add(3, std::memory_order_relaxed);

    const size_t nb_x = n_embd / 32;
    const size_t nb_h = n_ff / 32;
    if (!scratch_reserve(nb_x, n_ff)) return DESIREEIA_ERR_IO;
    if (!scratch_reserve_ffn(n_ff, n_embd)) return DESIREEIA_ERR_IO;
    // d_xin holds the raw activation the graph quantizes; it is normally
    // sized by the attention path, but the FFN can run without it.
    if (n_embd > g_scratch.xin_cap) {
        if (g_scratch.d_xin) cudaFree(g_scratch.d_xin);
        g_scratch.d_xin = nullptr;
        g_scratch.xin_cap = 0;
        invalidate_graphs();
        if (cudaMalloc(&g_scratch.d_xin, n_embd * sizeof(float)) != cudaSuccess) return DESIREEIA_ERR_IO;
        g_scratch.xin_cap = n_embd;
    }

    cudaStream_t stream = scratch_stream();
    int8_t* d_xq = reinterpret_cast<int8_t*>(g_scratch.d_stage);
    float* d_xscale = reinterpret_cast<float*>(g_scratch.d_stage + nb_x * 32);
    // The activation goes up raw and is quantized on device inside the
    // graph, so the graph's inputs are fixed addresses.
    cudaMemcpyAsync(g_scratch.d_xin, x, n_embd * sizeof(float), cudaMemcpyHostToDevice, stream);

    // up and gate read the same activation just uploaded.
    // The five kernels of this episode always run with the same
    // parameters for a given layer, so they are recorded once into a
    // graph and replayed afterwards as a single submission.
    auto record_ffn = [&](cudaStream_t st) {
        quantize_q8_0_kernel<<<(unsigned) nb_x, 32, 0, st>>>(g_scratch.d_xin, n_embd,
                                                              d_xq, d_xscale);
        Q80GroupDesc gu{};
        gu.n = 2;
        gu.qs[0] = static_cast<const int8_t*>(d_up_qs); gu.scale[0] = static_cast<const __half*>(d_up_scale);
        gu.y[0] = g_scratch.d_up; gu.bias[0] = nullptr; gu.rows[0] = (uint32_t) n_ff;
        gu.qs[1] = static_cast<const int8_t*>(d_gate_qs); gu.scale[1] = static_cast<const __half*>(d_gate_scale);
        gu.y[1] = g_scratch.d_gate; gu.bias[1] = nullptr; gu.rows[1] = (uint32_t) n_ff;
        launch_matmul_q8_0_group(gu, d_xq, d_xscale, nb_x, st);

        const int threads = 256;
        const unsigned blocks = (unsigned) ((n_ff + threads - 1) / threads);
        ffn_act_kernel<<<blocks, threads, 0, st>>>(g_scratch.d_up, g_scratch.d_gate,
                                                    n_ff, act_gelu);
        quantize_q8_0_kernel<<<(unsigned) nb_h, 32, 0, st>>>(g_scratch.d_up, n_ff,
                                                              g_scratch.d_hq, g_scratch.d_hscale);
        launch_matmul_q8_0(static_cast<const int8_t*>(d_down_qs), static_cast<const __half*>(d_down_scale),
                            g_scratch.d_hq, g_scratch.d_hscale, nb_h, n_embd, g_scratch.d_out, st);
    };

    // Keyed by the down-projection weights, which are unique per layer.
    CapturedGraph& cg = g_ffn_graphs[d_down_qs];
    if (!cg.exec) {
        if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
            return DESIREEIA_ERR_IO;
        }
        record_ffn(stream);
        if (cudaStreamEndCapture(stream, &cg.graph) != cudaSuccess) return DESIREEIA_ERR_IO;
        if (cudaGraphInstantiate(&cg.exec, cg.graph, nullptr, nullptr, 0) != cudaSuccess) {
            cudaGraphDestroy(cg.graph);
            cg.graph = nullptr;
            return DESIREEIA_ERR_IO;
        }
    }
    if (cudaGraphLaunch(cg.exec, stream) != cudaSuccess) return DESIREEIA_ERR_IO;
    if (cudaGetLastError() != cudaSuccess) return DESIREEIA_ERR_IO;

    cudaMemcpyAsync(out, g_scratch.d_out, n_embd * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) return DESIREEIA_ERR_IO;
    return DESIREEIA_OK;
}

// Original path (Phase 1 PoC): uploads the entire weight matrix on every
// call. Kept as a fallback for tensors NOT covered by the persistent
// weight cache (streaming/scratch_, weight cache disabled: in that case
// the weights change/get reread every step anyway, so device residency
// would have nothing to reuse) — correct but slow for the same reason
// diagnosed above, used only when matmul_q8_0_cuda_resident doesn't apply.
int matmul_q8_0_cuda(const uint8_t* q8_data, size_t rows, size_t cols, const float* x, float* y) {
    if (cols == 0 || cols % 32 != 0 || rows == 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    ScopedTimer prof_t(profile_counters().ns_cuda_upload);
    profile_counters().calls_cuda_upload.fetch_add(1, std::memory_order_relaxed);
    const size_t nb = cols / 32;

    std::vector<int8_t> xq;
    std::vector<float> xscale;
    std::vector<uint8_t> unused_signs;
    quantize_q8_0(x, cols, xq, xscale, unused_signs);
    if (xq.size() != nb * 32 || xscale.size() != nb) return DESIREEIA_ERR_NOT_SUPPORTED;

    std::vector<int8_t> w_qs;
    std::vector<uint16_t> w_scale;
    unpack_q8_0(q8_data, rows, nb, w_qs, w_scale);

    int8_t* d_w_qs = nullptr;
    uint16_t* d_w_scale = nullptr;
    int8_t* d_x_qs = nullptr;
    float* d_x_scale = nullptr;
    float* d_y = nullptr;
    int rc = DESIREEIA_OK;

    do {
        if (cudaMalloc(&d_w_qs, w_qs.size()) != cudaSuccess) { rc = DESIREEIA_ERR_IO; break; }
        if (cudaMalloc(&d_w_scale, w_scale.size() * sizeof(uint16_t)) != cudaSuccess) { rc = DESIREEIA_ERR_IO; break; }
        if (cudaMalloc(&d_x_qs, xq.size()) != cudaSuccess) { rc = DESIREEIA_ERR_IO; break; }
        if (cudaMalloc(&d_x_scale, xscale.size() * sizeof(float)) != cudaSuccess) { rc = DESIREEIA_ERR_IO; break; }
        if (cudaMalloc(&d_y, rows * sizeof(float)) != cudaSuccess) { rc = DESIREEIA_ERR_IO; break; }

        cudaMemcpy(d_w_qs, w_qs.data(), w_qs.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w_scale, w_scale.data(), w_scale.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x_qs, xq.data(), xq.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x_scale, xscale.data(), xscale.size() * sizeof(float), cudaMemcpyHostToDevice);

        launch_matmul_q8_0(d_w_qs, reinterpret_cast<const __half*>(d_w_scale),
                            d_x_qs, d_x_scale, nb, rows, d_y, nullptr);
        if (cudaGetLastError() != cudaSuccess) { rc = DESIREEIA_ERR_IO; break; }

        cudaMemcpy(y, d_y, rows * sizeof(float), cudaMemcpyDeviceToHost);
    } while (false);

    if (d_w_qs) cudaFree(d_w_qs);
    if (d_w_scale) cudaFree(d_w_scale);
    if (d_x_qs) cudaFree(d_x_qs);
    if (d_x_scale) cudaFree(d_x_scale);
    if (d_y) cudaFree(d_y);
    return rc;
}


// ═══════════════════════════════════════════════════════════════════════════
//  Tensor-core mat-mul for the batched prefill.
//
//  The dp4a batched kernels read every weight once per 8 tokens and run on
//  the plain integer units: on a 512-token chunk each matrix was read 64
//  times and the tensor cores stayed idle (Spark 4B: ~120 tok/s of prefill).
//
//  Here each matrix is expanded once per chunk to FP16 in a scratch buffer
//  (same values the CPU dequantizers produce), the chunk's activations are
//  converted to FP16 with a per-token scale, and the product runs on the
//  tensor cores (WMMA 16x16x16, FP32 accumulation). The per-token scale
//  keeps FP16 inputs in range - the input of the down projection can exceed
//  65504 - and is undone on the output: the product is linear per token, so
//  this adds no error.
// ═══════════════════════════════════════════════════════════════════════════

namespace {

// Q4_K/Q5_K 6-bit scale and minimum of sub-block j (0..7), as the CPU
// dequantizer unpacks them.
__device__ __forceinline__ void kq_scale_min(int j, const uint8_t* q, int& sc, int& mn) {
    if (j < 4) {
        sc = q[j] & 63;
        mn = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        mn = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

// One thread per weight: W[row][c] as FP16. Mirrors dequantize_row_* in
// quant/quant.cpp for the formats it handles; the caller never launches it
// for a format it does not know.
__global__ void dequant_f16_kernel(int fmt, const uint8_t* __restrict__ w, const __half* __restrict__ q8_scale,
                                    size_t rows, size_t cols, __half* __restrict__ out) {
    const size_t e = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= rows * cols) return;
    const size_t row = e / cols;
    const size_t c = e % cols;
    float v = 0.0f;
    if (fmt == 0) {                                   // Q8_0 (split layout on device)
        const int8_t q = reinterpret_cast<const int8_t*>(w)[e];
        v = (float) q * __half2float(q8_scale[row * (cols / 32) + c / 32]);
    } else if (fmt == DESIREEIA_CUDA_FMT_Q4_0) {      // 18 bytes: d, qs[16]
        const uint8_t* b = w + (row * (cols / 32) + c / 32) * 18;
        const int j = (int) (c % 32);
        const uint8_t q = b[2 + (j % 16)];
        const int n = (j < 16 ? (q & 0xF) : (q >> 4)) - 8;
        v = (float) n * __half2float(*reinterpret_cast<const __half*>(b));
    } else if (fmt == DESIREEIA_CUDA_FMT_Q4_K || fmt == DESIREEIA_CUDA_FMT_Q5_K) {
        const bool q5 = fmt == DESIREEIA_CUDA_FMT_Q5_K;
        const size_t bsz = q5 ? 176 : 144;
        const uint8_t* b = w + (row * (cols / 256) + c / 256) * bsz;
        const int within = (int) (c % 256);
        const int j64 = within / 64;
        const int l = within % 64;
        int sc, mn;
        kq_scale_min(2 * j64 + (l >= 32 ? 1 : 0), b + 4, sc, mn);
        const uint8_t* qs = b + (q5 ? 48 : 16) + j64 * 32;
        const uint8_t q = qs[l % 32];
        int n = l < 32 ? (q & 0xF) : (q >> 4);
        if (q5) {
            const uint8_t qh = b[16 + (l % 32)];
            const int bit = 2 * j64 + (l < 32 ? 0 : 1);
            if (qh & (1 << bit)) n += 16;
        }
        const float d = __half2float(*reinterpret_cast<const __half*>(b));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(b + 2));
        v = d * (float) sc * (float) n - dmin * (float) mn;
    } else if (fmt == DESIREEIA_CUDA_FMT_Q6_K) {      // 210 bytes: ql[128] qh[64] sc[16] d
        const uint8_t* b = w + (row * (cols / 256) + c / 256) * 210;
        const int within = (int) (c % 256);
        const int n128 = within / 128;
        const int r = within % 128;
        const int quad = r / 32;
        const int l = r % 32;
        const uint8_t* ql = b + n128 * 64;
        const uint8_t* qh = b + 128 + n128 * 32;
        const int8_t* scs = reinterpret_cast<const int8_t*>(b + 192) + n128 * 8;
        const int is = l / 16;
        int q;
        switch (quad) {
            case 0:  q = (ql[l] & 0xF)        | (((qh[l] >> 0) & 3) << 4); break;
            case 1:  q = (ql[l + 32] & 0xF)   | (((qh[l] >> 2) & 3) << 4); break;
            case 2:  q = (ql[l] >> 4)         | (((qh[l] >> 4) & 3) << 4); break;
            default: q = (ql[l + 32] >> 4)    | (((qh[l] >> 6) & 3) << 4); break;
        }
        const float d = __half2float(*reinterpret_cast<const __half*>(b + 208));
        v = d * (float) scs[is + 2 * quad] * (float) (q - 32);
    }
    out[e] = __float2half(v);
}

// Activations [m][k] float -> [m_pad][k] FP16, one block per token row.
// Rows past m are zero. inv_scale[t] receives the factor the row was
// divided by (1 when it already fits comfortably in FP16).
__global__ void act_to_f16_kernel(const float* __restrict__ x, size_t k, uint32_t m,
                                   __half* __restrict__ out, float* __restrict__ row_scale) {
    const uint32_t t = blockIdx.x;
    __half* dst = out + (size_t) t * k;
    if (t >= m) {
        for (size_t i = threadIdx.x; i < k; i += blockDim.x) dst[i] = __float2half(0.0f);
        if (threadIdx.x == 0) row_scale[t] = 1.0f;
        return;
    }
    const float* src = x + (size_t) t * k;
    extern __shared__ float amax_sm[];
    float m_local = 0.0f;
    for (size_t i = threadIdx.x; i < k; i += blockDim.x) m_local = fmaxf(m_local, fabsf(src[i]));
    amax_sm[threadIdx.x] = m_local;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) amax_sm[threadIdx.x] = fmaxf(amax_sm[threadIdx.x], amax_sm[threadIdx.x + s]);
        __syncthreads();
    }
    // Keep the largest input around 2^10: far from the FP16 ceiling (65504),
    // with the product of k terms accumulated in FP32 by the tensor cores.
    const float amax = amax_sm[0];
    const float scale = amax > 1024.0f ? amax / 1024.0f : 1.0f;
    const float inv = 1.0f / scale;
    for (size_t i = threadIdx.x; i < k; i += blockDim.x) dst[i] = __float2half(src[i] * inv);
    if (threadIdx.x == 0) row_scale[t] = scale;
}

// Y[t][r] = sum_k A[t][k] * W[r][k] on the tensor cores.
//   A [m_pad][k] FP16 (m_pad multiple of kGemmBM), W [rows][k] FP16,
//   Y [m_pad][rows] FP32.
// Block tile 128 tokens x 128 rows, 8 warps in a 4 x 2 grid, each warp
// 32 tokens x 64 rows (2 x 4 fragments of 16 x 16): every A fragment is
// used 4 times and every B fragment twice per load. Operand tiles go
// through shared memory in 32-wide k steps with 128-bit loads, double
// buffered: while the tensor cores work on one step, the next is already
// in registers, so one barrier per step instead of two. Rows are padded by
// 8 halves against bank conflicts. Requires k % 32 == 0, rows % 16 == 0.
constexpr int kGemmBM = 128;
constexpr int kGemmBN = 128;
constexpr int kGemmBK = 32;
constexpr int kGemmLd = kGemmBK + 8;

__global__ void __launch_bounds__(256) wmma_gemm_kernel(const __half* __restrict__ a,
                                                        const __half* __restrict__ wt,
                                                        float* __restrict__ y,
                                                        const float* __restrict__ row_scale,
                                                        size_t k, size_t rows, uint32_t m) {
    using namespace nvcuda;
    __shared__ __align__(32) __half as[2][kGemmBM][kGemmLd];
    __shared__ __align__(32) __half bs[2][kGemmBN][kGemmLd];

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int wm = warp / 2;   // 0..3 -> 32-token band
    const int wn = warp % 2;   // 0..1 -> 64-row band
    const size_t t0 = (size_t) blockIdx.y * kGemmBM;
    const size_t r0 = (size_t) blockIdx.x * kGemmBN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    // Each thread moves two 16-byte pieces of A and two of B per k step.
    const int4 zero = make_int4(0, 0, 0, 0);
    int4 ra[2], rb[2];
    auto fetch = [&](size_t kk) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = tid + i * 256;
            const int r = idx / 4, c = (idx % 4) * 8;
            ra[i] = *reinterpret_cast<const int4*>(a + (t0 + r) * k + kk + c);
            const size_t row = r0 + r;
            rb[i] = row < rows ? *reinterpret_cast<const int4*>(wt + row * k + kk + c) : zero;
        }
    };
    auto stash = [&](int buf) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = tid + i * 256;
            const int r = idx / 4, c = (idx % 4) * 8;
            *reinterpret_cast<int4*>(&as[buf][r][c]) = ra[i];
            *reinterpret_cast<int4*>(&bs[buf][r][c]) = rb[i];
        }
    };

    fetch(0);
    stash(0);
    __syncthreads();
    int buf = 0;
    for (size_t kk = 0; kk < k; kk += kGemmBK) {
        const bool more = kk + kGemmBK < k;
        if (more) fetch(kk + kGemmBK);           // next step in flight during the math
        #pragma unroll
        for (int ks = 0; ks < kGemmBK; ks += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb[4];
            #pragma unroll
            for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(fa[i], &as[buf][wm * 32 + i * 16][ks], kGemmLd);
            #pragma unroll
            for (int j = 0; j < 4; ++j) wmma::load_matrix_sync(fb[j], &bs[buf][wn * 64 + j * 16][ks], kGemmLd);
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
        if (more) stash(buf ^ 1);                // the other buffer is free since the last barrier
        __syncthreads();
        buf ^= 1;
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const size_t r = r0 + wn * 64 + j * 16;
            if (r < rows) {
                wmma::store_matrix_sync(y + (t0 + wm * 32 + i * 16) * rows + r, acc[i][j],
                                        (unsigned) rows, wmma::mem_row_major);
            }
        }
    // Undo the per-token input scale on this warp's own 32 x 64 tile while it
    // is still hot in L2 (the fragment element layout is unspecified, so this
    // goes through the stored values). Padding tokens (>= m) are left alone.
    __syncwarp();
    const int lane = tid % 32;
    for (int e = lane; e < 32 * 64; e += 32) {
        const size_t t = t0 + wm * 32 + e / 64;
        const size_t r = r0 + wn * 64 + e % 64;
        if (t < m && r < rows) {
            const float sc = row_scale[t];
            if (sc != 1.0f) y[t * rows + r] *= sc;
        }
    }
}

// FP16 copies of the matrices of ONE layer. The prefill runs layer by layer
// (every chunk of the prompt through layer l before layer l+1), so each
// matrix is expanded once per prompt and reused by every chunk.
constexpr int kTcSlots = 8;   // q, k, v, o, gate, up, down, attention gate
struct TensorCoreScratch {
    __half* w[kTcSlots] = {};  size_t w_cap[kTcSlots] = {};
    bool ready[kTcSlots] = {};
    __half* a = nullptr;  size_t a_cap = 0;      // FP16 activations [m_pad][k]
    float* scale = nullptr; size_t scale_cap = 0;
    // Int8 form of a slot (see the int8 section below): q [rows][k], scale and
    // minimum [rows][k/32]. is_i8 says which form the slot holds.
    bool is_i8[kTcSlots] = {};
    int8_t* w8[kTcSlots] = {};  size_t w8_cap[kTcSlots] = {};
    float* ws[kTcSlots] = {};   size_t ws_cap[kTcSlots] = {};
    float* wm[kTcSlots] = {};   size_t wm_cap[kTcSlots] = {};
    bool w_has_m[kTcSlots] = {};
    bool w_split[kTcSlots] = {};          // Q6_K: scales per 16 (low in ws, high in wm)
    int w_raw[kTcSlots] = {};             // 1 Q4_K / 2 Q5_K: the kernel reads the device blocks
    int w_hkq[kTcSlots] = {};             // FP16 product on the stored blocks: 1 Q4_K, 2 Q5_K, 3 Q8_0, 4 Q6_K
    const void* wsp[kTcSlots] = {};       // their half2 scales (ws, or the cached copy)
    float* ytmp = nullptr; size_t ytmp_cap = 0;  // result of a non-fused product that gets a residual
    const int8_t* w8p[kTcSlots] = {};         // int8 weights read by the kernel (the device Q8_0 itself, or w8)
    size_t rows_pad[kTcSlots] = {};
    int8_t* a8 = nullptr; size_t a8_cap = 0;     // int8 activations [m_pad][k]
    float* as = nullptr;  size_t as_cap = 0;     // their scales [m_pad][k/32]
    float* ax = nullptr;  size_t ax_cap = 0;     // block sums [m_pad][k/32]
    // Which conversions of the current input are valid (reset when the input changes).
    bool f16_in = false, i8_in = false;
};
TensorCoreScratch g_tc;

bool tc_grow(void** p, size_t& cap, size_t bytes) {
    if (bytes <= cap && *p) return true;
    if (*p) cudaFree(*p);
    *p = nullptr; cap = 0;
    if (cudaMalloc(p, bytes) != cudaSuccess) return false;
    cap = bytes;
    return true;
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════
//  Int8 tensor-core products (sm_80+)
//
//  The FP16 path above expands every weight to 16 bits and runs at the FP16
//  tensor-core rate. Ada/Ampere tensor cores multiply int8 at twice that
//  rate, and the weights are small integers to begin with. Here:
//    * weights: once per layer, each 32-wide block becomes 32 int8 plus a
//      float scale (and, for Q4_K/Q5_K, a float minimum): w = s*q - m, exact;
//    * activations: per token and per 32, int8 plus a scale, and the float
//      sum of the block for the minimum term;
//    * mma.sync m16n8k32: one mma covers exactly one 32-wide block, so its
//      int32 result is scaled (s_w * s_x) and accumulated in float right in
//      the registers - the accumulator layout of mma.sync is specified, which
//      is what WMMA does not give.
//  Q6_K (a scale every 16) keeps the FP16 path; so does anything below sm_80.
// ═══════════════════════════════════════════════════════════════════════════
namespace {


// One thread per 4 consecutive weights (one 32-bit word of the output), so
// neighbouring threads write neighbouring words; the thread holding the
// first word of a 32-block also writes its scale and minimum.
__global__ void dequant_i8_kernel(int fmt, const uint8_t* __restrict__ w, const __half* __restrict__ q8_scale,
                                   size_t rows, size_t cols, int8_t* __restrict__ q_out,
                                   float* __restrict__ s_out, float* __restrict__ m_out, size_t rows_pad) {
    const size_t nb = cols / 32;
    const size_t e = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= rows * cols / 4) return;
    const size_t row = e / (cols / 4);
    const size_t c = (e % (cols / 4)) * 4;           // first column of this word
    const size_t kb = c / 32;
    const int l0 = (int) (c % 32);
    int8_t v[4];
    float s = 0.0f, m = 0.0f;
    if (fmt == 0) {                                   // Q8_0, split layout on device
        const int8_t* src = reinterpret_cast<const int8_t*>(w) + row * cols + c;
        for (int i = 0; i < 4; ++i) v[i] = src[i];
        s = __half2float(q8_scale[row * nb + kb]);
    } else if (fmt == DESIREEIA_CUDA_FMT_Q4_0) {
        const uint8_t* b = w + (row * nb + kb) * 18;
        for (int i = 0; i < 4; ++i) {
            const int l = l0 + i;
            v[i] = (int8_t) ((l < 16 ? (b[2 + l] & 0xF) : (b[2 + l - 16] >> 4)) - 8);
        }
        s = __half2float(*reinterpret_cast<const __half*>(b));
    } else if (fmt == DESIREEIA_CUDA_FMT_Q6_K) {      // 210 bytes: ql[128] qh[64] sc[16] d
        const uint8_t* b = w + (row * (cols / 256) + kb / 8) * 210;
        const int within0 = (int) ((kb % 8) * 32);
        const int n128 = within0 / 128, quad = (within0 % 128) / 32;
        const uint8_t* ql = b + n128 * 64;
        const uint8_t* qh = b + 128 + n128 * 32;
        const int8_t* scs = reinterpret_cast<const int8_t*>(b + 192) + n128 * 8;
        for (int i = 0; i < 4; ++i) {
            const int l = l0 + i;
            int q;
            switch (quad) {
                case 0:  q = (ql[l] & 0xF)      | (((qh[l] >> 0) & 3) << 4); break;
                case 1:  q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4); break;
                case 2:  q = (ql[l] >> 4)       | (((qh[l] >> 4) & 3) << 4); break;
                default: q = (ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4); break;
            }
            v[i] = (int8_t) (q - 32);
        }
        if (l0 == 0) {                                // scale of the low and the high 16
            const float d = __half2float(*reinterpret_cast<const __half*>(b + 208));
            s = d * (float) scs[2 * quad];
            m = d * (float) scs[2 * quad + 1];
        }
    } else {                                          // Q4_K / Q5_K
        const bool q5 = fmt == DESIREEIA_CUDA_FMT_Q5_K;
        const size_t bsz = q5 ? 176 : 144;
        const uint8_t* b = w + (row * (cols / 256) + kb / 8) * bsz;
        const int sub = (int) (kb % 8);
        const bool hi = sub & 1;
        const uint8_t* qs = b + (q5 ? 48 : 16) + (sub / 2) * 32;
        for (int i = 0; i < 4; ++i) {
            const int l = l0 + i;
            int n = hi ? (qs[l] >> 4) : (qs[l] & 0xF);
            if (q5 && (b[16 + l] & (1 << sub))) n += 16;
            v[i] = (int8_t) n;
        }
        if (l0 == 0) {
            int sc, mn;
            kq_scale_min(sub, b + 4, sc, mn);
            s = __half2float(*reinterpret_cast<const __half*>(b)) * (float) sc;
            m = __half2float(*reinterpret_cast<const __half*>(b + 2)) * (float) mn;
        }
    }
    int32_t word;
    memcpy(&word, v, 4);
    reinterpret_cast<int32_t*>(q_out)[e] = word;
    if (l0 == 0) {                            // transposed: [block][row]
        s_out[kb * rows_pad + row] = s;
        if (m_out) m_out[kb * rows_pad + row] = m;
    }
}

// One warp per (token, 32-block): int8 with its scale, and the float sum.
// Tokens past m become zeros.
__global__ void act_to_i8_kernel(const float* __restrict__ x, size_t k, uint32_t m, uint32_t m_pad,
                                  int8_t* __restrict__ q, float* __restrict__ s, float* __restrict__ sum) {
    const size_t nb = k / 32;
    const size_t wid = ((size_t) blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (wid >= (size_t) m_pad * nb) return;
    const size_t t = wid / nb, kb = wid % nb;
    const float v = t < m ? x[t * k + kb * 32 + lane] : 0.0f;
    float amax = fabsf(v), tot = v;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
        tot += __shfl_xor_sync(0xffffffff, tot, o);
    }
    const float d = amax / 127.0f;
    q[t * k + kb * 32 + lane] = (int8_t) (d > 0.0f ? __float2int_rn(v / d) : 0);
    if (lane == 0) {                          // transposed: [block][token]
        s[kb * m_pad + t] = d;
        if (sum) sum[kb * m_pad + t] = tot;
    }
}

// Activations -> int8 for the int8 products (hkq_kernel I8): 4 threads
// per 32-block, 8 values each. The scale keeps 22 significant bits (so
// -M * scale is exact there) and quantizes with that same value; written
// as (d, -1.5 * 2^23 * d) pairs [block][token]. Tokens past m are zeros.
__global__ void act_to_i8q_kernel(const float* __restrict__ x, size_t k, uint32_t m, uint32_t m_pad,
                                  int8_t* __restrict__ q, float2* __restrict__ sc) {
    const uint32_t t = blockIdx.x;
    const size_t e0 = ((size_t) blockIdx.y * blockDim.x + threadIdx.x) * 8;
    const bool ok = e0 < k;                  // no early exit: whole-warp shuffles below
    float v[8];
    if (ok && t < m) {
        const float4 a = *reinterpret_cast<const float4*>(x + (size_t) t * k + e0);
        const float4 b = *reinterpret_cast<const float4*>(x + (size_t) t * k + e0 + 4);
        v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
    } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = 0.0f;
    }
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(v[i]));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
    // round the scale UP to 22 bits so |v / d| stays <= 127
    const uint32_t db = __float_as_uint(amax / 127.0f);
    const float d = __uint_as_float((db & 3u) ? (db | 3u) + 1u : db);
    const float inv = d > 0.0f ? 1.0f / d : 0.0f;
    uint32_t w[2];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        uint32_t o = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int qi = max(-127, min(127, __float2int_rn(v[h * 4 + i] * inv)));
            o |= (uint32_t) (qi & 0xFF) << (8 * i);
        }
        w[h] = o;
    }
    if (!ok) return;
    *reinterpret_cast<uint2*>(q + (size_t) t * k + e0) = make_uint2(w[0], w[1]);
    if ((threadIdx.x & 3) == 0) sc[(e0 / 32) * m_pad + t] = make_float2(d, -12582912.0f * d);
}

// A shared-memory float2 load the compiler keeps in program order (volatile
// asm), so it is not hoisted ahead of the products that precede it.
__device__ __forceinline__ float2 lds_f2_pinned(const float2* p) {
    float2 v;
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "r"(a));
    return v;
}

__device__ __forceinline__ void mma_s8_16832(int (&d)[4], const int (&a)[4], int b0, int b1, int c = 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
          "r"(c), "r"(c), "r"(c), "r"(c));
#else
    d[0] = d[1] = d[2] = d[3] = 0;   // never launched below sm_80 (host checks)
#endif
}

int g_sm_count() {
    static const int v = [] {
        int dev = 0, n = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return 1;
        if (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) return 1;
        return n > 0 ? n : 1;
    }();
    return v;
}

int cuda_sm_major() {
    static const int v = [] {
        int dev = 0, major = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) return 0;
        return major;
    }();
    return v;
}

bool i8_mma_usable(int fmt, size_t cols, size_t rows) {
    // On by default: reading the quantized weights directly (mmq3_kernel)
    // measured faster than expanding them to FP16 every prefill - 1.2-1.5x at
    // 512-2048 tokens. DESIREEIA_CUDA_INT8_MMA=0 uses the FP16 GEMM instead.
    static const bool enabled = [] {
        const char* v = std::getenv("DESIREEIA_CUDA_INT8_MMA");
        return !v || v[0] != '0';
    }();
    if (!enabled || cuda_sm_major() < 8 || cols % 64 != 0 || rows % 16 != 0 || rows < 64) return false;
    switch (fmt) {
        case 0:
        case DESIREEIA_CUDA_FMT_Q4_0: return true;
        case DESIREEIA_CUDA_FMT_Q4_K:
        case DESIREEIA_CUDA_FMT_Q5_K:
        case DESIREEIA_CUDA_FMT_Q6_K: {                          // Q6_K: two k16 products per block
            static const bool q6_fp16 = std::getenv("DESIREEIA_CUDA_Q6K_FP16") != nullptr;
            return !q6_fp16 && cols % 256 == 0;
        }
        default: return false;
    }
}

} // namespace

namespace {

__device__ __forceinline__ void mma_f16_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}

// ═══════════════════════════════════════════════════════════════════════════
//  FP16 tensor-core GEMM, pipelined (sm_80+)
//
//  Measured ceiling of the laptop GPU this was tuned on: 17.8 TMAC/s FP16,
//  35.5 int8. The WMMA kernel above reached ~5.6 (31%): one stage, loads
//  through registers, a barrier between every load and its use. Here:
//    * cp.async copies global -> shared directly, 3 stages in flight, so the
//      tensor cores never wait for memory;
//    * ldmatrix builds the mma fragments (one instruction per 16x16 of A or
//      2x(8x16) of B); rows padded to 40 halves (80 bytes), which makes the
//      8 row addresses of an ldmatrix fall on 8 different bank groups;
//    * mma.sync m16n8k16, FP32 accumulation; block 128 tokens x 128 rows,
//      8 warps of 64 x 32, 16 mma per warp per 16-wide k step;
//    * per-token input scale (see act_to_f16_kernel) undone in the epilogue.
// ═══════════════════════════════════════════════════════════════════════════
constexpr int kHgBM = 128, kHgBK = 32, kHgStages = 3;
constexpr int kHgLd = kHgBK + 8;     // halves per smem row: 80 bytes

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    const int n = valid ? 16 : 0;              // zero-fill past the matrix edge
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(n));
#endif
}
__device__ __forceinline__ void cp_async4(void* smem, const void* gmem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(s), "l"(gmem));
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
#endif
}

template <int BN>
constexpr size_t hgemm_smem_bytes() {
    return (size_t) kHgStages * (kHgBM + BN) * kHgLd * sizeof(__half);
}

// Y[t][r] = row_scale[t] * sum_k A[t][k] W[r][k]
//   A [m_pad][k] FP16 (m_pad multiple of 128), W [rows][k] FP16, k % 32 == 0.
// BN = 128: warps 64 x 32; BN = 256: warps 64 x 64 (each A fragment read
// from shared memory serves twice the products - used for the wide matrices).
template <int BN>
__global__ void __launch_bounds__(256) hgemm_kernel(const __half* __restrict__ A, const __half* __restrict__ W,
                                                     float* __restrict__ y, const float* __restrict__ row_scale,
                                                     size_t k, size_t rows) {
    extern __shared__ __align__(16) unsigned char hg_smem[];
    __half* As = reinterpret_cast<__half*>(hg_smem);                        // [S][BM][ld]
    __half* Bs = As + kHgStages * kHgBM * kHgLd;                             // [S][BN][ld]
    constexpr int WN = BN / 4, NT = WN / 8;          // warp width, 8-wide n-tiles per warp

    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int wm = warp / 4, wn = warp % 4;          // 2 x 4 warps: 64 tokens x 32 rows each
    const int g = lane >> 2, tq = lane & 3;
    // Token blocks run fastest (blockIdx.x): the blocks that share one slice
    // of weights run back to back and read it from L2, so each weight is
    // fetched from VRAM once per product instead of once per 128 tokens.
    const size_t t0 = (size_t) blockIdx.x * kHgBM;
    const size_t r0 = (size_t) blockIdx.y * BN;
    const int nk = (int) (k / kHgBK);

    // Each stage: 128 x 32 halves of A and of W = 512 chunks of 16 bytes each;
    // every thread issues 2 + 2.
    auto load_stage = [&](int stage, int kt) {
        const size_t kk = (size_t) kt * kHgBK;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int c = tid + i * 256;
            const int row = c / 4, col = (c % 4) * 8;
            cp_async16(As + ((size_t) stage * kHgBM + row) * kHgLd + col, A + (t0 + row) * k + kk + col, true);
        }
        #pragma unroll
        for (int i = 0; i < BN / 64; ++i) {
            const int c = tid + i * 256;
            const int row = c / 4, col = (c % 4) * 8;
            const size_t wr = r0 + row;
            cp_async16(Bs + ((size_t) stage * BN + row) * kHgLd + col,
                       W + (wr < rows ? wr : 0) * k + kk + col, wr < rows);
        }
    };

    float acc[4][NT][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < NT; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

    #pragma unroll
    for (int s = 0; s < kHgStages - 1; ++s) {
        if (s < nk) load_stage(s, s);
        cp_async_commit();
    }

    for (int kt = 0; kt < nk; ++kt) {
        cp_async_wait<kHgStages - 2>();
        __syncthreads();
        // Refill the stage consumed two iterations ago.
        const int nxt = kt + kHgStages - 1;
        if (nxt < nk) load_stage(nxt % kHgStages, nxt);
        cp_async_commit();

        const int st = kt % kHgStages;
        const __half* as = As + (size_t) st * kHgBM * kHgLd;
        const __half* bs = Bs + (size_t) st * BN * kHgLd;
        #pragma unroll
        for (int ks = 0; ks < kHgBK; ks += 16) {
            uint32_t af[4][4], bf[NT / 2][4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                // lanes 0-15: rows 0-15 at k ks; lanes 16-31: same rows at k ks+8
                const int row = wm * 64 + i * 16 + (lane % 16);
                ldmatrix_x4(af[i], as + row * kHgLd + ks + (lane / 16) * 8);
            }
            #pragma unroll
            for (int jj = 0; jj < NT / 2; ++jj) {
                // two n-tiles of 8: lanes 0-7 n0..7/k0, 8-15 n0..7/k8, 16-23 n8..15/k0, 24-31 n8..15/k8
                const int n = wn * WN + jj * 16 + (lane % 8) + (lane / 16) * 8;
                ldmatrix_x4(bf[jj], bs + n * kHgLd + ks + ((lane / 8) % 2) * 8);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < NT; ++j)
                    mma_f16_16816(acc[i][j], af[i], bf[j / 2][(j % 2) * 2], bf[j / 2][(j % 2) * 2 + 1]);
        }
    }
    cp_async_wait<0>();

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const size_t ta = t0 + wm * 64 + i * 16 + g;
        const float sa = row_scale[ta], sb = row_scale[ta + 8];
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            const size_t n = r0 + wn * WN + j * 8 + 2 * tq;
            if (n + 1 < rows && rows % 2 == 0) {
                *reinterpret_cast<float2*>(y + ta * rows + n) = make_float2(acc[i][j][0] * sa, acc[i][j][1] * sa);
                *reinterpret_cast<float2*>(y + (ta + 8) * rows + n) = make_float2(acc[i][j][2] * sb, acc[i][j][3] * sb);
            } else {
                if (n < rows) { y[ta * rows + n] = acc[i][j][0] * sa; y[(ta + 8) * rows + n] = acc[i][j][2] * sb; }
                if (n + 1 < rows) { y[ta * rows + n + 1] = acc[i][j][1] * sa; y[(ta + 8) * rows + n + 1] = acc[i][j][3] * sb; }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Int8 tensor-core GEMM straight on the quantized weights (sm_80+)
//
//  The FP16 path expands every weight of a layer to 16 bits before each
//  prefill: measured 39% of a 512-token prefill. Q8_0 weights are already
//  int8 rows on the device (split layout), so this kernel reads them as they
//  are; only their scales (1/32 of the data) are converted. Q4_K/Q5_K are
//  unpacked to int8 once per layer (half the bytes of FP16, no float math).
//
//  Same machinery as hgemm_kernel: cp.async 3-stage pipeline, ldmatrix (on
//  int8 data the b16 8x8 tiles are exactly the m16n8k32 A/B fragments),
//  token blocks fastest for L2 reuse of the weights. Activations are int8
//  with one scale per 32; each 32-wide block is one mma whose int32 result
//  is scaled in registers. Scales are stored transposed ([block][row]) so a
//  stage loads them with 16-byte copies.
// ═══════════════════════════════════════════════════════════════════════════
constexpr int kMq3BM = 128, kMq3BN = 128, kMq3BK = 64, kMq3Stages = 3;
constexpr int kMq3Ld = kMq3BK + 16;              // bytes per smem row: 80, conflict-free ldmatrix

__device__ __forceinline__ void cp_async16_cg(void* smem, const void* gmem, bool valid) { cp_async16(smem, gmem, valid); }

constexpr size_t mmq3_smem_bytes() {
    // per stage: A and W tiles, and for each of the 2 blocks: Sa, Sx (BM) + Sw, Mw (BN);
    // then one W tile the raw K-quant variants unpack into
    return (size_t) kMq3Stages * ((kMq3BM + kMq3BN) * kMq3Ld + 2 * (2 * kMq3BM + 2 * kMq3BN) * sizeof(float)) +
           (size_t) kMq3BN * kMq3Ld;
}

__device__ __forceinline__ void mma_s8_16816(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(0), "r"(0), "r"(0), "r"(0));
#else
    d[0] = d[1] = d[2] = d[3] = 0;
#endif
}

__device__ __forceinline__ float i2f_small3(int v) {
    return __int_as_float(v + 0x4B400000) - 12582912.0f;   // exact for |v| < 2^22
}

// Y[t][r] = sum_kb sa[kb][t] sw[kb][r] (qa.qw)_kb - sum_kb mw[kb][r] sx[kb][t]
//   A [m_pad][k] int8; W [rows][k] int8; sa/sx [nb][m_pad]; sw/mw [nb][rows_pad]
//   (rows_pad = rows rounded up to 4 so 16-byte copies stay inside).
// Dp4a = true: the activation comes as the producer kernels already
// quantized it for the dp4a path - int8 [tok][k], scale [tok][nb] and the
// int32 sum of each block - so no separate conversion pass is needed; the
// block sum for the minimum term is scale * integer sum.
// Split = true: the weights carry one scale per 16 (Q6_K): the low scale in
// sw, the high one in mw (no minimum term), and each 32-wide block is two
// k16 products - the halves of the k32 fragments - each with its scale.
// Raw = 1 (Q4_K) / 2 (Q5_K): W is the quantized blocks as stored on the
// device. A stage copies, per row, the block header and the bytes of its
// 64 weights (Q5_K also the 32 high-bit bytes) into the W tile; the block
// then unpacks them in shared memory - one nibble mask per word - into a
// separate int8 tile, with the scales and minimums of the two sub-blocks.
// No per-prefill expansion pass, no int8 copy of the weights in memory.
template <bool Dp4a, bool Split, int Raw>
__global__ void __launch_bounds__(256) mmq3_kernel(
        const int8_t* __restrict__ A, const float* __restrict__ sa, const float* __restrict__ sx,
        const int8_t* __restrict__ W, const float* __restrict__ sw, const float* __restrict__ mw,
        float* __restrict__ y, size_t k, size_t rows, size_t rows_pad, size_t m_pad) {
    extern __shared__ __align__(16) unsigned char mq3_smem[];
    const size_t stage_bytes = (size_t) (kMq3BM + kMq3BN) * kMq3Ld + 2 * (2 * kMq3BM + 2 * kMq3BN) * sizeof(float);
    auto As = [&](int st) { return reinterpret_cast<int8_t*>(mq3_smem + st * stage_bytes); };
    auto Bs = [&](int st) { return reinterpret_cast<int8_t*>(mq3_smem + st * stage_bytes + kMq3BM * kMq3Ld); };
    auto Sc = [&](int st, int blk, int which) {      // which: 0 Sa, 1 Sx (BM), 2 Sw, 3 Mw (BN)
        float* base = reinterpret_cast<float*>(mq3_smem + st * stage_bytes + (kMq3BM + kMq3BN) * kMq3Ld);
        return base + blk * (2 * kMq3BM + 2 * kMq3BN) + (which < 2 ? which * kMq3BM : 2 * kMq3BM + (which - 2) * kMq3BN);
    };

    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int wm = warp / 2, wn = warp % 2;          // 4 x 2 warps: 32 tokens x 64 rows each
    const int g = lane >> 2, tq = lane & 3;
    const size_t t0 = (size_t) blockIdx.x * kMq3BM;
    const size_t r0 = (size_t) blockIdx.y * kMq3BN;
    const size_t nb = k / 32;
    const int nk = (int) (k / kMq3BK);
    const bool has_m = Raw != 0 || (!Split && mw != nullptr);   // Split: mw carries the high scale
    const bool load_mw = mw != nullptr;

    auto load_stage = [&](int st, int ks) {
        const size_t kk = (size_t) ks * kMq3BK;
        // A and W: 128 rows x 64 bytes each = 512 chunks of 16 bytes: 2 + 2 per thread.
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int c = tid + i * 256;
            const int row = c / 4, col = (c % 4) * 16;
            cp_async16(As(st) + row * kMq3Ld + col, A + (t0 + row) * k + kk + col, true);
            if (!Raw) {
                const size_t wr = r0 + row;
                cp_async16(Bs(st) + row * kMq3Ld + col, W + (wr < rows ? wr : 0) * k + kk + col, wr < rows);
            }
        }
        if (Raw) {
            // Per row: header (d, dmin, scales) + [Q5_K: qh 32] + the 32 bytes of this 64.
            constexpr int NC = Raw == 2 ? 5 : 3;
            constexpr size_t bsz = Raw == 2 ? 176 : 144;
            const size_t row_bytes = (k / 256) * bsz;
            const size_t boff = (size_t) (ks / 4) * bsz;
            const int c = ks % 4;
            #pragma unroll
            for (int i = 0; i < (kMq3BN * NC + 255) / 256; ++i) {
                const int e = tid + i * 256;
                if (e < kMq3BN * NC) {
                    const int row = e / NC, part = e % NC;
                    size_t off;
                    if (part == 0)     off = 0;
                    else if (Raw == 1) off = 16 + c * 32 + (part - 1) * 16;
                    else               off = part < 3 ? 16 + (part - 1) * 16 : 48 + c * 32 + (part - 3) * 16;
                    const size_t wr = r0 + row;
                    cp_async16(Bs(st) + row * kMq3Ld + part * 16, W + (wr < rows ? wr : 0) * row_bytes + boff + off,
                               wr < rows);
                }
            }
        }
        if (!Dp4a) {
            // Scales of the 2 blocks: 4 arrays x 2 blocks x 128 floats = 256 chunks of 16 bytes: 1 per thread.
            const int blk = tid / 128, which = (tid / 32) % 4, part = tid % 32;
            const size_t kb = (size_t) ks * 2 + blk;
            float* dst = Sc(st, blk, which) + part * 4;
            if (which == 0)      cp_async16(dst, sa + kb * m_pad + t0 + part * 4, true);
            else if (which == 1) cp_async16(dst, sx + kb * m_pad + t0 + part * 4, has_m);
            else if (!Raw) {
                const size_t rr = r0 + part * 4;
                const float* src = (which == 2 ? sw : (load_mw ? mw : sw)) + kb * rows_pad + (rr < rows_pad ? rr : 0);
                cp_async16(dst, src, rr < rows_pad && (which == 2 || load_mw));
            }
        } else {
            // Activation scale and integer sum, one element each ([tok][nb] layout).
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int e = tid + i * 256;                 // 2 blocks x 2 arrays x 128 tokens
                const int blk = e / 256, which = (e / 128) % 2, t = e % 128;
                const size_t kb = (size_t) ks * 2 + blk;
                const size_t off = (t0 + t) * nb + kb;
                cp_async4(Sc(st, blk, which) + t, which == 0 ? (const void*) (sa + off) : (const void*) (sx + off));
            }
            // Weight scale and minimum: 2 blocks x 2 arrays x 32 chunks of 16 bytes.
            if (!Raw && tid < 128) {
                const int blk = tid / 64, which = 2 + (tid / 32) % 2, part = tid % 32;
                const size_t kb = (size_t) ks * 2 + blk;
                const size_t rr = r0 + part * 4;
                const float* src = (which == 2 ? sw : (load_mw ? mw : sw)) + kb * rows_pad + (rr < rows_pad ? rr : 0);
                cp_async16(Sc(st, blk, which) + part * 4, src, rr < rows_pad && (which == 2 || load_mw));
            }
        }
    };

    float acc[2][8][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

    #pragma unroll
    for (int s = 0; s < kMq3Stages - 1; ++s) {
        if (s < nk) load_stage(s, s);
        cp_async_commit();
    }

    for (int ks = 0; ks < nk; ++ks) {
        cp_async_wait<kMq3Stages - 2>();
        __syncthreads();
        const int nxt = ks + kMq3Stages - 1;
        if (nxt < nk) load_stage(nxt % kMq3Stages, nxt);
        cp_async_commit();

        const int st = ks % kMq3Stages;
        const int8_t* as = As(st);
        int8_t* const bu = reinterpret_cast<int8_t*>(mq3_smem + kMq3Stages * stage_bytes);
        if (Raw) {
            // Unpack: each 32-bit word of the 64's bytes gives 4 weights of
            // both sub-blocks (low and high nibbles).
            const int c = ks % 4;
            const unsigned char* raw = reinterpret_cast<const unsigned char*>(Bs(st));
            #pragma unroll
            for (int i = 0; i < kMq3BN * 8 / 256; ++i) {
                const int p = tid + i * 256;
                const int row = p / 8, jw = p % 8;
                const unsigned char* rr = raw + row * kMq3Ld;
                const uint32_t q = *reinterpret_cast<const uint32_t*>(rr + (Raw == 2 ? 48 : 16) + 4 * jw);
                uint32_t lo = q & 0x0F0F0F0Fu, hi = (q >> 4) & 0x0F0F0F0Fu;
                if (Raw == 2) {
                    const uint32_t h = *reinterpret_cast<const uint32_t*>(rr + 16 + 4 * jw);
                    lo |= ((h >> (2 * c)) & 0x01010101u) << 4;
                    hi |= ((h >> (2 * c + 1)) & 0x01010101u) << 4;
                }
                *reinterpret_cast<uint32_t*>(bu + row * kMq3Ld + 4 * jw) = lo;
                *reinterpret_cast<uint32_t*>(bu + row * kMq3Ld + 32 + 4 * jw) = hi;
            }
            {
                const int row = tid / 2, blk = tid % 2;
                const unsigned char* rr = raw + row * kMq3Ld;
                int sc, mn;
                kq_scale_min(2 * c + blk, rr + 4, sc, mn);
                Sc(st, blk, 2)[row] = __half2float(*reinterpret_cast<const __half*>(rr)) * (float) sc;
                Sc(st, blk, 3)[row] = __half2float(*reinterpret_cast<const __half*>(rr + 2)) * (float) mn;
            }
            __syncthreads();
        }
        const int8_t* bs = Raw ? bu : Bs(st);
        #pragma unroll
        for (int blk = 0; blk < 2; ++blk) {
            const int ko = blk * 32;
            const float* Sa = Sc(st, blk, 0);
            const float* Sx = Sc(st, blk, 1);
            const float* Sw = Sc(st, blk, 2);
            const float* Mw = Sc(st, blk, 3);
            uint32_t af[2][4];
            float sat[2][2], sxt[2][2], saq[2][2], sac[2][2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int rb = wm * 32 + i * 16;
                ldmatrix_x4(af[i], as + (rb + (lane % 16)) * kMq3Ld + ko + (lane / 16) * 16);
                sat[i][0] = Sa[rb + g]; sat[i][1] = Sa[rb + g + 8];
                if (Dp4a) {      // Sx holds the int32 block sum: scale * sum
                    sxt[i][0] = sat[i][0] * (float) __float_as_int(Sx[rb + g]);
                    sxt[i][1] = sat[i][1] * (float) __float_as_int(Sx[rb + g + 8]);
                } else {
                    sxt[i][0] = Sx[rb + g]; sxt[i][1] = Sx[rb + g + 8];
                }
                #pragma unroll
                for (int r = 0; r < 2; ++r) {
                    saq[i][r] = __uint_as_float(__float_as_uint(sat[i][r]) & ~3u);
                    sac[i][r] = -12582912.0f * saq[i][r];
                }
            }
            #pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                uint32_t bf[4];
                const int n = wn * 64 + jj * 16 + (lane % 8) + (lane / 16) * 8;
                ldmatrix_x4(bf, bs + n * kMq3Ld + ko + ((lane / 8) % 2) * 16);
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int j = jj * 2 + h;
                    const int nc = wn * 64 + j * 8 + 2 * tq;
                    const float sw0 = Sw[nc], sw1 = Sw[nc + 1];
                    if (Split) {
                        // two k16 products: low half (a0,a1 / b0) and high half (a2,a3 / b1)
                        const float hw0 = Mw[nc], hw1 = Mw[nc + 1];
                        #pragma unroll
                        for (int i = 0; i < 2; ++i) {
                            int dl[4], dh[4];
                            mma_s8_16816(dl, af[i][0], af[i][1], bf[h * 2]);
                            mma_s8_16816(dh, af[i][2], af[i][3], bf[h * 2 + 1]);
                            acc[i][j][0] += sat[i][0] * (i2f_small3(dl[0]) * sw0 + i2f_small3(dh[0]) * hw0);
                            acc[i][j][1] += sat[i][0] * (i2f_small3(dl[1]) * sw1 + i2f_small3(dh[1]) * hw1);
                            acc[i][j][2] += sat[i][1] * (i2f_small3(dl[2]) * sw0 + i2f_small3(dh[2]) * hw0);
                            acc[i][j][3] += sat[i][1] * (i2f_small3(dl[3]) * sw1 + i2f_small3(dh[3]) * hw1);
                        }
                        continue;
                    }
                    #pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        // The integer sum comes out already as the float M + v
                        // (accumulator seeded with M's bits), and
                        // fma(M + v, sa', -M sa') = v sa' before its single
                        // rounding: -M sa' is exact because sa' keeps 22
                        // significant bits (a 2^-22 relative change of the
                        // scale). Two FMAs per output instead of four ops.
                        int d[4];
                        mma_s8_16832(d, reinterpret_cast<const int (&)[4]>(af[i]), (int) bf[h * 2], (int) bf[h * 2 + 1],
                                     0x4B400000);
                        acc[i][j][0] = fmaf(fmaf(__int_as_float(d[0]), saq[i][0], sac[i][0]), sw0, acc[i][j][0]);
                        acc[i][j][1] = fmaf(fmaf(__int_as_float(d[1]), saq[i][0], sac[i][0]), sw1, acc[i][j][1]);
                        acc[i][j][2] = fmaf(fmaf(__int_as_float(d[2]), saq[i][1], sac[i][1]), sw0, acc[i][j][2]);
                        acc[i][j][3] = fmaf(fmaf(__int_as_float(d[3]), saq[i][1], sac[i][1]), sw1, acc[i][j][3]);
                    }
                    if (has_m) {
                        const float mw0 = Mw[nc], mw1 = Mw[nc + 1];
                        #pragma unroll
                        for (int i = 0; i < 2; ++i) {
                            acc[i][j][0] = fmaf(-mw0, sxt[i][0], acc[i][j][0]);
                            acc[i][j][1] = fmaf(-mw1, sxt[i][0], acc[i][j][1]);
                            acc[i][j][2] = fmaf(-mw0, sxt[i][1], acc[i][j][2]);
                            acc[i][j][3] = fmaf(-mw1, sxt[i][1], acc[i][j][3]);
                        }
                    }
                }
            }
        }
    }
    cp_async_wait<0>();

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const size_t ta = t0 + wm * 32 + i * 16 + g;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const size_t n = r0 + wn * 64 + j * 8 + 2 * tq;
            if (n + 1 < rows && rows % 2 == 0) {
                *reinterpret_cast<float2*>(y + ta * rows + n) = make_float2(acc[i][j][0], acc[i][j][1]);
                *reinterpret_cast<float2*>(y + (ta + 8) * rows + n) = make_float2(acc[i][j][2], acc[i][j][3]);
            } else {
                if (n < rows) { y[ta * rows + n] = acc[i][j][0]; y[(ta + 8) * rows + n] = acc[i][j][2]; }
                if (n + 1 < rows) { y[ta * rows + n + 1] = acc[i][j][1]; y[(ta + 8) * rows + n + 1] = acc[i][j][3]; }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  FP16 tensor-core GEMM straight on the quantized blocks (sm_80+)
//
//  The int8 product above pays a float rescale per 32-block of every output
//  on the CUDA cores (the activation scale changes every 32 values), which
//  caps it well below the tensor-core rate. The FP16 product has no such
//  step - the scales go into the weights - but used to need the whole layer
//  expanded to FP16 in memory before each prefill.
//
//  Here the weights stay as stored on the device (Q4_K / Q5_K blocks, Q8_0
//  int8 rows). A stage copies the bytes of 64 weights per row; each thread
//  then builds its own mma B fragments from them in registers: a byte
//  permute puts two 4/5/8-bit values under the exponent of 1024 in FP16
//  (0x6400 | q = 1024 + q, exact), one subtraction removes the offset and
//  one fma applies the block scale and minimum. Activations are the FP16
//  form with one scale per token, as for hgemm_kernel.
// ═══════════════════════════════════════════════════════════════════════════
constexpr int kHqBM = 128, kHqBK = 64;
// A rows are 64 halves (128 bytes, no padding); their eight 16-byte chunks
// are XOR-swizzled with the row (chunk ^ row % 8) so the 8 rows one
// ldmatrix reads land on distinct banks. Offset in halves:
__device__ __forceinline__ int hkq_a_off(int row, int chunk) { return row * kHqBK + ((chunk ^ (row & 7)) << 3); }
// Bytes per W row in a stage: Q4_K 32 (+pad), Q5_K 32 + 32 high bits,
// Q6_K 64 + 32 (+pad), Q8_0 64 (no pad: its four 16-byte chunks are
// swizzled with (row / 2) % 4 instead). Each layout spreads the 8 rows a
// warp reads over distinct banks.
__host__ __device__ constexpr int hkq_raw_ld(int raw) { return raw == 1 ? 48 : raw == 4 ? 112 : raw == 3 ? 64 : 80; }
// BN weight rows per block: 128 (2 x 4 warps of 64 tokens x 32 rows) or 256
// (1 x 8 warps of 128 tokens x 32 rows: each warp builds its weight
// fragments for all 128 tokens, none unpacked twice - for the wide
// matrices). Three stages when they fit in the 99 KB of shared memory a
// block can have, two otherwise.
// I8: the activation comes as int8 (64 bytes per row) with one float scale
// per token and 32-block, the stage's two blocks after the weight scales.
template <int Raw, int BN, bool I8 = false>
__host__ __device__ constexpr size_t hkq_stage_bytes() {
    // scales: two half2 per row (scale, -minimum of each block), or for Q8_0
    // (no minimum) one half2 holding the scales of the stage's two blocks
    return (size_t) kHqBM * kHqBK * (I8 ? 1 : sizeof(__half)) + (size_t) BN * hkq_raw_ld(Raw) +
           (Raw == 3 ? 1 : 2) * BN * sizeof(__half2) + (I8 ? 2 * kHqBM * sizeof(float2) : 0);
}
template <int Raw, int BN, bool I8 = false>
__host__ __device__ constexpr int hkq_stages() {
    return I8 && 4 * hkq_stage_bytes<Raw, BN, I8>() <= 101376 ? 4
         : 3 * hkq_stage_bytes<Raw, BN, I8>() <= 101376 ? 3 : 2;
}
template <int Raw, int BN, bool I8 = false>
constexpr size_t hkq_smem_bytes() {
    // the gated epilogue reuses the stage buffers for 128 x (BN / 2 + 4) floats
    return hkq_stages<Raw, BN, I8>() * hkq_stage_bytes<Raw, BN, I8>();
}

// Q6_K blocks are 210 bytes, so a block's data is only 2-byte aligned and a
// stage could not copy it in 16-byte pieces. Before a prefill the layer's
// Q6_K matrices are repacked (a few tens of MB, one pass): per row, for each
// 128 weights, their 64 low-bit bytes then their 32 high-bit bytes - 96
// bytes, 16-byte aligned. Scales stay in the original blocks (kq_scales).
__global__ void q6k_repack_kernel(const uint8_t* __restrict__ w, size_t rows, size_t cols, uint32_t* __restrict__ out) {
    const size_t nh = cols / 128;                     // 128-halves per row
    const size_t e = (size_t) blockIdx.x * blockDim.x + threadIdx.x;   // output word
    if (e >= rows * nh * 24) return;
    const size_t o = (e % 24) * 4;                    // byte in the 96
    const size_t hi = e / 24;                         // row * nh + half
    const size_t row = hi / nh, half = hi % nh;
    const uint8_t* b = w + (row * (cols / 256) + half / 2) * 210;
    const uint8_t* src = o < 64 ? b + (half % 2) * 64 + o : b + 128 + (half % 2) * 32 + (o - 64);
    const uint16_t* s16 = reinterpret_cast<const uint16_t*>(src);
    out[e] = (uint32_t) s16[0] | ((uint32_t) s16[1] << 16);
}

// Scale and minimum of every 32-block as the FP16 kernel reads them, half2
// transposed [block][row], rows past `rows` zero: (scale, -minimum) for
// Q4_K/Q5_K, (scale of the low 16, of the high 16) for Q6_K. Q8_0 has no
// minimum: one half2 per pair of blocks, (scale of block 2i, of 2i+1),
// [pair][row]. One thread per row and 256 weights: one block header read, eight
// outputs, neighbouring threads on neighbouring rows (coalesced writes).
__global__ void kq_scales_h2_kernel(int fmt, const uint8_t* __restrict__ w, const __half* __restrict__ q8_scale,
                                    size_t rows, size_t nb, size_t rows_pad, __half2* __restrict__ out) {
    const size_t ng = (nb + 7) / 8;
    const size_t e = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= rows_pad * ng) return;
    const size_t gi = e / rows_pad, r = e % rows_pad;
    if (fmt == 0) {
        #pragma unroll
        for (int j = 0; j < 8; j += 2) {
            const size_t kb = gi * 8 + j;
            if (kb >= nb) break;
            const float lo = r < rows ? __half2float(q8_scale[r * nb + kb]) : 0.0f;
            const float hi = r < rows && kb + 1 < nb ? __half2float(q8_scale[r * nb + kb + 1]) : 0.0f;
            out[(kb / 2) * rows_pad + r] = __floats2half2_rn(lo, hi);
        }
        return;
    }
    const bool q6 = fmt == DESIREEIA_CUDA_FMT_Q6_K;
    const size_t bsz = fmt == DESIREEIA_CUDA_FMT_Q5_K ? 176 : q6 ? 210 : 144;
    const uint8_t* b = fmt != 0 && r < rows ? w + (r * (nb / 8) + gi) * bsz : nullptr;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const size_t kb = gi * 8 + j;
        if (kb >= nb) break;
        float lo = 0.0f, hi = 0.0f;
        if (r < rows) {
            if (q6) {
                const float d = __half2float(*reinterpret_cast<const __half*>(b + 208));
                const int8_t* scs = reinterpret_cast<const int8_t*>(b + 192) + (j / 4) * 8 + (j % 4) * 2;
                lo = d * (float) scs[0];
                hi = d * (float) scs[1];
            } else {
                // the whole header in one 16-byte load (144/176-byte blocks are 16-aligned)
                union { uint4 v; uint8_t by[16]; __half h[8]; } hd;
                hd.v = *reinterpret_cast<const uint4*>(b);
                int s6, m6;
                kq_scale_min(j, hd.by + 4, s6, m6);
                lo = __half2float(hd.h[0]) * (float) s6;
                hi = -__half2float(hd.h[1]) * (float) m6;
            }
        }
        out[kb * rows_pad + r] = __floats2half2_rn(lo, hi);
    }
}

// Y[t][r] = row_scale[t] * sum_k A[t][k] W[r][k]
//   Raw: 1 Q4_K, 2 Q5_K (device blocks, k % 256 == 0), 3 Q8_0 (int8 rows [rows][k]),
//   4 Q6_K repacked by q6k_repack_kernel.
//   wsm: kq_scales_h2_kernel output. A [m_pad][k] FP16, m_pad multiple of 128.
//   Epilogue: + bias[r] and + resid[t][r] when given; with a residual only
//   the first m_valid token rows are written (y may then be a slice of a
//   longer buffer whose following rows belong to someone else).
// Gated: the block computes the same BN/2 rows of two matrices - W (up) in
// the first half of the tile, W2 (gate) in the second - and writes
// act(gate) * up: the gate half hands its results to the up half through
// shared memory. The two FFN inputs never reach memory, only their product.
// Several products on the same input in one launch (Q, K and V): the row
// tiles of the matrices follow each other along blockIdx.y and each block
// picks its matrix. n <= 1 means "only the kernel arguments".
struct HkqGroup {
    const uint8_t* w[3]; const __half2* s[3]; float* y[3]; const float* bias[3];
    uint32_t rows[3]; uint32_t rows_pad[3]; uint32_t tile0[3];
    int n;
};

template <int Raw, int BN, int WMW, bool Gated = false, bool I8 = false>
__global__ void __launch_bounds__(256) hkq_kernel(const __half* __restrict__ A, const uint8_t* __restrict__ W_a,
                                                  const __half2* __restrict__ wsm_a, float* __restrict__ y_a,
                                                  const float* __restrict__ row_scale,
                                                  size_t k, size_t rows_a, size_t rows_pad_a,
                                                  const float* __restrict__ bias_a, const float* __restrict__ resid,
                                                  uint32_t m_valid,
                                                  const uint8_t* __restrict__ W2 = nullptr,
                                                  const __half2* __restrict__ wsm2 = nullptr, int act_gelu = 0,
                                                  const HkqGroup grp = HkqGroup{},
                                                  const float* __restrict__ sa8 = nullptr, size_t m_pad8 = 0) {
    static_assert(!Gated || (BN == 256 && WMW == 1) || (BN == 128 && WMW == 2),
                  "gated: the up and gate halves must be whole warp columns");
    static_assert(!I8 || Raw == 3, "int8 activations: Q8_0 weights only");
    extern __shared__ __align__(16) unsigned char hq_smem[];
    constexpr int RLd = hkq_raw_ld(Raw);
    constexpr size_t SB = hkq_stage_bytes<Raw, BN, I8>();
    constexpr int S = hkq_stages<Raw, BN, I8>();
    constexpr int WNW = 8 / WMW, WN = BN / WNW, NT = WN / 8, MI = kHqBM / WMW / 16;   // warp grid, n/m tiles per warp
    constexpr size_t kABytes = (size_t) kHqBM * kHqBK * (I8 ? 1 : sizeof(__half));
    auto As = [&](int st) { return reinterpret_cast<__half*>(hq_smem + st * SB); };
    auto Ws = [&](int st) { return hq_smem + st * SB + kABytes; };
    auto Ss = [&](int st) { return reinterpret_cast<__half2*>(hq_smem + st * SB + kABytes + BN * RLd); };
    auto Sa = [&](int st) {                          // I8: [2 blocks][128 tokens] float2
        return reinterpret_cast<float*>(hq_smem + st * SB + kABytes + BN * RLd + BN * sizeof(__half2));
    };
    // I8 folds the activation scale into the products: the output needs none.
    auto rscale = [&](size_t t) { return I8 ? 1.0f : row_scale[t]; };

    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int wm = warp / WNW, wn = warp % WNW;      // WMW x WNW warps: MI*16 tokens x WN rows each
    const int g = lane >> 2, tq = lane & 3;
    // Token blocks fastest: the blocks sharing a slice of weights run back to back (L2).
    const size_t t0 = (size_t) blockIdx.x * kHqBM;
    const uint8_t* W = W_a;
    const __half2* wsm = wsm_a;
    float* y = y_a;
    const float* bias = bias_a;
    size_t rows = rows_a, rows_pad = rows_pad_a;
    uint32_t by = blockIdx.y;
    if (grp.n > 1) {
        int m = 0;
        for (int i = 1; i < grp.n; ++i) if (by >= grp.tile0[i]) m = i;
        W = grp.w[m]; wsm = grp.s[m]; y = grp.y[m]; bias = grp.bias[m];
        rows = grp.rows[m]; rows_pad = grp.rows_pad[m];
        by -= grp.tile0[m];
    }
    const size_t r0 = (size_t) by * (Gated ? BN / 2 : BN);
    const int nk = (int) (k / kHqBK);
    // Global row and matrix of tile row `row`.
    auto src_row = [&](int row, const uint8_t*& base) -> size_t {
        if (Gated && row >= BN / 2) { base = W2; return r0 + row - BN / 2; }
        base = W;
        return r0 + row;
    };

    auto load_stage = [&](int st, int ks) {
        const size_t kk = (size_t) ks * kHqBK;
        if (I8) {
            // A: 128 x 64 bytes = 512 chunks, 2 per thread, swizzled like the
            // Q8_0 weight rows; the scales of the two blocks: 64 chunks.
            const int8_t* A8 = reinterpret_cast<const int8_t*>(A);
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int c = tid + i * 256;
                const int row = c / 4, ch = c % 4;
                cp_async16(reinterpret_cast<unsigned char*>(As(st)) + row * 64 + ((ch ^ ((row >> 1) & 3)) << 4),
                           A8 + (t0 + row) * k + kk + ch * 16, true);
            }
            if (tid < 128) {                         // (sa, -M sa) of 2 blocks x 128 tokens
                const int blk = tid / 64, part = tid % 64;
                cp_async16(Sa(st) + blk * 2 * kHqBM + part * 4, sa8 + ((kk / 32 + blk) * m_pad8 + t0 + part * 2) * 2, true);
            }
        } else {
            // A: 128 x 64 halves = 1024 chunks of 16 bytes, 4 per thread.
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int c = tid + i * 256;
                const int row = c / 8, ch = c % 8;
                cp_async16(As(st) + hkq_a_off(row, ch), A + (t0 + row) * k + kk + ch * 8, true);
            }
        }
        if (Raw == 4) {
            // Repacked Q6_K (q6k_repack_kernel): the 96 bytes of this 128-half,
            // 6 chunks per row; two consecutive stages read the same half.
            const size_t row_bytes = (k / 128) * 96, off = (kk / 128) * 96;
            #pragma unroll
            for (int i = 0; i < 6 * BN / 256; ++i) {
                const int e = tid + i * 256;
                const int row = e / 6, part = e % 6;
                const uint8_t* wb;
                const size_t wr = src_row(row, wb);
                cp_async16(Ws(st) + row * RLd + part * 16, wb + (wr < rows ? wr : 0) * row_bytes + off + part * 16,
                           wr < rows);
            }
        }
        // W: Q4_K 2 chunks per row (the 32 bytes of these 64), Q5_K 2 + 2 (qs, qh), Q8_0 4.
        constexpr int NC = Raw == 1 ? 2 : 4;
        #pragma unroll
        for (int i = 0; i < (Raw == 4 ? 0 : NC * BN / 256); ++i) {
            const int e = tid + i * 256;
            const int row = e / NC, part = e % NC;
            const uint8_t* wb;
            const size_t wr = src_row(row, wb);
            const size_t rs = wr < rows ? wr : 0;
            const uint8_t* src;
            if (Raw == 3) {
                src = wb + rs * k + kk + part * 16;
            } else {
                constexpr size_t bsz = Raw == 2 ? 176 : 144;
                const uint8_t* b = wb + (rs * (k / 256) + kk / 256) * bsz;
                const int c = (int) ((kk / 64) % 4);
                if (Raw == 1) src = b + 16 + c * 32 + part * 16;
                else          src = part < 2 ? b + 48 + c * 32 + part * 16 : b + 16 + (part - 2) * 16;
            }
            const int pc = Raw == 3 ? part ^ ((row >> 1) & 3) : part;
            cp_async16(Ws(st) + row * RLd + pc * 16, src, wr < rows);
        }
        if (Raw == 3) {
            // Q8_0 scales, one half2 per row for the stage's two blocks: BN / 4 chunks.
            if (tid < BN / 4) {
                const bool second = Gated && tid * 4 >= BN / 2;
                const size_t rr = r0 + tid * 4 - (second ? BN / 2 : 0);
                const __half2* sb = second ? wsm2 : wsm;
                cp_async16(Ss(st) + tid * 4, sb + (kk / 64) * rows_pad + (rr < rows_pad ? rr : 0), rr < rows_pad);
            }
        } else if (tid < BN / 2) {
            // Scales of the 2 blocks: 2 x BN half2 = BN / 2 chunks.
            const int blk = tid / (BN / 4), part = tid % (BN / 4);
            const bool second = Gated && part * 4 >= BN / 2;
            const size_t rr = r0 + part * 4 - (second ? BN / 2 : 0);
            const __half2* sb = second ? wsm2 : wsm;
            cp_async16(Ss(st) + blk * BN + part * 4, sb + (kk / 32 + blk) * rows_pad + (rr < rows_pad ? rr : 0),
                       rr < rows_pad);
        }
    };

    float acc[MI][NT][4];
    #pragma unroll
    for (int i = 0; i < MI; ++i)
        #pragma unroll
        for (int j = 0; j < NT; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

    #pragma unroll
    for (int s = 0; s < S - 1; ++s) {
        if (s < nk) load_stage(s, s);
        cp_async_commit();
    }

    // Q8_0: signed byte ^ 0x80 = q + 128; Q6_K: q - 32
    const __half2 off = __float2half2_rn(Raw == 3 ? 1152.0f : Raw == 4 ? 1056.0f : 1024.0f);
    // I8 offsets in a stage. Swizzle (row / 2) % 4 depends only on the row
    // within its 8/16-row group: g for the weights, lane % 16 for A.
    const int i8_sa = ((lane % 16) >> 1) & 3, i8_sb = (g >> 1) & 3;
    const int i8_ar = (wm * MI * 16 + lane % 16) * 64, i8_br = (wn * WN + g) * RLd + 4 * tq;
    // chunk c of a row sits at base + ((c ^ swizzle) << 4) with bits 4-5 of
    // base clear, so chunk c is chunk 0's offset ^ (c << 4)
    const int i8_a0 = i8_ar + (((lane / 16) ^ i8_sa) << 4), i8_b0 = i8_br + (i8_sb << 4);
    const int i8_c = wn * WN + 2 * tq, i8_t = wm * MI * 16 + g;
    for (int ks = 0; ks < nk; ++ks) {
        cp_async_wait<S - 2>();
        __syncthreads();
        const int nxt = ks + S - 1;
        if (nxt < nk) load_stage(nxt % S, nxt);
        cp_async_commit();

        const int st = ks % S;
        const __half* as = As(st);
        const unsigned char* ws = Ws(st);
        const __half2* ss = Ss(st);
        const int sub0 = (ks % 4) * 2;               // Q5_K: first sub-block of these 64 in the 256
        const int qp = ks % 2;                        // Q6_K: second pair of quads of the 128-half
        if (I8) {
            // One k32 integer product per block and tile. The accumulator is
            // seeded with the bits of M = 1.5 * 2^23, so the integer sum v
            // comes out as the float M + v, and fma(M + v, sa, -M sa) is
            // v sa before its single rounding (the producer keeps sa to 22
            // significant bits, so -M sa is exact). Then one fma with the
            // weight scale: two per output. Offsets are per thread
            // constants (i8_*), only the stage base moves.
            const unsigned char* a8 = reinterpret_cast<const unsigned char*>(as);
            const float2* qc = reinterpret_cast<const float2*>(Sa(st));
            #pragma unroll 1
            for (int blk = 0; blk < 2; ++blk) {   // not unrolled: the next block's fragments would spill
                const int ao = i8_a0 ^ (blk << 5), bo0 = i8_b0 ^ (blk << 5), bo1 = bo0 ^ 16;
                uint32_t bq[NT][2];
                float sw[NT][2];
                #pragma unroll
                for (int j = 0; j < NT; ++j) {
                    bq[j][0] = *reinterpret_cast<const uint32_t*>(ws + bo0 + j * 8 * RLd);
                    bq[j][1] = *reinterpret_cast<const uint32_t*>(ws + bo1 + j * 8 * RLd);
                    const uint2 s01 = *reinterpret_cast<const uint2*>(ss + i8_c + j * 8);   // two half2
                    const __half2 s0 = *reinterpret_cast<const __half2*>(&s01.x);
                    const __half2 s1 = *reinterpret_cast<const __half2*>(&s01.y);
                    sw[j][0] = __half2float(blk ? __high2half(s0) : __low2half(s0));
                    sw[j][1] = __half2float(blk ? __high2half(s1) : __low2half(s1));
                }
                const float2* qb = qc + blk * kHqBM + i8_t;
                // Software-pipelined: the products of tile row i are issued
                // before the scales go on those of row i - 1, so a warp has
                // independent work while its integer products are in flight.
                int dp[NT][4];
                float2 pa = make_float2(0.0f, 0.0f), pb = pa;
                #pragma unroll
                for (int i = 0; i <= MI; ++i) {
                    int dc[NT][4];
                    uint32_t af[4];
                    float2 qa = make_float2(0.0f, 0.0f), qbb = qa;
                    if (i < MI) {
                        ldmatrix_x4(af, a8 + ao + i * 16 * 64);
                        // pinned loads: hoisted all at once they would spill
                        qa = lds_f2_pinned(qb + i * 16);
                        qbb = lds_f2_pinned(qb + i * 16 + 8);
                    }
                    #pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        if (i < MI)
                            mma_s8_16832(dc[j], reinterpret_cast<const int (&)[4]>(af), (int) bq[j][0], (int) bq[j][1],
                                         0x4B400000);
                        if (i > 0) {
                            float* a_ = acc[i - 1][j];
                            a_[0] = fmaf(fmaf(__int_as_float(dp[j][0]), pa.x, pa.y), sw[j][0], a_[0]);
                            a_[1] = fmaf(fmaf(__int_as_float(dp[j][1]), pa.x, pa.y), sw[j][1], a_[1]);
                            a_[2] = fmaf(fmaf(__int_as_float(dp[j][2]), pb.x, pb.y), sw[j][0], a_[2]);
                            a_[3] = fmaf(fmaf(__int_as_float(dp[j][3]), pb.x, pb.y), sw[j][1], a_[3]);
                        }
                    }
                    if (i < MI) {
                        #pragma unroll
                        for (int j = 0; j < NT; ++j)
                            #pragma unroll
                            for (int e = 0; e < 4; ++e) dp[j][e] = dc[j][e];
                        pa = qa; pb = qbb;
                    }
                }
            }
            continue;
        }
        #pragma unroll
        for (int kq = 0; kq < 4; ++kq) {             // k16 steps; the first two are block 0
            uint32_t af[MI][4];
            #pragma unroll
            for (int i = 0; i < MI; ++i) {
                const int row = wm * MI * 16 + i * 16 + (lane % 16);
                ldmatrix_x4(af[i], as + hkq_a_off(row, kq * 2 + lane / 16));
            }
            const int blk = kq / 2;
            #pragma unroll
            for (int j = 0; j < NT; ++j) {
                const int n = wn * WN + j * 8 + g;
                const unsigned char* wr = ws + n * RLd;
                const __half2 sm = Raw == 3 ? ss[n] : ss[blk * BN + n];
                const __half2 s2 = (Raw == 4 && (kq % 2)) || (Raw == 3 && blk) ? __high2half2(sm) : __low2half2(sm);
                const __half2 m2 = __high2half2(sm);
                uint32_t b[2];
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    // the fragment's two weights: k = 2*tq, 2*tq+1 (+8 for h = 1) of this k16
                    uint32_t x;
                    if (Raw == 3) {
                        const uint32_t v = *reinterpret_cast<const uint16_t*>(wr + ((kq ^ ((n >> 1) & 3)) << 4) + h * 8 + 2 * tq);
                        x = __byte_perm(v ^ 0x8080u, 0x64u, 0x4140);
                    } else if (Raw == 4) {
                        // quad 2*qp + blk of the 128-half: low bits in ql[32*blk + l]
                        // (low nibble for qp 0, high for 1), 2 high bits in qh[l]
                        const int l = (kq % 2) * 16 + h * 8 + 2 * tq;
                        const uint32_t v = *reinterpret_cast<const uint16_t*>(wr + 32 * blk + l);
                        uint32_t p = __byte_perm(v, 0u, 0x4140);
                        p = (qp ? p >> 4 : p) & 0x000F000Fu;
                        const uint32_t hb = *reinterpret_cast<const uint16_t*>(wr + 64 + l);
                        p |= ((__byte_perm(hb, 0u, 0x4140) >> (4 * qp + 2 * blk)) & 0x00030003u) << 4;
                        x = p | 0x64006400u;
                    } else {
                        // byte l holds weight l of the first block (low nibble) and of the second (high)
                        const int l = (kq % 2) * 16 + h * 8 + 2 * tq;
                        const uint32_t v = *reinterpret_cast<const uint16_t*>(wr + l);
                        uint32_t p = __byte_perm(v, 0u, 0x4140);            // byte 0 -> bits 0-7, byte 1 -> 16-23
                        p = (kq < 2 ? p : p >> 4) & 0x000F000Fu;
                        if (Raw == 2) {
                            const uint32_t hb = *reinterpret_cast<const uint16_t*>(wr + 32 + l);
                            p |= ((__byte_perm(hb, 0u, 0x4140) >> (sub0 + blk)) & 0x00010001u) << 4;
                        }
                        x = p | 0x64006400u;
                    }
                    __half2 t = __hsub2(*reinterpret_cast<const __half2*>(&x), off);
                    t = Raw >= 3 ? __hmul2(t, s2) : __hfma2(t, s2, m2);
                    b[h] = *reinterpret_cast<const uint32_t*>(&t);
                }
                #pragma unroll
                for (int i = 0; i < MI; ++i) mma_f16_16816(acc[i][j], af[i], b[0], b[1]);
            }
        }
    }
    cp_async_wait<0>();

    if (Gated) {
        // gate warps (second half of the tile rows) -> shared memory -> up warps
        constexpr int GLd = BN / 2 + 4;                 // padded: the 8 token rows a warp writes hit distinct banks
        __syncthreads();                                // every warp is done with the stage buffers
        float* G = reinterpret_cast<float*>(hq_smem);   // [128 tokens][GLd]
        const int tw = wm * MI * 16;                    // this warp's first token
        const bool is_gate = wn * WN >= BN / 2;
        const int cl0 = wn * WN - (is_gate ? BN / 2 : 0);
        if (is_gate) {
            #pragma unroll
            for (int i = 0; i < MI; ++i) {
                const int tl = tw + i * 16 + g;
                const float sa = rscale(t0 + tl), sb = rscale(t0 + tl + 8);
                #pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const int cl = cl0 + j * 8 + 2 * tq;
                    G[tl * GLd + cl] = acc[i][j][0] * sa;
                    G[tl * GLd + cl + 1] = acc[i][j][1] * sa;
                    G[(tl + 8) * GLd + cl] = acc[i][j][2] * sb;
                    G[(tl + 8) * GLd + cl + 1] = acc[i][j][3] * sb;
                }
            }
        }
        __syncthreads();
        if (!is_gate) {
            #pragma unroll
            for (int i = 0; i < MI; ++i) {
                const int tl = tw + i * 16 + g;
                const size_t ta = t0 + tl;
                const float sa = rscale(ta), sb = rscale(ta + 8);
                #pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const int cl = cl0 + j * 8 + 2 * tq;
                    const size_t n = r0 + cl;
                    const float v0 = ffn_act(G[tl * GLd + cl], act_gelu) * (acc[i][j][0] * sa);
                    const float v1 = ffn_act(G[tl * GLd + cl + 1], act_gelu) * (acc[i][j][1] * sa);
                    const float v2 = ffn_act(G[(tl + 8) * GLd + cl], act_gelu) * (acc[i][j][2] * sb);
                    const float v3 = ffn_act(G[(tl + 8) * GLd + cl + 1], act_gelu) * (acc[i][j][3] * sb);
                    if (n + 1 < rows && rows % 2 == 0) {
                        *reinterpret_cast<float2*>(y + ta * rows + n) = make_float2(v0, v1);
                        *reinterpret_cast<float2*>(y + (ta + 8) * rows + n) = make_float2(v2, v3);
                    } else {
                        if (n < rows) { y[ta * rows + n] = v0; y[(ta + 8) * rows + n] = v2; }
                        if (n + 1 < rows) { y[ta * rows + n + 1] = v1; y[(ta + 8) * rows + n + 1] = v3; }
                    }
                }
            }
        }
        return;
    }
    #pragma unroll
    for (int i = 0; i < MI; ++i) {
        const size_t ta = t0 + wm * MI * 16 + i * 16 + g;
        const float sa = rscale(ta), sb = rscale(ta + 8);
        const bool wa = !resid || ta < m_valid, wb = !resid || ta + 8 < m_valid;
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            const size_t n = r0 + wn * WN + j * 8 + 2 * tq;
            float v0 = acc[i][j][0] * sa, v1 = acc[i][j][1] * sa, v2 = acc[i][j][2] * sb, v3 = acc[i][j][3] * sb;
            if (bias) {
                const float b0 = n < rows ? bias[n] : 0.0f, b1 = n + 1 < rows ? bias[n + 1] : 0.0f;
                v0 += b0; v1 += b1; v2 += b0; v3 += b1;
            }
            if (resid) {
                if (wa && n < rows) v0 += resid[ta * rows + n];
                if (wa && n + 1 < rows) v1 += resid[ta * rows + n + 1];
                if (wb && n < rows) v2 += resid[(ta + 8) * rows + n];
                if (wb && n + 1 < rows) v3 += resid[(ta + 8) * rows + n + 1];
            }
            if (n + 1 < rows && rows % 2 == 0) {
                if (wa) *reinterpret_cast<float2*>(y + ta * rows + n) = make_float2(v0, v1);
                if (wb) *reinterpret_cast<float2*>(y + (ta + 8) * rows + n) = make_float2(v2, v3);
            } else {
                if (n < rows) { if (wa) y[ta * rows + n] = v0; if (wb) y[(ta + 8) * rows + n] = v2; }
                if (n + 1 < rows) { if (wa) y[ta * rows + n + 1] = v1; if (wb) y[(ta + 8) * rows + n + 1] = v3; }
            }
        }
    }
}

// Q8_0 scales (fp16 [rows][nb]) -> float, transposed [nb][rows_pad].
__global__ void q8_scales_t_kernel(const __half* __restrict__ s, size_t rows, size_t nb, size_t rows_pad,
                                   float* __restrict__ out) {
    const size_t e = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= rows_pad * nb) return;
    const size_t kb = e / rows_pad, r = e % rows_pad;
    out[e] = r < rows ? __half2float(s[r * nb + kb]) : 0.0f;
}

} // namespace

// Formats the tensor-core path can expand. Others keep the dp4a kernels.
static bool tc_supported(int fmt, size_t cols, size_t rows) {
    static const bool enabled = std::getenv("DESIREEIA_CUDA_NO_TENSOR_CORES") == nullptr;
    if (!enabled || rows % 16 != 0 || cols % 32 != 0 || rows < 64) return false;
    switch (fmt) {
        case 0:
        case DESIREEIA_CUDA_FMT_Q4_0: return true;
        case DESIREEIA_CUDA_FMT_Q4_K:
        case DESIREEIA_CUDA_FMT_Q5_K:
        case DESIREEIA_CUDA_FMT_Q6_K: return cols % 256 == 0;
        default: return false;
    }
}

// Expands a matrix into FP16 slot `slot` (once per layer). False when the
// format/shape is not covered or memory is short: the slot stays unready
// and the caller uses the dp4a path for that matrix.
static bool tc_prepare(int slot, int fmt, const void* w, const void* scale,
                       size_t rows, size_t cols, cudaStream_t stream) {
    g_tc.ready[slot] = false;
    g_tc.is_i8[slot] = false;
    g_tc.w_hkq[slot] = 0;
    // FP16 tensor cores reading the stored blocks (hkq_kernel): no expansion
    // pass, no per-block rescale. DESIREEIA_CUDA_NO_HKQ=1 turns it off,
    // DESIREEIA_CUDA_NO_HKQ_Q8=1 only for Q8_0.
    static const bool hkq_on = std::getenv("DESIREEIA_CUDA_NO_HKQ") == nullptr && cuda_sm_major() >= 8;
    static const bool hkq_q8 = std::getenv("DESIREEIA_CUDA_NO_HKQ_Q8") == nullptr;
    static const bool hkq_q6 = std::getenv("DESIREEIA_CUDA_NO_HKQ_Q6") == nullptr;
    const int hraw = !hkq_on ? 0
        : (fmt == DESIREEIA_CUDA_FMT_Q4_K && cols % 256 == 0) ? 1
        : (fmt == DESIREEIA_CUDA_FMT_Q5_K && cols % 256 == 0) ? 2
        : (fmt == DESIREEIA_CUDA_FMT_Q6_K && cols % 256 == 0 && hkq_q6) ? 4
        : (fmt == 0 && hkq_q8 && cols % 64 == 0) ? 3 : 0;
    if (hraw) {
        const size_t nb = cols / 32;
        const size_t rows_pad = (rows + 3) / 4 * 4;
        static const bool cache_on = std::getenv("DESIREEIA_CUDA_NO_PREP_CACHE") == nullptr;
        const size_t sc_bytes = rows_pad * nb * sizeof(__half2);
        const size_t rp_bytes = hraw == 4 ? rows * (cols / 128) * 24 * 4 : 0;
        if (cache_on) {
            auto it = g_hkq_prep.find(w);
            if (it != g_hkq_prep.end() && it->second.bytes == sc_bytes + rp_bytes) {
                g_tc.wsp[slot] = it->second.scales;
                g_tc.w8p[slot] = it->second.repack ? static_cast<const int8_t*>(it->second.repack)
                                                   : static_cast<const int8_t*>(w);
                g_tc.rows_pad[slot] = rows_pad;
                g_tc.w_hkq[slot] = hraw;
                g_tc.ready[slot] = true;
                return true;
            }
        }
        if (tc_grow((void**) &g_tc.ws[slot], g_tc.ws_cap[slot], rows_pad * nb * sizeof(__half2))) {
            kq_scales_h2_kernel<<<(unsigned) ((rows_pad * ((nb + 7) / 8) + 255) / 256), 256, 0, stream>>>(
                fmt, static_cast<const uint8_t*>(w), static_cast<const __half*>(scale), rows, nb, rows_pad,
                reinterpret_cast<__half2*>(g_tc.ws[slot]));
            g_tc.w8p[slot] = static_cast<const int8_t*>(w);
            if (hraw == 4) {
                const size_t words = rows * (cols / 128) * 24;
                if (!tc_grow((void**) &g_tc.w8[slot], g_tc.w8_cap[slot], words * 4)) return false;
                q6k_repack_kernel<<<(unsigned) ((words + 255) / 256), 256, 0, stream>>>(
                    static_cast<const uint8_t*>(w), rows, cols, reinterpret_cast<uint32_t*>(g_tc.w8[slot]));
                g_tc.w8p[slot] = g_tc.w8[slot];
            }
            g_tc.wsp[slot] = g_tc.ws[slot];
            g_tc.rows_pad[slot] = rows_pad;
            g_tc.w_hkq[slot] = hraw;
            g_tc.ready[slot] = true;
            // Keep a copy for the next prompts while at least 1 GB stays free.
            size_t free_b = 0, total_b = 0;
            if (cache_on && cudaMemGetInfo(&free_b, &total_b) == cudaSuccess &&
                free_b > sc_bytes + rp_bytes + ((size_t) 1 << 30)) {
                HkqPrep e;
                e.bytes = sc_bytes + rp_bytes;
                bool ok = cudaMalloc(&e.scales, sc_bytes) == cudaSuccess;
                if (ok && rp_bytes) ok = cudaMalloc(&e.repack, rp_bytes) == cudaSuccess;
                if (ok) {
                    cudaMemcpyAsync(e.scales, g_tc.ws[slot], sc_bytes, cudaMemcpyDeviceToDevice, stream);
                    if (rp_bytes) cudaMemcpyAsync(e.repack, g_tc.w8[slot], rp_bytes, cudaMemcpyDeviceToDevice, stream);
                    hkq_prep_erase(w);
                    g_hkq_prep[w] = e;
                } else {
                    if (e.scales) cudaFree(e.scales);
                    if (e.repack) cudaFree(e.repack);
                }
            }
            return true;
        }
    }
    if (i8_mma_usable(fmt, cols, rows)) {
        const size_t nb = cols / 32;
        const size_t rows_pad = (rows + 3) / 4 * 4;
        const bool split = fmt == DESIREEIA_CUDA_FMT_Q6_K;
        const bool has_m = fmt == DESIREEIA_CUDA_FMT_Q4_K || fmt == DESIREEIA_CUDA_FMT_Q5_K || split;  // wm: minimum, or Q6_K high scale
        const bool direct = fmt == 0;          // Q8_0: the device weights are int8 rows already
        static const bool kq_direct = std::getenv("DESIREEIA_CUDA_NO_KQ_DIRECT") == nullptr;
        const int raw = !kq_direct ? 0 : fmt == DESIREEIA_CUDA_FMT_Q4_K ? 1 : fmt == DESIREEIA_CUDA_FMT_Q5_K ? 2 : 0;
        if (raw) {                             // unpacked inside the product, per tile
            g_tc.w8p[slot] = static_cast<const int8_t*>(w);
            g_tc.rows_pad[slot] = rows_pad;
            g_tc.w_has_m[slot] = true;
            g_tc.w_split[slot] = false;
            g_tc.w_raw[slot] = raw;
            g_tc.is_i8[slot] = true;
            g_tc.ready[slot] = true;
            return true;
        }
        if (tc_grow((void**) &g_tc.ws[slot], g_tc.ws_cap[slot], rows_pad * nb * sizeof(float)) &&
            (direct || tc_grow((void**) &g_tc.w8[slot], g_tc.w8_cap[slot], rows * cols)) &&
            (!has_m || tc_grow((void**) &g_tc.wm[slot], g_tc.wm_cap[slot], rows_pad * nb * sizeof(float)))) {
            if (direct) {
                q8_scales_t_kernel<<<(unsigned) ((rows_pad * nb + 255) / 256), 256, 0, stream>>>(
                    static_cast<const __half*>(scale), rows, nb, rows_pad, g_tc.ws[slot]);
                g_tc.w8p[slot] = static_cast<const int8_t*>(w);
            } else {
                if (rows_pad != rows) {
                    cudaMemsetAsync(g_tc.ws[slot], 0, rows_pad * nb * sizeof(float), stream);
                    if (has_m) cudaMemsetAsync(g_tc.wm[slot], 0, rows_pad * nb * sizeof(float), stream);
                }
                const size_t nwords = rows * cols / 4;
                dequant_i8_kernel<<<(unsigned) ((nwords + 255) / 256), 256, 0, stream>>>(
                    fmt, static_cast<const uint8_t*>(w), static_cast<const __half*>(scale), rows, cols,
                    g_tc.w8[slot], g_tc.ws[slot], has_m ? g_tc.wm[slot] : nullptr, rows_pad);
                g_tc.w8p[slot] = g_tc.w8[slot];
            }
            g_tc.rows_pad[slot] = rows_pad;
            g_tc.w_has_m[slot] = has_m;
            g_tc.w_split[slot] = split;
            g_tc.w_raw[slot] = 0;
            g_tc.is_i8[slot] = true;
            g_tc.ready[slot] = true;
            return true;
        }
    }
    if (!tc_supported(fmt, cols, rows)) return false;
    if (!tc_grow((void**) &g_tc.w[slot], g_tc.w_cap[slot], rows * cols * sizeof(__half))) return false;
    const size_t nw = rows * cols;
    dequant_f16_kernel<<<(unsigned) ((nw + 255) / 256), 256, 0, stream>>>(
        fmt, static_cast<const uint8_t*>(w), static_cast<const __half*>(scale), rows, cols, g_tc.w[slot]);
    g_tc.ready[slot] = true;
    return true;
}

// Y[m][rows] = X[m][cols] * W^T with the FP16 matrix in `slot`. d_y must
// have room for m rounded up to kGemmBM token rows (the padding rows get
// written and are ignored). reuse_input: the previous call converted this
// same activation to FP16 (Q/K/V, up/gate share their input), skip it.
// xq/xs/xsum (optional): the same input already quantized by its producer
// (int8 [m][cols] padded to whole 128-token tiles, scale and int32 sum per
// 32); the int8 product then reads it directly.
// bias [rows] and resid [m][rows] (optional) are added to the result; with
// resid only the m real token rows of d_y are written.
__global__ void gemm_epi_kernel(float* __restrict__ y, const float* __restrict__ bias,
                                const float* __restrict__ resid, size_t rows, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = y[i];
    if (bias) v += bias[i % rows];
    if (resid) v += resid[i];
    y[i] = v;
}

static bool tc_gemm_run(int slot, const float* d_x, size_t cols, size_t rows, uint32_t m,
                        float* d_y, bool reuse_input, cudaStream_t stream,
                        const int8_t* xq, const float* xs, const int32_t* xsum,
                        const float* bias, const float* resid, bool& epi_done);

static bool tc_gemm(int slot, const float* d_x, size_t cols, size_t rows, uint32_t m,
                    float* d_y, bool reuse_input, cudaStream_t stream,
                    const int8_t* xq = nullptr, const float* xs = nullptr, const int32_t* xsum = nullptr,
                    const float* bias = nullptr, const float* resid = nullptr) {
    bool epi_done = false;
    // A residual output may be a slice of a longer buffer: the paths without
    // a fused epilogue write whole 128-token tiles, so they go through a
    // scratch result first.
    float* out = d_y;
    if (resid && !(g_tc.ready[slot] && g_tc.w_hkq[slot])) {
        const uint32_t m_pad = (m + kGemmBM - 1) / kGemmBM * kGemmBM;
        if (!tc_grow((void**) &g_tc.ytmp, g_tc.ytmp_cap, (size_t) m_pad * rows * sizeof(float))) return false;
        out = g_tc.ytmp;
    }
    if (!tc_gemm_run(slot, d_x, cols, rows, m, out, reuse_input, stream, xq, xs, xsum, bias, resid, epi_done)) {
        return false;
    }
    if (!epi_done && (bias || resid)) {
        const size_t n = (size_t) m * rows;
        gemm_epi_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>(out, bias, resid, rows, n);
    }
    if (out != d_y) cudaMemcpyAsync(d_y, out, (size_t) m * rows * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    return true;
}

// y [m][rows] = act(X W_gate^T) * (X W_up^T) in one product (hkq_kernel
// Gated), for two slots holding the same stored format. The FP16 input
// must already be in g_tc (the FFN norm writes it). False when the pair
// does not qualify - the caller runs the two products separately.
static bool tc_gemm_gated(int slot_up, int slot_gate, size_t cols, size_t rows, uint32_t m, float* d_y,
                          int act_gelu, cudaStream_t stream) {
    static const bool on = std::getenv("DESIREEIA_CUDA_NO_GATED") == nullptr;
    const int raw = g_tc.w_hkq[slot_up];
    // Q4_K and Q8_0: their 128 + 128-row tile keeps three pipeline stages.
    // Q5_K and Q6_K have two at 256 rows, and there both the 256 and the
    // 64 + 64 tile measured 3-6% slower than the two separate products.
    if (!on || !g_tc.f16_in || !g_tc.ready[slot_up] || !g_tc.ready[slot_gate] || (raw != 1 && raw != 3) ||
        g_tc.w_hkq[slot_gate] != raw || g_tc.rows_pad[slot_up] != g_tc.rows_pad[slot_gate]) {
        return false;
    }
    auto grant = [](auto kern, size_t bytes) {
        return cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) bytes) == cudaSuccess;
    };
    static const bool ok = grant(hkq_kernel<1, 256, 1, true>, hkq_smem_bytes<1, 256>()) &&
                           grant(hkq_kernel<3, 256, 1, true>, hkq_smem_bytes<3, 256>());
    // the epilogue exchange needs 128 x 132 floats of the stage memory
    static_assert(hkq_smem_bytes<1, 256>() >= 128 * 132 * sizeof(float) &&
                  hkq_smem_bytes<3, 256>() >= 128 * 132 * sizeof(float), "gated exchange buffer");
    if (!ok) return false;
    const uint32_t m_pad = (m + kGemmBM - 1) / kGemmBM * kGemmBM;
    const auto* wu = reinterpret_cast<const uint8_t*>(g_tc.w8p[slot_up]);
    const auto* wg = reinterpret_cast<const uint8_t*>(g_tc.w8p[slot_gate]);
    const auto* su = reinterpret_cast<const __half2*>(g_tc.wsp[slot_up]);
    const auto* sg = reinterpret_cast<const __half2*>(g_tc.wsp[slot_gate]);
    const size_t rp = g_tc.rows_pad[slot_up];
    const dim3 grid(m_pad / kHqBM, (unsigned) ((rows + 127) / 128));
    if (raw == 1) {
        hkq_kernel<1, 256, 1, true><<<grid, 256, hkq_smem_bytes<1, 256>(), stream>>>(
            g_tc.a, wu, su, d_y, g_tc.scale, cols, rows, rp, nullptr, nullptr, m, wg, sg, act_gelu, HkqGroup{});
    } else {
        hkq_kernel<3, 256, 1, true><<<grid, 256, hkq_smem_bytes<3, 256>(), stream>>>(
            g_tc.a, wu, su, d_y, g_tc.scale, cols, rows, rp, nullptr, nullptr, m, wg, sg, act_gelu, HkqGroup{});
    }
    return true;
}

// Up to three products of the SAME input (its FP16 form already in g_tc)
// in one launch, when the slots hold the same stored format: the small
// K/V projections alone fill a fraction of a wave. False when they do not
// qualify - the caller runs them one by one.
static bool tc_gemm_group(int n, const int* slots, const size_t* rows, float* const* ys,
                          const float* const* biases, size_t cols, uint32_t m, cudaStream_t stream) {
    static const bool on = std::getenv("DESIREEIA_CUDA_NO_QKV_GROUP") == nullptr;
    if (!on || n < 2 || n > 3 || !g_tc.f16_in) return false;
    const int raw = g_tc.w_hkq[slots[0]];
    if (raw == 0) return false;
    for (int i = 0; i < n; ++i) {
        if (!g_tc.ready[slots[i]] || g_tc.w_hkq[slots[i]] != raw) return false;
    }
    const uint32_t m_pad = (m + kGemmBM - 1) / kGemmBM * kGemmBM;
    const size_t sms = (size_t) g_sm_count();
    size_t tiles_w = 0, tiles_n = 0;
    for (int i = 0; i < n; ++i) { tiles_w += (rows[i] + 255) / 256; tiles_n += (rows[i] + 127) / 128; }
    const size_t tb = m_pad / kHqBM;
    const bool wide = std::getenv("DESIREEIA_CUDA_NO_HKQ_WIDE") == nullptr &&
                      (double) ((tb * tiles_w + sms - 1) / sms) * 1.7 < (double) ((tb * tiles_n + sms - 1) / sms);
    const int bn = wide ? 256 : 128;
    HkqGroup gdesc{};
    gdesc.n = n;
    uint32_t tile = 0;
    for (int i = 0; i < n; ++i) {
        gdesc.w[i] = reinterpret_cast<const uint8_t*>(g_tc.w8p[slots[i]]);
        gdesc.s[i] = reinterpret_cast<const __half2*>(g_tc.wsp[slots[i]]);
        gdesc.y[i] = ys[i];
        gdesc.bias[i] = biases[i];
        gdesc.rows[i] = (uint32_t) rows[i];
        gdesc.rows_pad[i] = (uint32_t) g_tc.rows_pad[slots[i]];
        gdesc.tile0[i] = tile;
        tile += (uint32_t) ((rows[i] + bn - 1) / bn);
    }
    const dim3 grid(m_pad / kHqBM, tile);
    auto run = [&](auto kern, size_t smem) {
        kern<<<grid, 256, smem, stream>>>(g_tc.a, gdesc.w[0], gdesc.s[0], gdesc.y[0], g_tc.scale, cols, rows[0],
                                          gdesc.rows_pad[0], gdesc.bias[0], nullptr, m, nullptr, nullptr, 0, gdesc, nullptr, (size_t) 0);
    };
    auto grant = [](auto kern, size_t bytes) {
        return cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) bytes) == cudaSuccess;
    };
    static const bool granted =
        grant(hkq_kernel<1, 256, 1>, hkq_smem_bytes<1, 256>()) && grant(hkq_kernel<2, 256, 1>, hkq_smem_bytes<2, 256>()) &&
        grant(hkq_kernel<3, 256, 1>, hkq_smem_bytes<3, 256>()) && grant(hkq_kernel<4, 256, 1>, hkq_smem_bytes<4, 256>()) &&
        grant(hkq_kernel<1, 128, 2>, hkq_smem_bytes<1, 128>()) && grant(hkq_kernel<2, 128, 2>, hkq_smem_bytes<2, 128>()) &&
        grant(hkq_kernel<3, 128, 2>, hkq_smem_bytes<3, 128>()) && grant(hkq_kernel<4, 128, 2>, hkq_smem_bytes<4, 128>());
    if (!granted) return false;
    if (wide) {
        if (raw == 1)      run(hkq_kernel<1, 256, 1>, hkq_smem_bytes<1, 256>());
        else if (raw == 2) run(hkq_kernel<2, 256, 1>, hkq_smem_bytes<2, 256>());
        else if (raw == 3) run(hkq_kernel<3, 256, 1>, hkq_smem_bytes<3, 256>());
        else               run(hkq_kernel<4, 256, 1>, hkq_smem_bytes<4, 256>());
    } else {
        if (raw == 1)      run(hkq_kernel<1, 128, 2>, hkq_smem_bytes<1, 128>());
        else if (raw == 2) run(hkq_kernel<2, 128, 2>, hkq_smem_bytes<2, 128>());
        else if (raw == 3) run(hkq_kernel<3, 128, 2>, hkq_smem_bytes<3, 128>());
        else               run(hkq_kernel<4, 128, 2>, hkq_smem_bytes<4, 128>());
    }
    return cudaGetLastError() == cudaSuccess;
}

static bool tc_gemm_run(int slot, const float* d_x, size_t cols, size_t rows, uint32_t m,
                        float* d_y, bool reuse_input, cudaStream_t stream,
                        const int8_t* xq, const float* xs, const int32_t* xsum,
                        const float* bias, const float* resid, bool& epi_done) {
    if (!g_tc.ready[slot]) return false;
    const uint32_t m_pad = (m + kGemmBM - 1) / kGemmBM * kGemmBM;
    if (!reuse_input) { g_tc.f16_in = false; g_tc.i8_in = false; }
    if (g_tc.is_i8[slot]) {
        const size_t nb = cols / 32;
        if (!tc_grow((void**) &g_tc.a8, g_tc.a8_cap, (size_t) m_pad * cols) ||
            !tc_grow((void**) &g_tc.as, g_tc.as_cap, (size_t) m_pad * nb * sizeof(float)) ||
            !tc_grow((void**) &g_tc.ax, g_tc.ax_cap, (size_t) m_pad * nb * sizeof(float))) {
            return false;
        }
        if (!g_tc.i8_in && !xq) {
            const size_t warps = (size_t) m_pad * nb;
            act_to_i8_kernel<<<(unsigned) ((warps * 32 + 255) / 256), 256, 0, stream>>>(
                d_x, cols, m, m_pad, g_tc.a8, g_tc.as, g_tc.ax);
            g_tc.i8_in = true;
        }
        auto grant = [](auto kern) {
            return cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) mmq3_smem_bytes()) == cudaSuccess;
        };
        static const bool smem_ok =
            grant(mmq3_kernel<false, false, 0>) && grant(mmq3_kernel<true, false, 0>) &&
            grant(mmq3_kernel<false, true, 0>) && grant(mmq3_kernel<true, true, 0>) &&
            grant(mmq3_kernel<false, false, 1>) && grant(mmq3_kernel<true, false, 1>) &&
            grant(mmq3_kernel<false, false, 2>) && grant(mmq3_kernel<true, false, 2>);
        if (!smem_ok) return false;
        const dim3 grid(m_pad / kMq3BM, (unsigned) ((rows + kMq3BN - 1) / kMq3BN));
        const int8_t* A = xq ? xq : g_tc.a8;
        const float* sa = xq ? xs : g_tc.as;
        const float* sx = xq ? reinterpret_cast<const float*>(xsum) : g_tc.ax;
        const float* mw = g_tc.w_has_m[slot] && !g_tc.w_raw[slot] ? g_tc.wm[slot] : nullptr;
        auto launch = [&](auto kern) {
            kern<<<grid, 256, mmq3_smem_bytes(), stream>>>(A, sa, sx, g_tc.w8p[slot], g_tc.ws[slot], mw, d_y, cols, rows,
                                                          g_tc.rows_pad[slot], m_pad);
        };
        const int raw = g_tc.w_raw[slot];
        if (xq) {
            if (raw == 1)                launch(mmq3_kernel<true, false, 1>);
            else if (raw == 2)           launch(mmq3_kernel<true, false, 2>);
            else if (g_tc.w_split[slot]) launch(mmq3_kernel<true, true, 0>);
            else                         launch(mmq3_kernel<true, false, 0>);
        } else {
            if (raw == 1)                launch(mmq3_kernel<false, false, 1>);
            else if (raw == 2)           launch(mmq3_kernel<false, false, 2>);
            else if (g_tc.w_split[slot]) launch(mmq3_kernel<false, true, 0>);
            else                         launch(mmq3_kernel<false, false, 0>);
        }
        return true;
    }
    static const bool hkq_i8 = std::getenv("DESIREEIA_CUDA_HKQ_I8") != nullptr;
    // Not when the producer already wrote this product's FP16 input itself
    // (the float at d_x is then not the activation).
    if (hkq_i8 && g_tc.w_hkq[slot] == 3 && !(reuse_input && g_tc.f16_in && !g_tc.i8_in)) {
        const size_t nb = cols / 32;
        if (!tc_grow((void**) &g_tc.a8, g_tc.a8_cap, (size_t) m_pad * cols) ||
            !tc_grow((void**) &g_tc.as, g_tc.as_cap, (size_t) m_pad * nb * sizeof(float2))) {
            return false;
        }
        if (!g_tc.i8_in) {
            act_to_i8q_kernel<<<dim3(m_pad, (unsigned) ((cols / 8 + 255) / 256)), 256, 0, stream>>>(
                d_x, cols, m, m_pad, g_tc.a8, reinterpret_cast<float2*>(g_tc.as));
            g_tc.i8_in = true;
        }
        auto grant = [](auto kern, size_t bytes) {
            return cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) bytes) == cudaSuccess;
        };
        static const bool ok = grant(hkq_kernel<3, 128, 2, false, true>, hkq_smem_bytes<3, 128, true>()) &&
                               grant(hkq_kernel<3, 256, 1, false, true>, hkq_smem_bytes<3, 256, true>());
        if (!ok) return false;
        const size_t sms = (size_t) g_sm_count();
        const size_t waves_w = ((size_t) (m_pad / kHqBM) * ((rows + 255) / 256) + sms - 1) / sms;
        const size_t waves_n = ((size_t) (m_pad / kHqBM) * ((rows + 127) / 128) + sms - 1) / sms;
        static const double i8_ratio = std::getenv("DESIREEIA_CUDA_I8_RATIO") ? atof(std::getenv("DESIREEIA_CUDA_I8_RATIO")) : 1.7;
        const bool wide = (double) waves_w * i8_ratio < (double) waves_n;
        const auto* A = reinterpret_cast<const __half*>(g_tc.a8);
        const auto* wq = reinterpret_cast<const uint8_t*>(g_tc.w8p[slot]);
        const auto* wsm = reinterpret_cast<const __half2*>(g_tc.wsp[slot]);
        auto run = [&](auto kern, int bn, size_t smem) {
            kern<<<dim3(m_pad / kHqBM, (unsigned) ((rows + bn - 1) / bn)), 256, smem, stream>>>(
                A, wq, wsm, d_y, nullptr, cols, rows, g_tc.rows_pad[slot], bias, resid, m, nullptr, nullptr, 0,
                HkqGroup{}, g_tc.as, (size_t) m_pad);
        };
        epi_done = true;
        if (wide) run(hkq_kernel<3, 256, 1, false, true>, 256, hkq_smem_bytes<3, 256, true>());
        else      run(hkq_kernel<3, 128, 2, false, true>, 128, hkq_smem_bytes<3, 128, true>());
        return true;
    }
    if (!tc_grow((void**) &g_tc.a, g_tc.a_cap, (size_t) m_pad * cols * sizeof(__half)) ||
        !tc_grow((void**) &g_tc.scale, g_tc.scale_cap, (size_t) m_pad * sizeof(float))) {
        return false;
    }
    if (!g_tc.f16_in) {
        act_to_f16_kernel<<<m_pad, 256, 256 * sizeof(float), stream>>>(d_x, cols, m, g_tc.a, g_tc.scale);
        g_tc.f16_in = true;
    }
    if (const int hraw = g_tc.w_hkq[slot]) {
        auto grant = [](auto kern, size_t bytes) {
            return cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) bytes) == cudaSuccess;
        };
        static const bool hkq_ok = grant(hkq_kernel<1, 128, 2>, hkq_smem_bytes<1, 128>()) &&
                                   grant(hkq_kernel<2, 128, 2>, hkq_smem_bytes<2, 128>()) &&
                                   grant(hkq_kernel<3, 128, 2>, hkq_smem_bytes<3, 128>()) &&
                                   grant(hkq_kernel<4, 128, 2>, hkq_smem_bytes<4, 128>());
        static const bool hkq_wide_ok = hkq_ok && std::getenv("DESIREEIA_CUDA_NO_HKQ_WIDE") == nullptr &&
                                   grant(hkq_kernel<1, 256, 1>, hkq_smem_bytes<1, 256>()) &&
                                   grant(hkq_kernel<2, 256, 1>, hkq_smem_bytes<2, 256>()) &&
                                   grant(hkq_kernel<3, 256, 1>, hkq_smem_bytes<3, 256>()) &&
                                   grant(hkq_kernel<4, 256, 1>, hkq_smem_bytes<4, 256>());
        if (!hkq_ok) return false;
        const auto* wq = reinterpret_cast<const uint8_t*>(g_tc.w8p[slot]);
        const auto* wsm = reinterpret_cast<const __half2*>(g_tc.wsp[slot]);
        const size_t rp = g_tc.rows_pad[slot];
        // One block per SM either way (shared memory), so the time is the
        // number of whole waves times the block time; a wide block costs
        // about 1.7 narrow ones (measured on full waves). Pick the cheaper:
        // e.g. 32 wide blocks on 20 SMs are 2 waves, 64 narrow ones 4.
        const size_t sms = (size_t) g_sm_count();
        const size_t waves_w = ((size_t) (m_pad / kHqBM) * ((rows + 255) / 256) + sms - 1) / sms;
        const size_t waves_n = ((size_t) (m_pad / kHqBM) * ((rows + 127) / 128) + sms - 1) / sms;
        const bool wide = hkq_wide_ok && (double) waves_w * 1.7 < (double) waves_n;
        auto run = [&](auto kern, int bn, size_t smem) {
            kern<<<dim3(m_pad / kHqBM, (unsigned) ((rows + bn - 1) / bn)), 256, smem, stream>>>(
                g_tc.a, wq, wsm, d_y, g_tc.scale, cols, rows, rp, bias, resid, m, nullptr, nullptr, 0, HkqGroup{}, nullptr, (size_t) 0);
        };
        epi_done = true;
        if (wide) {
            if (hraw == 1)      run(hkq_kernel<1, 256, 1>, 256, hkq_smem_bytes<1, 256>());
            else if (hraw == 2) run(hkq_kernel<2, 256, 1>, 256, hkq_smem_bytes<2, 256>());
            else if (hraw == 3) run(hkq_kernel<3, 256, 1>, 256, hkq_smem_bytes<3, 256>());
            else                run(hkq_kernel<4, 256, 1>, 256, hkq_smem_bytes<4, 256>());
        } else {
            if (hraw == 1)      run(hkq_kernel<1, 128, 2>, 128, hkq_smem_bytes<1, 128>());
            else if (hraw == 2) run(hkq_kernel<2, 128, 2>, 128, hkq_smem_bytes<2, 128>());
            else if (hraw == 3) run(hkq_kernel<3, 128, 2>, 128, hkq_smem_bytes<3, 128>());
            else                run(hkq_kernel<4, 128, 2>, 128, hkq_smem_bytes<4, 128>());
        }
        return true;
    }
    // Pipelined kernel (cp.async + ldmatrix + mma.sync) on sm_80+; the WMMA
    // one stays for sm_75 and as a fallback (DESIREEIA_CUDA_NO_HGEMM=1).
    static const bool hgemm_on = std::getenv("DESIREEIA_CUDA_NO_HGEMM") == nullptr;
    static const bool hgemm_ok = hgemm_on && cuda_sm_major() >= 8 &&
        cudaFuncSetAttribute(hgemm_kernel<128>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int) hgemm_smem_bytes<128>()) == cudaSuccess;
    static const bool hgemm_wide_ok = hgemm_ok && std::getenv("DESIREEIA_CUDA_NO_HGEMM_WIDE") == nullptr &&
        cudaFuncSetAttribute(hgemm_kernel<256>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int) hgemm_smem_bytes<256>()) == cudaSuccess;
    if (hgemm_ok) {
        // Wide tiles only while they still fill the GPU (K/V projections are narrow).
        const size_t blocks_wide = (size_t) (m_pad / kHgBM) * ((rows + 255) / 256);
        if (hgemm_wide_ok && blocks_wide >= 2 * (size_t) g_sm_count()) {
            hgemm_kernel<256><<<dim3(m_pad / kHgBM, (unsigned) ((rows + 255) / 256)), 256, hgemm_smem_bytes<256>(), stream>>>(
                g_tc.a, g_tc.w[slot], d_y, g_tc.scale, cols, rows);
        } else {
            hgemm_kernel<128><<<dim3(m_pad / kHgBM, (unsigned) ((rows + 127) / 128)), 256, hgemm_smem_bytes<128>(), stream>>>(
                g_tc.a, g_tc.w[slot], d_y, g_tc.scale, cols, rows);
        }
        return true;
    }
    wmma_gemm_kernel<<<dim3((unsigned) ((rows + kGemmBN - 1) / kGemmBN), m_pad / kGemmBM), 256, 0, stream>>>(
        g_tc.a, g_tc.w[slot], d_y, g_tc.scale, cols, rows, m);
    return true;
}

namespace {
void tensor_core_release() {
    for (int i = 0; i < kTcSlots; ++i) {
        if (g_tc.w[i]) cudaFree(g_tc.w[i]);
        if (g_tc.w8[i]) cudaFree(g_tc.w8[i]);
        if (g_tc.ws[i]) cudaFree(g_tc.ws[i]);
        if (g_tc.wm[i]) cudaFree(g_tc.wm[i]);
    }
    if (g_tc.a8) cudaFree(g_tc.a8);
    if (g_tc.as) cudaFree(g_tc.as);
    if (g_tc.ax) cudaFree(g_tc.ax);
    if (g_tc.a) cudaFree(g_tc.a);
    if (g_tc.scale) cudaFree(g_tc.scale);
    g_tc = TensorCoreScratch{};
}
} // namespace

// ═══════════════════════════════════════════════════════════════════════════
//  Batched whole-layer prefill.
//
//  cuda_layer_forward runs ONE token through a whole dense layer on device.
//  Prefill used a different path: every matrix of every layer went host ->
//  device -> host, with norms, RoPE, KV writes and residuals on the CPU in
//  between. Measured on a 4B model: 52-80 tok/s of prefill against ~48 tok/s
//  of single-token decode - prefill, the part that can use the whole GPU,
//  barely faster than decode.
//
//  This is the same layer for n_tok tokens at once: the activations of the
//  whole chunk stay on device across every layer (uploaded once before the
//  first, downloaded once after the last), each matrix is read ONCE for all
//  the tokens where a batched kernel exists for its format, and the only
//  per-layer host work is the KV mirror copy. Same maths and same order as
//  cuda_layer_forward; the host path in dense_forward.cpp stays the
//  definition of record and the fallback for anything this does not cover.
// ═══════════════════════════════════════════════════════════════════════════

namespace {

// y = W * x for TOK activation columns per block, W in Q8_0 (int8 rows +
// one half scale per 32). Same tiling as matmul_q4_k_batch_kernel: a warp
// owns a row, each weight block is loaded once and reused for every column
// of the tile. Activations are [token][sub-block] Q8_0.
template <int nwarps, int TOK>
__global__ void matmul_q8_0_batch_kernel(const int8_t* __restrict__ w_qs,
                                          const __half* __restrict__ w_scale,
                                          const int8_t* __restrict__ x_qs,
                                          const float* __restrict__ x_scale,
                                          size_t nb, size_t rows, uint32_t n_tok,
                                          float* __restrict__ y) {
    const size_t row = (size_t) blockIdx.x * nwarps + threadIdx.y;
    if (row >= rows) return;
    const uint32_t t0 = blockIdx.y * TOK;
    const int* wr = reinterpret_cast<const int*>(w_qs + row * nb * 32);
    const __half* sr = w_scale + row * nb;

    float acc[TOK];
    #pragma unroll
    for (int t = 0; t < TOK; ++t) acc[t] = 0.0f;

    for (size_t b = threadIdx.x; b < nb; b += DESIREEIA_CUDA_WARP) {
        const int4* w4 = reinterpret_cast<const int4*>(wr + b * 8);
        const int4 a0 = w4[0];
        const int4 a1 = w4[1];
        const float ws = __half2float(sr[b]);
        #pragma unroll
        for (int t = 0; t < TOK; ++t) {
            const uint32_t tk = t0 + (uint32_t) t;
            if (tk >= n_tok) continue;
            const size_t idx = (size_t) tk * nb + b;
            const int4* x4 = reinterpret_cast<const int4*>(x_qs + idx * 32);
            const int4 x0 = x4[0];
            const int4 x1 = x4[1];
            int s = desireeia_dp4a(a0.x, x0.x, 0);
            s = desireeia_dp4a(a0.y, x0.y, s);
            s = desireeia_dp4a(a0.z, x0.z, s);
            s = desireeia_dp4a(a0.w, x0.w, s);
            s = desireeia_dp4a(a1.x, x1.x, s);
            s = desireeia_dp4a(a1.y, x1.y, s);
            s = desireeia_dp4a(a1.z, x1.z, s);
            s = desireeia_dp4a(a1.w, x1.w, s);
            acc[t] += ws * x_scale[idx] * (float) s;
        }
    }

    #pragma unroll
    for (int t = 0; t < TOK; ++t) {
        #pragma unroll
        for (int off = DESIREEIA_CUDA_WARP / 2; off > 0; off >>= 1) {
            acc[t] += __shfl_xor_sync(0xffffffff, acc[t], off, DESIREEIA_CUDA_WARP);
        }
    }
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int t = 0; t < TOK; ++t) {
            const uint32_t tk = t0 + (uint32_t) t;
            if (tk < n_tok) y[(size_t) tk * rows + row] = acc[t];
        }
    }
}

// y[t][i] += b[i] over a [token][rows] buffer.
__global__ void add_bias_rows_kernel(float* __restrict__ y, const float* __restrict__ b,
                                      size_t rows, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += b[i % rows];
}

// rope_neox_kernel for a batch: blockIdx.y is the token, with its own
// cos/sin row in caches[token][n_rot]. Q is [token][q_dim], K [token][kv_dim].
__global__ void rope_neox_batch_kernel(float* __restrict__ q, float* __restrict__ k,
                                        const float* __restrict__ caches,
                                        uint32_t n_rot, uint32_t head_dim,
                                        uint32_t n_head, size_t q_dim, size_t kv_dim) {
    const uint32_t b = blockIdx.x;
    const size_t tok = blockIdx.y;
    const float* cache = caches + tok * n_rot;
    float* vh = (b < n_head ? q + tok * q_dim + (size_t) b * head_dim
                             : k + tok * kv_dim + (size_t) (b - n_head) * head_dim);
    const uint32_t half = n_rot / 2;
    for (uint32_t j = threadIdx.x; j < half; j += blockDim.x) {
        const float c = cache[2 * j + 0];
        const float s = cache[2 * j + 1];
        const float x0 = vh[j];
        const float x1 = vh[j + half];
        vh[j]        = x0 * c - x1 * s;
        vh[j + half] = x0 * s + x1 * c;
    }
}

// kv_quantize_kernel for a batch: blockIdx.y is the token, written at
// position pos0 + token. Same Q8_0 rounding as the single-token kernel.
__global__ void kv_quantize_batch_kernel(const float* __restrict__ ksrc, uint8_t* __restrict__ kbase,
                                          const float* __restrict__ vsrc, uint8_t* __restrict__ vbase,
                                          uint32_t head_dim, size_t row_bytes, uint32_t blocks_per_side,
                                          size_t pos_bytes, size_t kv_dim, uint32_t pos0) {
    const size_t tok = blockIdx.y;
    uint8_t* kdst = kbase + (size_t) (pos0 + tok) * pos_bytes;
    uint8_t* vdst = vbase + (size_t) (pos0 + tok) * pos_bytes;
    const bool is_v = blockIdx.x >= blocks_per_side;
    const uint32_t bid = is_v ? blockIdx.x - blocks_per_side : blockIdx.x;
    const float* src = (is_v ? vsrc : ksrc) + tok * kv_dim;
    uint8_t* dst = is_v ? vdst : kdst;

    const uint32_t nblk = head_dim / 32;
    const uint32_t h = bid / nblk;
    const uint32_t b = bid % nblk;
    const uint32_t j = threadIdx.x;

    const float x = src[(size_t) h * head_dim + b * 32 + j];
    float m = fabsf(x);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off, 32));
    }
    const float d = m > 0.0f ? m / 127.0f : 0.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;

    uint8_t* blk = dst + (size_t) h * row_bytes + (size_t) b * 34;
    if (j == 0) *reinterpret_cast<__half*>(blk) = __float2half(d);
    int qi = (int) rintf(x * id);
    qi = min(127, max(-127, qi));
    reinterpret_cast<int8_t*>(blk + 2)[j] = (int8_t) qi;
}

// Tiled prefill attention over the Q8_0 KV cache.
//
// attention_batch_kernel_q8 gives every (token, head) its own block, which
// streams all of that token's past K/V rows from global memory: nothing is
// shared between the tokens of a prompt, so a long prompt reads the cache
// O(n^2) times over (Spark 4B, 4096-token prompt: 67% of the prefill).
//
// Here a block takes FA_TQ consecutive query tokens of one head. The K and
// V rows of each FA_TK-position tile are loaded ONCE into shared memory, in
// their compact Q8_0 form (int8 + one scale per 32), and serve every query
// of the block. Each warp owns two queries; its 32 lanes each score one key
// of the tile, so the softmax max/sum are warp reductions, then every lane
// accumulates 32-wide slices of the output. Online softmax as in the
// per-token kernel; the causal mask and the sliding window are applied per
// query. K rows are padded by 4 bytes so the lanes (one key row each) hit
// different shared-memory banks. head_dim <= 256.
constexpr int kFaTQ = 16;
constexpr int kFaTK = 32;

__global__ void __launch_bounds__(256) attention_prefill_q8_kernel(
        const float* __restrict__ q_all, const uint8_t* __restrict__ kcache,
        const uint8_t* __restrict__ vcache, float* __restrict__ out_all,
        uint32_t head_dim, uint32_t heads_per_kv, uint32_t q_dim,
        size_t row_bytes, size_t pos_bytes, size_t layer_off,
        uint32_t pos0, uint32_t n_tok, uint32_t n_swa, float inv_d) {
    const uint32_t h = blockIdx.x;
    const uint32_t hkv = h / heads_per_kv;
    const uint32_t tq0 = blockIdx.y * kFaTQ;
    if (tq0 >= n_tok) return;
    const uint32_t nq = min((uint32_t) kFaTQ, n_tok - tq0);
    const uint32_t nblk = head_dim / 32;
    const uint32_t kstride = head_dim + 4;   // padded K row, in bytes

    extern __shared__ __align__(16) unsigned char fa_smem[];
    float* sq = reinterpret_cast<float*>(fa_smem);                  // [TQ][hd], pre-scaled
    float* skd = sq + kFaTQ * head_dim;                              // [TK][nblk]
    float* svd = skd + kFaTK * nblk;                                 // [TK][nblk]
    int8_t* sk = reinterpret_cast<int8_t*>(svd + kFaTK * nblk);      // [TK][kstride]
    int8_t* sv = sk + kFaTK * kstride;                               // [TK][hd]

    const uint32_t tid = threadIdx.x;
    for (uint32_t i = tid; i < kFaTQ * head_dim; i += blockDim.x) {
        const uint32_t qi = i / head_dim, d = i % head_dim;
        sq[i] = qi < nq ? q_all[(size_t) (tq0 + qi) * q_dim + (size_t) h * head_dim + d] * inv_d : 0.0f;
    }

    const uint32_t warp = tid / 32, lane = tid % 32;
    const uint32_t qa = warp * 2;                 // this warp's two queries
    float acc[2][8];
    float mrun[2] = { -INFINITY, -INFINITY };
    float lrun[2] = { 0.0f, 0.0f };
    #pragma unroll
    for (int qq = 0; qq < 2; ++qq)
        #pragma unroll
        for (int i = 0; i < 8; ++i) acc[qq][i] = 0.0f;

    const uint32_t pos_first = pos0 + tq0;
    const uint32_t pos_last = pos0 + tq0 + nq - 1;
    const uint32_t kv_begin = (n_swa > 0 && pos_first + 1 > n_swa) ? pos_first + 1 - n_swa : 0;

    for (uint32_t k0 = kv_begin; k0 <= pos_last; k0 += kFaTK) {
        __syncthreads();
        // K and V tile: one (key, 32-block) pair per thread and side.
        for (uint32_t idx = tid; idx < kFaTK * nblk; idx += blockDim.x) {
            const uint32_t j = idx / nblk, b = idx % nblk;
            const uint32_t kp = k0 + j;
            int8_t* kdst = sk + j * kstride + b * 32;
            int8_t* vdst = sv + j * head_dim + b * 32;
            if (kp <= pos_last) {
                const size_t off = layer_off + (size_t) kp * pos_bytes + (size_t) hkv * row_bytes + (size_t) b * 34;
                const uint8_t* kb = kcache + off;
                const uint8_t* vb = vcache + off;
                skd[j * nblk + b] = __half2float(*reinterpret_cast<const __half*>(kb));
                svd[j * nblk + b] = __half2float(*reinterpret_cast<const __half*>(vb));
                // Blocks are 34 bytes, so the int8 payload is only 2-byte aligned.
                const uint16_t* ks16 = reinterpret_cast<const uint16_t*>(kb + 2);
                const uint16_t* vs16 = reinterpret_cast<const uint16_t*>(vb + 2);
                uint16_t* kd16 = reinterpret_cast<uint16_t*>(kdst);
                uint16_t* vd16 = reinterpret_cast<uint16_t*>(vdst);
                #pragma unroll
                for (int e = 0; e < 16; ++e) { kd16[e] = ks16[e]; vd16[e] = vs16[e]; }
            } else {
                skd[j * nblk + b] = 0.0f;
                svd[j * nblk + b] = 0.0f;
                #pragma unroll
                for (int e = 0; e < 32; ++e) { kdst[e] = 0; vdst[e] = 0; }
            }
        }
        __syncthreads();

        const uint32_t kp = k0 + lane;            // this lane's key
        float p[2];
        #pragma unroll
        for (int qq = 0; qq < 2; ++qq) {
            const uint32_t qi = qa + qq;
            float s = -INFINITY;
            if (qi < nq) {
                const uint32_t qpos = pos0 + tq0 + qi;
                const bool valid = kp <= qpos && (n_swa == 0 || kp + n_swa > qpos);
                if (valid) {
                    const float* qv = sq + qi * head_dim;
                    const int8_t* kr = sk + lane * kstride;
                    float dot = 0.0f;
                    for (uint32_t b = 0; b < nblk; ++b) {
                        float bd = 0.0f;
                        #pragma unroll
                        for (int e = 0; e < 32; ++e) bd += qv[b * 32 + e] * (float) kr[b * 32 + e];
                        dot += skd[lane * nblk + b] * bd;
                    }
                    s = dot;
                }
            }
            // Online softmax for query qi over this tile.
            float mt = s;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffff, mt, off));
            const float mnew = fmaxf(mrun[qq], mt);
            float e = (s == -INFINITY || mnew == -INFINITY) ? 0.0f : __expf(s - mnew);
            float es = e;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) es += __shfl_xor_sync(0xffffffff, es, off);
            const float corr = (mrun[qq] == -INFINITY) ? 0.0f : __expf(mrun[qq] - mnew);
            lrun[qq] = lrun[qq] * corr + es;
            mrun[qq] = mnew;
            #pragma unroll
            for (int i = 0; i < 8; ++i) acc[qq][i] *= corr;
            p[qq] = e;
        }

        // Output: lane owns dimensions lane, lane+32, ... (hd/32 of them).
        for (uint32_t j = 0; j < kFaTK; ++j) {
            const float p0 = __shfl_sync(0xffffffff, p[0], j);
            const float p1 = __shfl_sync(0xffffffff, p[1], j);
            if (p0 == 0.0f && p1 == 0.0f) continue;
            const int8_t* vr = sv + j * head_dim;
            const float* vd = svd + j * nblk;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint32_t d = lane + 32 * i;
                if (d < head_dim) {
                    const float v = vd[d / 32] * (float) vr[d];
                    acc[0][i] += p0 * v;
                    acc[1][i] += p1 * v;
                }
            }
        }
    }

    #pragma unroll
    for (int qq = 0; qq < 2; ++qq) {
        const uint32_t qi = qa + qq;
        if (qi >= nq) continue;
        const float inv = lrun[qq] > 0.0f ? 1.0f / lrun[qq] : 0.0f;
        float* o = out_all + (size_t) (tq0 + qi) * q_dim + (size_t) h * head_dim;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t d = lane + 32 * i;
            if (d < head_dim) o[d] = acc[qq][i] * inv;
        }
    }
}

static size_t attention_prefill_smem(uint32_t head_dim) {
    const uint32_t nblk = head_dim / 32;
    return (size_t) kFaTQ * head_dim * sizeof(float) + 2 * (size_t) kFaTK * nblk * sizeof(float) +
           (size_t) kFaTK * (head_dim + 4) + (size_t) kFaTK * head_dim;
}

// Prefill attention on the tensor cores, over the Q8_0 KV cache.
//
// Block: one head, kTcaQ consecutive query tokens. For each tile of kTcaK
// key positions:
//   S = Q K^T         (WMMA, FP16 in, FP32 out)  -> shared memory
//   mask + online softmax per query row          -> P (FP16), row factors
//   O = O * corr + P V (WMMA, O kept in shared memory as FP32)
// K and V are expanded from Q8_0 to FP16 once per tile and shared by all the
// block's queries. O lives in shared memory rather than in accumulator
// fragments because the online softmax rescales it per ROW, and the element
// layout of a WMMA fragment is unspecified. Q is pre-multiplied by
// 1/sqrt(head_dim) so the scores come out already scaled.
// Requires head_dim % 16 == 0 and head_dim <= 256.
constexpr int kTcaQ = 16;
constexpr int kTcaK = 32;

__global__ void __launch_bounds__(128) attention_prefill_tc_kernel(
        const float* __restrict__ q_all, const uint8_t* __restrict__ kcache,
        const uint8_t* __restrict__ vcache, float* __restrict__ out_all,
        uint32_t head_dim, uint32_t heads_per_kv, uint32_t q_dim,
        size_t row_bytes, size_t pos_bytes, size_t layer_off,
        uint32_t pos0, uint32_t n_tok, uint32_t n_swa, float inv_d) {
    using namespace nvcuda;
    const uint32_t h = blockIdx.x;
    const uint32_t hkv = h / heads_per_kv;
    const uint32_t tq0 = blockIdx.y * kTcaQ;
    if (tq0 >= n_tok) return;
    const uint32_t nq = min((uint32_t) kTcaQ, n_tok - tq0);
    const uint32_t hd = head_dim;
    const uint32_t ldh = hd + 8;              // padded FP16 rows (Q, K, V)
    const uint32_t nblk = hd / 32;

    extern __shared__ __align__(32) unsigned char tca_smem[];
    __half* sq = reinterpret_cast<__half*>(tca_smem);                 // [TQ][ldh]
    __half* sk = sq + kTcaQ * ldh;                                     // [TK][ldh]
    __half* sv = sk + kTcaK * ldh;                                     // [TK][ldh]
    float* so = reinterpret_cast<float*>(sv + kTcaK * ldh);           // [TQ][hd]
    float* ss = so + kTcaQ * hd;                                       // [TQ][TK] scores
    __half* sp = reinterpret_cast<__half*>(ss + kTcaQ * kTcaK);       // [TQ][TK] probabilities
    float* srow = reinterpret_cast<float*>(sp + kTcaQ * kTcaK);       // m[TQ], l[TQ], corr[TQ]
    float* sm = srow;
    float* sl = srow + kTcaQ;
    float* sc = srow + 2 * kTcaQ;

    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid / 32, lane = tid % 32;

    for (uint32_t i = tid; i < kTcaQ * hd; i += blockDim.x) {
        const uint32_t qi = i / hd, d = i % hd;
        const float v = qi < nq ? q_all[(size_t) (tq0 + qi) * q_dim + (size_t) h * hd + d] * inv_d : 0.0f;
        sq[qi * ldh + d] = __float2half(v);
        so[i] = 0.0f;
    }
    if (tid < kTcaQ) { sm[tid] = -INFINITY; sl[tid] = 0.0f; }

    const uint32_t pos_first = pos0 + tq0;
    const uint32_t pos_last = pos0 + tq0 + nq - 1;
    const uint32_t kv_begin = (n_swa > 0 && pos_first + 1 > n_swa) ? pos_first + 1 - n_swa : 0;

    for (uint32_t k0 = kv_begin; k0 <= pos_last; k0 += kTcaK) {
        __syncthreads();
        // Expand the tile's K and V rows to FP16: one (key, 32-block) per step.
        for (uint32_t idx = tid; idx < kTcaK * nblk; idx += blockDim.x) {
            const uint32_t j = idx / nblk, b = idx % nblk;
            const uint32_t kp = k0 + j;
            __half* kd = sk + j * ldh + b * 32;
            __half* vd = sv + j * ldh + b * 32;
            if (kp <= pos_last) {
                const size_t off = layer_off + (size_t) kp * pos_bytes + (size_t) hkv * row_bytes + (size_t) b * 34;
                const uint8_t* kb = kcache + off;
                const uint8_t* vb = vcache + off;
                const float dk = __half2float(*reinterpret_cast<const __half*>(kb));
                const float dv = __half2float(*reinterpret_cast<const __half*>(vb));
                const int8_t* kq = reinterpret_cast<const int8_t*>(kb + 2);
                const int8_t* vq = reinterpret_cast<const int8_t*>(vb + 2);
                #pragma unroll
                for (int e = 0; e < 32; ++e) {
                    kd[e] = __float2half(dk * (float) kq[e]);
                    vd[e] = __float2half(dv * (float) vq[e]);
                }
            } else {
                #pragma unroll
                for (int e = 0; e < 32; ++e) { kd[e] = __float2half(0.0f); vd[e] = __float2half(0.0f); }
            }
        }
        __syncthreads();

        // S = Q K^T: 16 x 32 = two 16x16 fragments, warps 0 and 1.
        if (warp < 2) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> s;
            wmma::fill_fragment(s, 0.0f);
            for (uint32_t kk = 0; kk < hd; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::load_matrix_sync(fa, sq + kk, ldh);
                wmma::load_matrix_sync(fb, sk + warp * 16 * ldh + kk, ldh);
                wmma::mma_sync(s, fa, fb, s);
            }
            wmma::store_matrix_sync(ss + warp * 16, s, kTcaK, wmma::mem_row_major);
        }
        __syncthreads();

        // Mask + online softmax: warp w handles query rows w, w+4, w+8, w+12;
        // lane = key within the tile.
        for (uint32_t qi = warp; qi < kTcaQ; qi += 4) {
            const uint32_t qpos = pos0 + tq0 + qi;
            const uint32_t kp = k0 + lane;
            const bool valid = qi < nq && kp <= qpos && (n_swa == 0 || kp + n_swa > qpos);
            const float s = valid ? ss[qi * kTcaK + lane] : -INFINITY;
            float mt = s;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffff, mt, off));
            const float mold = sm[qi];
            const float mnew = fmaxf(mold, mt);
            const float e = (s == -INFINITY || mnew == -INFINITY) ? 0.0f : __expf(s - mnew);
            float es = e;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) es += __shfl_xor_sync(0xffffffff, es, off);
            sp[qi * kTcaK + lane] = __float2half(e);
            if (lane == 0) {
                const float corr = (mold == -INFINITY) ? 0.0f : __expf(mold - mnew);
                sc[qi] = corr;
                sl[qi] = sl[qi] * corr + es;
                sm[qi] = mnew;
            }
        }
        __syncthreads();

        // O = O * corr (per row), then O += P V on the tensor cores:
        // hd/16 output fragments of 16 x 16, spread over the 4 warps.
        for (uint32_t i = tid; i < kTcaQ * hd; i += blockDim.x) so[i] *= sc[i / hd];
        __syncthreads();
        for (uint32_t f = warp; f < hd / 16; f += 4) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> o;
            wmma::load_matrix_sync(o, so + f * 16, hd, wmma::mem_row_major);
            #pragma unroll
            for (int kk = 0; kk < kTcaK; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fp;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fv;
                wmma::load_matrix_sync(fp, sp + kk, kTcaK);
                wmma::load_matrix_sync(fv, sv + kk * ldh + f * 16, ldh);
                wmma::mma_sync(o, fp, fv, o);
            }
            wmma::store_matrix_sync(so + f * 16, o, hd, wmma::mem_row_major);
        }
    }
    __syncthreads();

    for (uint32_t i = tid; i < kTcaQ * hd; i += blockDim.x) {
        const uint32_t qi = i / hd, d = i % hd;
        if (qi >= nq) continue;
        const float l = sl[qi];
        out_all[(size_t) (tq0 + qi) * q_dim + (size_t) h * hd + d] = l > 0.0f ? so[i] / l : 0.0f;
    }
}

static size_t attention_prefill_tc_smem(uint32_t head_dim);
// Largest request (head_dim 256): what the attribute is raised to, once.
static size_t attention_prefill_smem_cap() { return attention_prefill_tc_smem(256); }

static size_t attention_prefill_tc_smem(uint32_t head_dim) {
    const size_t ldh = head_dim + 8;
    return (size_t) (kTcaQ + 2 * kTcaK) * ldh * sizeof(__half) +      // Q, K, V
           (size_t) kTcaQ * head_dim * sizeof(float) +                  // O
           (size_t) kTcaQ * kTcaK * sizeof(float) +                     // S
           (size_t) kTcaQ * kTcaK * sizeof(__half) +                    // P
           3 * (size_t) kTcaQ * sizeof(float);                          // m, l, corr
}

// Prefill attention on the tensor cores, one KV head per block (GQA).
//
// The block serves every query head that shares KV head hkv: its 32 rows
// are kGqaRows / heads_per_kv consecutive tokens x heads_per_kv heads, so
// each K/V tile is expanded from Q8_0 once for all of them (the per-head
// kernel expanded it heads_per_kv times).
//
// Two passes over the keys instead of an online softmax:
//   pass 1: S = Q K^T tile by tile, keep only each row's maximum;
//   pass 2: S again, P = exp(S - max) (final, never rescaled), row sums,
//           O += P V accumulated directly in tensor-core fragments.
// The output never leaves registers until the end - the online variant had
// to rescale O per row after every tile, through shared memory, because a
// fragment's element layout is unspecified. Recomputing Q K^T costs tensor
// core time, which is cheap; the shared-memory traffic it removes was not.
//
// 8 warps. Shared memory for head_dim 256 is ~57 KB (fits Turing's 64 KB).
// Requires head_dim % 32 == 0, head_dim <= 256, heads_per_kv dividing 32.
constexpr int kGqaRows = 32;
constexpr int kGqaK = 32;

__global__ void __launch_bounds__(256) attention_prefill_gqa_kernel(
        const float* __restrict__ q_all, const uint8_t* __restrict__ kcache,
        const uint8_t* __restrict__ vcache, float* __restrict__ out_all,
        uint32_t head_dim, uint32_t heads_per_kv, uint32_t q_dim,
        size_t row_bytes, size_t pos_bytes, size_t layer_off,
        uint32_t pos0, uint32_t n_tok, uint32_t n_swa, float inv_d) {
    using namespace nvcuda;
    const uint32_t hkv = blockIdx.x;
    const uint32_t hpk = heads_per_kv;
    const uint32_t tpb = kGqaRows / hpk;           // tokens per block
    const uint32_t tq0 = blockIdx.y * tpb;
    if (tq0 >= n_tok) return;
    const uint32_t nt = min(tpb, n_tok - tq0);
    const uint32_t hd = head_dim;
    const uint32_t ldh = hd + 8;
    const uint32_t nblk = hd / 32;

    extern __shared__ __align__(32) unsigned char gqa_smem[];
    __half* sq = reinterpret_cast<__half*>(gqa_smem);                  // [ROWS][ldh]
    __half* sk = sq + kGqaRows * ldh;                                    // [K][ldh]
    __half* sv = sk + kGqaK * ldh;                                       // [K][ldh]
    float* ss = reinterpret_cast<float*>(sv + kGqaK * ldh);             // [ROWS][K]
    __half* sp = reinterpret_cast<__half*>(ss + kGqaRows * kGqaK);      // [ROWS][K]
    float* smax = reinterpret_cast<float*>(sp + kGqaRows * kGqaK);      // [ROWS]
    float* ssum = smax + kGqaRows;                                       // [ROWS]

    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid / 32, lane = tid % 32;

    // Row r = token (r / hpk), head (hkv * hpk + r % hpk).
    for (uint32_t i = tid; i < kGqaRows * hd; i += blockDim.x) {
        const uint32_t r = i / hd, d = i % hd;
        const uint32_t t = r / hpk, hh = r % hpk;
        float v = 0.0f;
        if (t < nt) v = q_all[(size_t) (tq0 + t) * q_dim + (size_t) (hkv * hpk + hh) * hd + d] * inv_d;
        sq[r * ldh + d] = __float2half(v);
    }
    if (tid < kGqaRows) { smax[tid] = -INFINITY; ssum[tid] = 0.0f; }

    const uint32_t pos_first = pos0 + tq0;
    const uint32_t pos_last = pos0 + tq0 + nt - 1;
    const uint32_t kv_begin = (n_swa > 0 && pos_first + 1 > n_swa) ? pos_first + 1 - n_swa : 0;

    auto load_tile = [&](uint32_t k0, bool with_v) {
        for (uint32_t idx = tid; idx < kGqaK * nblk; idx += blockDim.x) {
            const uint32_t j = idx / nblk, b = idx % nblk;
            const uint32_t kp = k0 + j;
            __half* kd = sk + j * ldh + b * 32;
            __half* vd = sv + j * ldh + b * 32;
            if (kp <= pos_last) {
                const size_t off = layer_off + (size_t) kp * pos_bytes + (size_t) hkv * row_bytes + (size_t) b * 34;
                const uint8_t* kb = kcache + off;
                const float dk = __half2float(*reinterpret_cast<const __half*>(kb));
                const int8_t* kq = reinterpret_cast<const int8_t*>(kb + 2);
                #pragma unroll
                for (int e = 0; e < 32; ++e) kd[e] = __float2half(dk * (float) kq[e]);
                if (with_v) {
                    const uint8_t* vb = vcache + off;
                    const float dv = __half2float(*reinterpret_cast<const __half*>(vb));
                    const int8_t* vq = reinterpret_cast<const int8_t*>(vb + 2);
                    #pragma unroll
                    for (int e = 0; e < 32; ++e) vd[e] = __float2half(dv * (float) vq[e]);
                }
            } else {
                #pragma unroll
                for (int e = 0; e < 32; ++e) kd[e] = __float2half(0.0f);
                if (with_v) {
                    #pragma unroll
                    for (int e = 0; e < 32; ++e) vd[e] = __float2half(0.0f);
                }
            }
        }
    };
    // S = Q K^T for the tile: 32 x 32 = 4 fragments, one per warp 0..3.
    auto scores = [&]() {
        if (warp < 4) {
            const uint32_t fr = warp / 2, fc = warp % 2;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> s;
            wmma::fill_fragment(s, 0.0f);
            for (uint32_t kk = 0; kk < hd; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::load_matrix_sync(fa, sq + fr * 16 * ldh + kk, ldh);
                wmma::load_matrix_sync(fb, sk + fc * 16 * ldh + kk, ldh);
                wmma::mma_sync(s, fa, fb, s);
            }
            wmma::store_matrix_sync(ss + fr * 16 * kGqaK + fc * 16, s, kGqaK, wmma::mem_row_major);
        }
    };
    auto masked = [&](uint32_t r, uint32_t kp) {
        const uint32_t t = r / hpk;
        if (t >= nt) return true;
        const uint32_t qpos = pos0 + tq0 + t;
        return kp > qpos || (n_swa > 0 && kp + n_swa <= qpos);
    };

    // ── Pass 1: row maxima ─────────────────────────────────────────────
    for (uint32_t k0 = kv_begin; k0 <= pos_last; k0 += kGqaK) {
        __syncthreads();
        load_tile(k0, false);
        __syncthreads();
        scores();
        __syncthreads();
        // 8 warps x 4 rows; lane = key.
        for (uint32_t r = warp; r < kGqaRows; r += 8) {
            const float s = masked(r, k0 + lane) ? -INFINITY : ss[r * kGqaK + lane];
            float m = s;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
            if (lane == 0) smax[r] = fmaxf(smax[r], m);
        }
    }

    // ── Pass 2: final probabilities, sums, O += P V ────────────────────
    // O is 32 x hd = 2 x (hd/16) fragments; warp w owns columns w, w+8, ...
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(o[i][j], 0.0f);
    const uint32_t ncol = hd / 16;               // <= 16
    for (uint32_t k0 = kv_begin; k0 <= pos_last; k0 += kGqaK) {
        __syncthreads();
        load_tile(k0, true);
        __syncthreads();
        scores();
        __syncthreads();
        for (uint32_t r = warp; r < kGqaRows; r += 8) {
            const float m = smax[r];
            const bool dead = masked(r, k0 + lane) || m == -INFINITY;
            const float e = dead ? 0.0f : __expf(ss[r * kGqaK + lane] - m);
            sp[r * kGqaK + lane] = __float2half(e);
            float es = e;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) es += __shfl_xor_sync(0xffffffff, es, off);
            if (lane == 0) ssum[r] += es;
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const uint32_t col = warp + 8 * j;
            if (col >= ncol) continue;
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int kk = 0; kk < kGqaK; kk += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fp;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fv;
                    wmma::load_matrix_sync(fp, sp + i * 16 * kGqaK + kk, kGqaK);
                    wmma::load_matrix_sync(fv, sv + kk * ldh + col * 16, ldh);
                    wmma::mma_sync(o[i][j], fp, fv, o[i][j]);
                }
            }
        }
    }
    __syncthreads();

    // O out through shared memory (Q/K/V space, no longer needed): [ROWS][hd] floats.
    float* so = reinterpret_cast<float*>(gqa_smem);
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        const uint32_t col = warp + 8 * j;
        if (col >= ncol) continue;
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            wmma::store_matrix_sync(so + i * 16 * hd + col * 16, o[i][j], hd, wmma::mem_row_major);
    }
    __syncthreads();
    for (uint32_t i = tid; i < kGqaRows * hd; i += blockDim.x) {
        const uint32_t r = i / hd, d = i % hd;
        const uint32_t t = r / hpk, hh = r % hpk;
        if (t >= nt) continue;
        const float l = ssum[r];
        out_all[(size_t) (tq0 + t) * q_dim + (size_t) (hkv * hpk + hh) * hd + d] = l > 0.0f ? so[i] / l : 0.0f;
    }
}

static size_t attention_prefill_gqa_smem(uint32_t head_dim) {
    const size_t ldh = head_dim + 8;
    const size_t tiles = (size_t) (kGqaRows + 2 * kGqaK) * ldh * sizeof(__half) +
                         (size_t) kGqaRows * kGqaK * (sizeof(float) + sizeof(__half)) +
                         2 * (size_t) kGqaRows * sizeof(float);
    const size_t out = (size_t) kGqaRows * head_dim * sizeof(float);   // reuses the tile space
    return tiles > out ? tiles : out;
}

// Prefill attention in ONE pass over the keys (sm_80+).
//
// The kernel above (attention_prefill_gqa_kernel) runs two passes: the first
// computes every Q.K score just to find each row's maximum, the second
// computes them again for the softmax and P.V - because with WMMA the
// output fragments can't be rescaled in registers (their element layout is
// unspecified), so the softmax maximum had to be final before accumulating.
// Here the products use mma.sync directly (m16n8k16, FP16 in / FP32
// accumulate), whose accumulator layout IS specified:
//   * S = Q K^T for a tile of 32 keys stays in registers;
//   * online softmax: running maximum and sum per row, the output rescaled
//     in registers when the maximum grows;
//   * P becomes the A operand of the P.V product straight from the S
//     registers (the m16n8 accumulator of two adjacent key tiles is exactly
//     one m16n8k16 A fragment) - no trip through shared memory.
// One block = one KV head x 64 rows (64 / heads_per_kv tokens times the
// heads that share that KV head), 4 warps of 16 rows; each K/V tile is
// dequantized once for all of them. V is kept transposed so the B operand
// of P.V loads as 32-bit words. Scores are in log2 units (log2(e) folded
// into the Q scale) so the exponentials are exp2f.
constexpr int kFaRows = 64;
constexpr int kFaKeys = 32;
constexpr int kFaLdv = kFaKeys + 8;          // Vt row stride (halves): conflict-free B loads

template <int HD>
__global__ void __launch_bounds__(128) attention_prefill_fa_kernel(
        const float* __restrict__ q_all, const uint8_t* __restrict__ kcache,
        const uint8_t* __restrict__ vcache, float* __restrict__ out_all,
        uint32_t hpk, uint32_t q_dim, size_t row_bytes, size_t pos_bytes, size_t layer_off,
        uint32_t pos0, uint32_t n_tok, uint32_t n_swa, float q_scale) {
    constexpr int ldq = HD + 8;
    const uint32_t hkv = blockIdx.x;
    const uint32_t tpb = kFaRows / hpk;
    const uint32_t tq0 = blockIdx.y * tpb;
    if (tq0 >= n_tok) return;
    const uint32_t nt = min(tpb, n_tok - tq0);

    extern __shared__ __align__(16) unsigned char fa_smem[];
    __half* Qs = reinterpret_cast<__half*>(fa_smem);            // [64][ldq]
    __half* Ks = Qs + kFaRows * ldq;                             // [32][ldq]
    __half* Vt = Ks + kFaKeys * ldq;                             // [HD][kFaLdv]

    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int g = lane >> 2, tq = lane & 3;

    for (int i = tid; i < kFaRows * HD; i += blockDim.x) {
        const int r = i / HD, d = i % HD;
        const uint32_t t = r / hpk, hh = r % hpk;
        float v = 0.0f;
        if (t < nt) v = q_all[(size_t) (tq0 + t) * q_dim + (size_t) (hkv * hpk + hh) * HD + d] * q_scale;
        Qs[r * ldq + d] = __float2half(v);
    }

    const int rA = warp * 16 + g, rB = rA + 8;
    const uint32_t tA = rA / hpk, tB = rB / hpk;
    const bool okA = tA < nt, okB = tB < nt;
    const uint32_t qposA = pos0 + tq0 + tA, qposB = pos0 + tq0 + tB;

    float o[HD / 8][4];
    #pragma unroll
    for (int n = 0; n < HD / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float mA = -INFINITY, mB = -INFINITY, lA = 0.0f, lB = 0.0f;

    const uint32_t pos_first = pos0 + tq0, pos_last = pos0 + tq0 + nt - 1;
    const uint32_t kv_begin = (n_swa > 0 && pos_first + 1 > n_swa) ? pos_first + 1 - n_swa : 0;
    constexpr int nblk = HD / 32;

    for (uint32_t k0 = kv_begin; k0 <= pos_last; k0 += kFaKeys) {
        __syncthreads();
        for (int u = tid; u < kFaKeys * nblk; u += blockDim.x) {
            const int key = u / nblk, b = u % nblk;
            const uint32_t kp = k0 + key;
            __half* kd = Ks + key * ldq + b * 32;
            if (kp <= pos_last) {
                const size_t off = layer_off + (size_t) kp * pos_bytes + (size_t) hkv * row_bytes + (size_t) b * 34;
                const uint8_t* kb = kcache + off;
                const uint8_t* vb = vcache + off;
                const float dk = __half2float(*reinterpret_cast<const __half*>(kb));
                const float dv = __half2float(*reinterpret_cast<const __half*>(vb));
                const int8_t* kq = reinterpret_cast<const int8_t*>(kb + 2);
                const int8_t* vq = reinterpret_cast<const int8_t*>(vb + 2);
                #pragma unroll
                for (int e = 0; e < 32; ++e) {
                    kd[e] = __float2half(dk * (float) kq[e]);
                    Vt[(b * 32 + e) * kFaLdv + key] = __float2half(dv * (float) vq[e]);
                }
            } else {
                #pragma unroll
                for (int e = 0; e < 32; ++e) {
                    kd[e] = __float2half(0.0f);
                    Vt[(b * 32 + e) * kFaLdv + key] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        // S = Q K^T: 16 rows x 32 keys per warp = 4 key tiles of 8.
        float s[4][4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < HD; kk += 16) {
            uint32_t a[4];
            const __half* q0 = Qs + (warp * 16 + g) * ldq + kk + 2 * tq;
            a[0] = *reinterpret_cast<const uint32_t*>(q0);
            a[1] = *reinterpret_cast<const uint32_t*>(q0 + 8 * ldq);
            a[2] = *reinterpret_cast<const uint32_t*>(q0 + 8);
            a[3] = *reinterpret_cast<const uint32_t*>(q0 + 8 * ldq + 8);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const __half* k = Ks + (j * 8 + g) * ldq + kk + 2 * tq;
                mma_f16_16816(s[j], a, *reinterpret_cast<const uint32_t*>(k), *reinterpret_cast<const uint32_t*>(k + 8));
            }
        }

        // Mask, row maxima (a row's 32 scores sit in the 4 threads of a quad).
        float mxA = -INFINITY, mxB = -INFINITY;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
                const uint32_t kp = k0 + j * 8 + 2 * tq + e;
                const bool deadA = !okA || kp > qposA || (n_swa > 0 && kp + n_swa <= qposA);
                const bool deadB = !okB || kp > qposB || (n_swa > 0 && kp + n_swa <= qposB);
                if (deadA) s[j][e] = -INFINITY;
                if (deadB) s[j][2 + e] = -INFINITY;
                mxA = fmaxf(mxA, s[j][e]);
                mxB = fmaxf(mxB, s[j][2 + e]);
            }
        }
        #pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffff, mxA, off));
            mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffff, mxB, off));
        }
        const float nA = fmaxf(mA, mxA), nB = fmaxf(mB, mxB);
        const float cA = nA == -INFINITY ? 1.0f : exp2f(mA - nA);
        const float cB = nB == -INFINITY ? 1.0f : exp2f(mB - nB);
        mA = nA; mB = nB;
        lA *= cA; lB *= cB;
        #pragma unroll
        for (int n = 0; n < HD / 8; ++n) {
            o[n][0] *= cA; o[n][1] *= cA;
            o[n][2] *= cB; o[n][3] *= cB;
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
                s[j][e] = nA == -INFINITY ? 0.0f : exp2f(s[j][e] - nA);
                s[j][2 + e] = nB == -INFINITY ? 0.0f : exp2f(s[j][2 + e] - nB);
                lA += s[j][e];
                lB += s[j][2 + e];
            }
        }

        // O += P V: two k-steps of 16 keys; P's A fragment comes straight
        // from the S registers of key tiles 2ks and 2ks+1.
        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            uint32_t a[4];
            a[0] = pack_half2(s[2 * ks][0], s[2 * ks][1]);
            a[1] = pack_half2(s[2 * ks][2], s[2 * ks][3]);
            a[2] = pack_half2(s[2 * ks + 1][0], s[2 * ks + 1][1]);
            a[3] = pack_half2(s[2 * ks + 1][2], s[2 * ks + 1][3]);
            #pragma unroll
            for (int n = 0; n < HD / 8; ++n) {
                const __half* v = Vt + (n * 8 + g) * kFaLdv + ks * 16 + 2 * tq;
                mma_f16_16816(o[n], a, *reinterpret_cast<const uint32_t*>(v), *reinterpret_cast<const uint32_t*>(v + 8));
            }
        }
    }

    // Each thread summed its own columns: complete the row sums across the quad.
    #pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
        lA += __shfl_xor_sync(0xffffffff, lA, off);
        lB += __shfl_xor_sync(0xffffffff, lB, off);
    }
    const float iA = lA > 0.0f ? 1.0f / lA : 0.0f, iB = lB > 0.0f ? 1.0f / lB : 0.0f;
    float* outA = out_all + (size_t) (tq0 + tA) * q_dim + (size_t) (hkv * hpk + rA % hpk) * HD;
    float* outB = out_all + (size_t) (tq0 + tB) * q_dim + (size_t) (hkv * hpk + rB % hpk) * HD;
    #pragma unroll
    for (int n = 0; n < HD / 8; ++n) {
        const int d = n * 8 + 2 * tq;
        if (okA) { outA[d] = o[n][0] * iA; outA[d + 1] = o[n][1] * iA; }
        if (okB) { outB[d] = o[n][2] * iB; outB[d + 1] = o[n][3] * iB; }
    }
}

static size_t attention_prefill_fa_smem(uint32_t hd) {
    return ((size_t) (kFaRows + kFaKeys) * (hd + 8) + (size_t) hd * kFaLdv) * sizeof(__half);
}

// K/V of the current layer as FP16, [pos][kv_dim], converted from the Q8_0
// cache ONCE per layer (incrementally, chunk by chunk). The single-pass
// kernel above dequantizes every K/V tile again in every block of 64 query
// rows - on the CUDA cores, next to tensor-core math that is much cheaper.
// Converting from the Q8_0 values (not from the float K/V just computed)
// keeps the arithmetic exactly what decode and a restored session see.
__global__ void kv_q8_to_f16_kernel(const uint8_t* __restrict__ kcache, const uint8_t* __restrict__ vcache,
                                    size_t layer_off, size_t pos_bytes, size_t row_bytes,
                                    uint32_t n_kv, uint32_t hd, uint32_t p_begin, uint32_t p_end,
                                    __half* __restrict__ kh, __half* __restrict__ vh) {
    const uint32_t nb = hd / 32;
    const size_t per_pos = (size_t) n_kv * nb;
    const size_t e = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (size_t) (p_end - p_begin) * per_pos) return;
    const uint32_t pos = p_begin + (uint32_t) (e / per_pos);
    const uint32_t h = (uint32_t) ((e % per_pos) / nb), b = (uint32_t) (e % nb);
    const size_t off = layer_off + (size_t) pos * pos_bytes + (size_t) h * row_bytes + (size_t) b * 34;
    const size_t dst = (size_t) pos * n_kv * hd + (size_t) h * hd + (size_t) b * 32;
    const float dk = __half2float(*reinterpret_cast<const __half*>(kcache + off));
    const float dv = __half2float(*reinterpret_cast<const __half*>(vcache + off));
    const int8_t* kq = reinterpret_cast<const int8_t*>(kcache + off + 2);
    const int8_t* vq = reinterpret_cast<const int8_t*>(vcache + off + 2);
    #pragma unroll
    for (int i = 0; i < 32; i += 2) {
        *reinterpret_cast<__half2*>(kh + dst + i) = __floats2half2_rn(dk * kq[i], dk * kq[i + 1]);
        *reinterpret_cast<__half2*>(vh + dst + i) = __floats2half2_rn(dv * vq[i], dv * vq[i + 1]);
    }
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void* smem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
#endif
}

// Same algorithm as attention_prefill_fa_kernel, reading the FP16 K/V.
//   - K/V tiles of 32 keys arrive through cp.async into two buffers: the
//     next tile is in flight while the current one is multiplied.
//   - Q and K fragments come from ldmatrix; V is kept row-major and fed to
//     the P.V product through ldmatrix.trans.
//   - NW warps, 16 query rows each: 8 for head sizes up to 128 (twice the
//     warps per SM, each K/V tile serves 128 rows), 4 for 256 (shared
//     memory: Q plus two K/V buffers is exactly the 99 KB a block can have).
//   - The query tiles with the most keys (the last ones, causal) start first.
constexpr int fa16_warps(int hd) { return hd <= 128 ? 8 : 4; }

template <int HD, int NW>
__global__ void __launch_bounds__(NW * 32) attention_prefill_fa16_kernel(
        const float* __restrict__ q_all, const __half* __restrict__ kh, const __half* __restrict__ vh,
        float* __restrict__ out_all, uint32_t hpk, uint32_t n_kv, uint32_t q_dim,
        uint32_t pos0, uint32_t n_tok, uint32_t n_swa, float q_scale, __half* __restrict__ out_h) {
    constexpr int ld = HD + 8;
    constexpr int R = NW * 16;                                     // query rows per block
    const uint32_t hkv = blockIdx.x;
    const uint32_t tpb = R / hpk;
    const uint32_t tq0 = (gridDim.y - 1 - blockIdx.y) * tpb;
    if (tq0 >= n_tok) return;
    const uint32_t nt = min(tpb, n_tok - tq0);
    const size_t kv_dim = (size_t) n_kv * HD;

    extern __shared__ __align__(16) unsigned char fa16_smem[];
    __half* Qs = reinterpret_cast<__half*>(fa16_smem);           // [R][ld]
    auto Ks = [&](int b) { return Qs + R * ld + b * 2 * kFaKeys * ld; };            // [32][ld]
    auto Vs = [&](int b) { return Qs + R * ld + b * 2 * kFaKeys * ld + kFaKeys * ld; };

    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int g = lane >> 2, tq = lane & 3;

    const uint32_t pos_first = pos0 + tq0, pos_last = pos0 + tq0 + nt - 1;
    // Window start rounded down to a whole tile: the tile boundaries (and so
    // each row's summation order) no longer depend on where the chunk
    // starts, so a prompt split differently gives the same bits (the
    // extra keys are masked like any key outside the window).
    const uint32_t kv_begin = (n_swa > 0 && pos_first + 1 > n_swa) ? (pos_first + 1 - n_swa) / kFaKeys * kFaKeys : 0;
    constexpr int chunks = HD / 8;                                 // 16-byte pieces per key row
    auto load_tile = [&](int b, uint32_t k0) {
        for (int u = tid; u < kFaKeys * chunks; u += NW * 32) {
            const int key = u / chunks, c = (u % chunks) * 8;
            const uint32_t kp = k0 + key;
            const bool ok = kp <= pos_last;
            const size_t src = ok ? (size_t) kp * kv_dim + (size_t) hkv * HD + c : 0;
            cp_async16(Ks(b) + key * ld + c, kh + src, ok);
            cp_async16(Vs(b) + key * ld + c, vh + src, ok);
        }
    };
    load_tile(0, kv_begin);
    cp_async_commit();

    for (int i = tid; i < R * HD; i += NW * 32) {
        const int r = i / HD, d = i % HD;
        const uint32_t t = r / hpk, hh = r % hpk;
        float v = 0.0f;
        if (t < nt) v = q_all[(size_t) (tq0 + t) * q_dim + (size_t) (hkv * hpk + hh) * HD + d] * q_scale;
        Qs[r * ld + d] = __float2half(v);
    }

    const int rA = warp * 16 + g, rB = rA + 8;
    const uint32_t tA = rA / hpk, tB = rB / hpk;
    const bool okA = tA < nt, okB = tB < nt;
    const uint32_t qposA = pos0 + tq0 + tA, qposB = pos0 + tq0 + tB;

    float o[HD / 8][4];
    #pragma unroll
    for (int n = 0; n < HD / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float mA = -INFINITY, mB = -INFINITY, lA = 0.0f, lB = 0.0f;

    int buf = 0;
    for (uint32_t k0 = kv_begin; k0 <= pos_last; k0 += kFaKeys, buf ^= 1) {
        if (k0 + kFaKeys <= pos_last) load_tile(buf ^ 1, k0 + kFaKeys);
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();
        const __half* ks = Ks(buf);
        const __half* vs = Vs(buf);

        float s[4][4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < HD; kk += 16) {
            uint32_t a[4], b0[4], b1[4];
            ldmatrix_x4(a, Qs + (warp * 16 + (lane % 16)) * ld + kk + (lane / 16) * 8);
            // keys 0-15 and 16-31: lanes 0-7 n0..7/k0, 8-15 n0..7/k8, 16-23 n8..15/k0, 24-31 n8..15/k8
            const int kr = (lane % 8) + (lane / 16) * 8, kc = kk + ((lane / 8) % 2) * 8;
            ldmatrix_x4(b0, ks + kr * ld + kc);
            ldmatrix_x4(b1, ks + (16 + kr) * ld + kc);
            mma_f16_16816(s[0], a, b0[0], b0[1]);
            mma_f16_16816(s[1], a, b0[2], b0[3]);
            mma_f16_16816(s[2], a, b1[0], b1[1]);
            mma_f16_16816(s[3], a, b1[2], b1[3]);
        }

        float mxA = -INFINITY, mxB = -INFINITY;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
                const uint32_t kp = k0 + j * 8 + 2 * tq + e;
                if (!okA || kp > qposA || (n_swa > 0 && kp + n_swa <= qposA)) s[j][e] = -INFINITY;
                if (!okB || kp > qposB || (n_swa > 0 && kp + n_swa <= qposB)) s[j][2 + e] = -INFINITY;
                mxA = fmaxf(mxA, s[j][e]);
                mxB = fmaxf(mxB, s[j][2 + e]);
            }
        }
        #pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffff, mxA, off));
            mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffff, mxB, off));
        }
        const float nA = fmaxf(mA, mxA), nB = fmaxf(mB, mxB);
        const float cA = nA == -INFINITY ? 1.0f : exp2f(mA - nA);
        const float cB = nB == -INFINITY ? 1.0f : exp2f(mB - nB);
        mA = nA; mB = nB;
        if (cA != 1.0f || cB != 1.0f) {          // see attention_prefill_fa16q_kernel
            lA *= cA; lB *= cB;
            #pragma unroll
            for (int n = 0; n < HD / 8; ++n) {
                o[n][0] *= cA; o[n][1] *= cA;
                o[n][2] *= cB; o[n][3] *= cB;
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
                s[j][e] = nA == -INFINITY ? 0.0f : exp2f(s[j][e] - nA);
                s[j][2 + e] = nB == -INFINITY ? 0.0f : exp2f(s[j][2 + e] - nB);
                lA += s[j][e];
                lB += s[j][2 + e];
            }
        }

        #pragma unroll
        for (int kq = 0; kq < 2; ++kq) {
            uint32_t a[4];
            a[0] = pack_half2(s[2 * kq][0], s[2 * kq][1]);
            a[1] = pack_half2(s[2 * kq][2], s[2 * kq][3]);
            a[2] = pack_half2(s[2 * kq + 1][0], s[2 * kq + 1][1]);
            a[3] = pack_half2(s[2 * kq + 1][2], s[2 * kq + 1][3]);
            const int key = kq * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
            #pragma unroll
            for (int n2 = 0; n2 < HD / 16; ++n2) {
                uint32_t b[4];
                ldmatrix_x4_trans(b, vs + key * ld + n2 * 16 + (lane >> 4) * 8);
                mma_f16_16816(o[2 * n2], a, b[0], b[1]);
                mma_f16_16816(o[2 * n2 + 1], a, b[2], b[3]);
            }
        }
        __syncthreads();              // this buffer is refilled by the next iteration
    }
    cp_async_wait<0>();

    #pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
        lA += __shfl_xor_sync(0xffffffff, lA, off);
        lB += __shfl_xor_sync(0xffffffff, lB, off);
    }
    const float iA = lA > 0.0f ? 1.0f / lA : 0.0f, iB = lB > 0.0f ? 1.0f / lB : 0.0f;
    float* outA = out_all + (size_t) (tq0 + tA) * q_dim + (size_t) (hkv * hpk + rA % hpk) * HD;
    float* outB = out_all + (size_t) (tq0 + tB) * q_dim + (size_t) (hkv * hpk + rB % hpk) * HD;
    if (out_h) {                                  // FP16 input of the output projection
        __half* hA = out_h + (size_t) (tq0 + tA) * q_dim + (size_t) (hkv * hpk + rA % hpk) * HD;
        __half* hB = out_h + (size_t) (tq0 + tB) * q_dim + (size_t) (hkv * hpk + rB % hpk) * HD;
        #pragma unroll
        for (int n = 0; n < HD / 8; ++n) {
            const int d = n * 8 + 2 * tq;
            if (okA) *reinterpret_cast<__half2*>(hA + d) = __floats2half2_rn(o[n][0] * iA, o[n][1] * iA);
            if (okB) *reinterpret_cast<__half2*>(hB + d) = __floats2half2_rn(o[n][2] * iB, o[n][3] * iB);
        }
        return;
    }
    #pragma unroll
    for (int n = 0; n < HD / 8; ++n) {
        const int d = n * 8 + 2 * tq;
        if (okA) { outA[d] = o[n][0] * iA; outA[d + 1] = o[n][1] * iA; }
        if (okB) { outB[d] = o[n][2] * iB; outB[d + 1] = o[n][3] * iB; }
    }
}

// Head sizes up to 128: the Q fragments stay in registers for the whole key
// loop (they were read again from shared memory on every tile), tiles of 64
// keys (half the barriers and softmax rounds per key), 8 warps and two K/V
// buffers: 2 x 2 x 64 x (HD + 8) halves = 68 KB. Q passes through the second
// buffer once, before that buffer's first tile is requested. The causal and
// window masks are evaluated only on the tiles that cross them.
constexpr int kFa3Keys = 64;
__host__ __device__ constexpr int fa16q_keys(int hd) { return hd <= 128 ? kFa3Keys : 32; }

template <int HD>
__global__ void __launch_bounds__(256) attention_prefill_fa16q_kernel(
        const float* __restrict__ q_all, const __half* __restrict__ kh, const __half* __restrict__ vh,
        float* __restrict__ out_all, uint32_t hpk, uint32_t n_kv, uint32_t q_dim,
        uint32_t pos0, uint32_t n_tok, uint32_t n_swa, float q_scale, __half* __restrict__ out_h) {
    static_assert(HD <= 256, "Q fragments in registers: head size up to 256");
    constexpr int ld = HD + 8;
    constexpr int NW = 8, R = NW * 16;
    // Head size 256: 32-key tiles (the Q fragments and the output take
    // 192 registers), and Q is staged in the whole buffer area first.
    constexpr int KT = fa16q_keys(HD);
    const uint32_t hkv = blockIdx.x;
    const uint32_t tpb = R / hpk;
    const uint32_t tq0 = (gridDim.y - 1 - blockIdx.y) * tpb;
    if (tq0 >= n_tok) return;
    const uint32_t nt = min(tpb, n_tok - tq0);
    const size_t kv_dim = (size_t) n_kv * HD;

    extern __shared__ __align__(16) unsigned char fa3_smem[];
    __half* base = reinterpret_cast<__half*>(fa3_smem);
    auto Ks = [&](int b) { return base + b * 2 * KT * ld; };
    auto Vs = [&](int b) { return base + b * 2 * KT * ld + KT * ld; };

    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int g = lane >> 2, tq = lane & 3;

    const uint32_t pos_first = pos0 + tq0, pos_last = pos0 + tq0 + nt - 1;
    // Window start rounded down to a whole tile: the tile boundaries (and so
    // each row's summation order) no longer depend on where the chunk
    // starts, so a prompt split differently gives the same bits (the
    // extra keys are masked like any key outside the window).
    const uint32_t kv_begin = (n_swa > 0 && pos_first + 1 > n_swa) ? (pos_first + 1 - n_swa) / KT * KT : 0;
    constexpr int chunks = HD / 8;
    auto load_tile = [&](int b, uint32_t k0) {
        for (int u = tid; u < KT * chunks; u += NW * 32) {
            const int key = u / chunks, c = (u % chunks) * 8;
            const uint32_t kp = k0 + key;
            const bool ok = kp <= pos_last;
            const size_t src = ok ? (size_t) kp * kv_dim + (size_t) hkv * HD + c : 0;
            cp_async16(Ks(b) + key * ld + c, kh + src, ok);
            cp_async16(Vs(b) + key * ld + c, vh + src, ok);
        }
    };
    if (HD <= 128) load_tile(0, kv_begin);
    cp_async_commit();

    // Q -> buffer 1 (R rows fit exactly: R * ld == 2 * KT * ld; for HD 256
    // both buffers, R * ld == 4 * KT * ld) -> registers.
    {
        __half* Qs = HD <= 128 ? Ks(1) : Ks(0);
        for (int i = tid; i < R * HD; i += NW * 32) {
            const int r = i / HD, d = i % HD;
            const uint32_t t = r / hpk, hh = r % hpk;
            float v = 0.0f;
            if (t < nt) v = q_all[(size_t) (tq0 + t) * q_dim + (size_t) (hkv * hpk + hh) * HD + d] * q_scale;
            Qs[r * ld + d] = __float2half(v);
        }
    }
    __syncthreads();
    uint32_t qf[HD / 16][4];
    #pragma unroll
    for (int kk = 0; kk < HD / 16; ++kk)
        ldmatrix_x4(qf[kk], (HD <= 128 ? Ks(1) : Ks(0)) + (warp * 16 + (lane % 16)) * ld + kk * 16 + (lane / 16) * 8);
    __syncthreads();                                              // the staging buffer is free again
    if (HD > 128) {
        load_tile(0, kv_begin);
        cp_async_commit();
    }

    const int rA = warp * 16 + g, rB = rA + 8;
    const uint32_t tA = rA / hpk, tB = rB / hpk;
    const bool okA = tA < nt, okB = tB < nt;
    const uint32_t qposA = pos0 + tq0 + tA, qposB = pos0 + tq0 + tB;

    float o[HD / 8][4];
    #pragma unroll
    for (int n = 0; n < HD / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float mA = -INFINITY, mB = -INFINITY, lA = 0.0f, lB = 0.0f;

    int buf = 0;
    for (uint32_t k0 = kv_begin; k0 <= pos_last; k0 += KT, buf ^= 1) {
        if (k0 + KT <= pos_last) load_tile(buf ^ 1, k0 + KT);
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();
        const __half* ks = Ks(buf);
        const __half* vs = Vs(buf);

        float s[KT / 8][4];
        #pragma unroll
        for (int j = 0; j < KT / 8; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < HD / 16; ++kk) {
            const int kr = (lane % 8) + (lane / 16) * 8, kc = kk * 16 + ((lane / 8) % 2) * 8;
            #pragma unroll
            for (int j2 = 0; j2 < KT / 16; ++j2) {
                uint32_t b[4];
                ldmatrix_x4(b, ks + (j2 * 16 + kr) * ld + kc);
                mma_f16_16816(s[2 * j2], qf[kk], b[0], b[1]);
                mma_f16_16816(s[2 * j2 + 1], qf[kk], b[2], b[3]);
            }
        }

        // Masks only where the tile crosses the diagonal, the window edge,
        // the prompt end or the block's padding rows.
        const bool full = k0 + KT - 1 <= pos_first && (n_swa == 0 || k0 + n_swa > pos_last) && nt == tpb;
        float mxA = -INFINITY, mxB = -INFINITY;
        #pragma unroll
        for (int j = 0; j < KT / 8; ++j) {
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
                if (!full) {
                    const uint32_t kp = k0 + j * 8 + 2 * tq + e;
                    if (!okA || kp > qposA || (n_swa > 0 && kp + n_swa <= qposA)) s[j][e] = -INFINITY;
                    if (!okB || kp > qposB || (n_swa > 0 && kp + n_swa <= qposB)) s[j][2 + e] = -INFINITY;
                }
                mxA = fmaxf(mxA, s[j][e]);
                mxB = fmaxf(mxB, s[j][2 + e]);
            }
        }
        #pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffff, mxA, off));
            mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffff, mxB, off));
        }
        const float nA = fmaxf(mA, mxA), nB = fmaxf(mB, mxB);
        const float cA = nA == -INFINITY ? 1.0f : exp2f(mA - nA);
        const float cB = nB == -INFINITY ? 1.0f : exp2f(mB - nB);
        mA = nA; mB = nB;
        // Rescale only when a row's maximum moved: after the first tiles it
        // rarely does, and multiplying by exactly 1 changes nothing.
        if (cA != 1.0f || cB != 1.0f) {
            lA *= cA; lB *= cB;
            #pragma unroll
            for (int n = 0; n < HD / 8; ++n) {
                o[n][0] *= cA; o[n][1] *= cA;
                o[n][2] *= cB; o[n][3] *= cB;
            }
        }
        #pragma unroll
        for (int j = 0; j < KT / 8; ++j) {
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
                s[j][e] = nA == -INFINITY ? 0.0f : exp2f(s[j][e] - nA);
                s[j][2 + e] = nB == -INFINITY ? 0.0f : exp2f(s[j][2 + e] - nB);
                lA += s[j][e];
                lB += s[j][2 + e];
            }
        }

        #pragma unroll
        for (int kq = 0; kq < KT / 16; ++kq) {
            uint32_t a[4];
            a[0] = pack_half2(s[2 * kq][0], s[2 * kq][1]);
            a[1] = pack_half2(s[2 * kq][2], s[2 * kq][3]);
            a[2] = pack_half2(s[2 * kq + 1][0], s[2 * kq + 1][1]);
            a[3] = pack_half2(s[2 * kq + 1][2], s[2 * kq + 1][3]);
            const int key = kq * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
            #pragma unroll
            for (int n2 = 0; n2 < HD / 16; ++n2) {
                uint32_t b[4];
                ldmatrix_x4_trans(b, vs + key * ld + n2 * 16 + (lane >> 4) * 8);
                mma_f16_16816(o[2 * n2], a, b[0], b[1]);
                mma_f16_16816(o[2 * n2 + 1], a, b[2], b[3]);
            }
        }
        __syncthreads();
    }
    cp_async_wait<0>();

    #pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
        lA += __shfl_xor_sync(0xffffffff, lA, off);
        lB += __shfl_xor_sync(0xffffffff, lB, off);
    }
    const float iA = lA > 0.0f ? 1.0f / lA : 0.0f, iB = lB > 0.0f ? 1.0f / lB : 0.0f;
    float* outA = out_all + (size_t) (tq0 + tA) * q_dim + (size_t) (hkv * hpk + rA % hpk) * HD;
    float* outB = out_all + (size_t) (tq0 + tB) * q_dim + (size_t) (hkv * hpk + rB % hpk) * HD;
    if (out_h) {                                  // FP16 input of the output projection
        __half* hA = out_h + (size_t) (tq0 + tA) * q_dim + (size_t) (hkv * hpk + rA % hpk) * HD;
        __half* hB = out_h + (size_t) (tq0 + tB) * q_dim + (size_t) (hkv * hpk + rB % hpk) * HD;
        #pragma unroll
        for (int n = 0; n < HD / 8; ++n) {
            const int d = n * 8 + 2 * tq;
            if (okA) *reinterpret_cast<__half2*>(hA + d) = __floats2half2_rn(o[n][0] * iA, o[n][1] * iA);
            if (okB) *reinterpret_cast<__half2*>(hB + d) = __floats2half2_rn(o[n][2] * iB, o[n][3] * iB);
        }
        return;
    }
    #pragma unroll
    for (int n = 0; n < HD / 8; ++n) {
        const int d = n * 8 + 2 * tq;
        if (okA) *reinterpret_cast<float2*>(outA + d) = make_float2(o[n][0] * iA, o[n][1] * iA);
        if (okB) *reinterpret_cast<float2*>(outB + d) = make_float2(o[n][2] * iB, o[n][3] * iB);
    }
}

static size_t attention_prefill_fa16_smem(uint32_t hd, int nw) {
    return (size_t) (nw * 16 + 4 * kFaKeys) * (hd + 8) * sizeof(__half);
}

template <int HD>
static bool attention_prefill_fa16_launch(const float* q, const __half* kh, const __half* vh, float* out,
                                          uint32_t hpk, uint32_t n_kv, uint32_t q_dim, uint32_t pos0,
                                          uint32_t n_tok, uint32_t n_swa, float q_scale, cudaStream_t stream,
                                          __half* out_h = nullptr) {
    {
        static const bool qreg = std::getenv("DESIREEIA_CUDA_NO_FA16Q") == nullptr &&
                                 (HD <= 128 || std::getenv("DESIREEIA_CUDA_NO_FA16Q_256") == nullptr);
        if (qreg && 128 % hpk == 0) {
            constexpr size_t smem3 = (size_t) 4 * fa16q_keys(HD) * (HD + 8) * sizeof(__half);
            static const bool granted3 = cudaFuncSetAttribute(attention_prefill_fa16q_kernel<HD>,
                                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                              (int) smem3) == cudaSuccess;
            if (granted3) {
                const unsigned tpb3 = 128 / hpk;
                attention_prefill_fa16q_kernel<HD><<<dim3(n_kv, (n_tok + tpb3 - 1) / tpb3), 256, smem3, stream>>>(
                    q, kh, vh, out, hpk, n_kv, q_dim, pos0, n_tok, n_swa, q_scale, out_h);
                return true;
            }
        }
    }
    constexpr int NW = fa16_warps(HD);
    const size_t smem = attention_prefill_fa16_smem(HD, NW);
    static const bool granted = cudaFuncSetAttribute(attention_prefill_fa16_kernel<HD, NW>,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     (int) smem) == cudaSuccess;
    if (!granted || (NW * 16) % hpk != 0) return false;
    const unsigned tpb = NW * 16 / hpk;
    attention_prefill_fa16_kernel<HD, NW><<<dim3(n_kv, (n_tok + tpb - 1) / tpb), NW * 32, smem, stream>>>(
        q, kh, vh, out, hpk, n_kv, q_dim, pos0, n_tok, n_swa, q_scale, out_h);
    return true;
}

// Launches the single-pass kernel when the shape and the GPU allow it. The
// shared-memory limit is raised once per head size.
template <int HD>
static bool attention_prefill_fa_launch(const float* q, const uint8_t* kc, const uint8_t* vc, float* out,
                                        uint32_t hpk, uint32_t n_head_kv, uint32_t q_dim, size_t row_bytes,
                                        size_t pos_bytes, size_t layer_off, uint32_t pos0, uint32_t n_tok,
                                        uint32_t n_swa, float q_scale, cudaStream_t stream) {
    const size_t smem = attention_prefill_fa_smem(HD);
    static const bool granted = cudaFuncSetAttribute(attention_prefill_fa_kernel<HD>,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     (int) smem) == cudaSuccess;
    if (!granted) return false;
    const unsigned tpb = kFaRows / hpk;
    attention_prefill_fa_kernel<HD><<<dim3(n_head_kv, (n_tok + tpb - 1) / tpb), 128, smem, stream>>>(
        q, kc, vc, out, hpk, q_dim, row_bytes, pos_bytes, layer_off, pos0, n_tok, n_swa, q_scale);
    return true;
}

static bool attention_prefill_fa(const float* q, const uint8_t* kc, const uint8_t* vc, float* out,
                                 uint32_t hd, uint32_t hpk, uint32_t n_head_kv, uint32_t q_dim,
                                 size_t row_bytes, size_t pos_bytes, size_t layer_off,
                                 uint32_t pos0, uint32_t n_tok, uint32_t n_swa, float inv_d, cudaStream_t stream) {
    static const bool enabled = std::getenv("DESIREEIA_CUDA_NO_FA_ATTN") == nullptr;
    if (!enabled || cuda_sm_major() < 8 || hpk == 0 || kFaRows % hpk != 0) return false;
    const float q_scale = inv_d * 1.4426950408889634f;       // scores in log2 units
    switch (hd) {
        case 64:  return attention_prefill_fa_launch<64>(q, kc, vc, out, hpk, n_head_kv, q_dim, row_bytes, pos_bytes, layer_off, pos0, n_tok, n_swa, q_scale, stream);
        case 128: return attention_prefill_fa_launch<128>(q, kc, vc, out, hpk, n_head_kv, q_dim, row_bytes, pos_bytes, layer_off, pos0, n_tok, n_swa, q_scale, stream);
        case 256: return attention_prefill_fa_launch<256>(q, kc, vc, out, hpk, n_head_kv, q_dim, row_bytes, pos_bytes, layer_off, pos0, n_tok, n_swa, q_scale, stream);
        default:  return false;
    }
}

// Device buffers of the batched layer. x and rope cover the WHOLE prompt
// (the prefill runs layer by layer, so every token's activation must stay
// resident until the next layer); the others are per chunk.
struct LayerBatchScratch {
    float* x = nullptr;      // [n_total][n_embd]  prompt activations, live across layers
    float* rope = nullptr;   // [n_total][n_rot]   cos/sin of every prompt position
    size_t x_cap = 0, rope_cap = 0;   // in floats
    float* xn = nullptr;     // [tok][n_embd]  normalised (float)
    float* q = nullptr;      // [tok][q_dim]
    float* k = nullptr;      // [tok][kv_dim]
    float* v = nullptr;      // [tok][kv_dim]
    float* attn = nullptr;   // [tok][q_dim]
    float* y = nullptr;      // [tok][n_embd]  attention projection, then residual sum
    float* up = nullptr;     // [tok][n_ff]
    float* gate = nullptr;   // [tok][n_ff]
    float* out = nullptr;    // [tok][n_embd]
    float* gatev = nullptr;  // [tok][n_head]  per-head attention gate
    int8_t* xq = nullptr;    // [tok][max_dim] Q8_0 activation, for the dp4a fallback
    float* xs = nullptr;     // [tok][max_dim/32]
    int32_t* xsum = nullptr; // [tok][max_dim/32]
    size_t tok_cap = 0, embd_cap = 0, q_cap = 0, kv_cap = 0, ff_cap = 0, head_cap = 0;
    // FP16 K/V of the current layer for the attention (see kv_q8_to_f16_kernel):
    // K at kv16, V at kv16 + kv16_pos * kv_dim; positions [0, kv16_valid) filled.
    __half* kv16 = nullptr;
    size_t kv16_cap = 0;          // halves
    uint32_t kv16_pos = 0, kv16_valid = 0;
    float* rope_x[3] = {}; size_t rope_x_cap[3] = {};   // RoPE tables 1-3 (rope_slot)
};
LayerBatchScratch g_lb;

// Norm weights and biases of a layer, uploaded once (keyed like the decode
// graphs, by the device pointer of the layer's Wq).
struct LayerBatchConsts {
    float* bq = nullptr; float* bk = nullptr; float* bv = nullptr;
    float* attn_norm = nullptr; float* ffn_norm = nullptr;
    float* q_norm = nullptr; float* k_norm = nullptr;
    float* post_attn_norm = nullptr; float* post_ffn_norm = nullptr;
};
std::unordered_map<const void*, LayerBatchConsts> g_lb_consts;

} // namespace

// Batched dp4a mat-mul inside the layer, for matrices the tensor-core path
// does not cover: one kernel for all the tokens where the format has a
// batched kernel (Q8_0, Q4_K, Q6_K), otherwise one on-device mat-vec per
// token - still no host round trip, just more launches.
static void emit_matmul_batch(int fmt, const void* w, const void* scale,
                               const int8_t* d_xq, const float* d_xscale, const int32_t* d_xsum,
                               size_t cols, size_t rows, uint32_t n_tok, float* d_y,
                               cudaStream_t stream) {
    constexpr int kTok = 8;
    const dim3 block(DESIREEIA_CUDA_WARP, kMatmulQ80Warps);
    const dim3 grid((unsigned) ((rows + kMatmulQ80Warps - 1) / kMatmulQ80Warps),
                    (unsigned) ((n_tok + kTok - 1) / kTok));
    if (fmt == 0) {
        matmul_q8_0_batch_kernel<kMatmulQ80Warps, kTok><<<grid, block, 0, stream>>>(
            static_cast<const int8_t*>(w), static_cast<const __half*>(scale),
            d_xq, d_xscale, cols / 32, rows, n_tok, d_y);
        return;
    }
    if (fmt == DESIREEIA_CUDA_FMT_Q4_K && cols % 256 == 0) {
        matmul_q4_k_batch_kernel<kMatmulQ80Warps, kTok><<<grid, block, 0, stream>>>(
            static_cast<const uint8_t*>(w), d_xq, d_xscale, d_xsum, cols / 256, rows, n_tok, d_y);
        return;
    }
    if (fmt == DESIREEIA_CUDA_FMT_Q6_K && cols % 256 == 0) {
        matmul_q6_k_batch_kernel<kMatmulQ80Warps, kTok><<<grid, block, 0, stream>>>(
            static_cast<const uint8_t*>(w), d_xq, d_xscale, cols / 256, rows, n_tok, d_y);
        return;
    }
    const size_t nb = cols / 32;
    for (uint32_t t = 0; t < n_tok; ++t) {
        emit_matvec(fmt, w, scale, d_xq + (size_t) t * cols, d_xscale + (size_t) t * nb,
                    d_xsum + (size_t) t * nb, cols, rows, d_y + (size_t) t * rows, nullptr, stream);
    }
}

static bool lb_grow(float** p, size_t n) {
    if (*p) cudaFree(*p);
    *p = nullptr;
    return cudaMalloc(p, n * sizeof(float)) == cudaSuccess;
}

static bool lb_reserve(uint32_t n_tok, uint32_t n_total, size_t n_embd, size_t q_dim, size_t kv_dim,
                       size_t n_ff, uint32_t n_head, uint32_t n_rot) {
    auto& s = g_lb;
    // Whole-prompt buffers.
    const size_t need_x = (size_t) n_total * n_embd;
    if (need_x > s.x_cap) {
        if (!lb_grow(&s.x, need_x)) { s.x_cap = 0; return false; }
        s.x_cap = need_x;
    }
    const size_t need_rope = (size_t) n_total * std::max<uint32_t>(n_rot, 1);
    if (need_rope > s.rope_cap) {
        if (!lb_grow(&s.rope, need_rope)) { s.rope_cap = 0; return false; }
        s.rope_cap = need_rope;
    }
    // Per-chunk buffers.
    if ((n_tok + kGemmBM - 1) / kGemmBM * kGemmBM <= s.tok_cap && n_embd <= s.embd_cap && q_dim <= s.q_cap && kv_dim <= s.kv_cap &&
        n_ff <= s.ff_cap && n_head <= s.head_cap) {
        return true;
    }
    // Whole GEMM tiles of tokens: the tensor-core product writes the
    // padding rows of the last tile straight into these buffers.
    const size_t t = std::max<size_t>((n_tok + kGemmBM - 1) / kGemmBM * kGemmBM, s.tok_cap);
    const size_t e = std::max(n_embd, s.embd_cap), qd = std::max(q_dim, s.q_cap);
    const size_t kd = std::max(kv_dim, s.kv_cap), ff = std::max(n_ff, s.ff_cap);
    const size_t hd = std::max<size_t>(n_head, s.head_cap);
    const size_t maxd = std::max({e, qd, ff});
    bool ok = lb_grow(&s.xn, t * e) && lb_grow(&s.q, t * qd) &&
              lb_grow(&s.k, t * kd) && lb_grow(&s.v, t * kd) && lb_grow(&s.attn, t * qd) &&
              lb_grow(&s.y, t * e) && lb_grow(&s.up, t * ff) && lb_grow(&s.gate, t * ff) &&
              lb_grow(&s.out, t * e) && lb_grow(&s.gatev, t * hd) && lb_grow(&s.xs, t * (maxd / 32));
    if (ok) {
        if (s.xq) cudaFree(s.xq);
        if (s.xsum) cudaFree(s.xsum);
        s.xq = nullptr; s.xsum = nullptr;
        ok = cudaMalloc(&s.xq, t * maxd) == cudaSuccess &&
             cudaMalloc(&s.xsum, t * (maxd / 32) * sizeof(int32_t)) == cudaSuccess;
    }
    if (!ok) {
        s.tok_cap = s.embd_cap = s.q_cap = s.kv_cap = s.ff_cap = s.head_cap = 0;
        return false;
    }
    s.tok_cap = t; s.embd_cap = e; s.q_cap = qd; s.kv_cap = kd; s.ff_cap = ff; s.head_cap = hd;
    return true;
}

// Frees the per-layer constants (always) and, when asked, every prefill
// buffer too: called by invalidate_graphs and cuda_backend_shutdown, the
// same two points that release everything else on the device.
namespace {
void layer_batch_release(bool free_buffers) {
    for (auto& kv : g_lb_consts) {
        auto& c = kv.second;
        for (float* p : { c.bq, c.bk, c.bv, c.attn_norm, c.ffn_norm, c.q_norm, c.k_norm,
                          c.post_attn_norm, c.post_ffn_norm }) {
            if (p) cudaFree(p);
        }
    }
    g_lb_consts.clear();
    for (int i = 0; i < kTcSlots; ++i) g_tc.ready[i] = false;
    g_lb.kv16_valid = 0;          // the KV cache it mirrors may have moved
    if (!free_buffers) return;
    auto& s = g_lb;
    if (s.kv16) cudaFree(s.kv16);
    for (float* p : s.rope_x) if (p) cudaFree(p);
    for (float* p : { s.x, s.rope, s.xn, s.q, s.k, s.v, s.attn, s.y, s.up, s.gate, s.out, s.gatev, s.xs }) {
        if (p) cudaFree(p);
    }
    if (s.xq) cudaFree(s.xq);
    if (s.xsum) cudaFree(s.xsum);
    s = LayerBatchScratch{};
    tensor_core_release();
    hkq_prep_clear();             // a release asks for the memory back
}
} // namespace

// Tokens per chunk of the layer-by-layer prefill: 64 per SM (so a chunk is
// half as many 128-token tiles as the GPU has SMs, and the common product
// shapes fill whole waves), between 1024 and 2048. 20 SMs -> 1280.
int cuda_prefill_chunk_tokens() {
    const int n = 64 * g_sm_count();
    return n < 1024 ? 1024 : (n > 2048 ? 2048 : n);
}

// KV rows of the current prompt still to be mirrored to the host cache.
struct LbMirror { uint8_t* k; uint8_t* v; size_t off; size_t bytes; };
std::vector<LbMirror> g_lb_mirror;

int cuda_layer_forward_batch(const CudaLayerBatchArgs& a) {
    static const bool enabled = std::getenv("DESIREEIA_CUDA_NO_PREFILL_LAYER") == nullptr;
    if (!enabled) return DESIREEIA_ERR_NOT_SUPPORTED;
    if (!g_scratch.d_kcache || a.n_tok == 0 || a.n_total == 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    if (a.n_embd % 32 != 0 || a.q_dim % 32 != 0 || a.n_ff % 32 != 0 || a.head_dim % 32 != 0) {
        return DESIREEIA_ERR_NOT_SUPPORTED;
    }
    constexpr int athr = 128;
    if (a.head_dim > athr * DESIREEIA_ATTN_MAX_ACC) return DESIREEIA_ERR_NOT_SUPPORTED;

    ScopedTimer prof_t(profile_counters().ns_cuda_attn);
    profile_counters().calls_cuda_attn.fetch_add(a.n_tok, std::memory_order_relaxed);

    const uint32_t n_tok = a.n_tok;
    const uint32_t nblk = a.head_dim / 32;
    const size_t kv_row_bytes = (size_t) nblk * 34;
    const size_t pos_bytes = (size_t) a.n_head_kv * kv_row_bytes;
    if (!lb_reserve(n_tok, a.n_total, a.n_embd, a.q_dim, a.kv_dim, a.n_ff, a.n_head, a.n_rot)) {
        return DESIREEIA_ERR_NO_MEM;
    }
    auto& s = g_lb;

    LayerBatchConsts& c = g_lb_consts[a.wq_qs];
    if (!c.attn_norm) {
        auto upload = [](const float* src, size_t n, float** dst) {
            if (!src) return true;
            if (cudaMalloc(dst, n * sizeof(float)) != cudaSuccess) return false;
            return cudaMemcpy(*dst, src, n * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
        };
        if (!upload(a.bq, a.q_dim, &c.bq) || !upload(a.bk, a.kv_dim, &c.bk) ||
            !upload(a.bv, a.kv_dim, &c.bv) || !upload(a.attn_norm_w, a.n_embd, &c.attn_norm) ||
            !upload(a.ffn_norm_w, a.n_embd, &c.ffn_norm) || !upload(a.q_norm_w, a.head_dim, &c.q_norm) ||
            !upload(a.k_norm_w, a.head_dim, &c.k_norm) ||
            !upload(a.post_attn_norm_w, a.n_embd, &c.post_attn_norm) ||
            !upload(a.post_ffn_norm_w, a.n_embd, &c.post_ffn_norm)) {
            return DESIREEIA_ERR_IO;
        }
    }

    cudaStream_t stream = scratch_stream();
    if (a.upload_x) {
        g_lb_mirror.clear();                   // left over by a prompt that stopped midway
        cudaMemcpyAsync(s.x, a.x, (size_t) a.n_total * a.n_embd * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    }
    // RoPE table of this layer: slot 0 is s.rope, slots 1-3 the extra tables
    // of models that alternate bases.
    float* rope_base = s.rope;
    if (a.rope_caches && a.rope_slot > 0 && a.rope_slot < 4) {
        const int k = a.rope_slot - 1;
        const size_t need = (size_t) a.n_total * std::max<uint32_t>(a.n_rot, 1);
        if (need > s.rope_x_cap[k]) {
            if (s.rope_x[k]) cudaFree(s.rope_x[k]);
            s.rope_x[k] = nullptr; s.rope_x_cap[k] = 0;
            if (cudaMalloc(&s.rope_x[k], need * sizeof(float)) != cudaSuccess) return DESIREEIA_ERR_NO_MEM;
            s.rope_x_cap[k] = need;
        }
        rope_base = s.rope_x[k];
    }
    if (a.rope_caches && a.upload_rope) {
        cudaMemcpyAsync(rope_base, a.rope_caches, (size_t) a.n_total * a.n_rot * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    }

    // The layer's matrices as FP16, once per prompt (first chunk of the layer).
    enum { kQ, kK, kV, kO, kGate, kUp, kDown, kAg };
    if (a.prepare_weights) {
        tc_prepare(kQ, a.fmt_q, a.wq_qs, a.wq_scale, a.q_dim, a.n_embd, stream);
        tc_prepare(kK, a.fmt_k, a.wk_qs, a.wk_scale, a.kv_dim, a.n_embd, stream);
        tc_prepare(kV, a.fmt_v, a.wv_qs, a.wv_scale, a.kv_dim, a.n_embd, stream);
        tc_prepare(kO, a.fmt_o, a.wo_qs, a.wo_scale, a.n_embd, a.q_dim, stream);
        tc_prepare(kGate, a.fmt_gate, a.wgate_qs, a.wgate_scale, a.n_ff, a.n_embd, stream);
        tc_prepare(kUp, a.fmt_up, a.wup_qs, a.wup_scale, a.n_ff, a.n_embd, stream);
        tc_prepare(kDown, a.fmt_down, a.wdown_qs, a.wdown_scale, a.n_embd, a.n_ff, stream);
        if (a.wag_qs) tc_prepare(kAg, a.fmt_ag, a.wag_qs, a.wag_scale, a.n_head, a.n_embd, stream);
    }

    float* x = s.x + (size_t) a.x_row0 * a.n_embd;          // this chunk's rows
    const float* rope = rope_base + (size_t) a.x_row0 * a.n_rot;
    const int nthr = 256;
    const size_t nb_attn = a.q_dim / 32;
    const size_t nb_h = a.n_ff / 32;

    // Every mat-mul: tensor cores from the float activation when the matrix
    // was expanded, otherwise the dp4a batched kernels on its Q8_0 form
    // (both forms are produced by the kernel that feeds the product).
    // The FP16 form of an activation is reused by the next product only
    // when both read the same input and the previous one ran on the tensor
    // cores (a dp4a fallback in between leaves nothing to reuse).
    // fp16_valid: g_tc holds the FP16 form of the current input.
    // xq_valid: s.xq/xs/xsum hold its Q8_0 form (made by its producer, or on
    // demand here when a product has to fall back to the dp4a kernels).
    bool fp16_valid = false, xq_valid = false;
    auto mm = [&](int slot, int fmt, const void* w, const void* sc, const float* xf,
                  size_t cols, size_t rows, float* y, bool same_input,
                  const float* bias = nullptr, const float* resid = nullptr) {
        const bool reuse = same_input && fp16_valid;
        if (tc_gemm(slot, xf, cols, rows, n_tok, y, reuse, stream, xq_valid ? s.xq : nullptr, s.xs, s.xsum,
                    bias, resid)) {
            fp16_valid = true;
            return;
        }
        fp16_valid = false;
        if (!xq_valid) {
            quantize_q8_0_kernel<<<(unsigned) (n_tok * (cols / 32)), 32, 0, stream>>>(
                xf, (size_t) n_tok * cols, s.xq, s.xs, s.xsum);
            xq_valid = true;
        }
        emit_matmul_batch(fmt, w, sc, s.xq, s.xs, s.xsum, cols, rows, n_tok, y, stream);
        if (bias || resid) {
            const size_t n = (size_t) n_tok * rows;
            gemm_epi_kernel<<<(unsigned) ((n + nthr - 1) / nthr), nthr, 0, stream>>>(y, bias, resid, rows, n);
        }
    };
    // Every consumer of a norm on the FP16 tensor cores: the norm writes
    // their FP16 input itself.
    const uint32_t m_pad16 = (n_tok + kGemmBM - 1) / kGemmBM * kGemmBM;
    static const bool norm_f16 = std::getenv("DESIREEIA_CUDA_NO_NORM_F16") == nullptr;
    auto f16_slot = [](int slot) { return norm_f16 && g_tc.ready[slot] && !g_tc.is_i8[slot]; };
    auto f16_input = [&](size_t cols) {
        return tc_grow((void**) &g_tc.a, g_tc.a_cap, (size_t) m_pad16 * cols * sizeof(__half)) &&
               tc_grow((void**) &g_tc.scale, g_tc.scale_cap, (size_t) m_pad16 * sizeof(float));
    };

    const bool attn_f16 = f16_slot(kQ) && f16_slot(kK) && f16_slot(kV) && (!a.wag_qs || f16_slot(kAg)) &&
                          f16_input(a.n_embd);
    if (attn_f16) {
        rms_norm_f16_kernel<false><<<m_pad16, nthr, nthr * sizeof(float), stream>>>(
            x, nullptr, c.attn_norm, s.xn, g_tc.a, g_tc.scale, (uint32_t) a.n_embd, a.rms_eps, n_tok);
        g_tc.f16_in = true;
        g_tc.i8_in = false;
    } else {
        rms_norm_quant_kernel<false><<<n_tok, nthr, nthr * sizeof(float), stream>>>(
            x, nullptr, c.attn_norm, s.xn, s.xq, s.xs, s.xsum, (uint32_t) a.n_embd, a.rms_eps);
    }
    fp16_valid = attn_f16;
    xq_valid = !attn_f16;

    // Q, K and V in one launch when they share a stored format (or Q and K
    // when only V differs): the K/V projections alone fill a fraction of a
    // wave on short prompts.
    int grouped = 0;
    if (attn_f16 && g_tc.w_hkq[kQ] && g_tc.w_hkq[kQ] == g_tc.w_hkq[kK]) {
        const bool with_v = g_tc.w_hkq[kV] == g_tc.w_hkq[kQ];
        const int slots[3] = { kQ, kK, kV };
        const size_t rws[3] = { a.q_dim, a.kv_dim, a.kv_dim };
        float* const ys[3] = { s.q, s.k, s.v };
        const float* const bs[3] = { c.bq, c.bk, c.bv };
        if (tc_gemm_group(with_v ? 3 : 2, slots, rws, ys, bs, a.n_embd, n_tok, stream)) grouped = with_v ? 3 : 2;
    }
    if (grouped < 1) mm(kQ, a.fmt_q, a.wq_qs, a.wq_scale, s.xn, a.n_embd, a.q_dim, s.q, attn_f16, c.bq);
    if (grouped < 2) mm(kK, a.fmt_k, a.wk_qs, a.wk_scale, s.xn, a.n_embd, a.kv_dim, s.k, true, c.bk);
    if (grouped < 3) mm(kV, a.fmt_v, a.wv_qs, a.wv_scale, s.xn, a.n_embd, a.kv_dim, s.v, true, c.bv);
    if (a.wag_qs) mm(kAg, a.fmt_ag, a.wag_qs, a.wag_scale, s.xn, a.n_embd, a.n_head, s.gatev, true);

    if (c.q_norm) {
        rms_norm_heads_kernel<<<n_tok * a.n_head, 64, 64 * sizeof(float), stream>>>(
            s.q, c.q_norm, a.head_dim, a.rms_eps);
    }
    if (c.k_norm) {
        rms_norm_heads_kernel<<<n_tok * a.n_head_kv, 64, 64 * sizeof(float), stream>>>(
            s.k, c.k_norm, a.head_dim, a.rms_eps);
    }
    if (a.rope_caches) {
        rope_neox_batch_kernel<<<dim3(a.n_head + a.n_head_kv, n_tok), 64, 0, stream>>>(
            s.q, s.k, rope, a.n_rot, a.head_dim, a.n_head, a.q_dim, a.kv_dim);
    }

    const unsigned kvgrid = (unsigned) (a.n_head_kv * nblk);
    kv_quantize_batch_kernel<<<dim3(kvgrid * 2, n_tok), 32, 0, stream>>>(
        s.k, g_scratch.d_kcache + a.kv_layer_off, s.v, g_scratch.d_vcache + a.kv_layer_off,
        a.head_dim, kv_row_bytes, kvgrid, pos_bytes, a.kv_dim, a.pos0);

    __half* attn_out_h = nullptr;         // set when the attention writes the O input in FP16
    {
        static const bool tiled = std::getenv("DESIREEIA_CUDA_NO_TILED_ATTN") == nullptr;
        static const bool tc_attn = std::getenv("DESIREEIA_CUDA_NO_TC_ATTN") == nullptr;
        const float inv_d = 1.0f / sqrtf((float) a.head_dim);
        const size_t tc_smem = attention_prefill_tc_smem(a.head_dim);
        // The tensor-core kernel needs more than the default 48 KB of shared
        // memory for wide heads: ask once, fall back if the device refuses.
        static int tc_smem_granted = -1;
        if (tc_attn && tc_smem_granted < 0) {
            tc_smem_granted = cudaFuncSetAttribute(attention_prefill_tc_kernel,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   (int) attention_prefill_smem_cap()) == cudaSuccess ? 1 : 0;
        }
        static const bool gqa_attn = std::getenv("DESIREEIA_CUDA_NO_GQA_ATTN") == nullptr;
        static int gqa_smem_granted = -1;
        if (gqa_attn && gqa_smem_granted < 0) {
            gqa_smem_granted = cudaFuncSetAttribute(attention_prefill_gqa_kernel,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                    (int) attention_prefill_gqa_smem(256)) == cudaSuccess ? 1 : 0;
        }
        const uint32_t hpk = a.heads_per_kv;
        // FP16 K/V for this layer, converted once (positions not yet converted
        // for this layer: all of them on its first chunk).
        static const bool fa16_on = std::getenv("DESIREEIA_CUDA_NO_FA16_KV") == nullptr;
        bool fa16_ready = false;
        if (fa16_on && cuda_sm_major() >= 8 && hpk > 0 && kFaRows % hpk == 0 &&
            (a.head_dim == 64 || a.head_dim == 128 || a.head_dim == 256)) {
            const uint32_t end = a.pos0 + n_tok;
            const uint32_t prompt_end = a.pos0 - (uint32_t) a.x_row0 + (uint32_t) a.n_total;
            const size_t need = (size_t) 2 * prompt_end * a.kv_dim;
            if (a.prepare_weights || s.kv16_pos < prompt_end) s.kv16_valid = 0;
            if (need > s.kv16_cap) {
                if (s.kv16) cudaFree(s.kv16);
                s.kv16 = nullptr; s.kv16_cap = 0;
                if (cudaMalloc(&s.kv16, need * sizeof(__half)) == cudaSuccess) s.kv16_cap = need;
                s.kv16_valid = 0;
            }
            if (s.kv16) {
                s.kv16_pos = prompt_end;
                __half* kh = s.kv16;
                __half* vh = s.kv16 + (size_t) prompt_end * a.kv_dim;
                if (s.kv16_valid < end) {
                    const size_t work = (size_t) (end - s.kv16_valid) * a.n_head_kv * (a.head_dim / 32);
                    kv_q8_to_f16_kernel<<<(unsigned) ((work + 255) / 256), 256, 0, stream>>>(
                        g_scratch.d_kcache, g_scratch.d_vcache, a.kv_layer_off, pos_bytes, kv_row_bytes,
                        a.n_head_kv, a.head_dim, s.kv16_valid, end, kh, vh);
                    s.kv16_valid = end;
                }
                const float q_scale = inv_d * 1.4426950408889634f;
                // The output projection on the stored-block FP16 kernel takes the
                // attention output as FP16: written directly (a weighted mean of
                // FP16 V values, it cannot overflow), no float copy and no
                // conversion pass. Not with the per-head gate, which scales it.
                static const bool attn_h = std::getenv("DESIREEIA_CUDA_NO_ATTN_F16_OUT") == nullptr;
                if (attn_h && !a.wag_qs && g_tc.ready[kO] && g_tc.w_hkq[kO] && f16_input(a.q_dim)) attn_out_h = g_tc.a;
                switch (a.head_dim) {
                    case 64:  fa16_ready = attention_prefill_fa16_launch<64>(s.q, kh, vh, s.attn, hpk, a.n_head_kv, (uint32_t) a.q_dim, a.pos0, n_tok, a.n_swa, q_scale, stream, attn_out_h); break;
                    case 128: fa16_ready = attention_prefill_fa16_launch<128>(s.q, kh, vh, s.attn, hpk, a.n_head_kv, (uint32_t) a.q_dim, a.pos0, n_tok, a.n_swa, q_scale, stream, attn_out_h); break;
                    default:  fa16_ready = attention_prefill_fa16_launch<256>(s.q, kh, vh, s.attn, hpk, a.n_head_kv, (uint32_t) a.q_dim, a.pos0, n_tok, a.n_swa, q_scale, stream, attn_out_h); break;
                }
                if (!fa16_ready) attn_out_h = nullptr;
            }
        }
        if (fa16_ready) {
            // single-pass kernel on the FP16 K/V ran
        } else if (attention_prefill_fa(s.q, g_scratch.d_kcache, g_scratch.d_vcache, s.attn, a.head_dim, hpk,
                                 a.n_head_kv, (uint32_t) a.q_dim, kv_row_bytes, pos_bytes, a.kv_layer_off,
                                 a.pos0, n_tok, a.n_swa, inv_d, stream)) {
            // single-pass kernel ran
        } else if (gqa_attn && gqa_smem_granted == 1 && a.head_dim % 32 == 0 && a.head_dim <= 256 &&
            hpk > 0 && hpk <= (uint32_t) kGqaRows && kGqaRows % hpk == 0) {
            const uint32_t tpb = kGqaRows / hpk;
            attention_prefill_gqa_kernel<<<dim3(a.n_head_kv, (n_tok + tpb - 1) / tpb), 256,
                                           attention_prefill_gqa_smem(a.head_dim), stream>>>(
                s.q, g_scratch.d_kcache, g_scratch.d_vcache, s.attn,
                a.head_dim, hpk, (uint32_t) a.q_dim, kv_row_bytes, pos_bytes, a.kv_layer_off,
                a.pos0, n_tok, a.n_swa, inv_d);
        } else if (tc_attn && tc_smem_granted == 1 && a.head_dim % 16 == 0 && a.head_dim <= 256 &&
            tc_smem <= attention_prefill_smem_cap()) {
            attention_prefill_tc_kernel<<<dim3(a.n_head, (n_tok + kTcaQ - 1) / kTcaQ), 128, tc_smem, stream>>>(
                s.q, g_scratch.d_kcache, g_scratch.d_vcache, s.attn,
                a.head_dim, a.heads_per_kv, (uint32_t) a.q_dim, kv_row_bytes, pos_bytes, a.kv_layer_off,
                a.pos0, n_tok, a.n_swa, inv_d);
        } else if (tiled && a.head_dim <= 256) {
            attention_prefill_q8_kernel<<<dim3(a.n_head, (n_tok + kFaTQ - 1) / kFaTQ), 256,
                                          attention_prefill_smem(a.head_dim), stream>>>(
                s.q, g_scratch.d_kcache, g_scratch.d_vcache, s.attn,
                a.head_dim, a.heads_per_kv, (uint32_t) a.q_dim, kv_row_bytes, pos_bytes, a.kv_layer_off,
                a.pos0, n_tok, a.n_swa, inv_d);
        } else {
            attention_batch_kernel_q8<<<dim3(a.n_head, n_tok), athr, 2 * athr * sizeof(float), stream>>>(
                s.q, g_scratch.d_kcache, g_scratch.d_vcache, s.attn,
                a.head_dim, a.heads_per_kv, (uint32_t) a.q_dim, kv_row_bytes, pos_bytes, a.kv_layer_off,
                a.pos0, a.n_swa, inv_d);
        }
    }

    if (a.wag_qs) {
        attn_gate_apply_kernel<<<n_tok * a.n_head, 64, 0, stream>>>(s.attn, s.gatev, a.head_dim);
    }

    // The int8 copy of the attention output only feeds the dp4a fallback of
    // the output projection: skip it when that projection runs on the
    // tensor cores (it reads the float values).
    if (attn_out_h) {
        // the FP16 input is complete once its padding rows are zero and its
        // row scales are 1 (the attention output needs no scaling)
        f16_rows_finish_kernel<<<m_pad16, nthr, 0, stream>>>(g_tc.a, g_tc.scale, a.q_dim, n_tok);
        g_tc.f16_in = true;
        g_tc.i8_in = false;
        fp16_valid = true;
        xq_valid = false;
    } else {
        xq_valid = !g_tc.ready[kO] || g_tc.is_i8[kO];
        if (xq_valid) {
            quantize_q8_0_kernel<<<(unsigned) (n_tok * nb_attn), 32, 0, stream>>>(
                s.attn, (size_t) n_tok * a.q_dim, s.xq, s.xs, s.xsum);
        }
    }
    mm(kO, a.fmt_o, a.wo_qs, a.wo_scale, s.attn, a.q_dim, a.n_embd, s.y, attn_out_h != nullptr);
    if (c.post_attn_norm) {
        rms_norm_kernel<<<n_tok, nthr, nthr * sizeof(float), stream>>>(
            s.y, c.post_attn_norm, s.y, (uint32_t) a.n_embd, a.rms_eps);
    }
    // y += x (attention residual), then FFN norm (and its FP16 or Q8_0 form).
    const bool ffn_f16 = f16_slot(kUp) && f16_slot(kGate) && f16_input(a.n_embd);
    if (ffn_f16) {
        rms_norm_f16_kernel<true><<<m_pad16, nthr, nthr * sizeof(float), stream>>>(
            s.y, x, c.ffn_norm, s.xn, g_tc.a, g_tc.scale, (uint32_t) a.n_embd, a.rms_eps, n_tok);
        g_tc.f16_in = true;
        g_tc.i8_in = false;
    } else {
        rms_norm_quant_kernel<true><<<n_tok, nthr, nthr * sizeof(float), stream>>>(
            s.y, x, c.ffn_norm, s.xn, s.xq, s.xs, s.xsum, (uint32_t) a.n_embd, a.rms_eps);
    }
    fp16_valid = ffn_f16;
    xq_valid = !ffn_f16;

    // up and gate in one product when they share a stored format: s.up
    // receives act(gate) * up directly.
    const bool gated = ffn_f16 && tc_gemm_gated(kUp, kGate, a.n_embd, a.n_ff, n_tok, s.up, a.act_gelu, stream);
    if (!gated) {
        mm(kUp, a.fmt_up, a.wup_qs, a.wup_scale, s.xn, a.n_embd, a.n_ff, s.up, ffn_f16);
        mm(kGate, a.fmt_gate, a.wgate_qs, a.wgate_scale, s.xn, a.n_embd, a.n_ff, s.gate, true);
    }
    // Without a post-FFN norm the down projection adds the residual itself
    // and writes the layer output straight into x.
    static const bool resid_epi = std::getenv("DESIREEIA_CUDA_NO_RESID_EPI") == nullptr;
    const bool fuse_resid = resid_epi && !c.post_ffn_norm;
    float* const down_out = fuse_resid ? x : s.out;
    const float* const down_resid = fuse_resid ? s.y : nullptr;
    // Down projection on the FP16 tensor cores: the activation goes straight
    // to its FP16 input (no float result, no int8 copy). Otherwise, or if
    // that product fails, the float activation and its Q8_0 form as before.
    bool down_done = false;
    if (gated) {
        // s.up is the activation (float): the product converts it itself,
        // or quantizes it for the dp4a fallback.
        fp16_valid = false;
        xq_valid = false;
        mm(kDown, a.fmt_down, a.wdown_qs, a.wdown_scale, s.up, a.n_ff, a.n_embd, down_out, false,
           nullptr, down_resid);
        down_done = true;
    } else if (g_tc.ready[kDown] && !g_tc.is_i8[kDown] && a.n_ff % 2 == 0) {
        const uint32_t m_pad = (n_tok + kGemmBM - 1) / kGemmBM * kGemmBM;
        if (tc_grow((void**) &g_tc.a, g_tc.a_cap, (size_t) m_pad * a.n_ff * sizeof(__half)) &&
            tc_grow((void**) &g_tc.scale, g_tc.scale_cap, (size_t) m_pad * sizeof(float))) {
            ffn_act_f16_kernel<<<m_pad, 256, 0, stream>>>(s.up, s.gate, a.n_ff, n_tok, a.act_gelu, g_tc.a, g_tc.scale);
            g_tc.f16_in = true;
            g_tc.i8_in = false;
            down_done = tc_gemm(kDown, s.up, a.n_ff, a.n_embd, n_tok, down_out, true, stream,
                                nullptr, nullptr, nullptr, nullptr, down_resid);
        }
    }
    if (!down_done) {
        ffn_act_quant_kernel<<<(unsigned) (n_tok * nb_h), 32, 0, stream>>>(
            s.up, s.gate, s.xq, s.xs, s.xsum, (size_t) n_tok * a.n_ff, a.act_gelu);
        xq_valid = true;
        mm(kDown, a.fmt_down, a.wdown_qs, a.wdown_scale, s.up, a.n_ff, a.n_embd, down_out, false,
           nullptr, down_resid);
    }
    if (!fuse_resid) {
        if (c.post_ffn_norm) {
            post_norm_add_rows_kernel<<<n_tok, nthr, nthr * sizeof(float), stream>>>(
                s.out, c.post_ffn_norm, x, s.y, (uint32_t) a.n_embd, a.rms_eps);
        } else {
            const size_t n = (size_t) n_tok * a.n_embd;
            add2_kernel<<<(unsigned) ((n + nthr - 1) / nthr), nthr, 0, stream>>>(x, s.out, s.y, (uint32_t) n);
        }
    }
    if (cudaGetLastError() != cudaSuccess) return DESIREEIA_ERR_IO;

    // Host mirror of the KV rows just written: the device cache is the one
    // attention reads, but host paths (CPU attention for a later short
    // prompt, cache growth) read the host copy - it must not go stale.
    // A copy into pageable host memory blocks the caller until the GPU gets
    // there, which left the GPU idle while the next layer was being queued:
    // the copies are collected and made after the last layer, with the
    // prompt's single synchronisation (where errors also surface).
    if (a.host_kcache && a.host_vcache) {
        const size_t off = a.kv_layer_off + (size_t) a.pos0 * pos_bytes;
        g_lb_mirror.push_back({ static_cast<uint8_t*>(a.host_kcache) + off,
                                static_cast<uint8_t*>(a.host_vcache) + off, off, (size_t) n_tok * pos_bytes });
    }
    if (!a.download_x) return DESIREEIA_OK;
    for (const LbMirror& mc : g_lb_mirror) {
        cudaMemcpyAsync(mc.k, g_scratch.d_kcache + mc.off, mc.bytes, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(mc.v, g_scratch.d_vcache + mc.off, mc.bytes, cudaMemcpyDeviceToHost, stream);
    }
    g_lb_mirror.clear();
    {
        const size_t r0 = a.x_out_row0 < a.n_total ? a.x_out_row0 : 0;
        cudaMemcpyAsync(a.x_out + r0 * a.n_embd, s.x + r0 * a.n_embd, (size_t) (a.n_total - r0) * a.n_embd * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) return DESIREEIA_ERR_IO;
    return DESIREEIA_OK;
}

} // namespace desireeia
