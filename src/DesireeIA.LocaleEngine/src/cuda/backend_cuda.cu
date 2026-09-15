// Backend CUDA — vedi docs/CUDAPiano.md.
//
// Un solo kernel, per il formato Q8_0 (il piu' semplice: nessun nibble da
// spacchettare, i pesi sono gia' int8). Porta 1:1 la stessa matematica gia'
// validata su CPU in matmul_q8_0 (src/core/matmul.cpp): per ogni riga,
// somma su tutti i sotto-blocchi da 32 elementi di
// (scala_peso[riga,blocco] * scala_attivazione[blocco] * dot_int8(pesi, attivazione)).
// Nessun algoritmo nuovo, nessuna scelta di formato inventata — solo un
// porting su device di un kernel gia' corretto e misurato.
//
// STORIA (2026-09-14): la prima versione di questo file (matmul_q8_0_cuda,
// sotto) faceva upload dell'INTERA matrice pesi ad ogni singola chiamata
// (cudaMalloc + memcpy H2D + kernel + memcpy D2H + cudaFree per matvec).
// Misurato end-to-end contro il path CPU (desireeia-cli bench, Qwen2.5-
// Coder-3B Q8_0): 28x PIU' LENTO della CPU (0.49 vs 13.90 tok/s decode),
// causa isolata col profiler (quasi tutto il tempo in
// "seriale_fra_dispatch", non nel kernel stesso: overhead di trasferimento
// PCIe + malloc/free sincrono ripetuti a ogni token, per ogni matrice
// pesi). Aggiunte qui le funzioni "resident" (upload dei pesi UNA VOLTA,
// riusati per tutta la sessione): stesso principio della cache pesi
// lato CPU (LayerWeights/layer_cache_ in dense_forward.cpp), il pezzo
// mancante gia' previsto in docs/CUDAPiano.md sezione 4.

#include "../core/engine.h"
#include "../core/profile.h"
#include "../quant/quant.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <vector>
#include <unordered_map>

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
    // Fallback per architetture senza DP4A: stessa aritmetica, piu' lenta.
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

// Attivazione della FFN gated, fatta sul device per non dover riportare
// gate e up sull'host. Replica ESATTAMENTE le due varianti CPU di
// dense_forward.cpp: `ffn[i] = silu(gate[i]) * ffn[i]` e geglu_inplace
// (gelu tanh-approssimata sul ramo gate, poi prodotto). h finisce in
// `up_inout`, come la versione CPU che scrive in place su `ffn`.
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

// Quantizzazione Q8_0 dell'attivazione sul device: stessa formula di
// quantize_q8_0 (core/quant.cpp) — per blocco di 32, d = max|v|/127 (0 se
// tutto zero), q = clamp(rint(v/d), -127, 127), coda azzerata. Un warp per
// blocco, absmax via shuffle. Serve perche' l'intermedio della FFN nasce
// gia' su device: riportarlo sull'host solo per quantizzarlo vanificherebbe
// tutto il guadagno.
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

// Attenzione causale con GQA, un blocco per testa.
//
// Replica esattamente il ramo float (non kv_quantized_, senza ALiBi) di
// dense_forward.cpp: per ogni testa h, hkv = h / (n_head/n_head_kv);
//   scores[cc] = dot(q_h, k[cc][hkv]) * inv_d   per cc in [cc_start, pos]
//   softmax stabile sulla sola finestra [cc_start, pos]
//   out[d]    = somma_cc scores[cc] * v[cc][hkv][d]
// Il layout della KV cache e' lo stesso dell'host:
//   k[(layer*capacity + cc)*kv_dim + hkv*head_dim + d]
// cosi' la copia device e' un mirror bit-per-bit di quella host e le due
// non possono divergere nel layout.
//
// Un solo kernel invece di tre (punteggi / softmax / accumulo V) per non
// pagare tre lanci: le fasi sono separate da __syncthreads.
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
            const uint8_t* b = wr[r] + sblk * 144;
            uint8_t scv[2], mnv[2];
            get_scales_pair_k4(j / 2, b + 4, scv, mnv);
            const int* qs = reinterpret_cast<const int*>(b + 16) + qoff;
            int dot = 0;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                dot = desireeia_dp4a((qs[k] >> shift) & 0x0F0F0F0F, av[k], dot);
            }
            const float d    = __half2float(*reinterpret_cast<const __half*>(b));
            const float dmin = __half2float(*reinterpret_cast<const __half*>(b + 2));
            acc[r] += xs * (d * (float) scv[j & 1] * (float) dot -
                            dmin * (float) mnv[j & 1] * xsum);
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
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int vl = load_int_b2(ql, k);
                const int vh = load_int_b2(qh, k);
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
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int vl = load_int_b2(ql, k);
                const int vh = load_int_b2(qh, k);
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

// Lancio condiviso dai due entry point (resident e per-chiamata): stessa
// geometria, cosi' una modifica alla griglia vale per entrambi.
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
    // Ogni warp copre 2 righe (vedi il kernel), quindi un blocco ne copre
    // nwarps*2.
    const dim3 block(DESIREEIA_CUDA_WARP, kMatmulQ80Warps);
    const size_t rows_per_block = (size_t) kMatmulQ80Warps * 2;
    const unsigned grid = (unsigned) ((rows + rows_per_block - 1) / rows_per_block);
    matmul_q8_0_kernel<kMatmulQ80Warps><<<grid, block, 0, stream>>>(
        d_w_qs, d_w_scale, d_x_qs, d_x_scale, nb, rows, d_y, d_bias);
}

// Stream CUDA persistente + buffer di scratch device riutilizzati fra le
// chiamate.
//
// Perche': con i pesi gia' residenti su VRAM, la misura (vedi CUDAPiano.md)
// mostrava che il tempo di decode restava dominato NON dal kernel
// (in_dispatch=38ms su 576 chiamate) ma dall'overhead attorno: un
// cudaMalloc + cudaFree per i buffer di attivazione/output ad OGNI matvec
// (~36 per token) piu' il costo di submission sullo stream di default.
// Allocare una volta e riusare elimina il malloc/free per chiamata; uno
// stream esplicito evita la sincronizzazione implicita del null stream
// e permette memcpy asincroni ordinati con il kernel, con un solo punto
// di sincronizzazione a fine chiamata (il chiamante si aspetta y gia'
// pronto al ritorno).
//
// Thread-safety: lo stesso assunto gia' documentato per g_active_backend
// in dense_forward.cpp — il forward chiama i matvec in sequenza da un
// solo thread (il parallelismo CPU sta DENTRO i kernel CPU, non attorno
// a questi dispatch). Se in futuro piu' thread dovessero lanciare kernel
// CUDA insieme, questo stato va reso per-thread o protetto.
struct CudaScratch {
    cudaStream_t stream = nullptr;
    bool stream_ready = false;
    // [nb*32 byte di xq][nb float di xscale], una sola allocazione.
    // ATTENZIONE: le due viste dentro d_stage NON si memorizzano qui.
    // I buffer crescono e basta (nb_cap >= nb), quindi l'offset delle scale
    // dipende dall'nb DELLA CHIAMATA, non dalla capacita': memorizzarlo al
    // momento dell'allocazione faceva leggere al kernel le scale
    // all'offset sbagliato per ogni matrice piu' piccola della piu' grande
    // vista finora — output "!!!!" invece di testo, con il selftest (una
    // sola forma) che passava lo stesso.
    uint8_t* d_stage = nullptr;
    float* d_y = nullptr;
    size_t nb_cap = 0;    // capacita' in blocchi da 32 dell'attivazione
    size_t rows_cap = 0;  // capacita' in righe dell'output
    // Buffer host per la quantizzazione: riusati fra le chiamate, cosi'
    // quantize_q8_0 non rialloca tre vector ad ogni matvec (252 per token).
    std::vector<int8_t> xq;
    std::vector<float> xscale;
    std::vector<uint8_t> signs;
    // Buffer della FFN fusa: gate, up/h, h quantizzato e uscita restano
    // tutti su device per l'intero blocco FFN (vedi
    // matmul_q8_0_cuda_ffn_gated).
    float* d_gate = nullptr;
    float* d_up = nullptr;
    int8_t* d_hq = nullptr;
    float* d_hscale = nullptr;
    float* d_out = nullptr;
    size_t ff_cap = 0;    // capacita' in elementi di d_gate/d_up/d_hq
    size_t out_cap = 0;   // capacita' in elementi di d_out
    // KV cache su device: MIRROR bit-per-bit di k_cache_/v_cache_ host,
    // stesso layout [layer][pos][kv_head][head_dim]. Essendo uno specchio
    // scritto nell'unico punto di scrittura esistente (write_kv_cache),
    // non puo' divergere dall'originale: qualunque percorso (prefill,
    // decode, fallback CPU) passa comunque di li'.
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
    int32_t* d_xsum = nullptr;   // per-sub-block activation sums (Q4_K min term)
    size_t xsum_cap = 0;
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
void invalidate_graphs() {
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
}

cudaStream_t scratch_stream() {
    if (!g_scratch.stream_ready) {
        // Attesa attiva invece del blocking di default: il decode fa ~108
        // sincronizzazioni per token (una per gruppo di matvec), ciascuna
        // su un lavoro GPU di poche decine di microsecondi. Con lo
        // scheduling bloccante il costo di addormentare e risvegliare il
        // thread e' dello stesso ordine del lavoro atteso; con lo spin il
        // thread resta sul core e la latenza per sincronizzazione crolla.
        // Il trade-off (un core occupato ad aspettare) e' accettabile qui:
        // quel thread non avrebbe comunque altro da fare fino al risultato.
        cudaSetDeviceFlags(cudaDeviceScheduleSpin);
        if (cudaStreamCreate(&g_scratch.stream) != cudaSuccess) {
            g_scratch.stream = nullptr; // ricade sul null stream: corretto, solo piu' lento
        }
        g_scratch.stream_ready = true;
    }
    return g_scratch.stream;
}

// Cresce i buffer di scratch solo quando servono piu' grandi (le forme
// sono stabili per modello: dopo i primi token ogni matrice ha gia'
// trovato la sua capacita' e non si rialloca piu').
bool scratch_reserve(size_t nb, size_t rows) {
    if (nb > g_scratch.nb_cap || rows > g_scratch.rows_cap) invalidate_graphs();
    if (nb > g_scratch.nb_cap) {
        if (g_scratch.d_stage) cudaFree(g_scratch.d_stage);
        g_scratch.d_stage = nullptr;
        g_scratch.nb_cap = 0;
        // Un'unica allocazione device con lo stesso layout dello staging
        // host ([xq][xscale]), cosi' la H2D e' una sola copia contigua.
        // nb*32 e' multiplo di 4, quindi la parte float resta allineata.
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
bool scratch_reserve_attn(size_t q_dim, size_t kv_dim, size_t n_rot, size_t pos_bytes) {
    (void) pos_bytes;
    auto grow_f = [](float*& p, size_t& cap, size_t need) {
        if (need <= cap && p) return true;
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
    if (g_scratch.kvbuf_cap < kv_dim) {
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
        if (g_scratch.d_bk) cudaFree(g_scratch.d_bk);
        if (g_scratch.d_bv) cudaFree(g_scratch.d_bv);
        g_scratch.d_bk = nullptr; g_scratch.d_bv = nullptr; g_scratch.bkv_cap = 0;
        if (cudaMalloc(&g_scratch.d_bk, kv_dim * sizeof(float)) != cudaSuccess) return false;
        if (cudaMalloc(&g_scratch.d_bv, kv_dim * sizeof(float)) != cudaSuccess) return false;
        g_scratch.bkv_cap = kv_dim;
    }
    const size_t stage2_need = (q_dim / 32) * 32 + (q_dim / 32) * sizeof(float);
    if (stage2_need > g_scratch.stage2_cap) {
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

// Buffer della FFN fusa. Stessa politica grow-only degli altri.
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

// Spacchetta i byte block_q8_0 nativi (fp16 scale + 32 int8 qs, come letti
// dal GGUF) in due array piatti device-friendly: qs int8 contigui e scale
// gia' convertite fp16->fp32. Stessa logica sia per l'upload one-shot
// (resident) che per il path per-chiamata (fallback), fattorizzata qui per
// non duplicarla.
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

// Libera un buffer device allocato da una qualunque funzione qui sotto.
// Firma generica (void*) cosi' da poter essere usata come deleter di uno
// std::shared_ptr<void> lato C++ (dense_forward.h/.cpp) senza dover
// includere cuda_runtime.h li'.
void cuda_free_device(void* p) {
    if (p) cudaFree(p);
}

// Rilascia stream e buffer di scratch persistenti. Chiamata dal
// distruttore di DenseForward: senza, caricare piu' modelli nello stesso
// processo (la CLI e i test lo fanno) lascerebbe in giro uno stream e i
// buffer del modello precedente, dimensionati per forme che non servono
// piu'.
void cuda_backend_shutdown() {
    invalidate_graphs();
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

// Upload one-shot dei pesi Q8_0 di UNA matrice su device: ritorna due
// puntatori device (qs int8*, scale float*) da tenere per tutta la
// sessione e riusare ad ogni token via matmul_q8_0_cuda_resident, invece
// di ricaricarli da capo ogni volta (vedi nota in testa al file). Chiamata
// una sola volta per tensore, dal punto dove il peso entra nella cache
// persistente (DenseForward::load_matrix, solo se cache_enabled_).
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

// Come matmul_q8_0_cuda (sotto) ma con i pesi GIA' su device (da una
// precedente matmul_q8_0_cuda_upload_weights): ad ogni chiamata si
// quantizza e carica solo l'attivazione (poche decine di KB, non l'intera
// matrice pesi) e si scarica solo il risultato — il costo dominante
// misurato nella versione precedente (upload pesi ripetuto ogni token)
// sparisce.
int matmul_q8_0_cuda_resident(const void* d_qs, const void* d_scale, size_t rows, size_t cols,
                               const float* x, float* y) {
    if (!d_qs || !d_scale || cols == 0 || cols % 32 != 0 || rows == 0) {
        return DESIREEIA_ERR_NOT_SUPPORTED;
    }
    ScopedTimer prof_t(profile_counters().ns_cuda_resident);
    profile_counters().calls_cuda_resident.fetch_add(1, std::memory_order_relaxed);
    const size_t nb = cols / 32;

    // Buffer riusati fra le chiamate (nessun cudaMalloc/cudaFree qui) e
    // stream persistente: vedi la nota su CudaScratch in testa al file.
    if (!scratch_reserve(nb, rows)) return DESIREEIA_ERR_IO;

    // I vector di quantizzazione vivono nello scratch: quantize_q8_0 li
    // ridimensiona solo la prima volta, poi riusa la capacita' gia' presente.
    quantize_q8_0(x, cols, g_scratch.xq, g_scratch.xscale, g_scratch.signs);
    if (g_scratch.xq.size() != nb * 32 || g_scratch.xscale.size() != nb) {
        return DESIREEIA_ERR_NOT_SUPPORTED;
    }

    cudaStream_t stream = scratch_stream();

    // Viste calcolate sull'nb di QUESTA chiamata (vedi la nota su d_stage).
    // Due copie H2D direttamente dai vector, agli offset giusti dentro
    // l'unica allocazione: misurato piu' veloce dello staging in memoria
    // pinned (una sola copia ma preceduta da due memcpy host), che su
    // questi volumi costava piu' di quanto facesse risparmiare.
    int8_t* d_xq = reinterpret_cast<int8_t*>(g_scratch.d_stage);
    float* d_xscale = reinterpret_cast<float*>(g_scratch.d_stage + nb * 32);
    cudaMemcpyAsync(d_xq, g_scratch.xq.data(), nb * 32, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_xscale, g_scratch.xscale.data(), nb * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    launch_matmul_q8_0(static_cast<const int8_t*>(d_qs), static_cast<const __half*>(d_scale),
                        d_xq, d_xscale, nb, rows, g_scratch.d_y, stream);
    if (cudaGetLastError() != cudaSuccess) return DESIREEIA_ERR_IO;

    cudaMemcpyAsync(y, g_scratch.d_y, rows * sizeof(float), cudaMemcpyDeviceToHost, stream);
    // Unico punto di sincronizzazione: le operazioni sopra sono gia'
    // ordinate fra loro dallo stream, il chiamante si aspetta y pronto.
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

// Gruppo di matvec sulla stessa attivazione: una quantizzazione, una H2D,
// n kernel, n D2H asincrone, UNA sola sincronizzazione. Vedi la nota su
// CudaQ80Job in engine.h per il perche'.
int matmul_q8_0_cuda_resident_group(const CudaQ80Job* jobs, size_t n, size_t cols, const float* x) {
    if (!jobs || n == 0 || cols == 0 || cols % 32 != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    size_t rows_total = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!jobs[i].d_qs || !jobs[i].d_scale || jobs[i].rows == 0) return DESIREEIA_ERR_NOT_SUPPORTED;
        rows_total += jobs[i].rows;
    }

    ScopedTimer prof_t(profile_counters().ns_cuda_resident);
    // Contate come n chiamate, non una: cosi' il confronto con le misure
    // precedenti (us per matvec) resta leggibile.
    profile_counters().calls_cuda_resident.fetch_add((int64_t) n, std::memory_order_relaxed);

    const size_t nb = cols / 32;
    // d_y ospita le uscite di TUTTI i job, una dopo l'altra.
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

// --- KV cache su device (mirror di quella host) ---

// (Ri)alloca la KV cache device. Chiamata quando la capacita' host cambia
// (grow_cache): il contenuto valido viene ricaricato subito dopo con
// cuda_kv_cache_upload, cosi' lo specchio riparte allineato.
bool cuda_kv_cache_reserve(size_t total_bytes) {
    if (total_bytes <= g_scratch.kv_bytes && g_scratch.d_kcache) return true;
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

// Ricarica l'intera cache host su device (dopo una crescita/reset).
bool cuda_kv_cache_upload(const void* k_host, const void* v_host, size_t total_bytes) {
    if (!g_scratch.d_kcache || total_bytes > g_scratch.kv_bytes) return false;
    cudaStream_t stream = scratch_stream();
    cudaMemcpyAsync(g_scratch.d_kcache, k_host, total_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(g_scratch.d_vcache, v_host, total_bytes, cudaMemcpyHostToDevice, stream);
    return cudaStreamSynchronize(stream) == cudaSuccess;
}

// Specchia una singola posizione (k e v di un layer). Asincrona sullo
// stream condiviso: l'ordine con il kernel di attenzione che la legge e'
// garantito dallo stream, quindi non serve sincronizzare qui.
bool cuda_kv_cache_write(size_t byte_off, const void* k, const void* v, size_t bytes) {
    if (!g_scratch.d_kcache || byte_off + bytes > g_scratch.kv_bytes) return false;
    cudaStream_t stream = scratch_stream();
    cudaMemcpyAsync(g_scratch.d_kcache + byte_off, k, bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(g_scratch.d_vcache + byte_off, v, bytes, cudaMemcpyHostToDevice, stream);
    return true;
}

// Attenzione + proiezione di output in un solo episodio: q sale una volta,
// l'attenzione legge la KV cache gia' su device, e l'uscita
// dell'attenzione alimenta direttamente il matvec di `wo` senza mai
// tornare all'host. Scende solo `proj`.
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
        attention_kernel_q8<<<n_head, threads, 2 * threads * sizeof(float), stream>>>(
            g_scratch.d_q, g_scratch.d_kcache, g_scratch.d_vcache, g_scratch.d_attn,
            head_dim, heads_per_kv, row_bytes, row_bytes * (kv_dim / head_dim),
            layer_off, g_scratch.d_dyn, inv_d);
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

    // L'uscita dell'attenzione e' gia' su device: si quantizza li' e si
    // passa direttamente al matvec di wo, senza scendere e risalire.
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
    if (a.upload_dyn) {
        const uint32_t dyn[2] = { a.pos, a.cc_start };
        cudaMemcpyAsync(g_scratch.d_dyn, dyn, sizeof(dyn), cudaMemcpyHostToDevice, stream);
    }
    if (a.upload_x) {
        cudaMemcpyAsync(g_scratch.d_xin, a.x, a.n_embd * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    }
    if (a.rope_cache && a.upload_dyn) {
        cudaMemcpyAsync(g_scratch.d_rope, a.rope_cache, a.n_rot * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    }

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

        rms_norm_quant_kernel<false><<<1, norm_thr, norm_thr * sizeof(float), stream>>>(
            g_scratch.d_xin, nullptr, lg.d_attn_norm, g_scratch.d_xn,
            d_xq, d_xscale, g_scratch.d_xsum, (uint32_t) a.n_embd, a.rms_eps);
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

        // QK-norm, before RoPE, same order as the CPU path.
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
                g_scratch.d_qbuf, g_scratch.d_kbuf, g_scratch.d_rope,
                a.n_rot, a.head_dim, a.n_head);
        }

        const unsigned kvgrid = (unsigned) (a.n_head_kv * nblk);
        kv_quantize_kernel<<<kvgrid * 2, 32, 0, stream>>>(
            g_scratch.d_kbuf, g_scratch.d_kcache + a.kv_layer_off,
            g_scratch.d_vbuf, g_scratch.d_vcache + a.kv_layer_off,
            a.head_dim, kv_row_bytes, kvgrid, pos_bytes, g_scratch.d_dyn);

        const int athr = 128;
        attention_kernel_q8<<<a.n_head, athr, 2 * athr * sizeof(float), stream>>>(
            g_scratch.d_qbuf, g_scratch.d_kcache, g_scratch.d_vcache, g_scratch.d_attn,
            a.head_dim, a.heads_per_kv, kv_row_bytes, pos_bytes,
            a.kv_layer_off, g_scratch.d_dyn, 1.0f / sqrtf((float) a.head_dim));

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
        if (lg.d_post_attn_norm) {
            rms_norm_kernel<<<1, nthr, nthr * sizeof(float), stream>>>(
                g_scratch.d_y, lg.d_post_attn_norm, g_scratch.d_y,
                (uint32_t) a.n_embd, a.rms_eps);
        }
        // Attention residual, FFN norm and its quantisation in one launch:
        // d_y takes the residual sum (the layer still needs it) while
        // d_xn2 and d_xq take the normalised and quantised form.
        rms_norm_quant_kernel<true><<<1, norm_thr, norm_thr * sizeof(float), stream>>>(
            g_scratch.d_y, g_scratch.d_xin, lg.d_ffn_norm, g_scratch.d_xn2,
            d_xq, d_xscale, g_scratch.d_xsum, (uint32_t) a.n_embd, a.rms_eps);
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
        if (lg.d_post_ffn_norm) {
            rms_norm_kernel<<<1, nthr, nthr * sizeof(float), stream>>>(
                g_scratch.d_out, lg.d_post_ffn_norm, g_scratch.d_out,
                (uint32_t) a.n_embd, a.rms_eps);
        }
        // The new x is written back into the SAME buffer the layer read
        // from, so consecutive layers chain on device with no transfer in
        // between. proj (d_y) and fout (d_out) are distinct buffers, so
        // there is no aliasing hazard here.
        add2_kernel<<<embd_blocks, nthr, 0, stream>>>(g_scratch.d_xin, g_scratch.d_out,
                                                       g_scratch.d_y, (uint32_t) a.n_embd);

        if (cudaStreamEndCapture(stream, &lg.graph) != cudaSuccess) return DESIREEIA_ERR_IO;
        if (cudaGraphInstantiate(&lg.exec, lg.graph, nullptr, nullptr, 0) != cudaSuccess) {
            cudaGraphDestroy(lg.graph);
            lg.graph = nullptr;
            return DESIREEIA_ERR_IO;
        }
    }

    if (cudaGraphLaunch(lg.exec, stream) != cudaSuccess) return DESIREEIA_ERR_IO;
    if (cudaGetLastError() != cudaSuccess) return DESIREEIA_ERR_IO;
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

        const int threads = 128;
        attention_kernel_q8<<<a.n_head, threads, 2 * threads * sizeof(float), stream>>>(
            g_scratch.d_qbuf, g_scratch.d_kcache, g_scratch.d_vcache, g_scratch.d_attn,
            a.head_dim, a.heads_per_kv, kv_row_bytes, pos_bytes,
            a.kv_layer_off, g_scratch.d_dyn, 1.0f / sqrtf((float) a.head_dim));

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
// Prima questo blocco costava, per layer: 2 D2H da n_ff float (gate e up),
// l'attivazione e la riquantizzazione dell'intermedio su CPU, una H2D da
// n_ff, e 2 sincronizzazioni. L'intermedio (n_ff = 11008 sul modello di
// prova) non serve mai all'host: nasce e muore dentro la FFN.
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

    // up e gate leggono la stessa attivazione appena caricata.
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

// Path originale (Fase 1 PoC): upload dell'intera matrice pesi ad ogni
// chiamata. Tenuto come fallback per i tensori NON coperti dalla cache
// pesi persistente (streaming/scratch_, weight cache disabilitata: in
// quel caso i pesi cambiano/vengono riletti ad ogni step comunque, quindi
// la residenza su device non avrebbe nulla da riusare) — corretto ma
// lento per lo stesso motivo diagnosticato sopra, usato solo quando
// matmul_q8_0_cuda_resident non e' applicabile.
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

} // namespace desireeia
