// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

// GPU path of the ModernBERT-style encoder (see models/encoder_cuda.h). One
// stream, synchronous calls. The residual stream stays in float; every
// matrix product reads half-precision operands on the tensor cores and
// accumulates in float.

#include "models/encoder_cuda.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cmath>
#include <cstdio>
#include <cstring>

using namespace nvcuda;

namespace desireeia {

namespace {

constexpr int kBM = 128;
constexpr int kBN = 128;
constexpr int kBK = 32;
constexpr int kLd = kBK + 8;   // padded row length of the shared tiles (bank conflicts)

enum { EPI_STORE_F32 = 0, EPI_ADD_F32 = 1, EPI_STORE_F16 = 2 };

// C[t][r] (op)= sum_k A[t][k] * W[r][k], batched over blockIdx.z.
//   A [m_pad][k] half (row stride lda), W [rows][k] half (row stride ldw), both with
//   16-byte aligned rows; m_pad is a multiple of 128 and k a multiple of 32; rows a
//   multiple of 16. C has row stride ldc; the epilogue stores, adds into float, or
//   stores as half. 8 warps in a 4 x 2 grid, each owning 32 tokens x 64 rows.
template <int EPI>
__global__ void __launch_bounds__(256) gemm_kernel(const __half* __restrict__ a, size_t lda, size_t sa,
                                                   const __half* __restrict__ w, size_t ldw, size_t sw,
                                                   void* __restrict__ c, size_t ldc, size_t sc,
                                                   size_t k, size_t rows,
                                                   const float* __restrict__ bias, int relu) {
    __shared__ __align__(32) __half as[2][kBM][kLd];
    __shared__ __align__(32) __half bs[2][kBN][kLd];

    const __half* A = a + blockIdx.z * sa;
    const __half* W = w + blockIdx.z * sw;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int wm = warp / 2;
    const int wn = warp % 2;
    const size_t t0 = (size_t) blockIdx.y * kBM;
    const size_t r0 = (size_t) blockIdx.x * kBN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    const int4 zero = make_int4(0, 0, 0, 0);
    int4 ra[2], rb[2];
    auto fetch = [&](size_t kk) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = tid + i * 256;
            const int r = idx / 4, cc = (idx % 4) * 8;
            ra[i] = *reinterpret_cast<const int4*>(A + (t0 + r) * lda + kk + cc);
            const size_t row = r0 + r;
            rb[i] = row < rows ? *reinterpret_cast<const int4*>(W + row * ldw + kk + cc) : zero;
        }
    };
    auto stash = [&](int buf) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = tid + i * 256;
            const int r = idx / 4, cc = (idx % 4) * 8;
            *reinterpret_cast<int4*>(&as[buf][r][cc]) = ra[i];
            *reinterpret_cast<int4*>(&bs[buf][r][cc]) = rb[i];
        }
    };

    fetch(0);
    stash(0);
    __syncthreads();
    int buf = 0;
    for (size_t kk = 0; kk < k; kk += kBK) {
        const bool more = kk + kBK < k;
        if (more) fetch(kk + kBK);
#pragma unroll
        for (int ks = 0; ks < kBK; ks += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb[4];
#pragma unroll
            for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(fa[i], &as[buf][wm * 32 + i * 16][ks], kLd);
#pragma unroll
            for (int j = 0; j < 4; ++j) wmma::load_matrix_sync(fb[j], &bs[buf][wn * 64 + j * 16][ks], kLd);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
        if (more) stash(buf ^ 1);
        __syncthreads();
        buf ^= 1;
    }

    // The last barrier above means every warp is done with the tiles: reuse them as a
    // private 16 x 16 float scratch per warp for the epilogue.
    float* scratch = reinterpret_cast<float*>(&as[0][0][0]) + warp * 256;
    char* cbase = reinterpret_cast<char*>(c);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const size_t gc0 = r0 + wn * 64 + j * 16;
            wmma::store_matrix_sync(scratch, acc[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            if (gc0 < rows) {
                const size_t gr0 = t0 + wm * 32 + i * 16;
                for (int e = lane; e < 256; e += 32) {
                    const size_t gr = gr0 + (e >> 4), gc = gc0 + (e & 15);
                    float v = scratch[e];
                    if (bias) v += bias[gc];
                    if (relu) v = fmaxf(v, 0.0f);
                    if (EPI == EPI_STORE_F32) {
                        reinterpret_cast<float*>(cbase)[blockIdx.z * sc + gr * ldc + gc] = v;
                    } else if (EPI == EPI_ADD_F32) {
                        reinterpret_cast<float*>(cbase)[blockIdx.z * sc + gr * ldc + gc] += v;
                    } else {
                        reinterpret_cast<__half*>(cbase)[blockIdx.z * sc + gr * ldc + gc] = __float2half(v);
                    }
                }
            }
            __syncwarp();
        }
    }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    return v;
}

// Block-wide sum / max, result broadcast to every thread. `sh` holds 32 floats.
__device__ float block_sum(float v, float* sh) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32) v = warp_sum(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    v = sh[0];
    __syncthreads();
    return v;
}
__device__ float block_max(float v, float* sh) {
    v = warp_max(v);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nw) ? sh[threadIdx.x] : -3.0e38f;
    if (threadIdx.x < 32) v = warp_max(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    v = sh[0];
    __syncthreads();
    return v;
}

// LayerNorm without bias of one row held in `vals` (already in registers of the caller).
// Token embeddings are looked up and normalized into the float residual stream.
__global__ void embed_ln_kernel(const int* __restrict__ ids, const __half* __restrict__ tok,
                                const float* __restrict__ w, float* __restrict__ x, int d, float eps) {
    __shared__ float sh[32];
    const int t = blockIdx.x;
    const __half* row = tok + (size_t) ids[t] * d;
    float s = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) s += __half2float(row[i]);
    const float mean = block_sum(s, sh) / d;
    float q = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const float dv = __half2float(row[i]) - mean;
        q += dv * dv;
    }
    const float inv = rsqrtf(block_sum(q, sh) / d + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        x[(size_t) t * d + i] = (__half2float(row[i]) - mean) * inv * w[i];
}

template <bool OUT_HALF>
__global__ void ln_kernel(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                          void* __restrict__ out, int d, float eps) {
    __shared__ float sh[32];
    const float* row = x + (size_t) blockIdx.x * d;
    float s = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) s += row[i];
    const float mean = block_sum(s, sh) / d;
    float q = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const float dv = row[i] - mean;
        q += dv * dv;
    }
    const float inv = rsqrtf(block_sum(q, sh) / d + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const float y = (row[i] - mean) * inv * (w ? w[i] : 1.0f) + (b ? b[i] : 0.0f);
        if (OUT_HALF) reinterpret_cast<__half*>(out)[(size_t) blockIdx.x * d + i] = __float2half(y);
        else reinterpret_cast<float*>(out)[(size_t) blockIdx.x * d + i] = y;
    }
}

// qkv [n_pad][3d] float -> Q, K half [n_pad][d] with the rotary embedding applied
// (rotate-half layout) and V transposed to half [d][n_pad].
__global__ void rope_pack_kernel(const float* __restrict__ qkv, __half* __restrict__ q, __half* __restrict__ k,
                                 __half* __restrict__ vt, const float* __restrict__ inv_freq,
                                 int n_pad, int d, int hd) {
    const int t = blockIdx.x;
    const int half_hd = hd / 2;
    const float* row = qkv + (size_t) t * 3 * d;
    for (int idx = threadIdx.x; idx < d / 2; idx += blockDim.x) {
        const int h = idx / half_hd, j = idx % half_hd;
        float sn, cs;
        if (inv_freq) sincosf((float) t * inv_freq[j], &sn, &cs);   // null table: no rotation (the head)
        else { sn = 0.0f; cs = 1.0f; }
        const int base = h * hd;
        {
            const float x1 = row[base + j], x2 = row[base + j + half_hd];
            q[(size_t) t * d + base + j] = __float2half(x1 * cs - x2 * sn);
            q[(size_t) t * d + base + j + half_hd] = __float2half(x2 * cs + x1 * sn);
        }
        {
            const float x1 = row[d + base + j], x2 = row[d + base + j + half_hd];
            k[(size_t) t * d + base + j] = __float2half(x1 * cs - x2 * sn);
            k[(size_t) t * d + base + j + half_hd] = __float2half(x2 * cs + x1 * sn);
        }
    }
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        vt[(size_t) i * n_pad + t] = __float2half(row[2 * d + i]);
}

__global__ void add_row_kernel(float* __restrict__ x, const float* __restrict__ row, int d, size_t count) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) x[i] += row[i % d];
}

__global__ void to_half_kernel(const float* __restrict__ x, __half* __restrict__ y, size_t count) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) y[i] = __float2half(x[i]);
}

// Row softmax over the allowed keys of every (head, query). window_half < 0 means
// every key; otherwise only |i - j| <= window_half. Writes half probabilities,
// zero outside the window and in the padding.
__global__ void softmax_kernel(const float* __restrict__ s, __half* __restrict__ p, int n, int n_pad,
                               int window_half, float scale) {
    __shared__ float sh[32];
    const int i = blockIdx.x;
    const size_t base = ((size_t) blockIdx.y * n_pad + i) * n_pad;
    int lo = 0, hi = 0;
    if (i < n) {
        lo = 0;
        hi = n;
        if (window_half >= 0) {
            lo = i - window_half > 0 ? i - window_half : 0;
            hi = i + window_half + 1 < n ? i + window_half + 1 : n;
        }
    }
    float m = -3.0e38f;
    for (int j = lo + threadIdx.x; j < hi; j += blockDim.x) m = fmaxf(m, s[base + j] * scale);
    m = block_max(m, sh);
    float sum = 0.0f;
    for (int j = lo + threadIdx.x; j < hi; j += blockDim.x) sum += __expf(s[base + j] * scale - m);
    sum = block_sum(sum, sh);
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int j = threadIdx.x; j < n_pad; j += blockDim.x) {
        float v = 0.0f;
        if (j >= lo && j < hi) v = __expf(s[base + j] * scale - m) * inv;
        p[base + j] = __float2half(v);
    }
}

// wi [n_pad][2*ff] float -> act half [n_pad][ff] = gelu(first half) * second half.
__global__ void geglu_kernel(const float* __restrict__ wi, __half* __restrict__ act, size_t count, int ff) {
    const size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    const size_t row = idx / ff, j = idx % ff;
    const float a = wi[row * 2 * ff + j];
    const float b = wi[row * 2 * ff + ff + j];
    const float g = 0.5f * a * (1.0f + erff(a * 0.70710678118654752f));
    act[row * ff + j] = __float2half(g * b);
}

#define CK(call)                                                                      \
    do {                                                                              \
        cudaError_t e__ = (call);                                                     \
        if (e__ != cudaSuccess) { set_error(cudaGetErrorString(e__)); return false; } \
    } while (0)

}  // namespace

struct ModernBertCuda::Impl {
    struct LayerDev {
        __half *wqkv = nullptr, *wo = nullptr, *wi = nullptr, *mlp_wo = nullptr;
        float *attn_norm = nullptr, *mlp_norm = nullptr;
        bool is_global = false;
    };
    uint32_t d = 0, heads = 0, hd = 0, ff = 0;
    int window_half = 64;
    float eps = 1e-5f;
    __half* tok = nullptr;
    float *emb_norm = nullptr, *final_norm = nullptr;
    float *inv_global = nullptr, *inv_local = nullptr;
    std::vector<LayerDev> layers;
    struct HeadDev {
        __half *in_proj = nullptr, *out_proj = nullptr, *lin1 = nullptr, *lin2 = nullptr;
        float *in_proj_b = nullptr, *out_proj_b = nullptr, *lin1_b = nullptr, *lin2_b = nullptr;
        float *n1w = nullptr, *n1b = nullptr, *n2w = nullptr, *n2b = nullptr;
    };
    std::vector<HeadDev> head;
    float* type_emb = nullptr;
    uint32_t head_ffn = 0;
    std::string last_error;

    // workspace, sized for ws_pad tokens
    size_t ws_pad = 0;
    int* ids = nullptr;
    float *x = nullptr, *qkvf = nullptr, *scores = nullptr, *wif = nullptr, *outf = nullptr;
    __half *normed = nullptr, *q = nullptr, *k = nullptr, *vt = nullptr, *probs = nullptr, *attn = nullptr, *act = nullptr;

    void set_error(const char* m) { last_error = m; }

    template <typename T>
    bool upload(T** dst, const void* src, size_t bytes) {
        CK(cudaMalloc(reinterpret_cast<void**>(dst), bytes));
        CK(cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice));
        return true;
    }

    void free_workspace() {
        cudaFree(ids); cudaFree(x); cudaFree(qkvf); cudaFree(scores); cudaFree(wif); cudaFree(outf);
        cudaFree(normed); cudaFree(q); cudaFree(k); cudaFree(vt); cudaFree(probs); cudaFree(attn); cudaFree(act);
        ids = nullptr; x = qkvf = scores = wif = outf = nullptr;
        normed = q = k = vt = probs = attn = act = nullptr;
        ws_pad = 0;
    }

    ~Impl() {
        free_workspace();
        for (auto& l : layers) {
            cudaFree(l.wqkv); cudaFree(l.wo); cudaFree(l.wi); cudaFree(l.mlp_wo);
            cudaFree(l.attn_norm); cudaFree(l.mlp_norm);
        }
        cudaFree(tok); cudaFree(emb_norm); cudaFree(final_norm); cudaFree(inv_global); cudaFree(inv_local);
        free_head();
    }

    void free_head() {
        for (auto& h : head) {
            cudaFree(h.in_proj); cudaFree(h.out_proj); cudaFree(h.lin1); cudaFree(h.lin2);
            cudaFree(h.in_proj_b); cudaFree(h.out_proj_b); cudaFree(h.lin1_b); cudaFree(h.lin2_b);
            cudaFree(h.n1w); cudaFree(h.n1b); cudaFree(h.n2w); cudaFree(h.n2b);
        }
        head.clear();
        cudaFree(type_emb);
        type_emb = nullptr;
    }

    bool ensure_workspace(size_t n_pad) {
        if (n_pad <= ws_pad) return true;
        free_workspace();
        const size_t D = d, FF = ff, H = heads;
        CK(cudaMalloc(reinterpret_cast<void**>(&ids), n_pad * sizeof(int)));
        CK(cudaMalloc(reinterpret_cast<void**>(&x), n_pad * D * 4));
        CK(cudaMalloc(reinterpret_cast<void**>(&qkvf), n_pad * 3 * D * 4));
        CK(cudaMalloc(reinterpret_cast<void**>(&scores), H * n_pad * n_pad * 4));
        CK(cudaMalloc(reinterpret_cast<void**>(&wif), n_pad * 2 * FF * 4));
        CK(cudaMalloc(reinterpret_cast<void**>(&outf), n_pad * D * 4));
        CK(cudaMalloc(reinterpret_cast<void**>(&normed), n_pad * D * 2));
        CK(cudaMalloc(reinterpret_cast<void**>(&q), n_pad * D * 2));
        CK(cudaMalloc(reinterpret_cast<void**>(&k), n_pad * D * 2));
        CK(cudaMalloc(reinterpret_cast<void**>(&vt), D * n_pad * 2));
        CK(cudaMalloc(reinterpret_cast<void**>(&probs), H * n_pad * n_pad * 2));
        CK(cudaMalloc(reinterpret_cast<void**>(&attn), n_pad * D * 2));
        const size_t act_cols = FF > head_ffn ? FF : head_ffn;
        CK(cudaMalloc(reinterpret_cast<void**>(&act), n_pad * act_cols * 2));
        // Padding rows must hold finite values; they never receive anything else.
        CK(cudaMemset(x, 0, n_pad * D * 4));
        CK(cudaMemset(qkvf, 0, n_pad * 3 * D * 4));
        CK(cudaMemset(wif, 0, n_pad * 2 * FF * 4));
        CK(cudaMemset(normed, 0, n_pad * D * 2));
        CK(cudaMemset(q, 0, n_pad * D * 2));
        CK(cudaMemset(k, 0, n_pad * D * 2));
        CK(cudaMemset(vt, 0, D * n_pad * 2));
        CK(cudaMemset(attn, 0, n_pad * D * 2));
        CK(cudaMemset(act, 0, n_pad * act_cols * 2));
        ws_pad = n_pad;
        return true;
    }

    template <int EPI>
    bool gemm(const __half* a, size_t lda, size_t sa, const __half* w, size_t ldw, size_t sw, void* c, size_t ldc,
              size_t sc, size_t m_pad, size_t k_dim, size_t rows, size_t batch,
              const float* bias = nullptr, int relu = 0) {
        dim3 grid((unsigned) ((rows + kBN - 1) / kBN), (unsigned) (m_pad / kBM), (unsigned) batch);
        gemm_kernel<EPI><<<grid, 256>>>(a, lda, sa, w, ldw, sw, c, ldc, sc, k_dim, rows, bias, relu);
        CK(cudaGetLastError());
        return true;
    }

    // Multi-head attention over the padded sequence for the current q / k / vt, result in
    // `attn`. window_half < 0 attends to every position.
    bool attention(size_t n, size_t n_pad, int window_half) {
        const size_t H = heads, D = d;
        if (!gemm<EPI_STORE_F32>(q, D, hd, k, D, hd, scores, n_pad, n_pad * n_pad, n_pad, hd, n_pad, H)) return false;
        softmax_kernel<<<dim3((unsigned) n_pad, (unsigned) H), 128>>>(scores, probs, (int) n, (int) n_pad, window_half,
                                                                      1.0f / sqrtf((float) hd));
        CK(cudaGetLastError());
        return gemm<EPI_STORE_F16>(probs, n_pad, n_pad * n_pad, vt, n_pad, hd * n_pad, attn, D, hd, n_pad, n_pad, hd, H);
    }

    // The head layers over the residual stream outf (encoder output + type embedding).
    bool run_head(size_t n, size_t n_pad) {
        const size_t D = d;
        const int nn = (int) n, npad = (int) n_pad;
        for (const HeadDev& h : head) {
            ln_kernel<true><<<nn, 256>>>(outf, h.n1w, h.n1b, normed, (int) D, 1e-5f);
            CK(cudaGetLastError());
            if (!gemm<EPI_STORE_F32>(normed, D, 0, h.in_proj, D, 0, qkvf, 3 * D, 0, n_pad, D, 3 * D, 1, h.in_proj_b)) return false;
            rope_pack_kernel<<<nn, 256>>>(qkvf, q, k, vt, nullptr, npad, (int) D, (int) hd);
            CK(cudaGetLastError());
            if (!attention(n, n_pad, -1)) return false;
            if (!gemm<EPI_ADD_F32>(attn, D, 0, h.out_proj, D, 0, outf, D, 0, n_pad, D, D, 1, h.out_proj_b)) return false;
            ln_kernel<true><<<nn, 256>>>(outf, h.n2w, h.n2b, normed, (int) D, 1e-5f);
            CK(cudaGetLastError());
            if (!gemm<EPI_STORE_F16>(normed, D, 0, h.lin1, D, 0, act, head_ffn, 0, n_pad, D, head_ffn, 1, h.lin1_b, 1)) return false;
            if (!gemm<EPI_ADD_F32>(act, head_ffn, 0, h.lin2, head_ffn, 0, outf, D, 0, n_pad, head_ffn, D, 1, h.lin2_b)) return false;
        }
        return true;
    }

    bool run(const int32_t* host_ids, size_t n, std::vector<float>& out) {
        if (!encode(host_ids, n)) return false;
        const size_t D = d;
        out.resize(n * D);
        CK(cudaMemcpy(out.data(), outf, n * D * 4, cudaMemcpyDeviceToHost));
        return true;
    }

    bool run_with_head(const int32_t* host_ids, size_t n, int qtype, const size_t* rows, size_t n_rows,
                       std::vector<float>& out) {
        if (head.empty() || qtype < 0 || qtype > 2) return false;
        if (!encode(host_ids, n)) return false;
        const size_t D = d;
        const size_t n_pad = (n + kBM - 1) / kBM * kBM;
        add_row_kernel<<<(unsigned) ((n * D + 255) / 256), 256>>>(outf, type_emb + (size_t) qtype * D, (int) D, n * D);
        CK(cudaGetLastError());
        if (!run_head(n, n_pad)) return false;
        out.resize(n_rows * D);
        for (size_t i = 0; i < n_rows; ++i) {
            if (rows[i] >= n) return false;
            CK(cudaMemcpy(out.data() + i * D, outf + rows[i] * D, D * 4, cudaMemcpyDeviceToHost));
        }
        return true;
    }

    // Embeddings and every encoder layer; leaves the final-normed hidden state in outf.
    bool encode(const int32_t* host_ids, size_t n) {
        const size_t n_pad = (n + kBM - 1) / kBM * kBM;
        if (!ensure_workspace(n_pad)) return false;
        const size_t D = d, FF = ff, H = heads;
        const int nn = (int) n, npad = (int) n_pad;

        CK(cudaMemcpy(ids, host_ids, n * sizeof(int), cudaMemcpyHostToDevice));
        embed_ln_kernel<<<nn, 256>>>(ids, tok, emb_norm, x, (int) D, eps);
        CK(cudaGetLastError());

        for (size_t l = 0; l < layers.size(); ++l) {
            const LayerDev& L = layers[l];
            // attention
            if (L.attn_norm) {
                ln_kernel<true><<<nn, 256>>>(x, L.attn_norm, nullptr, normed, (int) D, eps);
            } else {
                to_half_kernel<<<(unsigned) ((n * D + 255) / 256), 256>>>(x, normed, n * D);
            }
            CK(cudaGetLastError());
            if (!gemm<EPI_STORE_F32>(normed, D, 0, L.wqkv, D, 0, qkvf, 3 * D, 0, n_pad, D, 3 * D, 1)) return false;
            rope_pack_kernel<<<nn, 256>>>(qkvf, q, k, vt, L.is_global ? inv_global : inv_local, npad, (int) D, (int) hd);
            CK(cudaGetLastError());
            // scores[h] = Q_h K_h^T over every (padded) position
            if (!gemm<EPI_STORE_F32>(q, D, hd, k, D, hd, scores, n_pad, n_pad * n_pad, n_pad, hd, n_pad, H)) return false;
            softmax_kernel<<<dim3((unsigned) n_pad, (unsigned) H), 128>>>(scores, probs, nn, npad,
                                                                          L.is_global ? -1 : window_half,
                                                                          1.0f / sqrtf((float) hd));
            CK(cudaGetLastError());
            // attn_h = P_h V_h, written straight into the head's columns
            if (!gemm<EPI_STORE_F16>(probs, n_pad, n_pad * n_pad, vt, n_pad, hd * n_pad, attn, D, hd, n_pad, n_pad, hd, H))
                return false;
            if (!gemm<EPI_ADD_F32>(attn, D, 0, L.wo, D, 0, x, D, 0, n_pad, D, D, 1)) return false;

            // feed-forward
            ln_kernel<true><<<nn, 256>>>(x, L.mlp_norm, nullptr, normed, (int) D, eps);
            CK(cudaGetLastError());
            if (!gemm<EPI_STORE_F32>(normed, D, 0, L.wi, D, 0, wif, 2 * FF, 0, n_pad, D, 2 * FF, 1)) return false;
            geglu_kernel<<<(unsigned) ((n * FF + 255) / 256), 256>>>(wif, act, n * FF, (int) FF);
            CK(cudaGetLastError());
            if (!gemm<EPI_ADD_F32>(act, FF, 0, L.mlp_wo, FF, 0, x, D, 0, n_pad, FF, D, 1)) return false;
        }

        ln_kernel<false><<<nn, 256>>>(x, final_norm, nullptr, outf, (int) D, eps);
        CK(cudaGetLastError());
        return true;
    }
};

ModernBertCuda::ModernBertCuda() : p_(new Impl) {}
ModernBertCuda::~ModernBertCuda() { delete p_; }

bool ModernBertCuda::device_available() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) {
        cudaGetLastError();   // clear the sticky error of a missing driver
        return false;
    }
    return n > 0;
}

bool ModernBertCuda::init(const CudaEncoderDesc& desc, std::string& err) {
    Impl& s = *p_;
    if (!device_available()) { err = "no CUDA device"; return false; }
    if (desc.heads == 0 || desc.d % desc.heads != 0 || (desc.d / desc.heads) % 2 != 0 ||
        desc.d % 8 != 0 || desc.ff % 32 != 0 || desc.d % 32 != 0) {
        err = "encoder dimensions are not supported by the GPU path";
        return false;
    }
    s.d = desc.d; s.heads = desc.heads; s.hd = desc.d / desc.heads; s.ff = desc.ff;
    s.eps = desc.eps;
    s.window_half = (int) (desc.local_window / 2);

    // Weights in half precision plus the token table: refuse early when they cannot fit.
    size_t need = (size_t) desc.n_vocab * desc.d * 2;
    for (size_t l = 0; l < desc.layers.size(); ++l)
        need += ((size_t) 3 * desc.d * desc.d + (size_t) desc.d * desc.d + (size_t) 2 * desc.ff * desc.d +
                 (size_t) desc.d * desc.ff) * 2;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) { cudaGetLastError(); err = "cannot query GPU memory"; return false; }
    const size_t workspace = (size_t) 512 << 20;   // generous: 1024 tokens need about 150 MB
    if (need + workspace > free_b) { err = "not enough free GPU memory"; return false; }

    auto fail = [&]() { err = s.last_error; return false; };
    if (!s.upload(&s.tok, desc.tok_emb, (size_t) desc.n_vocab * desc.d * 2)) return fail();
    if (!s.upload(&s.emb_norm, desc.emb_norm, (size_t) desc.d * 4)) return fail();
    if (!s.upload(&s.final_norm, desc.final_norm, (size_t) desc.d * 4)) return fail();

    // rotary frequencies for the two attention kinds
    const size_t half_hd = s.hd / 2;
    std::vector<float> fg(half_hd), fl(half_hd);
    for (size_t j = 0; j < half_hd; ++j) {
        fg[j] = (float) std::pow((double) desc.theta_global, -2.0 * (double) j / (double) s.hd);
        fl[j] = (float) std::pow((double) desc.theta_local, -2.0 * (double) j / (double) s.hd);
    }
    if (!s.upload(&s.inv_global, fg.data(), half_hd * 4)) return fail();
    if (!s.upload(&s.inv_local, fl.data(), half_hd * 4)) return fail();

    s.layers.resize(desc.layers.size());
    for (size_t l = 0; l < desc.layers.size(); ++l) {
        const CudaEncoderLayer& src = desc.layers[l];
        Impl::LayerDev& dst = s.layers[l];
        dst.is_global = src.is_global;
        if (!s.upload(&dst.wqkv, src.wqkv, (size_t) 3 * desc.d * desc.d * 2)) return fail();
        if (!s.upload(&dst.wo, src.wo, (size_t) desc.d * desc.d * 2)) return fail();
        if (!s.upload(&dst.wi, src.wi, (size_t) 2 * desc.ff * desc.d * 2)) return fail();
        if (!s.upload(&dst.mlp_wo, src.mlp_wo, (size_t) desc.d * desc.ff * 2)) return fail();
        if (!s.upload(&dst.mlp_norm, src.mlp_norm, (size_t) desc.d * 4)) return fail();
        if (src.attn_norm && !s.upload(&dst.attn_norm, src.attn_norm, (size_t) desc.d * 4)) return fail();
    }
    return true;
}

bool ModernBertCuda::forward(const int32_t* ids, size_t n, std::vector<float>& out) {
    if (n == 0 || n > max_tokens()) return false;
    return p_->run(ids, n, out);
}

bool ModernBertCuda::set_head(const CudaHeadDesc& hd, std::string& err) {
    Impl& s = *p_;
    if (s.d == 0 || hd.layers.empty() || !hd.type_emb || hd.ffn % 32 != 0 || hd.ffn == 0) {
        err = "decision head not supported by the GPU path";
        return false;
    }
    const size_t d = s.d, ffn = hd.ffn;
    const size_t need = hd.layers.size() * ((3 * d * d + d * d + ffn * d + d * ffn) * 2);
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) { cudaGetLastError(); err = "cannot query GPU memory"; return false; }
    if (need + ((size_t) 384 << 20) > free_b) { err = "not enough free GPU memory for the decision head"; return false; }

    s.free_workspace();   // the activation buffers depend on the head's feed-forward width
    s.free_head();
    s.head_ffn = hd.ffn;
    s.head.resize(hd.layers.size());
    auto fail = [&]() { err = s.last_error; s.free_head(); s.head_ffn = 0; s.free_workspace(); return false; };
    for (size_t i = 0; i < hd.layers.size(); ++i) {
        const CudaHeadLayer& src = hd.layers[i];
        Impl::HeadDev& dst = s.head[i];
        if (!s.upload(&dst.in_proj, src.in_proj, 3 * d * d * 2) || !s.upload(&dst.out_proj, src.out_proj, d * d * 2) ||
            !s.upload(&dst.lin1, src.lin1, ffn * d * 2) || !s.upload(&dst.lin2, src.lin2, d * ffn * 2) ||
            !s.upload(&dst.in_proj_b, src.in_proj_b, 3 * d * 4) || !s.upload(&dst.out_proj_b, src.out_proj_b, d * 4) ||
            !s.upload(&dst.lin1_b, src.lin1_b, ffn * 4) || !s.upload(&dst.lin2_b, src.lin2_b, d * 4) ||
            !s.upload(&dst.n1w, src.n1w, d * 4) || !s.upload(&dst.n1b, src.n1b, d * 4) ||
            !s.upload(&dst.n2w, src.n2w, d * 4) || !s.upload(&dst.n2b, src.n2b, d * 4))
            return fail();
    }
    if (!s.upload(&s.type_emb, hd.type_emb, 3 * d * 4)) return fail();
    return true;
}

bool ModernBertCuda::forward_head(const int32_t* ids, size_t n, int qtype, const size_t* rows, size_t n_rows,
                                  std::vector<float>& out) {
    if (n == 0 || n > max_tokens()) return false;
    return p_->run_with_head(ids, n, qtype, rows, n_rows, out);
}

}  // namespace desireeia
