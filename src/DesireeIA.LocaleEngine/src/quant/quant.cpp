#include "quant.h"
#include "desireeia/abi.h"
#include "iq_tables.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>

#define kvalues_mxfp4 kvalues_fp4
#define NGRID_IQ1S 2048
#define IQ1S_DELTA 0.125f
#define IQ1M_DELTA 0.125f

static inline void get_scale_min_k4(int j, const uint8_t* DESIREEIA_RESTRICT q, uint8_t* DESIREEIA_RESTRICT d, uint8_t* DESIREEIA_RESTRICT m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

void dequantize_row_q1_0(const block_q1_0 * DESIREEIA_RESTRICT x, float * DESIREEIA_RESTRICT y, int64_t k) {
    static const int qk = QK1_0;
    assert(k % qk == 0);
    const int nb = k / qk;
    for (int i = 0; i < nb; i++) {
        const float d = DESIREEIA_FP16_TO_FP32(x[i].d);
        const float neg_d = -d;
        for (int j = 0; j < qk; ++j) {
            const int byte_index = j / 8;
            const int bit_offset = j % 8;
            const uint8_t bit = (x[i].qs[byte_index] >> bit_offset) & 1;
            y[i*qk + j] = bit ? d : neg_d;
        }
    }
}

void dequantize_row_q2_0(const block_q2_0 * DESIREEIA_RESTRICT x, float * DESIREEIA_RESTRICT y, int64_t k) {
    static const int qk = QK2_0;
    assert(k % qk == 0);
    const int nb = k / qk;
    for (int i = 0; i < nb; i++) {
        const float d = DESIREEIA_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < qk; ++j) {
            const int byte_index = j / 4;
            const int bit_offset = (j % 4) * 2;
            const uint8_t q = (x[i].qs[byte_index] >> bit_offset) & 0x03;
            y[i*qk + j] = ((int)q - 1) * d;
        }
    }
}

void dequantize_row_q4_0(const block_q4_0 * DESIREEIA_RESTRICT x, float * DESIREEIA_RESTRICT y, int64_t k) {
    static const int qk = QK4_0;
    assert(k % qk == 0);
    const int nb = k / qk;
    for (int i = 0; i < nb; i++) {
        const float d = DESIREEIA_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;
            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

void dequantize_row_q4_1(const block_q4_1 * DESIREEIA_RESTRICT x, float * DESIREEIA_RESTRICT y, int64_t k) {
    static const int qk = QK4_1;
    assert(k % qk == 0);
    const int nb = k / qk;
    for (int i = 0; i < nb; i++) {
        const float d = DESIREEIA_FP16_TO_FP32(x[i].d);
        const float m = DESIREEIA_FP16_TO_FP32(x[i].m);
        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F);
            const int x1 = (x[i].qs[j] >>   4);
            y[i*qk + j + 0   ] = x0*d + m;
            y[i*qk + j + qk/2] = x1*d + m;
        }
    }
}

void dequantize_row_q5_0(const block_q5_0 * DESIREEIA_RESTRICT x, float * DESIREEIA_RESTRICT y, int64_t k) {
    static const int qk = QK5_0;
    assert(k % qk == 0);
    const int nb = k / qk;
    for (int i = 0; i < nb; i++) {
        const float d = DESIREEIA_FP16_TO_FP32(x[i].d);
        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));
        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;
            const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((x[i].qs[j] >>   4) | xh_1) - 16;
            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

void dequantize_row_q5_1(const block_q5_1 * DESIREEIA_RESTRICT x, float * DESIREEIA_RESTRICT y, int64_t k) {
    static const int qk = QK5_1;
    assert(k % qk == 0);
    const int nb = k / qk;
    for (int i = 0; i < nb; i++) {
        const float d = DESIREEIA_FP16_TO_FP32(x[i].d);
        const float m = DESIREEIA_FP16_TO_FP32(x[i].m);
        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));
        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;
            const int x0 = (x[i].qs[j] & 0x0F) | xh_0;
            const int x1 = (x[i].qs[j] >>   4) | xh_1;
            y[i*qk + j + 0   ] = x0*d + m;
            y[i*qk + j + qk/2] = x1*d + m;
        }
    }
}

void dequantize_row_q8_0(const block_q8_0 * DESIREEIA_RESTRICT x, float * DESIREEIA_RESTRICT y, int64_t k) {
    static const int qk = QK8_0;
    assert(k % qk == 0);
    const int nb = k / qk;
    for (int i = 0; i < nb; i++) {
        const float d = DESIREEIA_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }
}

void dequantize_row_mxfp4(const block_mxfp4 * DESIREEIA_RESTRICT x, float * DESIREEIA_RESTRICT y, int64_t k) {
    static const int qk = QK_MXFP4;
    assert(k % qk == 0);
    const int nb = k / qk;
    for (int i = 0; i < nb; i++) {
        const float d = DESIREEIA_E8M0_TO_FP32_HALF(x[i].e);
        for (int j = 0; j < qk/2; ++j) {
            const int8_t x0 = kvalues_mxfp4[x[i].qs[j] & 0x0F];
            const int8_t x1 = kvalues_mxfp4[x[i].qs[j] >>   4];
            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

void dequantize_row_nvfp4(const block_nvfp4 * DESIREEIA_RESTRICT x, float * DESIREEIA_RESTRICT y, int64_t k) {
    static const int qk = QK_NVFP4;
    static const int qk_sub = QK_NVFP4_SUB;
    static const int n_sub = QK_NVFP4 / QK_NVFP4_SUB;
    assert(k % qk == 0);
    const int nb = k / qk;
    for (int i = 0; i < nb; i++) {
        for (int s = 0; s < n_sub; s++) {
            const float d = desireeia_ue4m3_to_fp32(x[i].d[s]);
            float * yb = y + i*qk + s*qk_sub;
            for (int j = 0; j < qk_sub/2; ++j) {
                const int8_t v0 = kvalues_mxfp4[x[i].qs[s*(qk_sub/2) + j] & 0x0F];
                const int8_t v1 = kvalues_mxfp4[x[i].qs[s*(qk_sub/2) + j] >>   4];
                yb[j + 0       ] = v0*d;
                yb[j + qk_sub/2] = v1*d;
            }
        }
    }
}

static void dequantize_row_q8_1(const block_q8_1* DESIREEIA_RESTRICT x, float* DESIREEIA_RESTRICT y, int64_t k) {
    static const int qk = QK8_1;
    assert(k % qk == 0);
    const int nb = k / qk;
    for (int i = 0; i < nb; i++) {
        const float d = DESIREEIA_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }
}

void dequantize_row_q2_K(const block_q2_K * DESIREEIA_RESTRICT x, float * DESIREEIA_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const float d = DESIREEIA_FP16_TO_FP32(x[i].d);
        const float min = DESIREEIA_FP16_TO_FP32(x[i].dmin);
        const uint8_t * q = x[i].qs;
        int is = 0;
        float dl, ml;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;
                sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
}

void dequantize_row_q3_K(const block_q3_K * DESIREEIA_RESTRICT x, float * DESIREEIA_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    const int8_t * scales = (const int8_t*)aux;
    for (int i = 0; i < nb; i++) {
        const float d_all = DESIREEIA_FP16_TO_FP32(x[i].d);
        const uint8_t * DESIREEIA_RESTRICT q = x[i].qs;
        const uint8_t * DESIREEIA_RESTRICT hm = x[i].hmask;
        uint8_t m = 1;
        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        int is = 0;
        float dl;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l+ 0] >> shift) & 3) - ((hm[l+ 0] & m) ? 0 : 4));
                }
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4));
                }
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

void dequantize_row_q4_K(const block_q4_K * DESIREEIA_RESTRICT x, float * DESIREEIA_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const uint8_t * q = x[i].qs;
        const float d   = DESIREEIA_FP16_TO_FP32(x[i].d);
        const float min = DESIREEIA_FP16_TO_FP32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

void dequantize_row_q5_K(const block_q5_K * DESIREEIA_RESTRICT x, float * DESIREEIA_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const uint8_t * ql = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const float d = DESIREEIA_FP16_TO_FP32(x[i].d);
        const float min = DESIREEIA_FP16_TO_FP32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

void dequantize_row_q6_K(const block_q6_K * DESIREEIA_RESTRICT x, float * DESIREEIA_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const float d = DESIREEIA_FP16_TO_FP32(x[i].d);
        const uint8_t * DESIREEIA_RESTRICT ql = x[i].ql;
        const uint8_t * DESIREEIA_RESTRICT qh = x[i].qh;
        const int8_t  * DESIREEIA_RESTRICT sc = x[i].scales;
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void dequantize_row_tq1_0(const block_tq1_0 * DESIREEIA_RESTRICT x, float * DESIREEIA_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    for (int64_t i = 0; i < nb; ++i) {
        const float d = DESIREEIA_FP16_TO_FP32(x[i].d);
        for (size_t j = 0; j < sizeof(x->qs) - sizeof(x->qs) % 32; j += 32) {
            for (size_t n = 0; n < 5; ++n) {
                for (size_t m = 0; m < 32; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[n];
                    int16_t xi = ((uint16_t) q * 3) >> 8;
                    *y++ = (float) (xi - 1) * d;
                }
            }
        }
        for (size_t j = sizeof(x->qs) - sizeof(x->qs) % 32; j < sizeof(x->qs); j += 16) {
            for (size_t n = 0; n < 5; ++n) {
                for (size_t m = 0; m < 16; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[n];
                    int16_t xi = ((uint16_t) q * 3) >> 8;
                    *y++ = (float) (xi - 1) * d;
                }
            }
        }
        for (size_t n = 0; n < 4; ++n) {
            for (size_t j = 0; j < sizeof(x->qh); ++j) {
                uint8_t q = x[i].qh[j] * pow3[n];
                int16_t xi = ((uint16_t) q * 3) >> 8;
                *y++ = (float) (xi - 1) * d;
            }
        }
    }
}

void dequantize_row_tq2_0(const block_tq2_0 * DESIREEIA_RESTRICT x, float * DESIREEIA_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; ++i) {
        const float d = DESIREEIA_FP16_TO_FP32(x[i].d);
        for (size_t j = 0; j < sizeof(x->qs); j += 32) {
            for (size_t l = 0; l < 4; ++l) {
                for (size_t m = 0; m < 32; ++m) {
                    int8_t q = (x[i].qs[j + m] >> (l*2)) & 3;
                    *y++ = (float) (q - 1) * d;
                }
            }
        }
    }
}

void dequantize_row_iq2_xxs(const block_iq2_xxs * DESIREEIA_RESTRICT x, float * DESIREEIA_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;
    for (int i = 0; i < nb; i++) {
        const float d = DESIREEIA_FP16_TO_FP32(x[i].d);
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(aux32, x[i].qs + 4*ib32, 2*sizeof(uint32_t));
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}

void dequantize_row_iq2_xs(const block_iq2_xs * DESIREEIA_RESTRICT x, float * DESIREEIA_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    float db[2];
    for (int i = 0; i < nb; i++) {
        const float d = DESIREEIA_FP16_TO_FP32(x[i].d);
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >>  4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (x[i].qs[4*ib32 + l] & 511));
                const uint8_t  signs = ksigns_iq2xs[x[i].qs[4*ib32 + l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db[l/2] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}

void dequantize_row_iq2_s(const block_iq2_s * DESIREEIA_RESTRICT x, float * DESIREEIA_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    float db[2];
    for (int i = 0; i < nb; i++) {
        const float d = DESIREEIA_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * signs = qs + QK_K/8;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >>  4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const float dl = db[l/2];
                const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 4;
            signs += 4;
        }
    }
}

void dequantize_row_iq3_xxs(const block_iq3_xxs * DESIREEIA_RESTRICT x, float * DESIREEIA_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    uint32_t aux32;
    for (int i = 0; i < nb; i++) {
        const float d = DESIREEIA_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * scales_and_signs = qs + QK_K/4;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(&aux32, scales_and_signs + 4*ib32, sizeof(uint32_t));
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t  signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid + qs[2*l+0]);
                const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid + qs[2*l+1]);
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    y[j+4] = db * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
        }
    }
}

void dequantize_row_iq3_s(const block_iq3_s * DESIREEIA_RESTRICT x, float * DESIREEIA_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const float d = DESIREEIA_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * signs = x[i].signs;
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const float db1 = d * (1 + 2*(x[i].scales[ib32/2] & 0xf));
            const float db2 = d * (1 + 2*(x[i].scales[ib32/2] >>  4));
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[0] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[0] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db1 * grid1[j] * (signs[l] & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    y[j+4] = db1 * grid2[j] * (signs[l] & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[1] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[1] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db2 * grid1[j] * (signs[l] & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    y[j+4] = db2 * grid2[j] * (signs[l] & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
}

void dequantize_row_iq1_s(const block_iq1_s * DESIREEIA_RESTRICT x, float * DESIREEIA_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const float d = DESIREEIA_FP16_TO_FP32(x[i].d);
        const uint8_t  * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const float dl = d * (2*((qh[ib] >> 12) & 7) + 1);
            const float delta = qh[ib] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
            for (int l = 0; l < 4; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + (qs[l] | (((qh[ib] >> 3*l) & 7) << 8)));
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl * (grid[j] + delta);
                }
                y += 8;
            }
            qs += 4;
        }
    }
}

void dequantize_row_iq1_m(const block_iq1_m * DESIREEIA_RESTRICT x, float * DESIREEIA_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    float delta[4];
    uint16_t idx[4];
    iq1m_scale_t scale;
    for (int i = 0; i < nb; i++) {
        const uint16_t * sc = (const uint16_t *)x[i].scales;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        const float d = DESIREEIA_FP16_TO_FP32(scale.f16);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const float dl1 = d * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1);
            const float dl2 = d * (2*((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1);
            idx[0] = qs[0] | ((qh[0] << 8) & 0x700);
            idx[1] = qs[1] | ((qh[0] << 4) & 0x700);
            idx[2] = qs[2] | ((qh[1] << 8) & 0x700);
            idx[3] = qs[3] | ((qh[1] << 4) & 0x700);
            delta[0] = qh[0] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[1] = qh[0] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[2] = qh[1] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[3] = qh[1] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
            for (int l = 0; l < 2; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + idx[l]);
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl1 * (grid[j] + delta[l]);
                }
                y += 8;
            }
            for (int l = 2; l < 4; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + idx[l]);
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl2 * (grid[j] + delta[l]);
                }
                y += 8;
            }
            qs += 4;
            qh += 2;
        }
    }
}

void dequantize_row_iq4_nl(const block_iq4_nl * DESIREEIA_RESTRICT x, float * DESIREEIA_RESTRICT y, int64_t k) {
    assert(k % QK4_NL == 0);
    const int64_t nb = k / QK4_NL;
    for (int i = 0; i < nb; i++) {
        const uint8_t * qs = x[i].qs;
        const float d = DESIREEIA_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < QK4_NL/2; ++j) {
            y[j+       0] = d * kvalues_iq4nl[qs[j] & 0xf];
            y[j+QK4_NL/2] = d * kvalues_iq4nl[qs[j] >>  4];
        }
        y  += QK4_NL;
        qs += QK4_NL/2;
    }
}

void dequantize_row_iq4_xs(const block_iq4_xs * DESIREEIA_RESTRICT x, float * DESIREEIA_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const uint8_t * qs = x[i].qs;
        const float d = DESIREEIA_FP16_TO_FP32(x[i].d);
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const int ls = ((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl * kvalues_iq4nl[qs[j] & 0xf];
                y[j+16] = dl * kvalues_iq4nl[qs[j] >>  4];
            }
            y  += 32;
            qs += 16;
        }
    }
}

void dequantize_row_q8_K(const block_q8_K * DESIREEIA_RESTRICT x, float * DESIREEIA_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < QK_K; ++j) {
            *y++ = x[i].d * x[i].qs[j];
        }
    }
}



size_t desireeia_type_size(int type) {
    switch (type) {
    case DESIREEIA_QTYPE_F32:    return sizeof(float);
    case DESIREEIA_QTYPE_F16:    return sizeof(desireeia_half);
    case DESIREEIA_QTYPE_BF16:   return sizeof(uint16_t);
    case DESIREEIA_QTYPE_I8:     return sizeof(int8_t);
    case DESIREEIA_QTYPE_I16:    return sizeof(int16_t);
    case DESIREEIA_QTYPE_I32:    return sizeof(int32_t);
    case DESIREEIA_QTYPE_I64:    return sizeof(int64_t);
    case DESIREEIA_QTYPE_F64:    return sizeof(double);
    case DESIREEIA_QTYPE_Q1_0:   return sizeof(block_q1_0);
    case DESIREEIA_QTYPE_Q2_0:   return sizeof(block_q2_0);
    case DESIREEIA_QTYPE_Q4_0:   return sizeof(block_q4_0);
    case DESIREEIA_QTYPE_Q4_1:   return sizeof(block_q4_1);
    case DESIREEIA_QTYPE_Q5_0:   return sizeof(block_q5_0);
    case DESIREEIA_QTYPE_Q5_1:   return sizeof(block_q5_1);
    case DESIREEIA_QTYPE_Q8_0:   return sizeof(block_q8_0);
    case DESIREEIA_QTYPE_Q8_1:   return sizeof(block_q8_1);
    case DESIREEIA_QTYPE_Q2_K:   return sizeof(block_q2_K);
    case DESIREEIA_QTYPE_Q3_K:   return sizeof(block_q3_K);
    case DESIREEIA_QTYPE_Q4_K:   return sizeof(block_q4_K);
    case DESIREEIA_QTYPE_Q5_K:   return sizeof(block_q5_K);
    case DESIREEIA_QTYPE_Q6_K:   return sizeof(block_q6_K);
    case DESIREEIA_QTYPE_Q8_K:   return sizeof(block_q8_K);
    case DESIREEIA_QTYPE_IQ2_XXS: return sizeof(block_iq2_xxs);
    case DESIREEIA_QTYPE_IQ2_XS:  return sizeof(block_iq2_xs);
    case DESIREEIA_QTYPE_IQ3_XXS: return sizeof(block_iq3_xxs);
    case DESIREEIA_QTYPE_IQ1_S:   return sizeof(block_iq1_s);
    case DESIREEIA_QTYPE_IQ4_NL:  return sizeof(block_iq4_nl);
    case DESIREEIA_QTYPE_IQ3_S:   return sizeof(block_iq3_s);
    case DESIREEIA_QTYPE_IQ2_S:   return sizeof(block_iq2_s);
    case DESIREEIA_QTYPE_IQ4_XS:  return sizeof(block_iq4_xs);
    case DESIREEIA_QTYPE_IQ1_M:   return sizeof(block_iq1_m);
    case DESIREEIA_QTYPE_TQ1_0:   return sizeof(block_tq1_0);
    case DESIREEIA_QTYPE_TQ2_0:   return sizeof(block_tq2_0);
    case DESIREEIA_QTYPE_MXFP4:   return sizeof(block_mxfp4);
    case DESIREEIA_QTYPE_NVFP4:   return sizeof(block_nvfp4);
    default: return 0;
    }
}

int desireeia_blck_size(int type) {
    switch (type) {
    case DESIREEIA_QTYPE_F32:
    case DESIREEIA_QTYPE_F16:
    case DESIREEIA_QTYPE_BF16:
    case DESIREEIA_QTYPE_I8:
    case DESIREEIA_QTYPE_I16:
    case DESIREEIA_QTYPE_I32:
    case DESIREEIA_QTYPE_I64:
    case DESIREEIA_QTYPE_F64:    return 1;
    case DESIREEIA_QTYPE_Q1_0:   return QK1_0;
    case DESIREEIA_QTYPE_Q2_0:   return QK2_0;
    case DESIREEIA_QTYPE_Q4_0:
    case DESIREEIA_QTYPE_Q4_1:
    case DESIREEIA_QTYPE_Q5_0:
    case DESIREEIA_QTYPE_Q5_1:
    case DESIREEIA_QTYPE_Q8_0:
    case DESIREEIA_QTYPE_Q8_1:
    case DESIREEIA_QTYPE_IQ4_NL:
    case DESIREEIA_QTYPE_MXFP4:  return QK4_0;
    case DESIREEIA_QTYPE_NVFP4:  return QK_NVFP4;
    default:                  return QK_K;
    }
}

size_t desireeia_row_size(int type, int64_t ne0) {
    size_t type_size  = desireeia_type_size(type);
    int    blck       = desireeia_blck_size(type);
    int64_t nb = ne0 / blck + (ne0 % blck ? 1 : 0);
    return (size_t)(nb * (int64_t)type_size);
}

int desireeia_dequantize_row(int type, const void* DESIREEIA_RESTRICT x, float* DESIREEIA_RESTRICT y, int64_t n) {
    switch (type) {
    case DESIREEIA_QTYPE_F32: {
        const float* src = (const float*)x;
        memcpy(y, src, (size_t)(n * sizeof(float)));
        return DESIREEIA_OK;
    }
    case DESIREEIA_QTYPE_F16: {
        const desireeia_half* src = (const desireeia_half*)x;
        for (int64_t i = 0; i < n; ++i) y[i] = DESIREEIA_FP16_TO_FP32(src[i]);
        return DESIREEIA_OK;
    }
    case DESIREEIA_QTYPE_BF16: {
        const uint16_t* src = (const uint16_t*)x;
        for (int64_t i = 0; i < n; ++i) y[i] = desireeia_bf16_to_fp32(src[i]);
        return DESIREEIA_OK;
    }
    case DESIREEIA_QTYPE_F64: {
        const double* src = (const double*)x;
        for (int64_t i = 0; i < n; ++i) y[i] = (float)src[i];
        return DESIREEIA_OK;
    }
    case DESIREEIA_QTYPE_I8: {
        const int8_t* src = (const int8_t*)x;
        for (int64_t i = 0; i < n; ++i) y[i] = (float)src[i];
        return DESIREEIA_OK;
    }
    case DESIREEIA_QTYPE_I16: {
        const int16_t* src = (const int16_t*)x;
        for (int64_t i = 0; i < n; ++i) y[i] = (float)src[i];
        return DESIREEIA_OK;
    }
    case DESIREEIA_QTYPE_I32: {
        const int32_t* src = (const int32_t*)x;
        for (int64_t i = 0; i < n; ++i) y[i] = (float)src[i];
        return DESIREEIA_OK;
    }
    case DESIREEIA_QTYPE_I64: {
        const int64_t* src = (const int64_t*)x;
        for (int64_t i = 0; i < n; ++i) y[i] = (float)src[i];
        return DESIREEIA_OK;
    }
    case DESIREEIA_QTYPE_Q1_0:  dequantize_row_q1_0((const block_q1_0*)x, y, n); return DESIREEIA_OK;
    case DESIREEIA_QTYPE_Q2_0:  dequantize_row_q2_0((const block_q2_0*)x, y, n); return DESIREEIA_OK;
    case DESIREEIA_QTYPE_Q4_0:  dequantize_row_q4_0((const block_q4_0*)x, y, n); return DESIREEIA_OK;
    case DESIREEIA_QTYPE_Q4_1:  dequantize_row_q4_1((const block_q4_1*)x, y, n); return DESIREEIA_OK;
    case DESIREEIA_QTYPE_Q5_0:  dequantize_row_q5_0((const block_q5_0*)x, y, n); return DESIREEIA_OK;
    case DESIREEIA_QTYPE_Q5_1:  dequantize_row_q5_1((const block_q5_1*)x, y, n); return DESIREEIA_OK;
    case DESIREEIA_QTYPE_Q8_0:  dequantize_row_q8_0((const block_q8_0*)x, y, n); return DESIREEIA_OK;
    case DESIREEIA_QTYPE_Q8_1:  dequantize_row_q8_1((const block_q8_1*)x, y, n); return DESIREEIA_OK;
    case DESIREEIA_QTYPE_Q2_K:  dequantize_row_q2_K((const block_q2_K*)x, y, n); return DESIREEIA_OK;
    case DESIREEIA_QTYPE_Q3_K:  dequantize_row_q3_K((const block_q3_K*)x, y, n); return DESIREEIA_OK;
    case DESIREEIA_QTYPE_Q4_K:  dequantize_row_q4_K((const block_q4_K*)x, y, n); return DESIREEIA_OK;
    case DESIREEIA_QTYPE_Q5_K:  dequantize_row_q5_K((const block_q5_K*)x, y, n); return DESIREEIA_OK;
    case DESIREEIA_QTYPE_Q6_K:  dequantize_row_q6_K((const block_q6_K*)x, y, n); return DESIREEIA_OK;
    case DESIREEIA_QTYPE_Q8_K:  dequantize_row_q8_K((const block_q8_K*)x, y, n); return DESIREEIA_OK;
    case DESIREEIA_QTYPE_IQ2_XXS: dequantize_row_iq2_xxs((const block_iq2_xxs*)x, y, n); return DESIREEIA_OK;
    case DESIREEIA_QTYPE_IQ2_XS:  dequantize_row_iq2_xs((const block_iq2_xs*)x, y, n); return DESIREEIA_OK;
    case DESIREEIA_QTYPE_IQ3_XXS: dequantize_row_iq3_xxs((const block_iq3_xxs*)x, y, n); return DESIREEIA_OK;
    case DESIREEIA_QTYPE_IQ1_S:   dequantize_row_iq1_s((const block_iq1_s*)x, y, n); return DESIREEIA_OK;
    case DESIREEIA_QTYPE_IQ4_NL:  dequantize_row_iq4_nl((const block_iq4_nl*)x, y, n); return DESIREEIA_OK;
    case DESIREEIA_QTYPE_IQ3_S:   dequantize_row_iq3_s((const block_iq3_s*)x, y, n); return DESIREEIA_OK;
    case DESIREEIA_QTYPE_IQ2_S:   dequantize_row_iq2_s((const block_iq2_s*)x, y, n); return DESIREEIA_OK;
    case DESIREEIA_QTYPE_IQ4_XS:  dequantize_row_iq4_xs((const block_iq4_xs*)x, y, n); return DESIREEIA_OK;
    case DESIREEIA_QTYPE_IQ1_M:   dequantize_row_iq1_m((const block_iq1_m*)x, y, n); return DESIREEIA_OK;
    case DESIREEIA_QTYPE_TQ1_0:   dequantize_row_tq1_0((const block_tq1_0*)x, y, n); return DESIREEIA_OK;
    case DESIREEIA_QTYPE_TQ2_0:   dequantize_row_tq2_0((const block_tq2_0*)x, y, n); return DESIREEIA_OK;
    case DESIREEIA_QTYPE_MXFP4:   dequantize_row_mxfp4((const block_mxfp4*)x, y, n); return DESIREEIA_OK;
    case DESIREEIA_QTYPE_NVFP4:   dequantize_row_nvfp4((const block_nvfp4*)x, y, n); return DESIREEIA_OK;
    default: return DESIREEIA_ERR_NOT_SUPPORTED;
    }
}

