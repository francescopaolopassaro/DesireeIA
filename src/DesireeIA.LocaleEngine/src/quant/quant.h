// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#ifndef DESIREEIA_QUANT_H
#define DESIREEIA_QUANT_H

#include <stdint.h>
#include <cstring>
#include <cmath>
#include <cassert>

typedef uint16_t desireeia_half;
typedef uint16_t desireeia_fp16_t;
typedef uint32_t desireeia_half2;
#define DESIREEIA_EXTENSION
#define DESIREEIA_AGGR_S
#define DESIREEIA_AGGR_U
#define DESIREEIA_RESTRICT __restrict

#define QK_K 256
#define K_SCALE_SIZE 12

#define QK1_0 128
typedef struct {
    desireeia_half d;           
    uint8_t qs[QK1_0 / 8]; 
} block_q1_0;
static_assert(sizeof(block_q1_0) == sizeof(desireeia_half) + QK1_0 / 8, "wrong q1_0 block size/padding");

#define QK2_0 64
typedef struct {
    desireeia_half d;              
    uint8_t qs[QK2_0 / 4];   
} block_q2_0;
static_assert(sizeof(block_q2_0) == sizeof(desireeia_half) + QK2_0 / 4, "wrong q2_0 block size/padding");

#define QK4_0 32
typedef struct {
    desireeia_half d;           
    uint8_t qs[QK4_0 / 2]; 
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(desireeia_half) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
typedef struct {
    DESIREEIA_EXTENSION union {
        struct {
            desireeia_half d; 
            desireeia_half m; 
        } DESIREEIA_AGGR_S;
        desireeia_half2 dm;
    } DESIREEIA_AGGR_U;
    uint8_t qs[QK4_1 / 2]; 
} block_q4_1;
static_assert(sizeof(block_q4_1) == 2 * sizeof(desireeia_half) + QK4_1 / 2, "wrong q4_1 block size/padding");

#define QK_MXFP4 32
typedef struct {
    uint8_t e; 
    uint8_t qs[QK_MXFP4/2];
} block_mxfp4;
static_assert(sizeof(block_mxfp4) == sizeof(uint8_t) + QK_MXFP4/2, "wrong mxfp4 block size/padding");

#define QK_NVFP4 64
#define QK_NVFP4_SUB 16  
typedef struct {
    uint8_t d[QK_NVFP4/QK_NVFP4_SUB]; 
    uint8_t qs[QK_NVFP4/2];           
} block_nvfp4;
static_assert(sizeof(block_nvfp4) == sizeof(uint8_t)*(QK_NVFP4/QK_NVFP4_SUB) + QK_NVFP4/2, "wrong nvfp4 block size/padding");

#define QK5_0 32
typedef struct {
    desireeia_half d;           
    uint8_t qh[4];         
    uint8_t qs[QK5_0 / 2]; 
} block_q5_0;
static_assert(sizeof(block_q5_0) == sizeof(desireeia_half) + sizeof(uint32_t) + QK5_0 / 2, "wrong q5_0 block size/padding");

#define QK5_1 32
typedef struct {
    DESIREEIA_EXTENSION union {
        struct {
            desireeia_half d; 
            desireeia_half m; 
        } DESIREEIA_AGGR_S;
        desireeia_half2 dm;
    } DESIREEIA_AGGR_U;
    uint8_t qh[4];         
    uint8_t qs[QK5_1 / 2]; 
} block_q5_1;
static_assert(sizeof(block_q5_1) == 2 * sizeof(desireeia_half) + sizeof(uint32_t) + QK5_1 / 2, "wrong q5_1 block size/padding");

#define QK8_0 32
typedef struct {
    desireeia_half d;       
    int8_t  qs[QK8_0]; 
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(desireeia_half) + QK8_0, "wrong q8_0 block size/padding");

#define QK8_1 32
typedef struct {
    DESIREEIA_EXTENSION union {
        struct {
            desireeia_half d; 
            desireeia_half s; 
        } DESIREEIA_AGGR_S;
        desireeia_half2 ds;
    } DESIREEIA_AGGR_U;
    int8_t qs[QK8_1]; 
} block_q8_1;
static_assert(sizeof(block_q8_1) == 2*sizeof(desireeia_half) + QK8_1, "wrong q8_1 block size/padding");


typedef struct {
    uint8_t qs[(QK_K - 4 * QK_K / 64) / 5]; 
    uint8_t qh[QK_K/64]; 
    desireeia_half d;
} block_tq1_0;
static_assert(sizeof(block_tq1_0) == sizeof(desireeia_half) + QK_K / 64 + (QK_K - 4 * QK_K / 64) / 5, "wrong tq1_0 block size/padding");

typedef struct {
    uint8_t qs[QK_K/4]; 
    desireeia_half d;
} block_tq2_0;
static_assert(sizeof(block_tq2_0) == sizeof(desireeia_half) + QK_K / 4, "wrong tq2_0 block size/padding");


typedef struct {
    uint8_t scales[QK_K/16]; 
    uint8_t qs[QK_K/4];      
    DESIREEIA_EXTENSION union {
        struct {
            desireeia_half d;    
            desireeia_half dmin; 
        } DESIREEIA_AGGR_S;
        desireeia_half2 dm;
    } DESIREEIA_AGGR_U;
} block_q2_K;
static_assert(sizeof(block_q2_K) == 2*sizeof(desireeia_half) + QK_K/16 + QK_K/4, "wrong q2_K block size/padding");

typedef struct {
    uint8_t hmask[QK_K/8]; 
    uint8_t qs[QK_K/4];    
    uint8_t scales[12];    
    desireeia_half d;           
} block_q3_K;
static_assert(sizeof(block_q3_K) == sizeof(desireeia_half) + QK_K / 4 + QK_K / 8 + 12, "wrong q3_K block size/padding");

typedef struct {
    DESIREEIA_EXTENSION union {
        struct {
            desireeia_half d;    
            desireeia_half dmin; 
        } DESIREEIA_AGGR_S;
        desireeia_half2 dm;
    } DESIREEIA_AGGR_U;
    uint8_t scales[K_SCALE_SIZE]; 
    uint8_t qs[QK_K/2];           
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(desireeia_half) + K_SCALE_SIZE + QK_K/2, "wrong q4_K block size/padding");

typedef struct {
    DESIREEIA_EXTENSION union {
        struct {
            desireeia_half d;    
            desireeia_half dmin; 
        } DESIREEIA_AGGR_S;
        desireeia_half2 dm;
    } DESIREEIA_AGGR_U;
    uint8_t scales[K_SCALE_SIZE]; 
    uint8_t qh[QK_K/8];           
    uint8_t qs[QK_K/2];           
} block_q5_K;
static_assert(sizeof(block_q5_K) == 2*sizeof(desireeia_half) + K_SCALE_SIZE + QK_K/2 + QK_K/8, "wrong q5_K block size/padding");

typedef struct {
    uint8_t ql[QK_K/2];      
    uint8_t qh[QK_K/4];      
    int8_t  scales[QK_K/16]; 
    desireeia_half d;             
} block_q6_K;
static_assert(sizeof(block_q6_K) == sizeof(desireeia_half) + QK_K / 16 + 3*QK_K/4, "wrong q6_K block size/padding");

typedef struct {
    float   d;              
    int8_t  qs[QK_K];       
    int16_t bsums[QK_K/16]; 
} block_q8_K;
static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K + QK_K/16*sizeof(int16_t), "wrong q8_K block size/padding");

typedef struct {
    desireeia_half d;
    uint16_t qs[QK_K/8];
} block_iq2_xxs;
static_assert(sizeof(block_iq2_xxs) == sizeof(desireeia_half) + QK_K/8*sizeof(uint16_t), "wrong iq2_xxs block size/padding");

typedef struct {
    desireeia_half d;
    uint16_t qs[QK_K/8];
    uint8_t  scales[QK_K/32];
} block_iq2_xs;
static_assert(sizeof(block_iq2_xs) == sizeof(desireeia_half) + QK_K/8*sizeof(uint16_t) + QK_K/32, "wrong iq2_xs block size/padding");

typedef struct {
    desireeia_half d;
    uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/32];
    uint8_t scales[QK_K/32];
} block_iq2_s;
static_assert(sizeof(block_iq2_s) == sizeof(desireeia_half) + QK_K/4 + QK_K/16, "wrong iq2_s block size/padding");

typedef struct {
    desireeia_half d;
    uint8_t qs[3*QK_K/8];
} block_iq3_xxs;
static_assert(sizeof(block_iq3_xxs) == sizeof(desireeia_half) + 3*(QK_K/8), "wrong iq3_xxs block size/padding");

#define IQ3S_N_SCALE QK_K/64
typedef struct {
    desireeia_half d;
    uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/32];
    uint8_t signs[QK_K/8];
    uint8_t scales[IQ3S_N_SCALE];
} block_iq3_s;
static_assert(sizeof(block_iq3_s) == sizeof(desireeia_half) + 13*(QK_K/32) + IQ3S_N_SCALE, "wrong iq3_s block size/padding");

typedef struct {
    desireeia_half d;
    uint8_t  qs[QK_K/8];
    uint16_t qh[QK_K/32];
} block_iq1_s;
static_assert(sizeof(block_iq1_s) == sizeof(desireeia_half) + QK_K/8 + QK_K/16, "wrong iq1_s block size/padding");

typedef struct {
    uint8_t  qs[QK_K/8];      
    uint8_t  qh[QK_K/16];     
    uint8_t  scales[QK_K/32]; 
} block_iq1_m;
static_assert(sizeof(block_iq1_m) == QK_K/8 + QK_K/16 + QK_K/32, "wrong iq1_m block size/padding");

typedef union {
    desireeia_half f16;
    uint16_t  u16;
} iq1m_scale_t;

#define QK4_NL 32
typedef struct {
    desireeia_half d;
    uint8_t qs[QK4_NL/2];
} block_iq4_nl;
static_assert(sizeof(block_iq4_nl) == sizeof(desireeia_half) + QK4_NL/2, "wrong iq4_nl block size/padding");

typedef struct {
    desireeia_half d;
    uint16_t scales_h;
    uint8_t  scales_l[QK_K/64];
    uint8_t  qs[QK_K/2];
} block_iq4_xs;
static_assert(sizeof(block_iq4_xs) == sizeof(desireeia_half) + sizeof(uint16_t) + QK_K/64 + QK_K/2, "wrong iq4_xs block size/padding");

static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline float desireeia_compute_fp16_to_fp32(desireeia_fp16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = 0x1.0p-112f;
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

static inline desireeia_fp16_t desireeia_compute_fp32_to_fp16(float f) {
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}



static inline float desireeia_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;  

    if (x == 0) {
        bits = 0x00400000;
    }
    else {
        bits = (uint32_t) x << 23;
    }

    float result;  
    memcpy(&result, &bits, sizeof(float));
    return result;
}

static inline float desireeia_e8m0_to_fp32_half(uint8_t x) {
    uint32_t bits;

    if (x < 2) {
        bits = 0x00200000 << x;
    }
    else {
        bits = (uint32_t)(x - 1) << 23;
    }

    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}


static inline float desireeia_ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || x == 0x7F) {
        return 0.0f;
    }
    int   exp = (x >> 3) & 0xF;
    int   man = x & 0x7;
    float raw;
    if (exp == 0) {
        raw = ldexpf((float) man, -9);
    } else {
        raw = ldexpf(1.0f + (float) man / 8.0f, exp - 7);
    }
    return raw * 0.5f;
}

static inline uint8_t desireeia_fp32_to_ue4m3(float x) {
    if (!(x > 0.0f)) {
        return 0;
    }
    if (x > 448.0f) {
        x = 448.0f;
    }
    uint32_t bits;
    memcpy(&bits, &x, 4);
    int fp32_exp  = ((bits >> 23) & 0xFF) - 127;
    int fp32_man  = (bits >> 20) & 0x7;
    int ue4m3_exp = fp32_exp + 7;
    if (ue4m3_exp <= 0) {
        int man = (int) (x * 512.0f + 0.5f);
        if (man > 7) {
            man = 7;
        }
        if (man < 1) {
            return 0;
        }
        return (uint8_t) man;
    }
    if (ue4m3_exp >= 15) {
        return 0x7E;
    }
    int round_bit = (bits >> 19) & 1;
    int ue4m3_man = fp32_man + round_bit;
    if (ue4m3_man > 7) {
        ue4m3_man = 0;
        ue4m3_exp++;
        if (ue4m3_exp >= 15) {
            return 0x7E;
        }
    }
    return (uint8_t) ((ue4m3_exp << 3) | ue4m3_man);
}

static inline float desireeia_fp16_to_fp32(desireeia_half h) {
    return desireeia_compute_fp16_to_fp32(h);
}

static inline desireeia_half desireeia_fp32_to_fp16(float f) {
    return desireeia_compute_fp32_to_fp16(f);
}

typedef uint32_t desireeia_bf16_t;

static inline float desireeia_bf16_to_fp32(desireeia_bf16_t h) {
    uint32_t f32 = ((uint32_t)h) << 16;
    float r; memcpy(&r, &f32, sizeof(float));
    return r;
}

static inline desireeia_bf16_t desireeia_fp32_to_bf16(float f) {
    uint32_t tmp;
    memcpy(&tmp, &f, sizeof(float));
    tmp += 0x00007fffu + ((tmp >> 16) & 1);
    return (desireeia_bf16_t)(tmp >> 16);
}

#define DESIREEIA_FP16_TO_FP32(x) desireeia_fp16_to_fp32(x)
#define DESIREEIA_E8M0_TO_FP32(x) desireeia_e8m0_to_fp32(x)
#define DESIREEIA_E8M0_TO_FP32_HALF(x) desireeia_e8m0_to_fp32_half(x)
#define DESIREEIA_ASSERT(x) assert(x)
#define DESIREEIA_UNUSED(x) (void)(x)

enum DesireeIAQuantType {
    DESIREEIA_QTYPE_F32     = 0,
    DESIREEIA_QTYPE_F16     = 1,
    DESIREEIA_QTYPE_Q4_0    = 2,
    DESIREEIA_QTYPE_Q4_1    = 3,
    DESIREEIA_QTYPE_Q5_0    = 6,
    DESIREEIA_QTYPE_Q5_1    = 7,
    DESIREEIA_QTYPE_Q8_0    = 8,
    DESIREEIA_QTYPE_Q8_1    = 9,
    DESIREEIA_QTYPE_Q2_K    = 10,
    DESIREEIA_QTYPE_Q3_K    = 11,
    DESIREEIA_QTYPE_Q4_K    = 12,
    DESIREEIA_QTYPE_Q5_K    = 13,
    DESIREEIA_QTYPE_Q6_K    = 14,
    DESIREEIA_QTYPE_Q8_K    = 15,
    DESIREEIA_QTYPE_IQ2_XXS = 16,
    DESIREEIA_QTYPE_IQ2_XS  = 17,
    DESIREEIA_QTYPE_IQ3_XXS = 18,
    DESIREEIA_QTYPE_IQ1_S   = 19,
    DESIREEIA_QTYPE_IQ4_NL  = 20,
    DESIREEIA_QTYPE_IQ3_S   = 21,
    DESIREEIA_QTYPE_IQ2_S   = 22,
    DESIREEIA_QTYPE_IQ4_XS  = 23,
    DESIREEIA_QTYPE_I8      = 24,
    DESIREEIA_QTYPE_I16     = 25,
    DESIREEIA_QTYPE_I32     = 26,
    DESIREEIA_QTYPE_I64     = 27,
    DESIREEIA_QTYPE_F64     = 28,
    DESIREEIA_QTYPE_IQ1_M   = 29,
    DESIREEIA_QTYPE_BF16    = 30,
    DESIREEIA_QTYPE_TQ1_0   = 34,
    DESIREEIA_QTYPE_TQ2_0   = 35,
    DESIREEIA_QTYPE_MXFP4   = 39,
    DESIREEIA_QTYPE_NVFP4   = 40,
    DESIREEIA_QTYPE_Q1_0    = 41,
    DESIREEIA_QTYPE_Q2_0    = 42,
    DESIREEIA_QTYPE_UNKNOWN = -1
};

#if defined(_WIN32) && defined(DESIREEIA_BUILD)
#define DESIREEIA_INTERNAL __declspec(dllexport)
#else
#define DESIREEIA_INTERNAL
#endif

DESIREEIA_INTERNAL size_t desireeia_type_size(int type);
DESIREEIA_INTERNAL int    desireeia_blck_size(int type);
DESIREEIA_INTERNAL size_t desireeia_row_size(int type, int64_t ne0);
DESIREEIA_INTERNAL int desireeia_dequantize_row(int type, const void* DESIREEIA_RESTRICT x, float* DESIREEIA_RESTRICT y, int64_t n);

// Quantizes floats into Q4_K. n must be a multiple of QK_K (256).
// The inverse of dequantize_row_q4_K; see the notes on the definition.
DESIREEIA_INTERNAL void quantize_row_q4_K(const float* DESIREEIA_RESTRICT x,
                                          block_q4_K* DESIREEIA_RESTRICT y, int64_t n);

#endif



