#pragma once

#include "../TensorValue.h"
#include "../ModelParser.h"

typedef struct {
    const value_t *pos_bias_u, *pos_bias_v;
    const value_t *in_proj_weight, *in_proj_bias;
    const value_t *out_proj_weight, *out_proj_bias;
    const value_t *linear_pos_weight;
} Kaldi2EncSelfAttn_t;

void Kaldi2EncSelfAttn_Init(Kaldi2EncSelfAttn_t *self, ModelParser_t *parser);
static inline void Kaldi2EncSelfAttn_Destroy(Kaldi2EncSelfAttn_t *self) {}

void Kaldi2EncSelfAttn_Forward(Kaldi2EncSelfAttn_t *self, TensorValue_t *din, TensorValue_t *pe);
