#pragma once

#include "../TensorValue.h"
#include "../ModelParser.h"

typedef struct {
    const value_t *conv0_weight, *conv0_bias;
    const value_t *conv1_weight, *conv1_bias;
    const value_t *conv2_weight, *conv2_bias;
    const value_t *out_weight, *out_bias, *out_norm;
} Kaldi2EmbedLayer_t;

void Kaldi2EmbedLayer_Init(Kaldi2EmbedLayer_t *self, ModelParser_t *parser);
static inline void Kaldi2EmbedLayer_Destroy(Kaldi2EmbedLayer_t *self) {}

void Kaldi2EmbedLayer_Forward(Kaldi2EmbedLayer_t *self, TensorValue_t *din);
