#pragma once

#include "../TensorValue.h"
#include "../ModelParser.h"

typedef struct {
    const value_t *pointwise_conv1_weight, *pointwise_conv1_bias;
    const value_t *depthwise_conv_weight, *depthwise_conv_bias;
    const value_t *pointwise_conv2_weight, *pointwise_conv2_bias;
    TensorValue_t conv_cache;
} Kaldi2ConvModule_t;

void Kaldi2ConvModule_Init(Kaldi2ConvModule_t *self, ModelParser_t *parser);
static inline void Kaldi2ConvModule_Destroy(Kaldi2ConvModule_t *self) {
    TensorValue_Destroy(&self->conv_cache); }

void Kaldi2ConvModule_Reset(Kaldi2ConvModule_t *self);
void Kaldi2ConvModule_Forward(Kaldi2ConvModule_t *self, TensorValue_t *din);
