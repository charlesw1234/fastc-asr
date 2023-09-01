#pragma once

#include "EncSelfAttn.h"
#include "FeedForward.h"
#include "ConvModule.h"

typedef struct {
    Kaldi2EncSelfAttn_t self_attn;
    Kaldi2FeedForward_t feedforward, feedforward_macaron;
    Kaldi2ConvModule_t conv_module;
    const value_t *norm;
} Kaldi2SubEncoder_t;

void Kaldi2SubEncoder_Init(Kaldi2SubEncoder_t *self, ModelParser_t *parser);
void Kaldi2SubEncoder_Destroy(Kaldi2SubEncoder_t *self);

static inline void Kaldi2SubEncoder_Reset(Kaldi2SubEncoder_t *self) {
    Kaldi2ConvModule_Reset(&self->conv_module); }
void Kaldi2SubEncoder_Forward(Kaldi2SubEncoder_t *self, TensorValue_t *din, TensorValue_t *pe);
