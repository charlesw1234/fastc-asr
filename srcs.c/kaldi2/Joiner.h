#pragma once

#include "../TensorValue.h"
#include "../Vocab.h"
#include "../ModelParser.h"

typedef struct {
    size_t vocab_size;
    const value_t *encoder_proj_weight, *encoder_proj_bias;
    const value_t *decoder_proj_weight, *decoder_proj_bias;
    const value_t *output_linear_weight, *output_linear_bias;
} Kaldi2Joiner_t;

void Kaldi2Joiner_Init(Kaldi2Joiner_t *self, const Vocab_t *vocab, ModelParser_t *parser);
static inline void Kaldi2Joiner_Destroy(Kaldi2Joiner_t *self) {}

void Kaldi2Joiner_EncoderForward(Kaldi2Joiner_t *self, TensorValue_t *din);
void Kaldi2Joiner_DecoderForward(Kaldi2Joiner_t *self, TensorValue_t *din);
void Kaldi2Joiner_LinearForward(Kaldi2Joiner_t *self, value_t *encoder,
				value_t *decoder, TensorValue_t *dout);
