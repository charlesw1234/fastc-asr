#pragma once

#include "../TensorValue.h"
#include "../ModelParser.h"

typedef struct {
    size_t vocab_size;
    const value_t *embedding_weight;
    const value_t *conv_weight;
} Kaldi2Decoder_t;

void Kaldi2Decoder_Init(Kaldi2Decoder_t *self, size_t vocab_size, ModelParser_t *parser);
static inline void Kaldi2Decoder_Destroy(Kaldi2Decoder_t *self) {};

void Kaldi2Decoder_Forward(Kaldi2Decoder_t *self, int *hyps, TensorValue_t *dout);
