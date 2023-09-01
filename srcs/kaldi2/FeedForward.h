#pragma once

#include "../TensorValue.h"
#include "../ModelParser.h"

typedef struct {
    const value_t *w1_weight, *w1_bias;
    const value_t *w2_weight, *w2_bias;
} Kaldi2FeedForward_t;

void Kaldi2FeedForward_Init(Kaldi2FeedForward_t *self, ModelParser_t *parser);
static inline void Kaldi2FeedForward_Destroy(Kaldi2FeedForward_t *self) {}

void Kaldi2FeedForward_Forward(Kaldi2FeedForward_t *self, TensorValue_t *din);
