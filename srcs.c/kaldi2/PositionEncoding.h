#pragma once

#include "../TensorValue.h"

typedef struct {
    TensorValue_t pos_enc;
} Kaldi2PositionEncoding_t;
void Kaldi2PositionEncoding_Init(Kaldi2PositionEncoding_t *self, size_t max);
static inline void Kaldi2PositionEncoding_Destroy(Kaldi2PositionEncoding_t *self) {
    TensorValue_Destroy(&self->pos_enc); }

void Kaldi2PositionEncoding_Fetch(Kaldi2PositionEncoding_t *self,
				  size_t size, TensorValue_t *dout);
