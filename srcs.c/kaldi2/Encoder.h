#pragma once

#include "PositionEncoding.h"
#include "EmbedLayer.h"
#include "SubEncoder.h"

typedef struct {
    Kaldi2PositionEncoding_t *pos_enc;
    Kaldi2EmbedLayer_t embed;
    Kaldi2SubEncoder_t sub_encoder[12];
} Kaldi2Encoder_t;

void Kaldi2Encoder_Init(Kaldi2Encoder_t *self,
			Kaldi2PositionEncoding_t *pos_enc, ModelParser_t *parser);
void Kaldi2Encoder_Destroy(Kaldi2Encoder_t *self);

static inline void Kaldi2Encoder_Reset(Kaldi2Encoder_t *self) {}
void Kaldi2Encoder_Forward(Kaldi2Encoder_t *self, TensorValue_t *din);
