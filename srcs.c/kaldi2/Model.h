#pragma once

#include "../Model.h"
#include "../FeatureExtract.h"
#include "../Vocab.h"
#include "PositionEncoding.h"
#include "Encoder.h"
#include "Joiner.h"
#include "Decoder.h"

typedef struct {
    Model_t base;
    Vocab_t vocab;
    int model_fd;
    size_t model_size;
    value_t *model_body;
    FeatureExtract_t fe;
    Kaldi2PositionEncoding_t pos_enc;
    Kaldi2Encoder_t encoder;
    Kaldi2Joiner_t joiner;
    Kaldi2Decoder_t decoder;
} Kaldi2Model_t;

bool_t Kaldi2Model_Init(Kaldi2Model_t *self,
			const char *model_fpath, const char *vocab_fpath);
void Kaldi2Model_Destroy(Kaldi2Model_t *self);
