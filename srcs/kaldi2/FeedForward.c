#include "FeedForward.h"
#include "../utils.h"
#include <cblas.h>

void Kaldi2FeedForward_Init(Kaldi2FeedForward_t *self, ModelParser_t *parser) {
    self->w1_weight = ModelParser_GetAt(parser, 2048 * 512);
    self->w1_bias = ModelParser_GetAt(parser, 2048);
    self->w2_weight = ModelParser_GetAt(parser, 512 * 2048);
    self->w2_bias = ModelParser_GetAt(parser, 512);
}

void Kaldi2FeedForward_Forward(Kaldi2FeedForward_t *self, TensorValue_t *din) {
    TensorValue_t tmp;
    size_t mm = TensorValue_NumValues(din) / din->size[3];
    TensorValue_Init2(&tmp, mm, 2048);
    for (size_t index = 0; index < mm; ++index)
	memcpy(tmp.body + index * 2048, self->w1_bias, 2048 * sizeof(value_t));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, mm, 2048, 512, 1,
		din->body, 512, self->w1_weight, 512, 1, tmp.body, 2048);
    doubleswish(&tmp);
    for (size_t index = 0; index < mm; ++index)
	memcpy(din->body + index * 512, self->w2_bias, 512 * sizeof(value_t));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, mm, 512, 2048, 1,
		tmp.body, 2048, self->w2_weight, 2048, 1, din->body, 512);
    TensorValue_Destroy(&tmp);
}
