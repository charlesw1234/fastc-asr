#include "ConvModule.h"
#include "../utils.h"
#include <cblas.h>

void Kaldi2ConvModule_Init(Kaldi2ConvModule_t *self, ModelParser_t *parser) {
    self->pointwise_conv1_weight = ModelParser_GetAt(parser, 1024 * 512);
    self->pointwise_conv1_bias = ModelParser_GetAt(parser, 1024);
    self->depthwise_conv_weight = ModelParser_GetAt(parser, 512 * 31);
    self->depthwise_conv_bias = ModelParser_GetAt(parser, 512);
    self->pointwise_conv2_weight = ModelParser_GetAt(parser, 512 * 512);
    self->pointwise_conv2_bias = ModelParser_GetAt(parser, 512);
    TensorValue_Init2(&self->conv_cache, 14, 512);
    Kaldi2ConvModule_Reset(self);
}

void Kaldi2ConvModule_Reset(Kaldi2ConvModule_t *self) {
    TensorValue_t tmp;
    TensorValue_Init2(&tmp, 14, 1024);
    for (size_t index = 0; index < 14; ++index)
	memcpy(tmp.body + index * 1024, self->pointwise_conv1_bias, sizeof(value_t) * 1024);
    glu(&tmp, &self->conv_cache);
    TensorValue_Destroy(&tmp);
}
void Kaldi2ConvModule_Forward(Kaldi2ConvModule_t *self, TensorValue_t *din) {
    size_t mm = din->size[2];
    TensorValue_t tmp; TensorValue_Init2(&tmp, mm, 1024);
    for (size_t index = 0; index < mm; ++index)
	memcpy(tmp.body + index * 1024, self->pointwise_conv1_bias, sizeof(value_t) * 1024);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, mm, 1024, 512, 1,
		din->body, 512, self->pointwise_conv1_weight, 512, 1,
		tmp.body, 1024);
    glu(&tmp, din);
    TensorValue_Destroy(&tmp);

    TensorValue_t conv_in; TensorValue_Init2(&conv_in, 1, mm + 30);
    TensorValue_t blasin; TensorValue_Init2(&blasin, mm, 31);
    TensorValue_Zero(&conv_in);
    for (size_t i = 0; i < 512; ++i) {
	for (size_t j = 0; j < mm; ++j) {
	    size_t ii = j * 512 + i;
	    conv_in.body[j + 15] = din->body[ii];
	    din->body[ii] = self->depthwise_conv_bias[i];
	}
	for (size_t j = 0; j < mm; ++j)
	    memcpy(blasin.body + j * 31, conv_in.body + j, 31 * sizeof(value_t));
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mm, 1, 31, 1,
		    blasin.body, 31, self->depthwise_conv_weight + i * 31, 1, 1,
		    din->body + i, 512);
    }
    TensorValue_Destroy(&blasin);
    TensorValue_Destroy(&conv_in);
    doubleswish(din);

    TensorValue_CopyInit(&tmp, din);
    for (size_t index = 0; index < mm; ++index)
	memcpy(din->body + index * 512, self->pointwise_conv2_bias, 512 * sizeof(value_t));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, mm, 512, 512, 1,
		tmp.body, 512, self->pointwise_conv2_weight, 512, 1,
		din->body, 512);
    TensorValue_Destroy(&tmp);
}
