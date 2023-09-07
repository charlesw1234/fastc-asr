#include "SubEncoder.h"
#include "../utils.h"

void Kaldi2SubEncoder_Init(Kaldi2SubEncoder_t *self, ModelParser_t *parser) {
    Kaldi2EncSelfAttn_Init(&self->self_attn, parser);
    Kaldi2FeedForward_Init(&self->feedforward, parser);
    Kaldi2FeedForward_Init(&self->feedforward_macaron, parser);
    Kaldi2ConvModule_Init(&self->conv_module, parser);
    self->norm = ModelParser_GetAt(parser, 1);
}
void Kaldi2SubEncoder_Destroy(Kaldi2SubEncoder_t *self) {
    Kaldi2ConvModule_Destroy(&self->conv_module);
    Kaldi2FeedForward_Destroy(&self->feedforward_macaron);
    Kaldi2FeedForward_Destroy(&self->feedforward);
    Kaldi2EncSelfAttn_Destroy(&self->self_attn);
}

void Kaldi2SubEncoder_Forward(Kaldi2SubEncoder_t *self, TensorValue_t *din, TensorValue_t *pe) {
    TensorValue_t residual;
    TensorValue_CopyInit(&residual, din);
    Kaldi2FeedForward_Forward(&self->feedforward_macaron, &residual);
    TensorValue_Add(din, 1, &residual);
    TensorValue_Destroy(&residual);

    TensorValue_CopyInit(&residual, din);
    Kaldi2EncSelfAttn_Forward(&self->self_attn, din, pe);
    TensorValue_Add(din, 1, &residual);
    TensorValue_Destroy(&residual);

    TensorValue_CopyInit(&residual, din);
    Kaldi2ConvModule_Forward(&self->conv_module, din);
    TensorValue_Add(din, 1, &residual);
    TensorValue_Destroy(&residual);

    TensorValue_CopyInit(&residual, din);
    Kaldi2FeedForward_Forward(&self->feedforward, &residual);
    TensorValue_Add(din, 1, &residual);
    TensorValue_Destroy(&residual);

    basic_norm(din, *self->norm);
}
