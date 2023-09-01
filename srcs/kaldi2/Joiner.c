#include "Joiner.h"
#include <cblas.h>
#include <math.h>

void Kaldi2Joiner_Init(Kaldi2Joiner_t *self, const Vocab_t *vocab, ModelParser_t *parser) {
    self->vocab_size = Vocab_Size(vocab);
    self->encoder_proj_weight = ModelParser_GetAt(parser, 512 * 512);
    self->encoder_proj_bias = ModelParser_GetAt(parser, 512);
    self->decoder_proj_weight = ModelParser_GetAt(parser, 512 * 512);
    self->decoder_proj_bias = ModelParser_GetAt(parser, 512);
    self->output_linear_weight = ModelParser_GetAt(parser, self->vocab_size * 512);
    self->output_linear_bias = ModelParser_GetAt(parser, self->vocab_size);
}

void Kaldi2Joiner_EncoderForward(Kaldi2Joiner_t *self, TensorValue_t *din) {
    size_t index, mm = din->size[2];
    TensorValue_t dout;
    TensorValue_Init2(&dout, mm, 512);
    for (index = 0; index < mm; ++index)
        memcpy(dout.body + index * 512, self->decoder_proj_bias, sizeof(value_t) * 512);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, mm, 512, 512, 1,
                din->body, 512, self->encoder_proj_weight, 512, 1, dout.body, 512);
    TensorValue_Destroy(din);
    memcpy(din, &dout, sizeof(dout));
}
void Kaldi2Joiner_DecoderForward(Kaldi2Joiner_t *self, TensorValue_t *din) {
    TensorValue_t dout;
    TensorValue_Init2(&dout, 1, 512);
    memcpy(dout.body, self->decoder_proj_bias, sizeof(value_t) * 512);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, 512, 512, 1,
		din->body, 512, self->decoder_proj_weight, 512, 1, dout.body, 512);
    TensorValue_Destroy(din);
    memcpy(din, &dout, sizeof(dout));
}
void Kaldi2Joiner_LinearForward(Kaldi2Joiner_t *self, value_t *encoder,
				value_t *decoder, TensorValue_t *dout) {
    value_t din[512];
    for (size_t index = 0; index < 512; ++index)
	din[index] = tanh(encoder[index] + decoder[index]);
    memcpy(dout->body, self->output_linear_bias, self->vocab_size * sizeof(value_t));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, self->vocab_size, 512, 1,
		din, 512, self->output_linear_weight, 512, 1, dout->body, self->vocab_size);
}
