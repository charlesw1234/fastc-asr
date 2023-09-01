#include "Decoder.h"

void Kaldi2Decoder_Init(Kaldi2Decoder_t *self, size_t vocab_size, ModelParser_t *parser) {
    self->vocab_size = vocab_size;
    self->embedding_weight = ModelParser_GetAt(parser, vocab_size * 512);
    self->conv_weight = ModelParser_GetAt(parser, 512 * 2);
}
void Kaldi2Decoder_Forward(Kaldi2Decoder_t *self, int *hyps, TensorValue_t *dout) {
    TensorValue_t embed_out;
    TensorValue_Init2(&embed_out, 2, 512);
    for (size_t index = 0; index < 2; ++index)
	memcpy(embed_out.body + index * 512,
	       self->embedding_weight + hyps[index] * 512,
	       512 * sizeof(value_t));
    for (size_t index = 0; index < 512; ++index) {
	value_t value = embed_out.body[index] * self->conv_weight[index] +
	    embed_out.body[index + 512] * self->conv_weight[index + 512];
	dout->body[index] = value < 0 ? 0: value;
    }
    TensorValue_Destroy(&embed_out);
}
