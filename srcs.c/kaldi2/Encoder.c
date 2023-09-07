#include "Encoder.h"

void Kaldi2Encoder_Init(Kaldi2Encoder_t *self,
			Kaldi2PositionEncoding_t *pos_enc, ModelParser_t *parser) {
    self->pos_enc = pos_enc;
    Kaldi2EmbedLayer_Init(&self->embed, parser);
    size_t num_sub_encoder = sizeof(self->sub_encoder) / sizeof(self->sub_encoder[0]);
    for (size_t index = 0; index < num_sub_encoder; ++index)
	Kaldi2SubEncoder_Init(self->sub_encoder + index, parser);
}
void Kaldi2Encoder_Destroy(Kaldi2Encoder_t *self) {
    size_t num_sub_encoder = sizeof(self->sub_encoder) / sizeof(self->sub_encoder[0]);
    for (size_t index = 0; index < num_sub_encoder; ++index)
	Kaldi2SubEncoder_Destroy(self->sub_encoder + index);
    Kaldi2EmbedLayer_Destroy(&self->embed);
}

void Kaldi2Encoder_Forward(Kaldi2Encoder_t *self, TensorValue_t *din) {
    size_t Tmax = din->size[2];
    TensorValue_t pe_code;
    size_t num_sub_encoder = sizeof(self->sub_encoder) / sizeof(self->sub_encoder[0]);
    Kaldi2EmbedLayer_Forward(&self->embed, din);
    Kaldi2PositionEncoding_Fetch(self->pos_enc, Tmax, &pe_code);
    for (size_t index = 0; index < num_sub_encoder; ++index)
	Kaldi2SubEncoder_Forward(self->sub_encoder + index, din, &pe_code);
    TensorValue_Destroy(&pe_code);
}
