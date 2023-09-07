#include "PositionEncoding.h"
#include <math.h>

#include "../predefines/pos_enc_div_term.c"

void Kaldi2PositionEncoding_Init(Kaldi2PositionEncoding_t *self, size_t max) {
    TensorValue_Init2(&self->pos_enc, 2 * max - 1, 512);
    const value_t *div_term = (const value_t *)pos_enc_div_term_hex;
    value_t *body_at = self->pos_enc.body;
    for (int i = (int)max - 1; i >= -(int)max + 1; --i) {
	for (int j = 0; j < 256; ++j) {
	    value_t coe = i * div_term[j];
	    *body_at++ = sin(coe);
	    *body_at++ = cos(coe);
	}
    }
    assert(self->pos_enc.body + TensorValue_NumValues(&self->pos_enc) == body_at);
}

void Kaldi2PositionEncoding_Fetch(Kaldi2PositionEncoding_t *self,
				  size_t size, TensorValue_t *dout) {
    size_t all_size = size * 2 - 1;
    TensorValue_Init2(dout, all_size, 512);
    size_t start = self->pos_enc.size[2] / 2 - size + 1;
    memcpy(dout->body, self->pos_enc.body + start * 512, all_size * 512 * sizeof(value_t));
}
