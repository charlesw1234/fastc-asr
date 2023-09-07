#include "EncSelfAttn.h"
#include "../utils.h"
#include <cblas.h>

void Kaldi2EncSelfAttn_Init(Kaldi2EncSelfAttn_t *self, ModelParser_t *parser) {
    self->pos_bias_u = ModelParser_GetAt(parser, 8 * 64);
    self->pos_bias_v = ModelParser_GetAt(parser, 8 * 64);
    self->in_proj_weight = ModelParser_GetAt(parser, 1536 * 512);
    self->in_proj_bias = ModelParser_GetAt(parser, 1536);
    self->out_proj_weight = ModelParser_GetAt(parser, 512 * 512);
    self->out_proj_bias = ModelParser_GetAt(parser, 512);
    self->linear_pos_weight = ModelParser_GetAt(parser, 512 * 512);
}

void Kaldi2EncSelfAttn_Forward(Kaldi2EncSelfAttn_t *self, TensorValue_t *din, TensorValue_t *pe) {
    size_t Tmax = din->size[2];
    size_t Pmax = pe->size[2];
    size_t nn = 512 * 3;
    TensorValue_t linear_out, p;
    TensorValue_t q_with_bias_u, q_with_bias_v;
    TensorValue_t matrix_ac, matrix_bd, matrix_bd_new;
    TensorValue_t scores, tmp;

    TensorValue_Init2(&linear_out, Tmax, nn);
    TensorValue_Init2(&p, Pmax, 512);

    for (size_t index = 0; index < Tmax; ++index)
	memcpy(linear_out.body + index * nn, self->in_proj_bias, sizeof(value_t) * nn);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Tmax, nn, 512, 1,
		din->body, 512, self->in_proj_weight, 512, 1, linear_out.body, nn);
    TensorValue_Zero(&p);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Pmax, 512, 512, 1,
		pe->body, 512, self->linear_pos_weight, 512, 1, p.body, 512);

    TensorValue_Init2(&q_with_bias_u, Tmax, 512);
    TensorValue_Init2(&q_with_bias_v, Tmax, 512);

    for (size_t index = 0; index < Tmax; ++index)
	for (size_t j = 0; j < 512; ++j) {
	    size_t idx = index * 512 + j;
	    size_t ii = index * 512 * 3 + j;
	    value_t value = linear_out.body[ii] / 8;
	    q_with_bias_u.body[idx] = value + self->pos_bias_u[j];
	    q_with_bias_v.body[idx] = value + self->pos_bias_v[j];
	}

    TensorValue_Init3(&matrix_ac, Tmax, 8, Tmax);
    TensorValue_Init3(&matrix_bd, Tmax, 8, Pmax);
    TensorValue_Init3(&matrix_bd_new, Tmax, 8, Tmax);
    TensorValue_Zero(&matrix_ac);
    TensorValue_Zero(&matrix_bd);
    for (size_t index = 0; index < 8; ++index) {
	size_t offset1 = 64 * index;
	size_t offset2 = Tmax * index;
	size_t offset3 = Pmax * index;
	value_t *k_base_addr = linear_out.body + 512;
	cblas_sgemm(CblasRowMajor, CblasNoTrans ,CblasTrans, Tmax, Tmax, 64, 1,
		    q_with_bias_u.body + offset1, 512, k_base_addr + offset1,
		    512 * 3, 1, matrix_ac.body + offset2, Tmax * 8);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Tmax, Pmax, 64, 1,
		    q_with_bias_v.body + offset1, 512, p.body + offset1, 512, 1,
		    matrix_bd.body + offset3, Pmax * 8);
    }
    for (size_t index = 0; index < Tmax; ++index) {
	size_t offset3 = (Pmax) / 2 - index;
	for (size_t j = 0; j < 8; ++j) {
	    size_t offset1 = (index * 8 + j) * Tmax;
	    size_t offset2 = (index * 8 + j) * Pmax;
	    memcpy(matrix_bd_new.body + offset1,
		   matrix_bd.body + offset2 + offset3,
		   sizeof(value_t) * Tmax);
	}
    }

    TensorValue_Init3(&scores, Tmax, 8, Tmax);
    size_t body_size = TensorValue_NumValues(&scores);
    for (size_t index = 0; index < body_size; ++index)
	scores.body[index] = matrix_bd_new.body[index] + matrix_ac.body[index];
    for (size_t index = 0; index < Tmax * 8; ++index)
	softmax(scores.body + index * Tmax, Tmax, Tmax);

    TensorValue_Init2(&tmp, Tmax, 512);
    TensorValue_Zero(&tmp);
    for (size_t index = 0; index < 8; ++index) {
	size_t offset1 = Tmax * index;
	size_t offset2 = 64 * index;
	value_t *v_base_addr = linear_out.body + 1024;
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Tmax, 64, Tmax,
		    1, scores.body + offset1, Tmax * 8, v_base_addr + offset2,
		    512 * 3, 1, tmp.body + offset2, 512);
    }

    for (size_t index = 0; index < din->size[2]; ++index)
	memcpy(din->body + index * 512, self->out_proj_bias, 512 * sizeof(value_t));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, din->size[2], 512, 512,
		1, tmp.body, 512, self->out_proj_weight, 512, 1, din->body, 512);

    TensorValue_Destroy(&tmp);
    TensorValue_Destroy(&scores);
    TensorValue_Destroy(&matrix_bd_new);
    TensorValue_Destroy(&matrix_bd);
    TensorValue_Destroy(&matrix_ac);
    TensorValue_Destroy(&q_with_bias_v);
    TensorValue_Destroy(&q_with_bias_u);
    TensorValue_Destroy(&p);
    TensorValue_Destroy(&linear_out);
}
