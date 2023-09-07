#include "EmbedLayer.h"
#include "../utils.h"
#include <cblas.h>
#include <math.h>

void Kaldi2EmbedLayer_Init(Kaldi2EmbedLayer_t *self, ModelParser_t *parser) {
    self->conv0_weight = ModelParser_GetAt(parser, 72);
    self->conv0_bias = ModelParser_GetAt(parser, 8);
    self->conv1_weight = ModelParser_GetAt(parser, 2304);
    self->conv1_bias = ModelParser_GetAt(parser, 32);
    self->conv2_weight = ModelParser_GetAt(parser, 36864);
    self->conv2_bias = ModelParser_GetAt(parser, 128);
    self->out_weight = ModelParser_GetAt(parser, 512 * 2432);
    self->out_bias = ModelParser_GetAt(parser, 512);
    self->out_norm = ModelParser_GetAt(parser, 1);
}

static int *get_conv_index(size_t in_row, size_t in_column,
			   int kernel, int stride, int padding,
			   size_t *out_row, size_t *out_column) {
    *out_row = (size_t)(((int)in_row - kernel + stride + 2 * padding) / stride);
    *out_column = (size_t)(((int)in_column - kernel + stride + 2 * padding) / stride);
    size_t num_indexes = *out_row * *out_column * kernel * kernel;
    int *indexes = (int *)malloc(sizeof(int) * num_indexes);
    int *cur_indexes = indexes, *last_indexes = indexes + num_indexes;
    int column_start = 0 - padding, column_stop = (int)in_column - kernel + padding;
    int row_start = 0 - padding, row_stop = (int)in_row - kernel + padding;
    for (int row = row_start; row <= row_stop; row += stride) {
	for (int column = column_start; column <= column_stop; column += stride) {
	    for (int kernel_row = 0; kernel_row < kernel; ++kernel_row) {
		for (int kernel_column = 0; kernel_column < kernel; ++kernel_column) {
		    int sub_row = row + kernel_row;
		    int sub_column = column + kernel_column;
		    if ((sub_row >= 0) && (sub_row < in_row) &&
			(sub_column >= 0) && (sub_column < in_column))
			*cur_indexes++ = sub_row * in_column + sub_column;
		    else *cur_indexes++ = -1;
		}
	    }
	}
    }
    assert(cur_indexes == last_indexes);
    return indexes;
}
static void conv0_forward(Kaldi2EmbedLayer_t *self, TensorValue_t *din) {
    size_t conv_row, conv_column;
    size_t row = din->size[2], column = din->size[3];
    size_t kernel = 3, stride = 1, padding = 1, out_column = 8;
    int *conv_indexes = get_conv_index(row, column, kernel, stride, padding,
				       &conv_row, &conv_column);
    size_t conv_size = conv_row * conv_column;
    TensorValue_t blas_in, blas_out;
    TensorValue_Init2(&blas_in, conv_size, kernel * kernel);
    TensorValue_Init2(&blas_out, conv_size, out_column);
    size_t blas_in_body_size = TensorValue_NumValues(&blas_in);
    size_t blas_out_body_size = TensorValue_NumValues(&blas_out);

    for (size_t index = 0; index < blas_in_body_size; ++index) {
	int ii = conv_indexes[index];
	blas_in.body[index] = (ii == -1) ? 0: din->body[ii];
    }
    free(conv_indexes);
    TensorValue_Destroy(din);
    TensorValue_Init3(din, out_column, conv_row, conv_column);
    for (size_t index = 0; index < conv_size; ++index)
	memcpy(blas_out.body + index * out_column,
	       self->conv0_bias, out_column * sizeof(value_t));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		conv_size, out_column, kernel * kernel, 1,
		blas_in.body, kernel * kernel,
		self->conv0_weight, out_column, 1, blas_out.body, out_column);
    for (size_t index = 0; index < blas_out_body_size; ++index) {
	size_t kk = (index % out_column) * conv_size + (index / out_column);
	value_t value = blas_out.body[index];
	din->body[kk] = value / (1 + exp(-value + 1));
    }
    TensorValue_Destroy(&blas_out);
    TensorValue_Destroy(&blas_in);
}
static void conv1_forward(Kaldi2EmbedLayer_t *self, TensorValue_t *din) {
    size_t conv_row, conv_column;
    size_t row = din->size[2], column = din->size[3];
    size_t kernel = 3, stride = 2, padding = 0, out_column = 32;
    int *conv_indexes = get_conv_index(row, column, kernel, stride, padding,
				       &conv_row, &conv_column);
    size_t conv_size = conv_row * conv_column;
    TensorValue_t blas_in, blas_out;
    TensorValue_Init2(&blas_in, conv_size, kernel * kernel);
    TensorValue_Init2(&blas_out, conv_size, out_column);
    size_t blas_in_body_size = TensorValue_NumValues(&blas_in);
    size_t blas_out_body_size = TensorValue_NumValues(&blas_out);

    for (size_t index = 0; index < conv_size; ++index)
	memcpy(blas_out.body + index * out_column,
	       self->conv1_bias, out_column * sizeof(value_t));
    for (size_t index = 0; index < 8; ++ index) {
	value_t *sub_conv_in = din->body + index * row * column;
	const value_t *sub_weight = self->conv1_weight + index * out_column * kernel * kernel;
	for (size_t mm = 0; mm < blas_in_body_size; ++mm)
	    blas_in.body[mm] = sub_conv_in[conv_indexes[mm]];
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		    conv_size, out_column, kernel * kernel, 1,
		    blas_in.body, kernel * kernel, sub_weight, out_column, 1,
		    blas_out.body, out_column);
    }
    free(conv_indexes);
    TensorValue_Destroy(din);
    TensorValue_Init3(din, out_column, conv_row, conv_column);
    for (size_t index = 0; index < blas_out_body_size; ++index) {
	size_t kk = (index % out_column) * conv_size + (index / out_column);
	value_t value = blas_out.body[index];
	din->body[kk] = value / (1 + exp(-value + 1));
    }
    TensorValue_Destroy(&blas_out);
    TensorValue_Destroy(&blas_in);
}
static void conv2_forward(Kaldi2EmbedLayer_t *self, TensorValue_t *din) {
    size_t conv_row, conv_column;
    size_t row = din->size[2], column = din->size[3];
    size_t kernel = 3, stride = 2, padding = 0, out_column = 128;
    int *conv_indexes = get_conv_index(row, column, kernel, stride, padding,
				       &conv_row, &conv_column);
    size_t conv_size = conv_row * conv_column;
    TensorValue_t blas_in, blas_out;
    TensorValue_Init2(&blas_in, conv_size, kernel * kernel);
    TensorValue_Init2(&blas_out, conv_size, out_column);
    size_t blas_in_body_size = TensorValue_NumValues(&blas_in);
    size_t blas_out_body_size = TensorValue_NumValues(&blas_out);

    for (size_t index = 0; index < conv_size; ++index)
	memcpy(blas_out.body + index * out_column,
	       self->conv2_bias, out_column * sizeof(value_t));
    for (size_t index = 0; index < 32; ++ index) {
	value_t *sub_conv_in = din->body + index * row * column;
	const value_t *sub_weight = self->conv2_weight + index * out_column * kernel * kernel;
	for (size_t mm = 0; mm < blas_in_body_size; ++mm)
	    blas_in.body[mm] = sub_conv_in[conv_indexes[mm]];
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		    conv_size, out_column, kernel * kernel, 1,
		    blas_in.body, kernel * kernel, sub_weight, out_column, 1,
		    blas_out.body, out_column);
    }
    free(conv_indexes);
    TensorValue_Destroy(din);
    TensorValue_Init3(din, conv_row, out_column, conv_column);
    for (size_t index = 0; index < blas_out_body_size; ++index) {
	size_t ii = index / (out_column * 19);
	size_t jj = (index >> 7) % 19;
	size_t kk = index & 0x7F;
	size_t hh = ii * out_column * 19 + kk * 19 + jj;
	value_t value = blas_out.body[index];
	din->body[hh] = value / (1 + exp(-value + 1));
    }
    TensorValue_Destroy(&blas_out);
    TensorValue_Destroy(&blas_in);
}
static void linear_out_forward(Kaldi2EmbedLayer_t *self, TensorValue_t *din) {
    size_t Tmax = din->size[1];
    TensorValue_t dout;
    TensorValue_Init2(&dout, Tmax, 512);
    for (size_t index = 0; index < Tmax; ++index)
	memcpy(dout.body + index * 512, self->out_bias, 512 * sizeof(value_t));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Tmax, 512, 19 * 128, 1,
		din->body, 19 * 128, self->out_weight, 512, 1,
		dout.body, 512);
    TensorValue_Destroy(din);
    memcpy(din, &dout, sizeof(dout));
}
/*
static void norm_forward(Kaldi2EmbedLayer_t *self, TensorValue_t *din) {
    size_t Tmax = din->size[2];
    for (size_t i = 0; i < Tmax; ++i) {
	value_t sum = 0;
	for (size_t j = 0; j < 512; ++j) {
	    value_t value = din->body[i * 512 + j];
	    sum += value * value;
	}
	value_t mean = sqrt(sum / 512 + *self->out_norm);
	for (size_t j = 0; j < 512; ++j)
	    din->body[i * 512 + j] /= mean;
    }
}
*/

void Kaldi2EmbedLayer_Forward(Kaldi2EmbedLayer_t *self, TensorValue_t *din) {
    conv0_forward(self, din);
    conv1_forward(self, din);
    conv2_forward(self, din);
    linear_out_forward(self, din);
    basic_norm(din, *self->out_norm);
}
