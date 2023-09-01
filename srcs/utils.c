#include "utils.h"
#include <math.h>

void findmax(value_t *din, size_t len, value_t *max_value, size_t *max_at) {
    assert(len > 0);
    *max_value = din[0];
    for (size_t index = 1; index < len; ++index)
	if (din[index] > *max_value) { *max_value = din[index]; *max_at = index; }
}

void basic_norm(TensorValue_t *din, value_t norm) {
    size_t Tmax = din->size[2];
    for (size_t i = 0; i < Tmax; ++i) {
	value_t sum = 0;
	for (size_t j = 0; j < 512; ++j) {
	    size_t ii = i * 512 + j;
	    sum += din->body[ii] * din->body[ii];
	}
	float mean = sqrt(sum / 512 + norm);
	for (size_t j = 0; j < 512; ++j) {
	    size_t ii = i * 512 + j;
	    din->body[ii] = din->body[ii] / mean;
	}
    }
}

void relu(TensorValue_t *din) {
    size_t body_size = TensorValue_NumValues(din);
    for (size_t index = 0; index < body_size; ++index) {
	value_t value = din->body[index];
	din->body[index] = value < 0 ? 0: value;
    }
}

void swish(TensorValue_t *din) {
    size_t body_size = TensorValue_NumValues(din);
    for (size_t index = 0; index < body_size; ++index) {
	value_t value = din->body[index];
	din->body[index] = value / (1 + exp(-value));
    }
}

void sigmoid(TensorValue_t *din) {
    size_t body_size = TensorValue_NumValues(din);
    for (size_t index = 0; index < body_size; ++index) {
	value_t value = din->body[index];
	din->body[index] = 1 / (1 + exp(-value));
    }
}

void doubleswish(TensorValue_t *din) {
    size_t body_size = TensorValue_NumValues(din);
    for (size_t index = 0; index < body_size; ++index) {
	value_t value = din->body[index];
	din->body[index] = value / (1 + exp(-value + 1));
    }
}

void softmax(value_t *din, size_t mask, size_t len) {
    value_t sum = 0, max = -INFINITY;
    value_t *tmp = (value_t *)malloc(mask * sizeof(value_t));
    for (size_t index = 0; index < mask; ++index)
	if (max < din[index]) max = din[index];
    for (size_t index = 0; index < mask; ++index) {
	tmp[index] = exp(din[index] - max);
	sum += tmp[index];
    }
    for (size_t index = 0; index < mask; ++index) din[index] = tmp[index] / sum;
    free(tmp);
    for (size_t index = mask; index < len; ++index) din[index] = 0;
}

void log_softmax(value_t *din, size_t len) {
    value_t sum = 0;
    value_t *tmp = (value_t *)malloc(len * sizeof(value_t));
    for (size_t index = 0; index < len; ++index)
	sum += tmp[index] = exp(din[index]);
    for (size_t index = 0; index < len; ++index)
	din[index] = log(tmp[index] / sum);
    free(tmp);
}

void glu(const TensorValue_t *din, TensorValue_t *dout) {
    size_t i, j, mm = TensorValue_NumValues(din) / 1024;
    for (i = 0; i < mm; ++i)
	for (j = 0; j < 512; ++j) {
	    size_t in_off = i * 1024 + j;
	    size_t out_off = i * 512 + j;
	    value_t a = din->body[in_off];
	    value_t b = din->body[in_off + 512];
	    dout->body[out_off] = a / (1 + exp(-b));
	}
}
