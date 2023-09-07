#pragma once

#include "TensorValue.h"

void findmax(value_t *din, size_t len, value_t *max_value, size_t *max_at);
void basic_norm(TensorValue_t *din, value_t norm);
void relu(TensorValue_t *din);
void swish(TensorValue_t *din);
void sigmoid(TensorValue_t *din);
void doubleswish(TensorValue_t *din);
void softmax(value_t *din, size_t mask, size_t len);
void log_softmax(value_t *din, size_t len);
void glu(const TensorValue_t *din, TensorValue_t *dout);
