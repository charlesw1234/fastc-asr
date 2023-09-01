#pragma once

#include "asrcdefs.h"

typedef struct {
    size_t size[4];
    value_t *body;
}  TensorValue_t;

void TensorValue_Init(TensorValue_t *self, size_t a, size_t b, size_t c, size_t d);
void TensorValue_CopyInit(TensorValue_t *self, const TensorValue_t *in);
static inline void TensorValue_Destroy(TensorValue_t *self) { free(self->body); }

static inline void TensorValue_Init1(TensorValue_t *self, size_t a) {
    TensorValue_Init(self, 1, 1, 1, a); }
static inline void TensorValue_Init2(TensorValue_t *self, size_t a, size_t b) {
    TensorValue_Init(self, 1, 1, a, b); }
static inline void TensorValue_Init3(TensorValue_t *self, size_t a, size_t b, size_t c) {
    TensorValue_Init(self, 1, a, b, c); }
static inline size_t TensorValue_NumValues(const TensorValue_t *self) {
    return self->size[0] * self->size[1] * self->size[2] * self->size[3]; }

void TensorValue_Zero(TensorValue_t *self);
void TensorValue_Add(TensorValue_t *self, value_t coe, const TensorValue_t *in);

void TensorValue_SaveDigest(const TensorValue_t *self, FILE *logfp, const char *prefix);
