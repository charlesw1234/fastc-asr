#pragma once

#include "asrcdefs.h"

typedef struct {
    const value_t *body, *at, *end;
} ModelParser_t;

static inline void ModelParser_Init(ModelParser_t *self, const value_t *body, size_t size) {
    self->body = self->at = body; self->end = body + size; }
static inline void ModelParser_Destroy(ModelParser_t *self) {}

const value_t *ModelParser_GetAt(ModelParser_t *self, size_t size);
