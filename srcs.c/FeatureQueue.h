#pragma once

#include "asrcdefs.h"
#include "TensorValue.h"

#define OVERLAPPED_FRAMES 3
#define FRAME_SIZE        80

typedef struct {
    size_t window_size, first_unused_frame;
    TensorValue_t *body, *start, *accepter, *stop;
}  FeatureQueue_t;

void FeatureQueue_Init(FeatureQueue_t *self, size_t window_size, size_t limit);
void FeatureQueue_Destroy(FeatureQueue_t *self);

void FeatureQueue_ReInit(FeatureQueue_t *self, size_t window_size);
static inline void FeatureQueue_Reset(FeatureQueue_t *self) { self->first_unused_frame = 0; }
void FeatureQueue_PushFrame(FeatureQueue_t *self, const value_t *frame);
void FeatureQueue_PushLastFrame(FeatureQueue_t *self, const value_t *frame);
bool_t FeatureQueue_Pop(FeatureQueue_t *self, TensorValue_t *feature);

static inline size_t FeatureQueue_Size(const FeatureQueue_t *self) {
    return self->accepter - self->start; }
