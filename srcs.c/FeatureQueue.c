#include "FeatureQueue.h"

static inline void next_accepter(FeatureQueue_t *self) {
    if (++self->accepter < self->stop) return;
    size_t limit = 2 * (self->stop - self->body);
    TensorValue_t *new_body = (TensorValue_t *)
	realloc(self->body, sizeof(self->body[0]) * limit);
    assert(new_body != NULL);
    self->stop = new_body + limit;
    self->start = new_body + (self->start - self->body);
    self->accepter = new_body + (self->accepter - self->body);
    self->body = new_body;
}

void FeatureQueue_Init(FeatureQueue_t *self, size_t window_size, size_t limit) {
    self->window_size = window_size; self->first_unused_frame = 0;
    assert(self->window_size > OVERLAPPED_FRAMES); assert(limit > 0);
    self->body = (TensorValue_t *)malloc(sizeof(TensorValue_t) * limit);
    self->start = self->accepter = self->body; self->stop = self->body + limit;
    TensorValue_Init2(self->accepter, self->window_size, FRAME_SIZE);
}
void FeatureQueue_Destroy(FeatureQueue_t *self) {
    for (TensorValue_t *at = self->start; at <= self->accepter; ++at)
	TensorValue_Destroy(at);
    free(self->body);
}

void FeatureQueue_ReInit(FeatureQueue_t *self, size_t window_size) {
    TensorValue_Destroy(self->accepter);
    self->window_size = window_size;
    TensorValue_Init2(self->accepter, window_size, FRAME_SIZE);
    self->first_unused_frame = 0;
}
void FeatureQueue_PushFrame(FeatureQueue_t *self, const value_t *frame) {
    memcpy(self->accepter->body + FRAME_SIZE * self->first_unused_frame++,
	   frame, sizeof(frame[0]) * FRAME_SIZE);
    if (self->first_unused_frame < self->window_size) return;
    next_accepter(self);
    TensorValue_Init2(self->accepter, self->window_size, FRAME_SIZE);
    memcpy(self->accepter->body,
	   (self->accepter - 1)-> body +
	   (self->window_size - OVERLAPPED_FRAMES) * FRAME_SIZE,
	   sizeof(frame[0]) * OVERLAPPED_FRAMES * FRAME_SIZE);
    self->first_unused_frame = OVERLAPPED_FRAMES;
}
void FeatureQueue_PushLastFrame(FeatureQueue_t *self, const value_t *frame) {
    memcpy(self->accepter->body + FRAME_SIZE * self->first_unused_frame++,
	   frame, sizeof(frame[0]) * FRAME_SIZE);
    self->accepter->size[2] = self->first_unused_frame;
    next_accepter(self);
    TensorValue_Init2(self->accepter, self->window_size, FRAME_SIZE);
    self->first_unused_frame = 0;
}
bool_t FeatureQueue_Pop(FeatureQueue_t *self, TensorValue_t *feature) {
    if (self->start >= self->accepter) return FALSE;
    memcpy(feature, self->start++, sizeof(self->body[0]));
    return TRUE;
}
