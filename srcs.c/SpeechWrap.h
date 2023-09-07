#pragma once

#include "asrcdefs.h"

typedef struct {
    size_t head_size, tail_size;
    value_t head[400]; // max size of all types of window size.
    const value_t *tail;
}  SpeechWrap_t;

static inline void SpeechWrap_Init(SpeechWrap_t *self) {
    self->head_size = self->tail_size = 0; }
static inline void SpeechWrap_Destroy(SpeechWrap_t *self) {}
static inline void SpeechWrap_Reset(SpeechWrap_t *self) {
    self->head_size = self->tail_size = 0; }
static inline size_t SpeechWrap_Size(SpeechWrap_t *self) {
    return self->head_size + self->tail_size; }
static inline float SpeechWrap_Get(const SpeechWrap_t *self, size_t idx) {
    return idx < self->head_size ? self->head[idx]: self->tail[idx - self->head_size]; }
void SpeechWrap_SetTail(SpeechWrap_t *self, const value_t *din, size_t len);
void SpeechWrap_Consume(SpeechWrap_t *self, size_t window_start);
