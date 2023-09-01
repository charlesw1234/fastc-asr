#include "SpeechWrap.h"

void SpeechWrap_SetTail(SpeechWrap_t *self, const value_t *din, size_t len) {
    assert(self->tail_size == 0);
    self->tail = din; self->tail_size = len;
}
void SpeechWrap_Consume(SpeechWrap_t *self, size_t window_start) {
    assert(self->head_size <= window_start); // all head must be consumed.
    size_t remain_size = self->head_size + self->tail_size - window_start;
    assert(remain_size < sizeof(self->head) / sizeof(self->head[0]));
    if (remain_size > 0)
	memcpy(self->head, self->tail + (window_start - self->head_size),
	       sizeof(self->head[0]) * remain_size);
    self->head_size = remain_size;
    self->tail_size = 0;
}
