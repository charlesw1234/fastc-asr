#pragma once

#include "asrcdefs.h"

typedef struct {
    int fd;
    size_t fsize, nvocabs;
    char *start;
    const char **vocab_starts;
} Vocab_t;

bool_t Vocab_Init(Vocab_t *self, const char *vocab_fpath);
void Vocab_Destroy(Vocab_t *self);

static inline size_t Vocab_Size(const Vocab_t *self) { return self->nvocabs; }
bool_t Vocab_Vector2String(const Vocab_t *self, char *strbuf, size_t strbuf_size,
			   const int *hyps, size_t vector_size);
