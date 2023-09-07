#pragma once

#include "asrcdefs.h"

typedef struct {
    size_t start, end, len;
}  AudioFrame_t;

typedef struct {
}  Audio_t;

void Audio_Init(Audio_t *self, int data_type);
void Audio_Destroy(Audio_t *self);
void Audio_LoadWAV(Audio_t *self, const char *fpath);
