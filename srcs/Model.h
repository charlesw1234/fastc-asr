#pragma once

#include "asrcdefs.h"

typedef struct Model Model_t;
struct Model {
    void (*destroy)(Model_t *self);
    void (*reset)(Model_t *self);
    bool_t (*forward)(Model_t *self, char *result, size_t result_size,
		      const value_t *din, size_t len, flag_t flag);
    bool_t (*forward_chunk)(Model_t *self, char *result, size_t result_size,
			    const value_t *din, size_t len, flag_t flag);
    FILE *logfp;
    const value_t *window;
    size_t window_size;
    bool_t fqueue_reinit;
    void (*global_cmvn)(Model_t *self, value_t *dout);
};

static inline void Model_Init(Model_t *self, FILE *logfp,
			      const value_t *window, size_t window_size,
			      bool_t fqueue_reinit) {
    self->logfp = logfp;
    self->window = window; self->window_size = window_size;
    self->fqueue_reinit = fqueue_reinit; }
static inline void Model_Destroy(Model_t *self) { self->destroy(self); }
static inline void Model_Reset(Model_t *self) { self->reset(self); }
static inline bool_t Model_Forward(Model_t *self, char *result, size_t result_size,
				   const value_t *din, size_t len, flag_t flag) {
    return self->forward(self, result, result_size, din, len, flag); }
static inline bool_t Model_ForwardChunk(Model_t *self, char *result, size_t result_size,
					const value_t *din, size_t len, flag_t flag) {
    return self->forward_chunk(self, result, result_size, din, len, flag); }
static inline void Model_GlobalCMVN(Model_t *self, value_t *dout) {
    self->global_cmvn(self, dout); }

static inline void Model_DestroyV(Model_t *self) {}
