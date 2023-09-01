#pragma once

#include "asrcdefs.h"
#include <fftw3.h>
#include "SpeechWrap.h"
#include "FeatureQueue.h"
#include "Model.h"

#define FFT_SIZE 512

typedef struct {
    Model_t *model;
    SpeechWrap_t speech;
    FeatureQueue_t fqueue;
    value_t fft_input[FFT_SIZE];
    // when value_t is float, use fftwf.
    fftwf_complex fft_out[FFT_SIZE];
    fftwf_plan plan;
}  FeatureExtract_t;

void FeatureExtract_Init(FeatureExtract_t *self, Model_t *model);
void FeatureExtract_Destroy(FeatureExtract_t *self);

static inline void FeatureExtract_Reset(FeatureExtract_t *self) {
    SpeechWrap_Reset(&self->speech); FeatureQueue_Reset(&self->fqueue); }
static inline size_t FeatureExtract_Size(const FeatureExtract_t *self) {
    return FeatureQueue_Size(&self->fqueue); }

void FeatureExtract_Insert(FeatureExtract_t *self, const value_t *din, size_t len, flag_t flag);
static inline bool_t FeatureExtract_Fetch(FeatureExtract_t *self, TensorValue_t *feature) {
    return FeatureQueue_Pop(&self->fqueue, feature); }
