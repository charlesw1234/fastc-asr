#include "Model.h"
#include "../ModelParser.h"
#include "../utils.h"
#include "../predefine_coe.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <math.h>
#include <unistd.h>

static void Kaldi2Model_DestroyV(Model_t *baseself) {
    Kaldi2Model_t *self = (Kaldi2Model_t *)baseself;
    Kaldi2Decoder_Destroy(&self->decoder);
    Kaldi2Joiner_Destroy(&self->joiner);
    Kaldi2Encoder_Destroy(&self->encoder);
    Kaldi2PositionEncoding_Destroy(&self->pos_enc);
    FeatureExtract_Destroy(&self->fe);
    munmap(self->model_body, self->model_size);
    close(self->model_fd);
    Vocab_Destroy(&self->vocab);
    Model_DestroyV(baseself);
}
static void Kaldi2Model_ResetV(Model_t *baseself) {
    Kaldi2Model_t *self = (Kaldi2Model_t *)baseself;
    FeatureExtract_Reset(&self->fe);
}

static bool_t Kaldi2Model_GreedySearch(Kaldi2Model_t *self,
				       char *result, size_t result_size,
				       TensorValue_t *encoder_out) {
    int hyps[512], *hyps_at = hyps + 2;
    int *hyps_last = hyps + sizeof(hyps) / sizeof(hyps[0]);
    TensorValue_t decoder_out;
    hyps[0] = hyps[1] = 0;
    Kaldi2Joiner_EncoderForward(&self->joiner, encoder_out);
    TensorValue_Init2(&decoder_out, 1, 512);
    Kaldi2Decoder_Forward(&self->decoder, hyps, &decoder_out);
    Kaldi2Joiner_DecoderForward(&self->joiner, &decoder_out);
    for (size_t index = 0; index < encoder_out->size[2]; ++index) {
	value_t *sub_encoder_out = encoder_out->body + index * 512;
	value_t *sub_decoder_out = decoder_out.body;
	TensorValue_t logit;
	TensorValue_Init2(&logit, 1, Vocab_Size(&self->vocab));
	Kaldi2Joiner_LinearForward(&self->joiner, sub_encoder_out, sub_decoder_out, &logit);
	value_t max_value; size_t max_at;
	findmax(logit.body, Vocab_Size(&self->vocab), &max_value, &max_at);
	TensorValue_Destroy(&logit);
	if (max_at == 0) continue;
	if (hyps_at >= hyps_last) return FALSE;
	*hyps_at++ = max_at;
	Kaldi2Decoder_Forward(&self->decoder, hyps_at - 2, &decoder_out);
	Kaldi2Joiner_DecoderForward(&self->joiner, &decoder_out);
    }
    TensorValue_Destroy(&decoder_out);
    return Vocab_Vector2String(&self->vocab, result, result_size,
			       hyps + 2, hyps_at - (hyps + 2));
}
static bool_t Kaldi2Model_ForwardV(Model_t *baseself, char *result, size_t result_size,
				   const value_t *din, size_t len, flag_t flag) {
    Kaldi2Model_t *self = (Kaldi2Model_t *)baseself;
    TensorValue_t feature;
    FeatureExtract_Insert(&self->fe, din, len, flag);
    FeatureExtract_Fetch(&self->fe, &feature);
    Kaldi2Encoder_Forward(&self->encoder, &feature);
    bool_t succ_ornot = Kaldi2Model_GreedySearch(self, result, result_size, &feature);
    TensorValue_Destroy(&feature);
    return succ_ornot;
}

static bool_t Kaldi2Model_ForwardChunkV(Model_t *baseself, char *result, size_t result_size,
					const value_t *din, size_t len, flag_t flag) {
    //Kaldi2Model_t *self = (Kaldi2Model_t *)baseself;
    snprintf(result, result_size, "Not implemented yet.");
    return FALSE;
}

static void Kaldi2Model_GlobalCMVNV(Model_t *baseself, value_t *dout) {
    uint32_t value = 0x34000000;
    value_t min_resol = *((value_t *)&value);
    for (size_t frame_at = 0; frame_at < FRAME_SIZE; ++frame_at)
	dout[frame_at] = log(dout[frame_at] < min_resol ? min_resol: dout[frame_at]);
}

bool_t Kaldi2Model_Init(Kaldi2Model_t *self,
			const char *model_fpath,
			const char *vocab_fpath) {
    ModelParser_t parser;
    struct stat model_st;
    self->base.destroy = Kaldi2Model_DestroyV;
    self->base.reset = Kaldi2Model_ResetV;
    self->base.forward = Kaldi2Model_ForwardV;
    self->base.forward_chunk = Kaldi2Model_ForwardChunkV;
    Model_Init(&self->base, (const value_t *)window_hex, window_hex_size, TRUE);
    self->base.global_cmvn = Kaldi2Model_GlobalCMVNV;
    if (!Vocab_Init(&self->vocab, vocab_fpath)) goto errquit0;
    if ((self->model_fd = open(model_fpath, O_RDONLY)) < 0) goto errquit1;
    if (fstat(self->model_fd, &model_st) < 0) goto errquit2;
    self->model_size = (size_t)model_st.st_size;
    self->model_body = (value_t *)
	mmap(NULL, (size_t)model_st.st_size, PROT_READ, MAP_SHARED, self->model_fd, 0);
    if (self->model_body == NULL) goto errquit2;
    ModelParser_Init(&parser, self->model_body, self->model_size);
    FeatureExtract_Init(&self->fe, &self->base);
    Kaldi2PositionEncoding_Init(&self->pos_enc, 5000);
    Kaldi2Encoder_Init(&self->encoder, &self->pos_enc, &parser);
    Kaldi2Decoder_Init(&self->decoder, Vocab_Size(&self->vocab), &parser);
    Kaldi2Joiner_Init(&self->joiner, &self->vocab, &parser);
    ModelParser_Destroy(&parser);
    return TRUE;
 errquit2: close(self->model_fd);
 errquit1: Vocab_Destroy(&self->vocab);
 errquit0: return FALSE;
}
