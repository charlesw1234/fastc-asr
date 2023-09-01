#include "kaldi2/Model.h"
#include <sys/types.h>
#include <unistd.h>
#include <sndfile.h>
#include <openssl/evp.h>

static const char *logfpath_format = "tests/test-kaldi2-%u.jsons";
static const char *model_fpath = "models/k2_rnnt2_cli/wenet_params.bin";
static const char *vocab_fpath = "models/k2_rnnt2_cli/vocab.txt";
int main(int argc, char *argv[]) {
    char logfpath[128]; FILE *logfp;
    SF_INFO sfinfo; SNDFILE *sndfile;
    value_t *frames; char result[512];
    Kaldi2Model_t model;
    OpenSSL_add_all_digests();
    snprintf(logfpath, sizeof(logfpath), logfpath_format, (unsigned)getpid());
    logfp = fopen(logfpath, "w"); assert(logfp != NULL);
    printf("Log at: %s\n", logfpath);
    printf("Using: %s\n", sf_version_string());
    for (int index = 1; index < argc; ++index) {
	fprintf(logfp, "{ \"wavfpath\": \"%s\" }\n", argv[index]);
	sndfile = sf_open(argv[index], SFM_READ, &sfinfo);
	printf("%s:\n", argv[index]);
	printf("\tSample Rate: %d\n", (int)sfinfo.samplerate);
	printf("\tChannels: %d\n", (int)sfinfo.channels);
	printf("\tSections: %d\n", (int)sfinfo.sections);
	printf("\tFrames: %d\n", (int)sfinfo.frames);
	frames = (value_t *)malloc(sizeof(value_t) * sfinfo.frames);
	printf("\tLoad frames: %d\n", (int)sf_readf_float(sndfile, frames, sfinfo.frames));
	sf_close(sndfile);
	printf("\tModel Init: %s\n",
	       Kaldi2Model_Init(&model, logfp, model_fpath, vocab_fpath) ? "True": "False");
	Model_Reset(&model.base);
	printf("\tModel Forward: %s\n",
	       Model_Forward(&model.base, result, sizeof(result),
			     frames, sfinfo.frames, S_END) ? "True": "False");
	free(frames);
	printf("result: \"%s\"\n", result);
	Model_Destroy(&model.base);
    }
    fclose(logfp);
    return 0;
}
