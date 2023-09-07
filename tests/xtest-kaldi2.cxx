#include "kaldi2/ModelImp.hxx"
#include "ComDefine.h"
#include <sndfile.h>
#include <openssl/evp.h>

static const char *model_dirpath = "models/k2_rnnt2_cli";
int main(int argc, char *argv[]) {
    SF_INFO sfinfo; SNDFILE *sndfile;
    float *frames;
    OpenSSL_add_all_digests();
    printf("Using: %s\n", sf_version_string());
    for (int index = 1; index < argc; ++index) {
	sndfile = sf_open(argv[index], SFM_READ, &sfinfo);
	printf("%s:\n", argv[index]);
	printf("\tSample Rate: %d\n", (int)sfinfo.samplerate);
	printf("\tChannels: %d\n", (int)sfinfo.channels);
	printf("\tSections: %d\n", (int)sfinfo.sections);
	printf("\tFrames: %d\n", (int)sfinfo.frames);
	frames = (float *)malloc(sizeof(float) * sfinfo.frames);
	printf("\tLoad frames: %d\n", (int)sf_readf_float(sndfile, frames, sfinfo.frames));
	sf_close(sndfile);
	Model *model = new kaldi2::ModelImp(model_dirpath, 2);
	model->reset();
	printf("result: %s\n", model->forward(frames, sfinfo.frames, S_END).c_str());
	free(frames);
	delete model;
    }
    return 0;
}
