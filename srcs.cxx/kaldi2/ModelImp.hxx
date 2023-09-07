
#ifndef K2_MODELIMP_H
#define K2_MODELIMP_H

#include "../Model.hxx"
#include <stdint.h>
#include <string>

#include "../FeatureExtract.hxx"
#include "../Tensor.hxx"
#include "../Vocab.hxx"
#include "Decoder.hxx"
#include "Encoder.hxx"
#include "Joiner.hxx"
#include "ModelParams.hxx"
#include "PositionEncoding.hxx"

using namespace kaldi2;

namespace kaldi2 {

class ModelImp : public Model {
  private:
    FILE *logfp;
    FeatureExtract *fe;
    kaldi2::ModelParamsHelper *p_helper;

    PositionEncoding *pos_enc;
    Encoder *encoder;
    Joiner *joiner;
    Decoder *decoder;
    Vocab *vocab;

  public:
    ModelImp(const char *path, int mode);
    ~ModelImp();
    void reset();
    string forward_chunk(float *din, int len, int flag);
    string forward(float *din, int len, int flag);
    string rescoring();

    string greedy_search(Tensor<float> *&encoder_out);
};

} // namespace kaldi2
#endif
