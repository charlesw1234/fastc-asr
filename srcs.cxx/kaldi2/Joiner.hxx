
#ifndef K2_JOINER_H
#define K2_JOINER_H

#include <stdint.h>

#include "../Tensor.hxx"
#include "ModelParams.hxx"

using namespace kaldi2;
namespace kaldi2 {

class Joiner {
  private:
    JoinerParams *params;

  public:
    Joiner(JoinerParams *params);
    ~Joiner();
    void encoder_forward(Tensor<float> *&din);
    void decoder_forward(Tensor<float> *&din);
    void linear_forward(float *encoder, float *decoder, Tensor<float> *dout);
};

} // namespace kaldi2

#endif
