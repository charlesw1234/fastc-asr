
#ifndef K2_DECODER_H
#define K2_DECODER_H

#include <stdint.h>

#include "../Tensor.hxx"
#include "ModelParams.hxx"

using namespace kaldi2;
namespace kaldi2 {

class Decoder {
  private:
    DecoderParams *params;
    int vocab_size;

  public:
    Decoder(DecoderParams *params, int vocab_size);
    ~Decoder();
    void forward(int *hyps, Tensor<float> *&dout);
};

} // namespace kaldi2

#endif
