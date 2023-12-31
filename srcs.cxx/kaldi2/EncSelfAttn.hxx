

#ifndef K2_ENCSELFATTN_H
#define K2_ENCSELFATTN_H

#include <stdint.h>

#include "../Tensor.hxx"
#include "FeedForward.hxx"
#include "ModelParams.hxx"

using namespace kaldi2;
namespace kaldi2 {
class EncSelfAttn {
  private:
    EncSelfAttnParams *params;

  public:
    EncSelfAttn(EncSelfAttnParams *params);
    ~EncSelfAttn();
    void forward(Tensor<float> *din, Tensor<float> *pe);
};
} // namespace kaldi2

#endif
