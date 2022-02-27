#include "wrapper.hpp"
#include <upfirdn2d.cpp>

extern "C" void *upfirdn2d_ffi(const void *x, const void *f, int upx, int upy,
                               int downx, int downy, int padx0, int padx1,
                               int pady0, int pady1, bool flip, float gain) {
  torch::Tensor tensor_ =
      upfirdn2d(*reinterpret_cast<const torch::Tensor *>(x),
                *reinterpret_cast<const torch::Tensor *>(f), upx, upy, downx,
                downy, padx0, padx1, pady0, pady1, flip, gain);
  torch::Tensor *tensor = new torch::Tensor(tensor_);
  return reinterpret_cast<void *>(tensor);
}
