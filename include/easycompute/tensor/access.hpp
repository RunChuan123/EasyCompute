#pragma once

#include "easycompute/tensor/tensor.hpp"

namespace ec::detail {

// Narrow internal bridge: compute backends can inspect storage metadata without
// making TensorImpl part of the public Tensor API.
class TensorAccess {
public:
  static const TensorImpl& get(const Tensor& tensor) { return tensor.impl(); }
  static TensorImpl& get(Tensor& tensor) {
    (void)tensor.impl();
    return *tensor.impl_;
  }
};

}  // namespace ec::detail
