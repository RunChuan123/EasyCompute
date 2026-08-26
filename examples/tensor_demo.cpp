#include <iostream>

#include "easycompute/easycompute.hpp"

int main() {
  using ec::DType;
  using ec::Tensor;

  const auto a = Tensor::arange({2, 3}, DType::Float32);
  const auto b = Tensor::full({2, 3}, 0.5F, DType::Float16).to(DType::Float32);
  const auto c = a * a + b;

  std::cout << "a = " << a << '\n';
  std::cout << "a.T = " << a.transpose(0, 1) << '\n';
  std::cout << "a*a+b = " << c << '\n';

  if (ec::cuda_available()) {
    const auto gpu = a.to(ec::Device::cuda());
    std::cout << "CUDA a+a = " << gpu + gpu << '\n';
  }
}

