#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "easycompute/easycompute.hpp"

namespace {

int failures = 0;

void check(bool condition, const char* expression, int line) {
  if (!condition) { ++failures; std::cerr << "FAIL line " << line << ": " << expression << '\n'; }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

bool close(float lhs, float rhs, float tolerance = 1.0e-3F) { return std::fabs(lhs - rhs) <= tolerance; }

void test_int_tuple() {
  const ec::layout::Shape shape{{2, 3}, 4};
  CHECK(shape.str() == "((2,3),4)");
  CHECK(shape.product() == 24);
  CHECK(shape.depth() == 2);
  CHECK(shape.flatten() == std::vector<std::int64_t>({2, 3, 4}));
}

void test_layout() {
  using ec::layout::Layout;
  const auto left = Layout::left({2, 3});
  const auto right = Layout::right({2, 3});
  CHECK(left.str() == "(2,3):(1,2)");
  CHECK(right.str() == "(2,3):(3,1)");
  CHECK(left(ec::layout::Coord{1, 2}) == 5);
  CHECK(right(ec::layout::Coord{1, 2}) == 5);

  const auto transposed = right.transpose(0, 1);
  CHECK(transposed.str() == "(3,2):(1,3)");
  CHECK(transposed(ec::layout::Coord{2, 1}) == 5);
  CHECK(transposed.transpose(0, 1) == right);

  const Layout a({6, 2}, {8, 2});
  const Layout b({4, 3}, {3, 1});
  const auto composed = ec::layout::composition(ec::layout::LayoutFunction(a), ec::layout::LayoutFunction(b));
  CHECK(composed(0) == 0);
  CHECK(composed(1) == 24);
  CHECK(composed(2) == 2);
  CHECK(composed(3) == 26);
}

void test_dtype() {
  for (float value : {0.0F, 1.0F, -2.5F, 0.3333F, 65504.0F}) {
    const ec::Float16 half(value);
    CHECK(close(static_cast<float>(half), value, std::max(1.0e-3F, std::fabs(value) * 1.0e-3F)));
  }
}

void test_tensor_on(ec::Device device) {
  for (auto dtype : {ec::DType::Float32, ec::DType::Float16}) {
    const auto a = ec::Tensor::arange({2, 3}, dtype, device);
    const auto b = ec::Tensor::full({2, 3}, 2.0F, dtype, device);
    const auto c = a * b + b;
    const auto values = c.to_vector();
    for (std::size_t i = 0; i < values.size(); ++i) CHECK(close(values[i], static_cast<float>(i) * 2.0F + 2.0F));

    const auto t = a.transpose(0, 1);
    CHECK(t.shape().str() == "(3,2)");
    CHECK(t.stride().str() == "(1,3)");
    CHECK(t.to_vector() == std::vector<float>({0, 3, 1, 4, 2, 5}));
    const auto packed = t.contiguous();
    CHECK(packed.is_contiguous());
    CHECK(packed.to_vector() == t.to_vector());
    CHECK(a.at({1, 2}) == 5.0F);
  }
}

}  // namespace

int main() {
  test_int_tuple();
  test_layout();
  test_dtype();
  test_tensor_on(ec::Device::cpu());
  if (ec::cuda_available()) test_tensor_on(ec::Device::cuda());
  if (failures != 0) { std::cerr << failures << " check(s) failed\n"; return 1; }
  std::cout << "all EasyCompute tests passed\n";
  return 0;
}

