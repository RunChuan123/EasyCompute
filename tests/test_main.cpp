#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "easycompute/easycompute.hpp"

namespace {

int failures = 0;
int mock_api_token = 0;

void check(bool condition, const char* expression, int line) {
  if (!condition) { ++failures; std::cerr << "FAIL line " << line << ": " << expression << '\n'; }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

bool close(float lhs, float rhs, float tolerance = 1.0e-3F) { return std::fabs(lhs - rhs) <= tolerance; }

ec::Status register_mock_plugin(ec::Runtime&, ec::RegistrationTransaction& transaction) {
  return transaction.add(ec::CapabilityProvider{
      ec::ExtensionKey{"test.runtime", "mock_capability", 1},
      ec::ExtensionKey{"test.plugin", "mock", 1}, {}, {}, 1, 0, 7, &mock_api_token, {}});
}

ec::Status reject_mock_plugin(ec::Runtime&, ec::RegistrationTransaction& transaction) {
  const auto status = transaction.add(ec::CapabilityProvider{
      ec::ExtensionKey{"test.runtime", "rejected_capability", 1},
      ec::ExtensionKey{"test.plugin", "rejected", 1}, {}, {}, 1, 0, 0, &mock_api_token, {}});
  if (!status.ok()) return status;
  return {ec::StatusCode::InvalidArgument, "intentional registration failure", "test"};
}

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

void test_open_device_identity() {
  constexpr ec::DeviceType mock_type{42, "mock"};
  constexpr ec::Device mock_device{mock_type, 3};
  CHECK(mock_device.type() == mock_type);
  CHECK(mock_device.index() == 3);
  CHECK(mock_device.str() == "mock:3");
}

void test_identity_and_capabilities() {
  ec::InternTable ids;
  const ec::ExtensionKey add{"easycompute.core", "add", 1};
  const auto first = ids.intern(add);
  CHECK(first.valid());
  CHECK(ids.intern(add) == first);
  CHECK(ids.lookup(first) == add);
  CHECK(ids.size() == 1);

  ec::InternTable registry_ids;
  ec::CapabilityRegistry registry(registry_ids);
  const ec::ExtensionKey capability{"test.runtime", "kernel", 1};
  const ec::ExtensionKey slow{"test.provider", "slow", 1};
  const ec::ExtensionKey fast{"test.provider", "fast", 1};
  auto first_transaction = registry.begin();
  CHECK(first_transaction.add({capability, slow, {}, {}, 1, 0, 1, &mock_api_token, {}}).ok());
  CHECK(first_transaction.add({capability, fast, {}, {}, 1, 0, 10, &mock_api_token, {}}).ok());
  const auto initial_snapshot = registry.snapshot();
  CHECK(first_transaction.commit().ok());
  CHECK(initial_snapshot->epoch == 0);
  CHECK(initial_snapshot->providers.empty());
  const auto providers = registry.query(capability);
  CHECK(providers.size() == 2);
  CHECK(providers[0].provider == fast);
  CHECK(providers[0].capability_id.valid());
  CHECK(providers[0].provider_id.valid());
  CHECK(registry.epoch() == 1);

  bool deferred_ran = false;
  auto duplicate = registry.begin();
  CHECK(duplicate.add({capability, slow, {}, {}, 1, 0, 2, &mock_api_token, {}}).ok());
  duplicate.defer([&] { deferred_ran = true; return ec::Status::success(); });
  const auto duplicate_status = duplicate.commit();
  CHECK(!duplicate_status.ok());
  CHECK(duplicate_status.code() == ec::StatusCode::AlreadyExists);
  CHECK(!deferred_ran);

  auto stale = registry.begin();
  auto winner = registry.begin();
  CHECK(winner.add({ec::ExtensionKey{"test.runtime", "other", 1}, fast,
                    {}, {}, 1, 0, 0, &mock_api_token, {}}).ok());
  CHECK(winner.commit().ok());
  CHECK(!stale.commit().ok());
}

void test_runtime_and_plugins() {
  ec::Runtime first;
  ec::Runtime second;
  ec::ensure_builtin_plugins_registered(first);
  ec::ensure_builtin_plugins_registered(second);
  CHECK(&first.devices() != &second.devices());
  CHECK(&first.kernels() != &second.kernels());
  CHECK(first.plugins().is_loaded(ec::ExtensionKey{"easycompute.backend", "cpu", 1}));
  CHECK(!first.capabilities().query(
      ec::ExtensionKey{"easycompute.runtime", "memory_backend", 1}).empty());

  const ec::StaticPluginDescriptor mock{
      ec::ExtensionKey{"test.plugin", "mock", 1}, {1, 0, 0}, {{1, 0, 0}, {2, 0, 0}},
      {}, &register_mock_plugin};
  CHECK(first.plugins().load_static(mock).ok());
  CHECK(first.plugins().load_static(mock).ok());
  CHECK(first.capabilities().query(
      ec::ExtensionKey{"test.runtime", "mock_capability", 1}).size() == 1);

  const ec::StaticPluginDescriptor rejected{
      ec::ExtensionKey{"test.plugin", "rejected", 1}, {1, 0, 0}, {{1, 0, 0}, {2, 0, 0}},
      {}, &reject_mock_plugin};
  CHECK(!first.plugins().load_static(rejected).ok());
  CHECK(first.capabilities().query(
      ec::ExtensionKey{"test.runtime", "rejected_capability", 1}).empty());

  const ec::StaticPluginDescriptor missing_dependency{
      ec::ExtensionKey{"test.plugin", "dependent", 1}, {1, 0, 0}, {{1, 0, 0}, {2, 0, 0}},
      {{ec::ExtensionKey{"test.plugin", "missing", 1}, {{1, 0, 0}, {2, 0, 0}}}},
      &register_mock_plugin};
  const auto dependency_status = second.plugins().load_static(missing_dependency);
  CHECK(!dependency_status.ok());
  CHECK(dependency_status.code() == ec::StatusCode::FailedPrecondition);
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

    const auto doubled_transpose = t + t;
    CHECK(doubled_transpose.to_vector() == std::vector<float>({0, 6, 2, 8, 4, 10}));
  }
}

}  // namespace

int main() {
  test_int_tuple();
  test_layout();
  test_dtype();
  test_open_device_identity();
  test_identity_and_capabilities();
  test_runtime_and_plugins();
  CHECK(ec::device_available(ec::Device::cpu()));
  CHECK(ec::is_host_accessible(ec::Device::cpu()));
  test_tensor_on(ec::Device::cpu());
  if (ec::cuda_available()) test_tensor_on(ec::Device::cuda());
  if (failures != 0) { std::cerr << failures << " check(s) failed\n"; return 1; }
  std::cout << "all EasyCompute tests passed\n";
  return 0;
}
