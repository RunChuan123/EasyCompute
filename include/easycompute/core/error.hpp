#pragma once

#include <stdexcept>

namespace ec {

class Error : public std::runtime_error { public: using std::runtime_error::runtime_error; };
class LayoutError final : public Error { public: using Error::Error; };
class DTypeError final : public Error { public: using Error::Error; };
class DeviceError final : public Error { public: using Error::Error; };
class TensorError final : public Error { public: using Error::Error; };

}  // namespace ec

