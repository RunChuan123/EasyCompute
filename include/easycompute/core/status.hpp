#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace ec {

enum class StatusCode {
  Ok = 0,
  InvalidArgument,
  NotFound,
  AlreadyExists,
  Unsupported,
  FailedPrecondition,
  Internal,
  VersionMismatch,
};

class Status {
public:
  Status() = default;
  Status(StatusCode code, std::string message, std::string domain = "easycompute.core")
      : code_(code), message_(std::move(message)), domain_(std::move(domain)) {}

  static Status success() { return {}; }
  bool ok() const { return code_ == StatusCode::Ok; }
  StatusCode code() const { return code_; }
  const std::string& message() const { return message_; }
  const std::string& domain() const { return domain_; }

private:
  StatusCode code_{StatusCode::Ok};
  std::string message_;
  std::string domain_;
};

template <typename T>
class Result {
public:
  Result(T value) : value_(std::move(value)) {}
  Result(Status status) : status_(std::move(status)) {
    if (status_.ok()) throw std::invalid_argument("a failed Result requires a non-OK status");
  }

  bool ok() const { return status_.ok(); }
  const Status& status() const { return status_; }
  T& value() & {
    if (!ok()) throw std::logic_error(status_.message());
    return *value_;
  }
  const T& value() const& {
    if (!ok()) throw std::logic_error(status_.message());
    return *value_;
  }
  T&& value() && {
    if (!ok()) throw std::logic_error(status_.message());
    return std::move(*value_);
  }

private:
  Status status_;
  std::optional<T> value_;
};

}  // namespace ec
