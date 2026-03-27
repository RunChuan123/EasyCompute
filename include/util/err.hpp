#pragma once

#include <exception>
#include <iostream>
#include <string>
#include <sstream>

namespace EC
{
    // 基础异常类：所有EasyCompute异常的父类
    class ECException : public std::exception {
    public:
        // 构造函数：接收错误信息
        explicit ECException(const std::string& msg) : err_msg_(msg) {}
        
        // 核心：重写what()方法，返回错误信息（const noexcept是必须的）
        const char* what() const noexcept override {
            return err_msg_.c_str();
        }

        // 可选：析构函数（基类是虚析构，这里可默认）
        ~ECException() override = default;

    protected:
        std::string err_msg_; // 存储错误信息
    };
    
    class TensorException : public ECException {
    public:
        TensorException(const std::string& msg, size_t tensor_id = 0) : ECException(msg) {
            // 可选：拼接更详细的错误信息（比如tensor ID）
            std::ostringstream oss;
            oss << "[TensorException] (ID: " << tensor_id << ") " << msg;
            err_msg_ = oss.str();
        }
    };
    class ShapeException : public ECException {
    public:
        ShapeException(const std::string& msg) : ECException(msg) {
            // 可选：拼接更详细的错误信息（比如tensor ID）
            std::ostringstream oss;
            oss << "[ShapeException] "  << msg;
            err_msg_ = oss.str();
        }
    };
    class DataException : public ECException {
    public:
        DataException(const std::string& msg) : ECException(msg) {
            // 可选：拼接更详细的错误信息（比如tensor ID）
            std::ostringstream oss;
            oss << "[DataException] "  << msg;
            err_msg_ = oss.str();
        }
    };
    class TypeException : public ECException {
    public:
        TypeException(const std::string& msg) : ECException(msg) {
            // 可选：拼接更详细的错误信息（比如tensor ID）
            std::ostringstream oss;
            oss << "[TypeException] "  << msg;
            err_msg_ = oss.str();
        }
    };
    class GraphException : public ECException {
    public:
        GraphException(const std::string& msg) : ECException(msg) {
            // 可选：拼接更详细的错误信息（比如tensor ID）
            std::ostringstream oss;
            oss << "[GraphException] "  << msg;
            err_msg_ = oss.str();
        }
    };
    class ExecuteException : public ECException {
    public:
        ExecuteException(const std::string& msg) : ECException(msg) {
            // 可选：拼接更详细的错误信息（比如tensor ID）
            std::ostringstream oss;
            oss << "[ExecuteException] "  << msg;
            err_msg_ = oss.str();
        }
    };
    class ContextException : public ECException {
    public:
        ContextException(const std::string& msg) : ECException(msg) {
            // 可选：拼接更详细的错误信息（比如tensor ID）
            std::ostringstream oss;
            oss << "[ContextException] " << msg;
            err_msg_ = oss.str();
        }
    };
    class TraceException : public ECException {
    public:
        TraceException(const std::string& msg) : ECException(msg) {
            // 可选：拼接更详细的错误信息（比如tensor ID）
            std::ostringstream oss;
            oss << "[TraceException] " << msg;
            err_msg_ = oss.str();
        }
    };
    class FunctionException : public ECException {
    public:
        FunctionException(const std::string& msg) : ECException(msg) {
            // 可选：拼接更详细的错误信息（比如tensor ID）
            std::ostringstream oss;
            oss << "[FunctionException] " << msg;
            err_msg_ = oss.str();
        }
    };
    class BufferException : public ECException {
    public:
        BufferException(const std::string& msg) : ECException(msg) {
            // 可选：拼接更详细的错误信息（比如tensor ID）
            std::ostringstream oss;
            oss << "[BufferException] " << msg;
            err_msg_ = oss.str();
        }
    };
    class DeviceException : public ECException {
    public:
        DeviceException(const std::string& msg) : ECException(msg) {
            // 可选：拼接更详细的错误信息（比如tensor ID）
            std::ostringstream oss;
            oss << "[DeviceException] " << msg;
            err_msg_ = oss.str();
        }
    };

} // namespace EC
