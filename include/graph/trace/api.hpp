
#pragma once
#include <cstdint>
#include <vector>
#include <optional>
#include "tensor/TOp.hpp"

// namespace EC::AT { struct Tensor; }          // forward declare Tensor
// // namespace EC::Gr {  }

// namespace EC::Tr {

// using ValueId = int32_t;
// // enum class OpType : uint16_t { ew_add, ew_mul, sin /*...*/ };

// // 非 owning 的 trace 接口
// struct ITracer {
//     virtual ~ITracer() = default;

//     virtual bool is_tracing() const = 0;

//     // 把一个 Tensor 变成 graph 里的 ValueId（symbolic 的直接返回；真实的就 capture 成 Input/Const/Param）
//     virtual ValueId value_of(const AT::Tensor& t) = 0;

//     // 创建一个输出 Value，并返回它的 id（也可以让 record_op 返回 outputs）
//     virtual ValueId make_value_like(const AT::Tensor& like, bool requires_grad) = 0;

//     // 记录一个 op node：inputs -> outputs
//     virtual void record_op(TOp op,
//                            const std::vector<ValueId>& inputs,
//                            const std::vector<ValueId>& outputs) = 0;
// };

// // 全局/线程局部 tracer 指针（由 TraceGuard 设置）
// ITracer* current_tracer();
// void set_current_tracer(ITracer* t);

// } // namespace EC::Graph