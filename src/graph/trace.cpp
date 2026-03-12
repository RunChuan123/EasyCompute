#include "graph/trace/api.hpp"

namespace EC::Tr{
static thread_local ITracer* tls_tracer = nullptr;

ITracer* current_tracer() { return tls_tracer; }

// 仅给 TraceGuard/trace.cpp 用：设置当前 tracer
void set_current_tracer(ITracer* t) { tls_tracer = t; }

}