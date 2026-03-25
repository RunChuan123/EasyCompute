
#pragma once

#include "ctx.hpp"


namespace EC::Tr
{

struct TraceGuard{

    explicit TraceGuard(TraceContext* ctx);
    ~TraceGuard();

    TraceGuard(const TraceGuard&)=delete;
    TraceGuard& operator=(const TraceGuard&)=delete;

private:
    TraceMode prev_mode_ = TraceMode::Off;
    TraceContext* prev_ctx_ = nullptr;
};

inline bool is_tracing(){
    return current_trace_mode() == TraceMode::On && current_tracer() != nullptr;
}

}