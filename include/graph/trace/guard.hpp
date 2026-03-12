
#pragma once

#include "ctx.hpp"





// namespace EC::Tr
// {
//     struct TraceGuard{
//     ExecMode prev_mode;
//     ITracer* prev_ctx;

//     explicit TraceGuard(ITracer& now)
//     :prev_mode(g_mode),prev_ctx(current_tracer()){
//         g_mode = ExecMode::Trace;
//     }
//     ~TraceGuard(){
//         g_mode = prev_mode;
//         set_current_tracer(prev_ctx);
//     }
// };
// } // namespace EC::Tr
