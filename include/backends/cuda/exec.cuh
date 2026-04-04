#pragma once
#include "backends/device_enabled.hpp"
#include "backends/abstract.hpp"

#ifdef EC_ENABLE_CUDA

namespace EC::Dev
{

struct CUDASTREAM final : public IStream{
public:
    CudaStream(DI device, cudaStream_t s, bool own = true);
    ~CudaStream();

    DI device() const override;
    void submit(Task* task) override;
    void synchronize() override;
    void wait_event(IEvent ev) override;
    IEvent record_event() override;

    cudaStream_t native() const { return stream_; }

private:
    DI device_;
    cudaStream_t stream_;
    bool own_;
};

}
#endif