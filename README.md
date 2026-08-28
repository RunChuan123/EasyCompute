# EasyCompute v2

EasyCompute is an educational tensor runtime being rebuilt around explicit layout algebra,
storage ownership, CPU/CUDA devices, and a stable graph IR.

The current vertical slice provides:

- nested integer tuples and affine `Layout(shape, stride)` values;
- general function composition through `LayoutFunction`;
- `float32` and IEEE binary16 storage;
- CPU and CUDA allocators with one-owner `Storage` lifetime;
- strided Tensor views, device/dtype copies, addition, multiplication, and printing;
- CPU tests that automatically exercise CUDA when a GPU is available.
- isolated runtimes with transactional, versioned capability and static-plugin registration.

```bash
cmake -S . -B build -DEC_ENABLE_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/tensor_demo
```

CUDA is enabled automatically when `nvcc` is available.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for invariants and the roadmap.
