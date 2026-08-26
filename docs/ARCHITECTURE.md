# EasyCompute v2 architecture

## Non-negotiable invariants

1. `Storage` exclusively owns an allocation and releases it through the allocator that created it.
2. `Layout` owns no memory. It is a coordinate-to-offset function represented by congruent nested shape and stride tuples.
3. A Tensor view owns a new `TensorImpl`, shares `Storage`, and carries its own layout and storage offset.
4. Tensor copies share `TensorImpl`; `clone()` allocates new storage.
5. DType and Device combinations are either tested or rejected. There are no advertised placeholder types.
6. Eager autograd graphs and serializable trace graphs are separate systems.
7. Every commit on the rewrite branch must build and pass the CPU test suite.

## Layout model

An affine layout is a function from a congruent nested coordinate to a linear offset:

```text
L(c) = sum(flatten(c)[i] * flatten(stride)[i])
```

Linear coordinates follow CuTe's colexicographic convention. Tensor presentation uses row-major
logical coordinate enumeration and asks the layout for the corresponding physical offset.

`LayoutFunction` is the type-erased boundary for general layout functions. Composition is exact:

```text
(B o A)(x) = B(A(x))
```

It deliberately exposes no stride because not every composition is affine. Future layout work
will add canonicalization, complement, logical divide/product, tractability checks, and static
integer tuple specializations without changing the Tensor/Storage ownership boundary.

## Planned graph boundary

The graph IR will use integer IDs rather than owning pointers:

```text
Graph -> Value[] + Node[] + inputs[] + outputs[]
Node  -> opcode + input ValueIds + output ValueIds + typed attributes
Value -> TensorMeta + producer/users + stable name
```

Serialization will be versioned and deterministic. Import performs schema, ID, topology, dtype,
device, shape, and attribute validation before producing an executable graph. Constants will be
stored in a separate versioned tensor-data section rather than embedded as process pointers.

## Roadmap

1. Finish affine/nested layout laws and diagnostics.
2. Finish CPU/CUDA Tensor vertical slice and sanitizer tests.
3. Add operation schemas and shape/dtype inference.
4. Add eager autograd and mutation version counters.
5. Add versioned graph IR, text/binary import/export, trace capture, and replay.
6. Add reductions and matrix multiplication.
7. Add layout algebra operations used by tiled CUDA kernels.
