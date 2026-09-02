# split-K nondeterminism

Does split-K GEMM with atomicAdd accumulation actually produce a different
answer run to run, and if it does, is the difference large enough to matter?

## Result

Nondeterminism in this kernel is decided by the block-to-element grid mapping,
not by the number of splits. With the split index on the slowest-varying grid
dimension, the partial sums for a given output arrive in the same order every
time and the result is bit-identical across 200 runs. Moving the split index to
the fastest-varying dimension puts those same partial sums next to each other in
launch order, they race, and the result changes on every run. Same arithmetic,
same split count, same inputs.

| splits | blocks far apart | blocks adjacent | largest relative difference |
|---:|---:|---:|---:|
| 2 | 0 / 4096 | 0 / 4096 | none |
| 4 | 0 / 4096 | 3015 / 4096 | 5.12e-05 |
| 8 | 0 / 4096 | 4092 / 4096 | 1.33e-04 |
| 32 | 0 / 4096 | 4096 / 4096 | 3.08e-04 |

Counts are output elements that changed at least once across 200 runs on one
B200, with identical inputs every time.

The two-split row is the check that the measurement is honest. The arrival order
changed for every element and the answer never moved, because two values add to
the same result in either order. Floating-point addition is commutative; it is
associativity that fails, and you need at least three terms before order can
change anything.

Before any of this was rearranged, the original benchmark ran 96 configurations
of K, split factor, dtype and input distribution, 500 times each, and produced
zero differing elements everywhere. That result was not measuring a quiet GPU.
It was measuring the safe layout.

For reinforcement learning, the harness separates two quantities that are easy
to confuse. Run-to-run nondeterminism at a fixed split factor was exactly zero.
The difference between a split=1 reduction and an atomic split-K reduction,
which is what you see when generation and training use different kernels, was
9.4e-07 to 1.9e-06 in logprob terms, far below the PPO clipping band.

Fixed-order reduction cost 0 to 2 percent over the atomic version, so determinism
is close to free here. Split-K itself bought nothing at these shapes, because
4096 blocks already saturate 148 SMs.

## Scope

These are custom microbenchmark kernels: one output element per block, ordinary
scalar FMA, a shared-memory tree reduction, one atomicAdd per block. No tiling,
no tensor cores, no CUTLASS, no cuBLAS. Everything here describes this kernel on
this GPU while it is otherwise idle, and says nothing about how a production
GEMM behaves, since tile sizes, occupancy and reduction strategy are exactly the
things this experiment suggests are decisive.

The relative differences come from synthetic heavy-tailed inputs where large
terms cancel and leave a small result, which is the regime where addition order
matters most. They are maximums, not typical errors.

The contention run tested only weak pressure on the reproducible layout and
should not be read as evidence that contention has no effect.

## Programs

`splitk_layout.cu` is the main experiment. It recovers the true accumulation
order from the return value of the output atomicAdd, which is the accumulator
value immediately before each split landed. Since C starts at zero, the split
that saw zero went first, the one that saw the first partial went second, and
walking that chain gives the real order. A separate counter cannot do this,
because it is an independent atomic and a block can win the output atomicAdd and
lose the counter race.

`splitk_bench.cu` measures run-to-run differences in ULPs and absolute and
relative error. `splitk_rl.cu` reports logprob and PPO effects. `splitk_timing.cu`
compares atomic, fixed-order and no split-K. `splitk_contend.cu` repeats the
measurement with a background kernel resident. `splitk_order.cu` records global
block arrival order and is kept only for context, since a scrambled global order
does not show that the partials for one output swapped places.

## Running it

```
./run_all.sh sm_100        # sm_90 for H100, sm_89 for 4090
```

Everything builds first and stops on the first compile error, then runs, and the
output lands in b200-final-results.txt with the commit hash, driver version and
device string at the top. Every kernel launch is followed by cudaGetLastError and
cudaDeviceSynchronize, because a launch that silently fails produces a table of
perfect zeros that looks like a finding.

Split factor 1 is the control. Each block owns a distinct output element, so no
two blocks accumulate into the same address and the result must be bit-exact. If
a split=1 row ever varies, the harness is broken and nothing else means anything.
