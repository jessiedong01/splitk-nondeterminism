# split-K nondeterminism

Does split-K GEMM with `atomicAdd` accumulation actually produce different
results run to run, and does it matter for RL?

## Scope, read this first

These are **custom microbenchmark kernels**, not production GEMMs. One output
element per block, ordinary scalar FMA, a shared-memory tree reduction, and a
single `atomicAdd` per block. No tiling, no tensor cores, no `tcgen05`, no
CUTLASS, no cuBLAS.

Any result here describes **this kernel on this GPU while it is otherwise idle**.
It does not establish how cuBLAS or CUTLASS split-K behave. Their block layouts,
tile sizes, occupancy, and reduction strategies are different, and this
experiment suggests those are exactly the things that decide the answer.

## Programs

| file | question |
|---|---|
| `splitk_bench.cu` | Does the output change across runs? ULP, absolute and relative error. |
| `splitk_order.cu` | Do the blocks actually arrive in a different order each run? |
| `splitk_layout.cu` | Does the block-to-element mapping change the answer? |
| `splitk_rl.cu` | Logprob and PPO effects, run-to-run vs reduction-method. |

## Two different quantities

`splitk_rl` reports both, because they are easy to confuse:

- **NONDET** — atomic run 0 vs atomic runs 1..N at the *same* split factor.
  This is run-to-run nondeterminism.
- **ALGO** — split=1 vs atomic split-K. This is the difference between two
  reduction algorithms. It shows up when generation and training use different
  kernels. It is deterministic and it is not nondeterminism.

The PPO clip test seeds old-policy logprobs so ratios span [0.75, 1.25]. Starting
every ratio at exactly 1.0 and asking whether numerical noise pushes it past
0.8 or 1.2 is a rigged question with a guaranteed answer.

## Build

```
nvcc -O3 -arch=sm_100 splitk_bench.cu  -o splitk_bench     # sm_90 for H100
nvcc -O3 -arch=sm_100 splitk_order.cu  -o splitk_order
nvcc -O3 -arch=sm_100 splitk_layout.cu -o splitk_layout
nvcc -O3 -arch=sm_100 splitk_rl.cu     -o splitk_rl
```

`splits = 1` is the control. Each block owns a distinct output element, so no
two blocks accumulate into the same address and the result must be bit-exact.
If a `splits = 1` row shows any variation, the harness is broken.

## Running everything

```
./run_all.sh sm_100        # sm_90 for H100, sm_89 for 4090
```

Builds all six programs first and stops on the first compile error, so a typo
costs a minute rather than a session. Every kernel launch is followed by
`cudaGetLastError` + `cudaDeviceSynchronize`, because a launch that silently
fails produces a table of perfect zeros that looks like a finding.

Output lands in `b200-final-results.txt` with the commit hash, driver version
and device string at the top.

## What each program answers

- `splitk_order` — global block arrival order, via a counter. Kept for context
  only. Weak on its own: a scrambled global order does **not** show that the
  partials for one output element swapped places, and a counter is a second
  independent atomic, so a block can win the output `atomicAdd` and lose the
  counter race. That records an order which never happened.
- `splitk_layout` — the real per-element measurement, and the exact one. The
  probe uses the **return value of the output `atomicAdd`**: the accumulator
  value immediately before each split was added. Since `C` starts at zero, the
  split that saw `old == 0` went first, the one that saw `old == p_first` went
  second, and walking that chain recovers the true accumulation order. Runs
  under two block mappings (`s` slowest vs `s` fastest) and reports how often
  the order changed across runs, plus an `ambig` rate for the rare cases where
  two orders produce identical intermediates. Instrumented kernels are used
  only for order study; `splitk_bench` stays uninstrumented.
- `splitk_bench` — run-to-run differences: frequency, ULP, absolute, relative.
- `splitk_rl` — NONDET vs ALGO, plus a synthetic clip-boundary test. Flip
  counts are reported both as events out of `(runs-1) x tokens` and as distinct
  tokens out of `tokens`. The headline is the perturbation window, not the flip
  count: the seeded ratio grid is far coarser than the noise, so zero flips is
  expected and says little.
- `splitk_contend` — the same measurement idle and with a background kernel
  resident. Scheduling pressure is the most likely way to expose variation.
- `splitk_timing` — atomic split-K vs fixed-order split-K vs no split-K. This
  is the cost of determinism **in this scalar microbenchmark**, not the general
  cost of a deterministic GEMM.
