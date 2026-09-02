# split-K nondeterminism

Atomic split-K can give different answers across runs. These tests measure how much of that depends on where the split blocks sit in the grid, and how large the differences get.

## Result

When the blocks for each output were far apart, their partial answers were added in the same order every time. Every output matched across 200 runs.

When those blocks were placed next to each other, they added their answers in different orders. With 32 splits, all 4096 outputs changed at least once.

The calculation, inputs and number of splits stayed the same. Only the block arrangement changed.

| splits | blocks far apart | blocks next to each other | largest difference |
|---:|---:|---:|---:|
| 2 | 0 / 4096 | 0 / 4096 | none |
| 4 | 0 / 4096 | 3015 / 4096 | 0.005% |
| 8 | 0 / 4096 | 4092 / 4096 | 0.013% |
| 32 | 0 / 4096 | 4096 / 4096 | 0.031% |

The counts show how many outputs changed at least once across 200 runs on one B200.

The two-split result is also useful. The order changed, but every output still matched. This is because `a + b` gives the same result as `b + a`. With at least three partial answers, changing the order can also change where rounding happens.

## The first test

Before changing the block arrangement, the benchmark covered 96 combinations of:

- K size
- number of splits
- fp32, fp16 and bf16 inputs
- different input values

Each setup was repeated 500 times. Every output matched. Sixteen of the 96 use a single split, where each block owns its own output and nothing can compete, so those cannot vary either way.

This first result did not mean atomic split-K was always repeatable. It meant this particular block arrangement kept adding the partial answers in the same order.

## Token probabilities

The RL test separates two different kinds of changes:

- running the same atomic split-K kernel again
- switching between no split-K and split-K

Repeating the same layout did not change the log probabilities.

Switching between no split-K and split-K changed them by around `9.4e-7` to `1.9e-6`. This is a change caused by using different ways to do the calculation, not run-to-run nondeterminism.

## Speed

Making the addition order fixed was around 0–2% slower than the atomic version in this test.

Split-K itself did not help much at these matrix sizes. There was already enough work to use the B200, so adding more splits mostly added extra work.

## Limits

These are small custom test kernels. Each block calculates one output using regular multiply-add instructions.

They do not use the tiled tensor-core kernels found in cuBLAS or CUTLASS. The results show what happened in this kernel on one otherwise idle B200. They should not be treated as results for every split-K implementation.

The inputs also included some unusually large values that cancel each other. This makes changes from the addition order easier to see. The percentages above are the largest differences, not the usual difference.

## Programs

`splitk_layout.cu` is the main experiment. It records the value already stored in the output each time `atomicAdd` is called. This is used to recover the order in which the partial answers were added.

`splitk_bench.cu` measures run-to-run differences.

`splitk_rl.cu` checks changes to token log probabilities and a PPO-style calculation.

`splitk_timing.cu` compares no split-K, atomic split-K and fixed-order split-K.

`splitk_contend.cu` adds a background kernel. This only created light competition for the GPU, so its result should not be used to claim that other GPU work has no effect.

`splitk_order.cu` records when blocks finish across the whole GPU. It is included for context, but this alone cannot show whether the blocks working on one output changed order.

## Run

```bash
./run_all.sh sm_100
```

Use `sm_90` for an H100 or `sm_89` for an RTX 4090.

The script builds every program before running them. Results are saved to `b200-final-results.txt`.

Split factor 1 is the control. Every block has its own output, so there are no blocks competing to update the same value. If this result ever changes across runs, something is wrong with the test.
