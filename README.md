# How nondeterministic is split-K, actually?

"Split-K GEMMs are nondeterministic" is repeated constantly and rarely quantified.
This measures the magnitude, and what makes it larger.

## Why it happens

When K is large but M and N are small, one thread block per output tile does not fill
the GPU. Split-K divides the K dimension across blocks, each computes a partial sum, and
the partials are combined with `atomicAdd`. Floating-point addition is not associative,
so the order the partials land in changes the result in the low bits. That order depends
on which block finishes first, which varies between runs.

## Method

`C[8,128] = A[8,K] x B[K,128]` in fp32. Identical inputs every time, so any difference
between runs is nondeterminism rather than a different problem.

Each configuration runs 50 times. Run 0 is the reference; the other 49 are compared
against it. Reported per configuration:

- how many runs differed from the reference at all
- the largest gap in **ULPs**, the number of representable floats between two values
- the largest relative difference

Sweeps K over 1024, 4096, 16384, 65536 and the split factor over 1, 2, 4, 8, 16, 32.

The reduction inside each block is a deterministic tree, so the only source of variation
is the ordering of the `splits` atomic adds.

**Split factor 1 is the control.** One block per output element, no cross-block atomics,
so it must be bit-identical on every run. If it is not, the harness is wrong and nothing
else in the table means anything.

## Build and run

```bash
nvcc -O3 -arch=sm_90 splitk_bench.cu -o splitk_bench
./splitk_bench
```

Set `-arch` to match the card: `sm_80` A100, `sm_89` RTX 4090, `sm_90` H100, `sm_100` B200.

## Hypothesis, recorded before running

Split factor 1 is bit-exact. Variation appears from split factor 2 upward and grows with
both the split factor and K, since both increase the number of partial sums whose ordering
can change.

## Results

Not yet run.

## What this cannot show

Whether the differences change downstream outcomes in RL or other logprob-sensitive
training. That needs a training run, not a microbenchmark. What the numbers do give is a
noise floor to compare against the gaps that actually matter in those settings.
