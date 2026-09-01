#pragma once
#include <vector>
#include <cstdio>

// Exact reduction-order probe.
//
// The instrumented kernel records, for each (element, split), the value the
// accumulator held immediately BEFORE that split was added -- the return value
// of the real output atomicAdd -- together with the partial it added.
//
// C starts at zero, so the split that saw old == 0 went first, the split that
// saw old == p_first went second, and so on. Walking that chain recovers the
// true accumulation order.
//
// This is exact where a counter-based probe is not: incrementing a separate
// counter is a second, independent atomic, so a block can win the output
// atomicAdd and then lose the counter race. That records an order which never
// happened.
//
// Returns true if the order was recovered unambiguously.
static inline bool reconstruct_order(const float *oldv, const float *part,
                                     int S, std::vector<int> &order) {
    order.clear();
    std::vector<char> used(S, 0);
    float acc = 0.f;
    for (int step = 0; step < S; step++) {
        int found = -1, matches = 0;
        for (int s = 0; s < S; s++) {
            if (used[s]) continue;
            if (oldv[s] == acc) { matches++; if (found < 0) found = s; }
        }
        if (matches != 1) return false;      // ambiguous: equal intermediates
        used[found] = 1;
        order.push_back(found);
        acc = acc + part[found];             // fp32, same rounding as atomicAdd
    }
    return true;
}
