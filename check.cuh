#pragma once
#include <cstdio>
#include <cstdlib>

#define CK(x) do {                                                        \
    cudaError_t e_ = (x);                                                 \
    if (e_ != cudaSuccess) {                                              \
        fprintf(stderr, "\nCUDA ERROR at %s:%d\n  %s\n  -> %s\n",         \
                __FILE__, __LINE__, #x, cudaGetErrorString(e_));          \
        exit(1);                                                          \
    }                                                                     \
} while (0)

// call immediately after every kernel launch
#define CKLAUNCH() do {                                                   \
    CK(cudaGetLastError());                                               \
    CK(cudaDeviceSynchronize());                                          \
} while (0)
