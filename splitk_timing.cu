// What does determinism cost? atomic split-K vs fixed-order split-K vs no split-K.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "check.cuh"

#define M       16
#define NCOL    256
#define THREADS 256
#define ITERS   50

__global__ void nosplit(float *C, const float *A, const float *B, int K) {
    int n = blockIdx.x, m = blockIdx.y;
    float acc = 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        acc += A[m * K + k] * B[(size_t)k * NCOL + n];
    __shared__ float sm[THREADS];
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (int off = blockDim.x >> 1; off; off >>= 1) {
        if (threadIdx.x < off) sm[threadIdx.x] += sm[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) C[m * NCOL + n] = sm[0];
}

__global__ void atomicSplit(float *C, const float *A, const float *B, int K, int splits) {
    int n = blockIdx.x, m = blockIdx.y, s = blockIdx.z;
    int kb = (K + splits - 1) / splits;
    int k0 = s * kb, k1 = min(k0 + kb, K);
    float acc = 0.f;
    for (int k = k0 + threadIdx.x; k < k1; k += blockDim.x)
        acc += A[m * K + k] * B[(size_t)k * NCOL + n];
    __shared__ float sm[THREADS];
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (int off = blockDim.x >> 1; off; off >>= 1) {
        if (threadIdx.x < off) sm[threadIdx.x] += sm[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(&C[m * NCOL + n], sm[0]);
}

// pass 1: each split writes its own slot, no atomics
__global__ void fixedPart(float *P, const float *A, const float *B, int K, int splits) {
    int n = blockIdx.x, m = blockIdx.y, s = blockIdx.z;
    int kb = (K + splits - 1) / splits;
    int k0 = s * kb, k1 = min(k0 + kb, K);
    float acc = 0.f;
    for (int k = k0 + threadIdx.x; k < k1; k += blockDim.x)
        acc += A[m * K + k] * B[(size_t)k * NCOL + n];
    __shared__ float sm[THREADS];
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (int off = blockDim.x >> 1; off; off >>= 1) {
        if (threadIdx.x < off) sm[threadIdx.x] += sm[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) P[(size_t)s * M * NCOL + m * NCOL + n] = sm[0];
}

// pass 2: sum the slots in a fixed order, s ascending
__global__ void fixedReduce(float *C, const float *P, int splits) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= M * NCOL) return;
    float acc = 0.f;
    for (int s = 0; s < splits; s++) acc += P[(size_t)s * M * NCOL + e];
    C[e] = acc;
}

int main() {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("%s (sm_%d%d)\n", p.name, p.major, p.minor);
    printf("Cost of determinism IN THIS CUSTOM KERNEL. Not a general result. Fixed-order split-K uses two kernels and a\n");
    printf("preallocated workspace; both are included in its time.\n");
    printf("repro = fixed-order result bit-identical across %d runs.\n\n", ITERS);

    int Ks[] = {4096, 65536};
    int Sv[] = {2, 4, 8, 16, 32};
    size_t outN = (size_t)M * NCOL;

    for (int ki = 0; ki < 2; ki++) {
        int K = Ks[ki];
        std::vector<float> hA((size_t)M * K), hB((size_t)K * NCOL);
        srand(1234);
        for (auto &v : hA) v = (float)rand() / RAND_MAX - 0.5f;
        for (auto &v : hB) v = (float)rand() / RAND_MAX - 0.5f;

        float *dA, *dB, *dC, *dP;
        CK(cudaMalloc(&dA, hA.size() * 4));
        CK(cudaMalloc(&dB, hB.size() * 4));
        CK(cudaMalloc(&dC, outN * 4));
        CK(cudaMalloc(&dP, outN * 32 * 4));            // workspace, preallocated
        CK(cudaMemcpy(dA, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice));

        cudaEvent_t t0, t1;
        CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
        auto time_it = [&](auto fn) {
            fn(); CKLAUNCH();                            // warmup
            CK(cudaEventRecord(t0));
            for (int i = 0; i < ITERS; i++) fn();
            CK(cudaEventRecord(t1));
            CK(cudaEventSynchronize(t1));
            CK(cudaGetLastError());
            float ms; CK(cudaEventElapsedTime(&ms, t0, t1));
            return ms / ITERS;
        };

        float tNo = time_it([&] {
            nosplit<<<dim3(NCOL, M), THREADS>>>(dC, dA, dB, K);
        });

        printf("K = %d,  no split-K = %.4f ms\n", K, tNo);
        printf("%7s %12s %12s %12s %10s %7s\n",
               "splits", "atomic ms", "fixed ms", "fixed/atomic", "vs nosplit", "repro");

        for (int si = 0; si < 5; si++) {
            int S = Sv[si];
            float tAtom = time_it([&] {
                CK(cudaMemsetAsync(dC, 0, outN * 4));
                atomicSplit<<<dim3(NCOL, M, S), THREADS>>>(dC, dA, dB, K, S);
            });
            float tFix = time_it([&] {
                fixedPart<<<dim3(NCOL, M, S), THREADS>>>(dP, dA, dB, K, S);
                fixedReduce<<<(outN + 255) / 256, 256>>>(dC, dP, S);
            });

            // is the fixed-order path actually bit-reproducible?
            std::vector<float> ref(outN), cur(outN);
            bool repro = true;
            for (int i = 0; i < ITERS; i++) {
                fixedPart<<<dim3(NCOL, M, S), THREADS>>>(dP, dA, dB, K, S);
                CK(cudaGetLastError());
                fixedReduce<<<(outN + 255) / 256, 256>>>(dC, dP, S);
                CKLAUNCH();
                CK(cudaMemcpy(cur.data(), dC, outN * 4, cudaMemcpyDeviceToHost));
                if (i == 0) ref = cur;
                else for (size_t e = 0; e < outN; e++)
                    if (cur[e] != ref[e]) { repro = false; break; }
            }
            printf("%7d %12.4f %12.4f %12.2fx %9.2fx %7s\n",
                   S, tAtom, tFix, tFix / tAtom, tNo / tAtom, repro ? "yes" : "NO");
        }
        printf("\n");
        CK(cudaEventDestroy(t0)); CK(cudaEventDestroy(t1));
        CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(dC)); CK(cudaFree(dP));
    }
    return 0;
}
