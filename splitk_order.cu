#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "check.cuh"

#define M       16
#define NCOL    256
#define THREADS 256
#define RUNS    100

__device__ int g_slot;

__global__ void probe(float *C, const float *A, const float *B,
                      int K, int splits, int *order) {
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
    if (threadIdx.x == 0) {
        atomicAdd(&C[m * NCOL + n], sm[0]);
        int slot = atomicAdd(&g_slot, 1);          // arrival ticket
        order[slot] = (s * gridDim.y + m) * gridDim.x + n;
    }
}

int main() {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("%s (sm_%d%d)\n", p.name, p.major, p.minor);
    printf("Do the split-K atomics actually arrive in a different order each run?\n");
    printf("%d runs per config. Ticket order recorded via a global atomic counter.\n\n", RUNS);
    printf("%8s %7s %8s %14s %14s\n",
           "K", "splits", "blocks", "runs identical", "mean pos match");

    int Ks[] = {4096, 65536};
    int Sv[] = {1, 2, 4, 8, 16, 32};
    size_t outN = (size_t)M * NCOL;

    for (int ki = 0; ki < 2; ki++) {
        int K = Ks[ki];
        std::vector<float> hA((size_t)M * K), hB((size_t)K * NCOL);
        srand(1234);
        for (auto &v : hA) v = (float)rand() / RAND_MAX - 0.5f;
        for (auto &v : hB) v = (float)rand() / RAND_MAX - 0.5f;

        float *dA, *dB, *dC;
        CK(cudaMalloc(&dA, hA.size() * 4));
        CK(cudaMalloc(&dB, hB.size() * 4));
        CK(cudaMalloc(&dC, outN * 4));
        CK(cudaMemcpy(dA, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice));

        for (int si = 0; si < 6; si++) {
            int S = Sv[si];
            size_t nblk = outN * S;
            int *dOrd; cudaMalloc(&dOrd, nblk * sizeof(int));
            std::vector<int> ref(nblk), cur(nblk);
            int identical = 0; double posMatch = 0;

            for (int r = 0; r < RUNS; r++) {
                int zero = 0;
                CK(cudaMemcpyToSymbol(g_slot, &zero, sizeof(int)));
                CK(cudaMemset(dC, 0, outN * 4));
                probe<<<dim3(NCOL, M, S), THREADS>>>(dC, dA, dB, K, S, dOrd);
                CKLAUNCH();
                CK(cudaMemcpy(cur.data(), dOrd, nblk * sizeof(int), cudaMemcpyDeviceToHost));
                if (r == 0) { ref = cur; continue; }
                size_t same = 0;
                for (size_t i = 0; i < nblk; i++) same += (cur[i] == ref[i]);
                if (same == nblk) identical++;
                posMatch += (double)same / nblk;
            }
            printf("%8d %7d %8zu %11d/%-3d %13.2f%%\n",
                   K, S, nblk, identical, RUNS - 1, 100.0 * posMatch / (RUNS - 1));
            CK(cudaFree(dOrd));
        }
        printf("\n");
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    return 0;
}
