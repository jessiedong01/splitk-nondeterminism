#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

#define M       8
#define NCOL    128
#define THREADS 256
#define RUNS    50

__global__ void splitk(float *C, const float *A, const float *B, int K, int splits) {
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

static int ulps(float a, float b) {
    if (a == b) return 0;
    int32_t x, y;
    memcpy(&x, &a, 4);
    memcpy(&y, &b, 4);
    if (x < 0) x = 0x80000000 - x;
    if (y < 0) y = 0x80000000 - y;
    return abs(x - y);
}

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("%s (sm_%d%d)\n\n", p.name, p.major, p.minor);
    printf("C[%d,%d] = A[%d,K] x B[K,%d], fp32, %d runs per config\n\n", M, NCOL, M, NCOL, RUNS);

    int Ks[]      = {1024, 4096, 16384, 65536};
    int splitsv[] = {1, 2, 4, 8, 16, 32};
    size_t outN   = (size_t)M * NCOL;

    printf("%8s %7s %10s %10s %14s\n", "K", "splits", "runs diff", "max ulp", "max rel diff");

    for (int ki = 0; ki < 4; ki++) {
        int K = Ks[ki];
        std::vector<float> hA((size_t)M * K), hB((size_t)K * NCOL);
        srand(1234);
        for (auto &v : hA) v = (float)rand() / RAND_MAX - 0.5f;
        for (auto &v : hB) v = (float)rand() / RAND_MAX - 0.5f;

        float *dA, *dB, *dC;
        cudaMalloc(&dA, hA.size() * 4);
        cudaMalloc(&dB, hB.size() * 4);
        cudaMalloc(&dC, outN * 4);
        cudaMemcpy(dA, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice);

        for (int si = 0; si < 6; si++) {
            int S = splitsv[si];
            std::vector<float> ref(outN), cur(outN);
            int diffRuns = 0, maxUlp = 0;
            double maxRel = 0.0;

            for (int r = 0; r < RUNS; r++) {
                cudaMemset(dC, 0, outN * 4);
                splitk<<<dim3(NCOL, M, S), THREADS>>>(dC, dA, dB, K, S);
                cudaMemcpy(cur.data(), dC, outN * 4, cudaMemcpyDeviceToHost);
                if (r == 0) { ref = cur; continue; }

                bool differs = false;
                for (size_t i = 0; i < outN; i++) {
                    if (cur[i] != ref[i]) {
                        differs = true;
                        maxUlp = std::max(maxUlp, ulps(cur[i], ref[i]));
                        double rel = fabs((double)cur[i] - ref[i]) / std::max(1e-30, fabs((double)ref[i]));
                        maxRel = std::max(maxRel, rel);
                    }
                }
                if (differs) diffRuns++;
            }
            printf("%8d %7d %6d/%-3d %10d %14.3e\n", K, S, diffRuns, RUNS - 1, maxUlp, maxRel);
        }
        printf("\n");
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    return 0;
}
