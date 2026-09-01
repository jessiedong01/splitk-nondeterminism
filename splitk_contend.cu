// Does GPU contention expose nondeterminism that an idle GPU hides?
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include "check.cuh"

#define M       16
#define NCOL    256
#define THREADS 256
#define RUNS    300

__global__ void splitk(float *C, const float *A, const float *B,
                       int K, int splits, int *elemCtr, int *arrival) {
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
        int elem = m * NCOL + n;
        atomicAdd(&C[elem], sm[0]);
        int pos = atomicAdd(&elemCtr[elem], 1);
        arrival[(size_t)elem * splits + pos] = s;
    }
}

// persistent background load, stopped by a host-visible flag
__global__ void hog(volatile int *stop, float *sink) {
    float x = threadIdx.x * 1e-3f;
    while (*stop == 0) {
        for (int i = 0; i < 2048; i++) x = fmaf(x, 1.0000001f, 1e-7f);
    }
    if (x == 123456.f) sink[0] = x;   // never true, keeps the loop live
}

static int ulps(float a, float b) {
    if (a == b) return 0;
    int32_t x, y;
    memcpy(&x, &a, 4); memcpy(&y, &b, 4);
    if (x < 0) x = 0x80000000 - x;
    if (y < 0) y = 0x80000000 - y;
    return abs(x - y);
}

int main() {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("%s (sm_%d%d), %d SMs\n", p.name, p.major, p.minor, p.multiProcessorCount);
    printf("Same measurement on an idle GPU and with a background kernel resident.\n");
    printf("ord chg = %% of output elements whose partial-sum arrival order\n");
    printf("          differed from run 0 at least once.\n\n");

    const int K = 65536;
    int Sv[] = {2, 4, 8, 32};
    size_t outN = (size_t)M * NCOL;

    std::vector<float> hA((size_t)M * K), hB((size_t)K * NCOL);
    srand(1234);
    auto gen = [&](std::vector<float> &v) {
        for (auto &x : v) {
            double u = (double)rand() / RAND_MAX;
            double n = ((double)rand() / RAND_MAX - 0.5) * 2.0;
            x = (float)(u < 0.01 ? n * 20.0 : n * 0.1);
        }
    };
    gen(hA); gen(hB);

    float *dA, *dB, *dC, *dSink; int *dCtr, *dArr;
    CK(cudaMalloc(&dA, hA.size() * 4));
    CK(cudaMalloc(&dB, hB.size() * 4));
    CK(cudaMalloc(&dC, outN * 4));
    CK(cudaMalloc(&dSink, 4));
    CK(cudaMalloc(&dCtr, outN * sizeof(int)));
    CK(cudaMalloc(&dArr, outN * 32 * sizeof(int)));
    CK(cudaMemcpy(dA, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice));

    int *hStop; CK(cudaHostAlloc(&hStop, sizeof(int), cudaHostAllocMapped));
    int *dStop; CK(cudaHostGetDevicePointer(&dStop, hStop, 0));
    cudaStream_t hogStream, workStream;
    CK(cudaStreamCreateWithFlags(&hogStream,  cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&workStream, cudaStreamNonBlocking));

    for (int contended = 0; contended < 2; contended++) {
        if (contended) {
            *hStop = 0;
            // one block per SM: perturbs scheduling without filling the device
            hog<<<p.multiProcessorCount, 256, 0, hogStream>>>(dStop, dSink);
            CK(cudaGetLastError());
        }
        printf("--- %s ---\n", contended ? "CONTENDED (background kernel resident)"
                                         : "IDLE");
        printf("%7s %10s %12s %10s %12s %12s\n",
               "splits", "ord chg", "elems dif", "max ulp", "max abs", "max rel");

        for (int si = 0; si < 4; si++) {
            int S = Sv[si];
            std::vector<float> ref(outN), cur(outN);
            std::vector<int> arrRef((size_t)outN * S), arr((size_t)outN * S);
            std::vector<char> touched(outN, 0), ordChanged(outN, 0);
            int maxUlp = 0; double maxAbs = 0, maxRel = 0;

            for (int r = 0; r < RUNS; r++) {
                CK(cudaMemsetAsync(dC, 0, outN * 4, workStream));
                CK(cudaMemsetAsync(dCtr, 0, outN * sizeof(int), workStream));
                splitk<<<dim3(NCOL, M, S), THREADS, 0, workStream>>>(
                    dC, dA, dB, K, S, dCtr, dArr);
                CK(cudaGetLastError());
                CK(cudaMemcpyAsync(cur.data(), dC, outN * 4,
                                   cudaMemcpyDeviceToHost, workStream));
                CK(cudaMemcpyAsync(arr.data(), dArr, (size_t)outN * S * sizeof(int),
                                   cudaMemcpyDeviceToHost, workStream));
                CK(cudaStreamSynchronize(workStream));

                if (r == 0) { ref = cur; arrRef = arr; continue; }
                for (size_t e = 0; e < outN; e++) {
                    for (int i = 0; i < S; i++)
                        if (arr[e * S + i] != arrRef[e * S + i]) { ordChanged[e] = 1; break; }
                    if (cur[e] == ref[e]) continue;
                    touched[e] = 1;
                    maxUlp = std::max(maxUlp, ulps(cur[e], ref[e]));
                    double ae = fabs((double)cur[e] - (double)ref[e]);
                    maxAbs = std::max(maxAbs, ae);
                    if (ref[e] != 0.f) maxRel = std::max(maxRel, ae / fabs((double)ref[e]));
                }
            }
            size_t nd = 0, nord = 0;
            for (size_t e = 0; e < outN; e++) { nd += touched[e]; nord += ordChanged[e]; }
            printf("%7d %9.2f%% %6zu/%-5zu %10d %12.3e %12.3e\n",
                   S, 100.0 * nord / outN, nd, outN, maxUlp, maxAbs, maxRel);
        }
        if (contended) {
            *hStop = 1;
            CK(cudaStreamSynchronize(hogStream));
        }
        printf("\n");
    }
    return 0;
}
