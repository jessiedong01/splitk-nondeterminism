#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define M       8
#define NCOL    128
#define THREADS 256
#define RUNS    50

template <typename T>
__device__ float toF(T v);
template <> __device__ float toF<float>(float v)          { return v; }
template <> __device__ float toF<__half>(__half v)        { return __half2float(v); }
template <> __device__ float toF<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__global__ void splitk(float *C, const T *A, const T *B, int K, int splits) {
    int n = blockIdx.x, m = blockIdx.y, s = blockIdx.z;
    int kb = (K + splits - 1) / splits;
    int k0 = s * kb, k1 = min(k0 + kb, K);

    float acc = 0.f;
    for (int k = k0 + threadIdx.x; k < k1; k += blockDim.x)
        acc += toF(A[m * K + k]) * toF(B[(size_t)k * NCOL + n]);

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
    memcpy(&x, &a, 4); memcpy(&y, &b, 4);
    if (x < 0) x = 0x80000000 - x;
    if (y < 0) y = 0x80000000 - y;
    return abs(x - y);
}
static float to_fp16(float f) { return __half2float(__float2half(f)); }
static float to_bf16(float f) { return __bfloat162float(__float2bfloat16(f)); }

template <typename T>
static void sweep(const char *name, float (*cast)(float)) {
    int Ks[] = {1024, 4096, 16384, 65536};
    int Sv[] = {1, 2, 4, 8, 16, 32};
    size_t outN = (size_t)M * NCOL;

    printf("\n=== %s inputs, fp32 accumulate ===\n", name);
    printf("%8s %7s %9s %8s %8s %8s %8s %8s %9s\n",
           "K", "splits", "elems dif", "p50 ulp", "p99 ulp", "max ulp",
           "surv f16", "surv bf16", "argmax");

    for (int ki = 0; ki < 4; ki++) {
        int K = Ks[ki];
        std::vector<float> hA((size_t)M * K), hB((size_t)K * NCOL);
        srand(1234);
        for (auto &v : hA) v = (float)rand() / RAND_MAX - 0.5f;
        for (auto &v : hB) v = (float)rand() / RAND_MAX - 0.5f;
        std::vector<T> tA(hA.size()), tB(hB.size());
        for (size_t i = 0; i < hA.size(); i++) tA[i] = (T)hA[i];
        for (size_t i = 0; i < hB.size(); i++) tB[i] = (T)hB[i];

        T *dA, *dB; float *dC;
        cudaMalloc(&dA, tA.size() * sizeof(T));
        cudaMalloc(&dB, tB.size() * sizeof(T));
        cudaMalloc(&dC, outN * 4);
        cudaMemcpy(dA, tA.data(), tA.size() * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, tB.data(), tB.size() * sizeof(T), cudaMemcpyHostToDevice);

        for (int si = 0; si < 6; si++) {
            int S = Sv[si];
            std::vector<float> ref(outN), cur(outN);
            std::vector<int> refArg(M);
            std::vector<int> allUlp;
            size_t everDiff = 0, survF16 = 0, survBF16 = 0;
            int argFlips = 0;
            std::vector<char> touched(outN, 0);

            for (int r = 0; r < RUNS; r++) {
                cudaMemset(dC, 0, outN * 4);
                splitk<T><<<dim3(NCOL, M, S), THREADS>>>(dC, dA, dB, K, S);
                cudaMemcpy(cur.data(), dC, outN * 4, cudaMemcpyDeviceToHost);

                if (r == 0) {
                    ref = cur;
                    for (int m = 0; m < M; m++)
                        refArg[m] = (int)(std::max_element(ref.begin() + m * NCOL,
                                          ref.begin() + (m + 1) * NCOL) - (ref.begin() + m * NCOL));
                    continue;
                }
                for (size_t i = 0; i < outN; i++) {
                    if (cur[i] == ref[i]) continue;
                    touched[i] = 1;
                    allUlp.push_back(ulps(cur[i], ref[i]));
                    if (to_fp16(cur[i]) != to_fp16(ref[i])) survF16++;
                    if (to_bf16(cur[i]) != to_bf16(ref[i])) survBF16++;
                }
                for (int m = 0; m < M; m++) {
                    int a = (int)(std::max_element(cur.begin() + m * NCOL,
                                  cur.begin() + (m + 1) * NCOL) - (cur.begin() + m * NCOL));
                    if (a != refArg[m]) argFlips++;
                }
            }
            for (size_t i = 0; i < outN; i++) everDiff += touched[i];

            int p50 = 0, p99 = 0, mx = 0;
            if (!allUlp.empty()) {
                std::sort(allUlp.begin(), allUlp.end());
                p50 = allUlp[allUlp.size() / 2];
                p99 = allUlp[(size_t)(allUlp.size() * 0.99)];
                mx  = allUlp.back();
            }
            double totalDiff = (double)allUlp.size();
            printf("%8d %7d %6zu/%-3zu %8d %8d %8d %7.1f%% %8.1f%% %6d/%-3d\n",
                   K, S, everDiff, outN, p50, p99, mx,
                   totalDiff ? 100.0 * survF16 / totalDiff : 0.0,
                   totalDiff ? 100.0 * survBF16 / totalDiff : 0.0,
                   argFlips, (RUNS - 1) * M);
        }
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
}

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("%s (sm_%d%d)\n", p.name, p.major, p.minor);
    printf("C[%d,%d] = A[%d,K] x B[K,%d], %d runs per config, identical inputs\n", M, NCOL, M, NCOL, RUNS);
    printf("elems dif = output elements that ever differed from run 0\n");
    printf("surv f16/bf16 = share of differences that survive rounding the result to that format\n");
    printf("argmax = row argmax changes, out of (runs-1) x rows\n");

    sweep<float>("fp32", to_fp16);
    sweep<__half>("fp16", to_fp16);
    sweep<__nv_bfloat16>("bf16", to_bf16);
    return 0;
}
