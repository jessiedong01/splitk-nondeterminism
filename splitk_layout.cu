#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>

#define M       16
#define NCOL    256
#define THREADS 256
#define RUNS    200

// mapping 0: s is slowest  -> grid(NCOL, M, S), contributions to one element are M*NCOL apart
// mapping 1: s is fastest  -> grid(S, NCOL, M), contributions to one element are adjacent
__global__ void splitk_map(float *C, const float *A, const float *B,
                           int K, int splits, int mapping,
                           int *elemCtr, int *arrival) {
    int m, n, s;
    if (mapping == 0) { n = blockIdx.x; m = blockIdx.y; s = blockIdx.z; }
    else              { s = blockIdx.x; n = blockIdx.y; m = blockIdx.z; }

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
        int pos = atomicAdd(&elemCtr[elem], 1);      // per-element arrival slot
        arrival[(size_t)elem * splits + pos] = s;
    }
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
    cudaGetDeviceProperties(&p, 0);
    printf("%s (sm_%d%d)\n", p.name, p.major, p.minor);
    printf("Same math, same split count. Only the block->element mapping changes.\n");
    printf("s slowest = contributions to one output are M*NCOL blocks apart.\n");
    printf("s fastest = contributions to one output are adjacent.\n");
    printf("in-order = %% of output elements whose S partials arrived as s=0,1,2,...\n\n");

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

    float *dA, *dB, *dC; int *dCtr, *dArr;
    cudaMalloc(&dA, hA.size() * 4);
    cudaMalloc(&dB, hB.size() * 4);
    cudaMalloc(&dC, outN * 4);
    cudaMalloc(&dCtr, outN * sizeof(int));
    cudaMalloc(&dArr, outN * 32 * sizeof(int));
    cudaMemcpy(dA, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice);

    printf("%10s %7s %10s %12s %10s %10s\n",
           "mapping", "splits", "in-order", "elems dif", "p99 ulp", "max ulp");

    for (int mapping = 0; mapping < 2; mapping++) {
        for (int si = 0; si < 4; si++) {
            int S = Sv[si];
            std::vector<float> ref(outN), cur(outN);
            std::vector<int> arr((size_t)outN * S);
            std::vector<char> touched(outN, 0);
            std::vector<int> allUlp;
            double inOrderPct = 0;

            for (int r = 0; r < RUNS; r++) {
                cudaMemset(dC, 0, outN * 4);
                cudaMemset(dCtr, 0, outN * sizeof(int));
                dim3 grid = mapping == 0 ? dim3(NCOL, M, S) : dim3(S, NCOL, M);
                splitk_map<<<grid, THREADS>>>(dC, dA, dB, K, S, mapping, dCtr, dArr);
                cudaMemcpy(cur.data(), dC, outN * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(arr.data(), dArr, (size_t)outN * S * sizeof(int),
                           cudaMemcpyDeviceToHost);

                size_t ordered = 0;
                for (size_t e = 0; e < outN; e++) {
                    bool ok = true;
                    for (int i = 0; i < S; i++)
                        if (arr[e * S + i] != i) { ok = false; break; }
                    ordered += ok;
                }
                inOrderPct += 100.0 * ordered / outN;

                if (r == 0) { ref = cur; continue; }
                for (size_t i = 0; i < outN; i++) {
                    if (cur[i] == ref[i]) continue;
                    touched[i] = 1;
                    allUlp.push_back(ulps(cur[i], ref[i]));
                }
            }
            size_t everDiff = 0;
            for (size_t i = 0; i < outN; i++) everDiff += touched[i];
            int p99 = 0, mx = 0;
            if (!allUlp.empty()) {
                std::sort(allUlp.begin(), allUlp.end());
                p99 = allUlp[(size_t)(allUlp.size() * 0.99)];
                mx  = allUlp.back();
            }
            printf("%10s %7d %9.2f%% %7zu/%-4zu %10d %10d\n",
                   mapping == 0 ? "s slowest" : "s fastest", S,
                   inOrderPct / RUNS, everDiff, outN, p99, mx);
        }
        printf("\n");
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dCtr); cudaFree(dArr);
    return 0;
}
