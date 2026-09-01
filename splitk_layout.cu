// Does the block->element mapping decide whether split-K is reproducible?
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include "check.cuh"
#include "order_probe.cuh"

#define M       16
#define NCOL    256
#define THREADS 256
#define RUNS    200

// mapping 0: s slowest -> grid(NCOL, M, S), contributions to one element M*NCOL apart
// mapping 1: s fastest -> grid(S, NCOL, M), contributions to one element adjacent
__global__ void splitk_probe(float *C, const float *A, const float *B,
                             int K, int splits, int mapping,
                             float *oldv, float *part) {
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
        float before = atomicAdd(&C[elem], sm[0]);   // the real atomic, its own return
        oldv[(size_t)elem * splits + s] = before;
        part[(size_t)elem * splits + s] = sm[0];
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
    CK(cudaGetDeviceProperties(&p, 0));
    printf("%s (sm_%d%d)\n", p.name, p.major, p.minor);
    printf("Same arithmetic, same split count. Only the block->element mapping differs.\n");
    printf("s slowest = the S partials for one output are M*NCOL blocks apart.\n");
    printf("s fastest = the S partials for one output are adjacent.\n\n");
    printf("Order is reconstructed from the output atomicAdd's own return value,\n");
    printf("not from a separate counter, so it is the order that actually produced C.\n");
    printf("  in-order = %% of elements accumulated as s=0,1,2,...\n");
    printf("  ord chg  = %% of elements whose order differed from run 0 at least once\n");
    printf("  ambig    = %% of (element,run) pairs where order was not recoverable\n");
    printf("Inputs are synthetic heavy-tailed values, not data from a real model.\n\n");

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

    float *dA, *dB, *dC, *dOld, *dPart;
    CK(cudaMalloc(&dA, hA.size() * 4));
    CK(cudaMalloc(&dB, hB.size() * 4));
    CK(cudaMalloc(&dC, outN * 4));
    CK(cudaMalloc(&dOld,  outN * 32 * 4));
    CK(cudaMalloc(&dPart, outN * 32 * 4));
    CK(cudaMemcpy(dA, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice));

    printf("%10s %7s %10s %10s %8s %12s %10s %11s\n",
           "mapping", "splits", "in-order", "ord chg", "ambig",
           "elems dif", "max ulp", "max rel");

    for (int mapping = 0; mapping < 2; mapping++) {
        for (int si = 0; si < 4; si++) {
            int S = Sv[si];
            std::vector<float> ref(outN), cur(outN);
            std::vector<float> hOld((size_t)outN * S), hPart((size_t)outN * S);
            std::vector<std::vector<int>> ordRef(outN);
            std::vector<char> touched(outN, 0), ordChanged(outN, 0);
            std::vector<int> ord;
            size_t ambig = 0, canonical = 0, ordSamples = 0;
            int maxUlp = 0; double maxRel = 0;

            for (int r = 0; r < RUNS; r++) {
                CK(cudaMemset(dC, 0, outN * 4));
                dim3 grid = mapping == 0 ? dim3(NCOL, M, S) : dim3(S, NCOL, M);
                splitk_probe<<<grid, THREADS>>>(dC, dA, dB, K, S, mapping, dOld, dPart);
                CKLAUNCH();
                CK(cudaMemcpy(cur.data(), dC, outN * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(hOld.data(), dOld, (size_t)outN * S * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(hPart.data(), dPart, (size_t)outN * S * 4, cudaMemcpyDeviceToHost));

                for (size_t e = 0; e < outN; e++) {
                    ordSamples++;
                    if (!reconstruct_order(&hOld[e * S], &hPart[e * S], S, ord)) {
                        ambig++;
                        continue;
                    }
                    bool canon = true;
                    for (int i = 0; i < S; i++) if (ord[i] != i) { canon = false; break; }
                    canonical += canon;
                    if (r == 0) ordRef[e] = ord;
                    else if (!ordRef[e].empty() && ord != ordRef[e]) ordChanged[e] = 1;
                }

                if (r == 0) { ref = cur; continue; }
                for (size_t i = 0; i < outN; i++) {
                    if (cur[i] == ref[i]) continue;
                    touched[i] = 1;
                    maxUlp = std::max(maxUlp, ulps(cur[i], ref[i]));
                    double ae = fabs((double)cur[i] - (double)ref[i]);
                    if (ref[i] != 0.f) maxRel = std::max(maxRel, ae / fabs((double)ref[i]));
                }
            }
            size_t everDiff = 0, nOrd = 0;
            for (size_t i = 0; i < outN; i++) { everDiff += touched[i]; nOrd += ordChanged[i]; }
            printf("%10s %7d %9.2f%% %9.2f%% %7.3f%% %6zu/%-5zu %10d %11.3e\n",
                   mapping == 0 ? "s slowest" : "s fastest", S,
                   100.0 * canonical / ordSamples, 100.0 * nOrd / outN,
                   100.0 * ambig / ordSamples, everDiff, outN, maxUlp, maxRel);
        }
        printf("\n");
    }
    CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(dC));
    CK(cudaFree(dOld)); CK(cudaFree(dPart));
    return 0;
}
