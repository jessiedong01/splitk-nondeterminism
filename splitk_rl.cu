#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#define ROWS    8
#define VOCAB   256
#define THREADS 256
#define RUNS    500

__global__ void splitk(float *C, const float *A, const float *B, int K, int splits) {
    int n = blockIdx.x, m = blockIdx.y, s = blockIdx.z;
    int kb = (K + splits - 1) / splits;
    int k0 = s * kb, k1 = min(k0 + kb, K);
    float acc = 0.f;
    for (int k = k0 + threadIdx.x; k < k1; k += blockDim.x)
        acc += A[m * K + k] * B[(size_t)k * VOCAB + n];
    __shared__ float sm[THREADS];
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (int off = blockDim.x >> 1; off; off >>= 1) {
        if (threadIdx.x < off) sm[threadIdx.x] += sm[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(&C[m * VOCAB + n], sm[0]);
}

static void log_softmax(const float *logits, std::vector<double> &out) {
    double mx = -1e300;
    for (int i = 0; i < VOCAB; i++) mx = std::max(mx, (double)logits[i]);
    double s = 0;
    for (int i = 0; i < VOCAB; i++) s += exp((double)logits[i] - mx);
    double lse = mx + log(s);
    out.resize(VOCAB);
    for (int i = 0; i < VOCAB; i++) out[i] = (double)logits[i] - lse;
}
static double kl(const std::vector<double> &lp, const std::vector<double> &lq) {
    double d = 0;
    for (int i = 0; i < VOCAB; i++) d += exp(lp[i]) * (lp[i] - lq[i]);
    return d;
}
static double pct(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, (size_t)(v.size() * q))];
}

int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("%s (sm_%d%d)\n", p.name, p.major, p.minor);
    printf("logits[%d,%d] = hidden[%d,K] x lm_head[K,%d], %d runs per config\n", ROWS, VOCAB, ROWS, VOCAB, RUNS);
    printf("split=1 (no cross-block atomics) is the deterministic generation-time reference.\n");
    printf("ratio = exp(logprob_atomic - logprob_deterministic), PPO recompute of the same token.\n\n");

    int Ks[] = {4096, 16384};
    int Sv[] = {2, 4, 8, 16, 32};
    size_t outN = (size_t)ROWS * VOCAB;

    for (int ki = 0; ki < 2; ki++) {
        int K = Ks[ki];
        std::vector<float> hA((size_t)ROWS * K), hB((size_t)K * VOCAB);
        srand(7);
        auto gen = [&](std::vector<float> &v) {
            for (auto &x : v) {
                double u = (double)rand() / RAND_MAX;
                double n = ((double)rand() / RAND_MAX - 0.5) * 2.0;
                x = (float)(u < 0.01 ? n * 2.0 : n * 0.02);
            }
        };
        gen(hA); gen(hB);

        float *dA, *dB, *dC;
        cudaMalloc(&dA, hA.size() * 4); cudaMalloc(&dB, hB.size() * 4); cudaMalloc(&dC, outN * 4);
        cudaMemcpy(dA, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice);

        std::vector<float> det(outN);
        cudaMemset(dC, 0, outN * 4);
        splitk<<<dim3(VOCAB, ROWS, 1), THREADS>>>(dC, dA, dB, K, 1);
        cudaMemcpy(det.data(), dC, outN * 4, cudaMemcpyDeviceToHost);

        std::vector<std::vector<double>> detLP(ROWS);
        std::vector<int> topTok(ROWS), midTok(ROWS);
        for (int m = 0; m < ROWS; m++) {
            log_softmax(det.data() + m * VOCAB, detLP[m]);
            std::vector<int> idx(VOCAB);
            for (int i = 0; i < VOCAB; i++) idx[i] = i;
            std::sort(idx.begin(), idx.end(),
                      [&](int a, int b){ return detLP[m][a] > detLP[m][b]; });
            topTok[m] = idx[0];
            midTok[m] = idx[VOCAB / 2];
        }

        printf("K = %d\n", K);
        printf("%7s %12s %12s %12s %12s %12s %10s\n", "splits",
               "max dlp top", "p99 dlp top", "max dlp mid", "max |r-1|", "max KL", "clip>0.2");

        for (int si = 0; si < 5; si++) {
            int S = Sv[si];
            std::vector<double> dTop, dMid, rDev, kls;
            int outside = 0;
            std::vector<float> cur(outN);

            for (int r = 0; r < RUNS; r++) {
                cudaMemset(dC, 0, outN * 4);
                splitk<<<dim3(VOCAB, ROWS, S), THREADS>>>(dC, dA, dB, K, S);
                cudaMemcpy(cur.data(), dC, outN * 4, cudaMemcpyDeviceToHost);
                for (int m = 0; m < ROWS; m++) {
                    std::vector<double> lp;
                    log_softmax(cur.data() + m * VOCAB, lp);
                    double dt = fabs(lp[topTok[m]] - detLP[m][topTok[m]]);
                    double dm = fabs(lp[midTok[m]] - detLP[m][midTok[m]]);
                    double ratio = exp(lp[topTok[m]] - detLP[m][topTok[m]]);
                    dTop.push_back(dt); dMid.push_back(dm);
                    rDev.push_back(fabs(ratio - 1.0));
                    kls.push_back(kl(detLP[m], lp));
                    if (fabs(ratio - 1.0) > 0.2) outside++;
                }
            }
            printf("%7d %12.3e %12.3e %12.3e %12.3e %12.3e %6d/%-4d\n", S,
                   *std::max_element(dTop.begin(), dTop.end()), pct(dTop, 0.99),
                   *std::max_element(dMid.begin(), dMid.end()),
                   *std::max_element(rDev.begin(), rDev.end()),
                   *std::max_element(kls.begin(), kls.end()),
                   outside, RUNS * ROWS);
        }
        printf("\n");
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    return 0;
}
