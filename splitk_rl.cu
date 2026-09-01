#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define ROWS    8
#define VOCAB   256
#define THREADS 256
#define RUNS    500
#define NTOK    (ROWS * VOCAB)

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

static void log_softmax(const float *logits, double *out) {
    double mx = -1e300;
    for (int i = 0; i < VOCAB; i++) mx = std::max(mx, (double)logits[i]);
    double s = 0;
    for (int i = 0; i < VOCAB; i++) s += exp((double)logits[i] - mx);
    double lse = mx + log(s);
    for (int i = 0; i < VOCAB; i++) out[i] = (double)logits[i] - lse;
}
static void all_lp(const std::vector<float> &lg, std::vector<double> &lp) {
    lp.resize(NTOK);
    for (int m = 0; m < ROWS; m++) log_softmax(&lg[m * VOCAB], &lp[m * VOCAB]);
}

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("%s (sm_%d%d)\n", p.name, p.major, p.minor);
    printf("logits[%d,%d] = hidden[%d,K] x lm_head[K,%d], %d runs per config\n\n",
           ROWS, VOCAB, ROWS, VOCAB, RUNS);
    printf("NONDET  = atomic run 0 vs atomic runs 1..%d, same split factor.\n", RUNS - 1);
    printf("          This is run-to-run nondeterminism.\n");
    printf("ALGO    = split=1 vs atomic split-K. Different reduction, not nondeterminism.\n");
    printf("          Matters when generation and training use different kernels.\n");
    printf("flips   = tokens whose PPO clip status changed, out of %d.\n", NTOK);
    printf("          Ratios are pre-seeded across [0.75,1.25] so the 0.8/1.2\n");
    printf("          boundaries are populated. Without this the test is vacuous.\n");

    int Ks[] = {4096, 16384};
    int Sv[] = {2, 4, 8, 16, 32};
    size_t outN = (size_t)ROWS * VOCAB;

    // pre-seed old-policy logprobs so ratios straddle the clip boundaries
    std::vector<double> ratioTarget(NTOK);
    for (int i = 0; i < NTOK; i++) ratioTarget[i] = 0.75 + 0.50 * i / (NTOK - 1);

    for (int ki = 0; ki < 2; ki++) {
        int K = Ks[ki];
        std::vector<float> hA((size_t)ROWS * K), hB((size_t)K * VOCAB);
        srand(1234);
        auto gen = [&](std::vector<float> &v) {
            for (auto &x : v) {
                double u = (double)rand() / RAND_MAX;
                double n = ((double)rand() / RAND_MAX - 0.5) * 2.0;
                x = (float)(u < 0.01 ? n * 2.0 : n * 0.02);
            }
        };
        gen(hA); gen(hB);

        float *dA, *dB, *dC;
        cudaMalloc(&dA, hA.size() * 4);
        cudaMalloc(&dB, hB.size() * 4);
        cudaMalloc(&dC, outN * 4);
        cudaMemcpy(dA, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice);

        auto launch = [&](int S, std::vector<float> &out) {
            out.resize(outN);
            cudaMemset(dC, 0, outN * 4);
            splitk<<<dim3(VOCAB, ROWS, S), THREADS>>>(dC, dA, dB, K, S);
            cudaMemcpy(out.data(), dC, outN * 4, cudaMemcpyDeviceToHost);
        };

        std::vector<float> lg1; launch(1, lg1);
        std::vector<double> lp1; all_lp(lg1, lp1);

        printf("\nK = %d\n", K);
        printf("%7s %13s %13s %13s %13s %8s\n",
               "splits", "NONDET dlp", "NONDET |r-1|", "ALGO dlp", "ALGO |r-1|", "flips");

        for (int si = 0; si < 5; si++) {
            int S = Sv[si];
            std::vector<float> lg0, lgr;
            std::vector<double> lp0, lpr;
            launch(S, lg0); all_lp(lg0, lp0);

            double ndDlp = 0, algDlp = 0;
            int flips = 0;
            for (int i = 0; i < NTOK; i++)
                algDlp = std::max(algDlp, fabs(lp0[i] - lp1[i]));

            for (int r = 1; r < RUNS; r++) {
                launch(S, lgr); all_lp(lgr, lpr);
                for (int i = 0; i < NTOK; i++) {
                    double d = fabs(lpr[i] - lp0[i]);
                    ndDlp = std::max(ndDlp, d);
                    // old-policy logprob implied by the seeded target ratio
                    double oldLp = lp0[i] - log(ratioTarget[i]);
                    double r0 = exp(lp0[i] - oldLp), r1 = exp(lpr[i] - oldLp);
                    bool c0 = (r0 < 0.8 || r0 > 1.2), c1 = (r1 < 0.8 || r1 > 1.2);
                    if (c0 != c1) flips++;
                }
            }
            printf("%7d %13.3e %13.3e %13.3e %13.3e %8d\n",
                   S, ndDlp, exp(ndDlp) - 1.0, algDlp, exp(algDlp) - 1.0, flips);
        }
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    return 0;
}
