#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: %s (%s)\n", msg, cudaGetErrorString(err));
        exit(0);
    }
}

struct ClassStats {
    double avgR, avgG, avgB;
    double invCov[3][3];
    double logDet;
};

__constant__ ClassStats c_stats[32];
__constant__ int c_numClasses;


__global__ void classifyKernel(uchar4* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        uchar4 p = data[i];
        int bestClass = 0;
        double maxF = -1e300;
        for (int c = 0; c < c_numClasses; c++) {
            double dR = (double)(p.x - c_stats[c].avgR);
            double dG = (double)(p.y - c_stats[c].avgG);
            double dB = (double)(p.z - c_stats[c].avgB);
            double m0 = dR * c_stats[c].invCov[0][0] + dG * c_stats[c].invCov[1][0] + dB * c_stats[c].invCov[2][0];
            double m1 = dR * c_stats[c].invCov[0][1] + dG * c_stats[c].invCov[1][1] + dB * c_stats[c].invCov[2][1];
            double m2 = dR * c_stats[c].invCov[0][2] + dG * c_stats[c].invCov[1][2] + dB * c_stats[c].invCov[2][2];
            double mah = m0 * dR + m1 * dG + m2 * dB;
            double f = -mah - c_stats[c].logDet;

            if (f > maxF) {
                maxF = f;
                bestClass = c;
            }
        }
        data[i].w = (unsigned char)bestClass;
    }
}

int main() {
    char inPath[256], outPath[256];
    if (scanf("%s %s", inPath, outPath) != 2) return 0;

    FILE* f = fopen(inPath, "rb");
    if (!f) return 0;
    int w, h;
    fread(&w, sizeof(int), 1, f);
    fread(&h, sizeof(int), 1, f);
    int n = w * h;
    std::vector<uchar4> image(n);
    fread(image.data(), sizeof(uchar4), n, f);
    fclose(f);

    int nc;
    scanf("%d", &nc);
    std::vector<ClassStats> h_stats(nc);

    for (int i = 0; i < nc; i++) {
        int np;
        scanf("%d", &np);
        std::vector<uchar4> sample(np);
        double sumR = 0, sumG = 0, sumB = 0;

        for (int j = 0; j < np; j++) {
            int sx, sy;
            scanf("%d %d", &sx, &sy);
            uchar4 p = image[sy * w + sx];
            sample[j] = p;
            sumR += (double)p.x;
            sumG += (double)p.y;
            sumB += (double)p.z;
        }
        h_stats[i].avgR = sumR / np;
        h_stats[i].avgG = sumG / np;
        h_stats[i].avgB = sumB / np;

        double cov[3][3] = {0};
        for (int j = 0; j < np; j++) {
            double dR = (double)(sample[j].x - h_stats[i].avgR);
            double dG = (double)(sample[j].y - h_stats[i].avgG);
            double dB = (double)(sample[j].z - h_stats[i].avgB);
            cov[0][0] += dR * dR; cov[0][1] += dR * dG; cov[0][2] += dR * dB;
            cov[1][0] += dG * dR; cov[1][1] += dG * dG; cov[1][2] += dG * dB;
            cov[2][0] += dB * dR; cov[2][1] += dB * dG; cov[2][2] += dB * dB;
        }
        for(int r = 0; r < 3; ++r) {
            for(int c = 0; c < 3; ++c) {
                cov[r][c] /= (double)(np - 1);
            }
        }

        double det = cov[0][0] * (cov[1][1] * cov[2][2] - cov[1][2] * cov[2][1])
                        - cov[0][1] * (cov[1][0] * cov[2][2] - cov[1][2] * cov[2][0]) + cov[0][2] * (cov[1][0] * cov[2][1] - cov[1][1] * cov[2][0]);

        h_stats[i].logDet = log(fabs(det));

        double invDet = 1.0 / det;
        h_stats[i].invCov[0][0] = (cov[1][1]*cov[2][2] - cov[1][2]*cov[2][1]) * invDet;
        h_stats[i].invCov[0][1] = -(cov[0][1]*cov[2][2] - cov[0][2]*cov[2][1]) * invDet;
        h_stats[i].invCov[0][2] = (cov[0][1]*cov[1][2] - cov[0][2]*cov[1][1]) * invDet;
        h_stats[i].invCov[1][0] = -(cov[1][0]*cov[2][2] - cov[1][2]*cov[2][0]) * invDet;
        h_stats[i].invCov[1][1] = (cov[0][0]*cov[2][2] - cov[0][2]*cov[2][0]) * invDet;
        h_stats[i].invCov[1][2] = -(cov[0][0]*cov[1][2] - cov[0][2]*cov[1][0]) * invDet;
        h_stats[i].invCov[2][0] = (cov[1][0]*cov[2][1] - cov[1][1]*cov[2][0]) * invDet;
        h_stats[i].invCov[2][1] = -(cov[0][0]*cov[2][1] - cov[0][1]*cov[2][0]) * invDet;
        h_stats[i].invCov[2][2] = (cov[0][0]*cov[1][1] - cov[0][1]*cov[1][0]) * invDet;
    }

    checkCuda(cudaMemcpyToSymbol(c_stats, h_stats.data(), sizeof(ClassStats) * nc), "Copy stats fail");
    checkCuda(cudaMemcpyToSymbol(c_numClasses, &nc, sizeof(int)), "Copy nc fail");

    uchar4* d_data;
    checkCuda(cudaMalloc(&d_data, n * sizeof(uchar4)), "Malloc fail");
    checkCuda(cudaMemcpy(d_data, image.data(), n * sizeof(uchar4), cudaMemcpyHostToDevice), "H2D fail");

    classifyKernel<<<256, 256>>>(d_data, n);
    checkCuda(cudaDeviceSynchronize(), "Sync fail");

    checkCuda(cudaMemcpy(image.data(), d_data, n * sizeof(uchar4), cudaMemcpyDeviceToHost), "D2H fail");
    FILE* fout = fopen(outPath, "wb");
    fwrite(&w, sizeof(int), 1, fout);
    fwrite(&h, sizeof(int), 1, fout);
    fwrite(image.data(), sizeof(uchar4), n, fout);
    fclose(fout);
    cudaFree(d_data);
    return 0;
}