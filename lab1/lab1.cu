#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>

void checkCuda(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        fprintf(stderr, "ERROR: %s (%s)\n", message, cudaGetErrorString(result));
        exit(0);
    }
}

__global__ void vecMaxKernel(const double* a, const double* b, double* c, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        c[i] = (a[i] > b[i]) ? a[i] : b[i];
    }
}

int main() {
    int n;
    if (scanf("%d", &n) != 1) return 0;
    if (n <= 0) return 0;

    std::vector<double> h_a(n), h_b(n), h_c(n);
    for (int i = 0; i < n; ++i) {
        scanf("%lf", &h_a[i]);
    }
    for (int i = 0; i < n; ++i) {
        scanf("%lf", &h_b[i]);
    }

    double *d_a, *d_b, *d_c;
    checkCuda(cudaMalloc(&d_a, n * sizeof(double)), "Failed to allocate memory for d_a");
    checkCuda(cudaMalloc(&d_b, n * sizeof(double)), "Failed to allocate memory for d_b");
    checkCuda(cudaMalloc(&d_c, n * sizeof(double)), "Failed to allocate memory for d_c");

    checkCuda(cudaMemcpy(d_a, h_a.data(), n * sizeof(double), cudaMemcpyHostToDevice), "Copy h_a to d_a failed");
    checkCuda(cudaMemcpy(d_b, h_b.data(), n * sizeof(double), cudaMemcpyHostToDevice), "Copy h_b to d_b failed");

    int threadsPerBlock = 1024;
    int blocksPerGrid = 256;

    vecMaxKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    checkCuda(cudaGetLastError(), "Kernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "Sync failed");

    checkCuda(cudaMemcpy(h_c.data(), d_c, n * sizeof(double), cudaMemcpyDeviceToHost), "Copy D2H failed");

    for (int i = 0; i < n; ++i) {
        printf("%.10e ", h_c[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}