#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

void checkCuda(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        fprintf(stderr, "ERROR: %s (%s)\n", message, cudaGetErrorString(result));
        exit(0);
    }
}

struct abs_comp {
    __device__ bool operator()(double a, double b) {
        return fabs(a) < fabs(b);
    }
};

__global__ void copy_column_kernel(double* matrix, double* col_buf, int n, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = k + idx;
    if (offset < n) {
        col_buf[idx] = matrix[offset * n + k];
    }
}

__global__ void swap_rows_kernel(double* matrix, int n, int r1, int r2, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + k;
    if (idx < n) {
        double tmp = matrix[r1 * n + idx];
        matrix[r1 * n + idx] = matrix[r2 * n + idx];
        matrix[r2 * n + idx] = tmp;
    }
}

__global__ void eliminate_kernel(double* matrix, int n, int k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int col = k + 1 + x;
    int row = k + 1 + y;
    for (int i = row; i < n; i += blockDim.y * gridDim.y) {
        double factor = matrix[i * n + k] / matrix[k * n + k];
        for (int j = col; j < n; j += blockDim.x * gridDim.x) {
            matrix[i * n + j] -= factor * matrix[k * n + j];
        }
    }
}

int main() {
    int n;
    if (scanf("%d", &n) != 1) return 0;
    std::vector<double> h_matrix(n * n);
    for (int i = 0; i < n * n; i++) {
        scanf("%lf", &h_matrix[i]);
    }

    double *d_matrix, *d_col_buf;
    checkCuda(cudaMalloc(&d_matrix, n * n * sizeof(double)), "Matrix malloc fail");
    checkCuda(cudaMalloc(&d_col_buf, n * sizeof(double)), "Col buf malloc fail");
    checkCuda(cudaMemcpy(d_matrix, h_matrix.data(), n * n * sizeof(double), cudaMemcpyHostToDevice), "Memcpy fail");

    double det_sign = 1.0;
    int threads1D = 256;
    int blocks1D = 256;
    dim3 threads2D(16, 16);
    dim3 blocks2D(16, 16);

    for (int k = 0; k < n; k++) {
        int elements_left = n - k;
        copy_column_kernel<<<blocks1D, threads1D>>>(d_matrix, d_col_buf, n, k);
        thrust::device_ptr<double> d_ptr = thrust::device_pointer_cast(d_col_buf);
        auto max_it = thrust::max_element(d_ptr, d_ptr + elements_left, abs_comp());
        int pivot_in_col = max_it - d_ptr;
        int pivot_row = k + pivot_in_col;

        double pivot_val;
        checkCuda(cudaMemcpy(&pivot_val, d_matrix + pivot_row * n + k, sizeof(double), cudaMemcpyDeviceToHost), "Get pivot fail");

        if (fabs(pivot_val) < 1e-7) {
            printf("%.10e\n", 0.0);
            cudaFree(d_matrix);
            cudaFree(d_col_buf);
            return 0;
        }

        if (pivot_row != k) {
            swap_rows_kernel<<<blocks1D, threads1D>>>(d_matrix, n, k, pivot_row, k);
            det_sign *= -1.0;
        }

        eliminate_kernel<<<blocks2D, threads2D>>>(d_matrix, n, k);
    }

    std::vector<double> res_matrix(n * n);
    checkCuda(cudaMemcpy(res_matrix.data(), d_matrix, n * n * sizeof(double), cudaMemcpyDeviceToHost), "Copy back fail");

    double det = det_sign;
    for (int i = 0; i < n; i++) {
        det *= res_matrix[i * n + i];
    }

    printf("%.10e\n", det);
    cudaFree(d_matrix);
    cudaFree(d_col_buf);

    return 0;
}