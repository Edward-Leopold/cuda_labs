#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

void checkCuda(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        fprintf(stderr, "ERROR: %s (%s)\n", message, cudaGetErrorString(result));
        exit(0);
    }
}

__global__ void ssaaFunc(cudaTextureObject_t tex, uchar4* dst, int wn, int hn, int sw, int sh) {
    int scaleX = sw / wn;
    int scaleY = sh / hn;

    int idxX = blockIdx.x * blockDim.x + threadIdx.x;
    int idxY = blockIdx.y * blockDim.y + threadIdx.y;

    int strideX = blockDim.x * gridDim.x;
    int strideY = blockDim.y * gridDim.y;

    for (int y = idxY; y < hn; y += strideY) {
        for (int x = idxX; x < wn; x += strideX) {
            int r = 0, g = 0, b = 0;

            for (int i = 0; i < scaleY; i++) {
                for (int j = 0; j < scaleX; j++) {
                    uchar4 p = tex2D<uchar4>(tex, x * scaleX + j, y * scaleY + i);
                    r += p.x;
                    g += p.y;
                    b += p.z;
                }
            }

            int count = scaleX * scaleY;
            dst[y * wn + x] = make_uchar4((unsigned char)(r / count), (unsigned char)(g / count), (unsigned char)(b / count), 0 );
        }
    }
}

int main() {
    char inPath[1024], outPath[1024];
    int wn, hn;

    if (scanf("%s %s %d %d", inPath, outPath, &wn, &hn) != 4) return 0;

    FILE* f_in = fopen(inPath, "rb");
    if (!f_in) {
        fprintf(stderr, "ERROR: Cannot open input file\n");
        return 0;
    }

    int sw, sh;
    fread(&sw, sizeof(int), 1, f_in);
    fread(&sh, sizeof(int), 1, f_in);

    size_t numPixels = (size_t)(sw * sh);
    std::vector<uchar4> h_in(numPixels);
    fread(h_in.data(), sizeof(uchar4), numPixels, f_in);
    fclose(f_in);

    cudaArray* arr;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    checkCuda(cudaMallocArray(&arr, &channelDesc, sw, sh), "Array malloc failed");
    checkCuda(cudaMemcpy2DToArray(arr, 0, 0, h_in.data(), sw * sizeof(uchar4), sw * sizeof(uchar4), sh, cudaMemcpyHostToDevice), "Copy to array failed");

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj = 0;
    checkCuda(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL), "Texture creation failed");

    uchar4* d_out;
    size_t outSize = (size_t)wn * hn * sizeof(uchar4);
    checkCuda(cudaMalloc(&d_out, outSize), "Output malloc failed");

    dim3 threads(16, 16);
    dim3 blocks(32, 32);
    ssaaFunc<<<blocks, threads>>>(texObj, d_out, wn, hn, sw, sh);

    checkCuda(cudaGetLastError(), "Kernel failed");
    checkCuda(cudaDeviceSynchronize(), "Sync failed");

    std::vector<uchar4> h_out(wn * hn);
    checkCuda(cudaMemcpy(h_out.data(), d_out, outSize, cudaMemcpyDeviceToHost), "Copy D2H failed");
    FILE* f_out = fopen(outPath, "wb");
    if (f_out) {
        fwrite(&wn, sizeof(int), 1, f_out);
        fwrite(&hn, sizeof(int), 1, f_out);
        fwrite(h_out.data(), sizeof(uchar4), wn * hn, f_out);
        fclose(f_out);
    }

    checkCuda(cudaDestroyTextureObject(texObj), "Destroy texture failed");
    cudaFreeArray(arr);
    cudaFree(d_out);

    return 0;
}