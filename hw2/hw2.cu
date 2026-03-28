#include <stdio.h>
#include <stdlib.h>

void swap(float *a, float *b) {
    float temp = *a;
    *a = *b;
    *b = temp;
}

int main() {
    int n;
    if (scanf("%d", &n) != 1) {
        fprintf(stderr, "ERROR: Failed to read input data\n");
        return 0;
    }

    float *arr = (float*) malloc(n * sizeof(float));
    if (arr == NULL) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        return 0;
    }
    for (int i = 0; i < n; i++) {
        if (scanf("%f", &arr[i]) != 1) {
            fprintf(stderr, "ERROR: Failed to read array element\n");
            free(arr);
            return 0;
        }
    }

    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1; j++) {
            if (arr[j] > arr[j + 1]){
                swap(&arr[j], &arr[j + 1]);
            }
        }
    }

    for (int i = 0; i < n; i++) {
        printf("%.6e ", arr[i]);
    }

    free(arr);
    return 0;
}