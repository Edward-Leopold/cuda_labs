#include <stdio.h>
#include <math.h>

int main() {
	float a, b, c;
    
	if (scanf("%f %f %f", &a, &b, &c) != 3) {
		fprintf(stderr, "ERROR: Failed to read input data\n");
		return 0;
	}

	if (a == 0) {
		if (b == 0 && c == 0) {
			printf("any\n");
		} else if (b == 0) {
			printf("incorrect\n");
		} else {
			printf("%.6f\n", -c / b);
		}

		return 0;
	}

	float D = (b * b) - (4 * a * c);
	if (D > 0){
		printf("%.6f %.6f", (-b + sqrtf(D)) / (2 * a), (-b - sqrtf(D)) / (2 * a));
	} else if (D == 0){
		printf("%.6f", (-b) / (2 * a));
	} else {
		printf("imaginary");
	}


	return 0;
}