#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cublas_v2.h>
#include "multi_block_globalMem.h"
#include "multi_block_sharedMem.h"
#include "multi_block_warps.h"

#define cudaCheckErr() {                                                                           \
    cudaError_t err = cudaGetLastError();                                                          \
    if(err != cudaSuccess){                                                                        \
    printf("Cuda Error in align: %s, %d %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
    exit(0);                                                                                   \
    }                                                                                              \
}

// do vector ops of sum of a*b (where a and b are vectors) So basically the inner product

void runTest(char* lab, int (*fptr)(float* a, float* b, float* res, int size, int threads), float* src_a, float* src_b, float* src_res, int size, int threads, int reps, float checkSum){

    float *d_a, *d_b, *d_res;

    cudaMalloc(&d_a, sizeof(float) * size * reps);
    cudaMalloc(&d_b, sizeof(float) * size * reps);
    cudaMalloc(&d_res, sizeof(float) * size * reps);

    cudaEvent_t start, stop;
    float time_ms;

    cudaEventCreate (&start);
    cudaEventCreate(&stop);

    cudaCheckErr();

    // creates the input reps times
    for (int i = 0; i < reps; i ++){
        int offset = size * i;
        cudaMemcpy(d_a + offset, src_a, sizeof(float)*size , cudaMemcpyHostToDevice);
        cudaMemcpy(d_b + offset, src_b, sizeof(float)*size , cudaMemcpyHostToDevice);
    }
    // warm up
    // fptr(d_a, d_b, d_res, size, threads);

    cudaDeviceSynchronize();
    cudaCheckErr();

    cudaEventRecord(start);
    int swap = 0;
    for(int i = 0; i < reps; i ++){
        int offset = size * i;
        swap = fptr(d_a + offset, d_b + offset, d_res, size, threads);
    }

    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaCheckErr();

    cudaEventElapsedTime(&time_ms, start, stop);
    float time_s = time_ms / (float) 1e3;

    float GB = (float) size * sizeof(float) * reps * 2;
    float GBs = GB / time_s / (float)1e9;

    if(swap){
        float * temp = d_res;
        d_res = d_a;
        d_a = temp;
    }

    cudaMemcpy(src_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErr();

    if (src_res[0] != checkSum){
        printf("%s with %d threads and size %d: result is %f, but should be %f \n",lab, threads, size, src_res[0], (float) checkSum);
    } else {
        printf("%s with %d threads and size %d TIME: %fs GB/s: %f\n", lab, threads, size, time_s, GBs);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);
}


void vecInitOnes(float * a, int size){

    for (int i = 0; i < size; i++){
        a[i] = 1.0;
    }
}

void vecInitGauss (float * a, int size){
    for (int i = 0; i < size; i++){
        a[i] = i;
    }
}

int main(){

    int reps = 1;

    for (int i = 9; i < 10; i ++){         // Threads
        for (int j = 24; j < 25; j++){   // Number of Elements
            
            int size = 1 << j;
            int threads = 1 << i;

            float checkSum = ((size - 1) * (size - 1) + (size - 1)) / 2;
            checkSum = size;


            // Initialize a, b and the result in "normal" memory
            float * a = (float*) malloc (sizeof(float)*size);
            float * b = (float*) malloc (sizeof(float)*size);
            float * res = (float*) malloc (sizeof(float) * size);

            vecInitOnes(a, size);
            vecInitOnes(b, size);

            // callNaiveGlobalMem(size, threads);
            // runTest("multi_block_globalMem", multi_block_globalMem, a, b, res, size, threads, reps, checkSum);
            // callNaiveSharedMem(size, threads);
            // runTest("multi_block_SharedMem", multi_block_sharedMem, a, b, res, size, threads, reps, checkSum);
            
            // callNaiveWarpRed(size, threads);
            runTest("multi_block_Warps", multi_block_warps, a, b, res, size, threads, reps, checkSum);
            

            free(a);
            free(b);
            free(res);
        }
    }
}



