#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cublas_v2.h>

// try to do vector ops of sum of a*b (where a and b are vectors) So basically the inner product

__global__ void printArr(float * a){
    // This is just a function, to help debug
    int id = threadIdx.x + (blockIdx.x * blockDim.x);
    printf("From PrintArr: res[%d] = %f\n", id, a[id]);
}

__global__ void align (float* a, float* res, int targetSize, int stride){
    int id = threadIdx.x + (blockIdx.x * blockDim.x);
    res[id] = a[id * stride];
    // printf("I put a[%d] = %f into res[%d]\n", id* stride, a[id*stride], id);
}

__global__ void naiveGlobalMem(float * a, float * b, float* res, int size){
   
    /* since every thread adds up two numbers (in the first iteration)
    we need double the block-starting point to keep working on disjoint parts of the input */
    int id = threadIdx.x + 2*(blockIdx.x * blockDim.x);
    int stepSize = size/2;
    
    int iterations = 1;
    int logSize = size;

    while(logSize > 1){
        logSize /= 2;
        iterations++;
    }

    // first iteration: 
    res[id] = (a[id] * b[id]) + (a[id + stepSize] * b[id + stepSize]);
    stepSize /= 2; 

    for(int i = 1; i < iterations; i++){
        if (id - 2 * (blockDim.x * blockIdx.x) < stepSize){
            res[id] += res[id + stepSize];
            stepSize /= 2;
        }
        __syncthreads();
    }
}

__global__ void naiveSharedMem(float * a, float * b, float* res, int size){


    // There are 3 Arrays, each of size 64 elements
    __shared__ float rab[3 * 64];
    int maxIndx = 3 * 64 - 1;
    int id = threadIdx.x;
    
    int s_a_i = id + size;
    int s_b_i = id + (2 * size);

    if (s_b_i + 32 > maxIndx){
        printf("I am Thread %d, I tried to access s_b_i + size = %d, but the Max Index is: %d \n", id, s_b_i + 32, maxIndx);
    }
    // we have 2*threads many elements, so we need that each threads adds 2 pairs of elements!
    rab[s_a_i] = a[id];
    rab[s_b_i] = b[id];
    rab[s_a_i + 32] = a[id + 32];
    rab[s_b_i + 32] = b[id + 32];


    // if (id == 0){
    //     printf("rab[s_a_i] = rab[%d] = %f\n rab[s_b_i] = rab[%d] = %f\n", s_a_i,rab[s_a_i], s_b_i, rab[s_b_i]);
    //     printf("rab[s_a_i+4] = rab[%d] = %f\n rab[s_b_i+4] = rab[%d] = %f\n", s_a_i +4,rab[s_a_i+4], s_b_i+4, rab[s_b_i+4]);
    // }
    // note that, we have 2*threads many elements! Hence we do the first iteration here!
    rab[id] = (rab[s_a_i] * rab[s_b_i]) + (rab[s_a_i + 32] * rab[s_b_i + 32]);
    int stepSize = size/4;
    
    int iterations = 1;
    int logSize = size;

    while(logSize > 1){
        logSize /= 2;
        iterations++;
    }
    //printf("rab[%d] = %f\n", id, rab[id]);
    __syncthreads();
    for(int i = 1; i < iterations; i++){
        if (id < stepSize){
            // printf("I am Tread %d, this is iteration %d, and I will add cell %d\n", id, i, id + stepSize);
            // printf("I am Tread %d, my value is %f and I will add %f\n", id, rab[id], rab[id+ stepSize]);
            rab[id] += rab[id + stepSize];
            stepSize /= 2;
        }
        
       // printf("rab[%d] = %f\n", id, rab[id]);
        __syncthreads();
    }

    
        res[id] = rab[id];
}

__global__ void naiveWarpRed(float* a, float* b, float* res, int size){

    int id = threadIdx.x;

    float mySum = a[id] * b[id] + a[id + size / 2] * b[id + size / 2];
    for(int i = size / 4; i > 0; i /= 2){
        int condition = id < i * 2;
        unsigned mask = __ballot_sync(0xffffffff, condition);
        if(condition){
            mySum += __shfl_down_sync(mask, mySum, i);
        }
        // no sync here, because the shuffle syncs!
    }
    if(id == 0){
        res[id] = mySum;
    }
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

void vecInit(float * a, float size){
    srand(time(NULL));

    for(int i = 0; i < size; i++){
        float r = rand() % 64;
        a[i] = r;
    }
}

void callCublas(int size){
    cublasHandle_t h;
    cublasCreate(&h);

    cudaEvent_t start, stop;
    float time;

    cudaEventCreate (&start);
    cudaEventCreate(&stop);

    float * a = (float*) malloc (sizeof(float)*size);
    float * b = (float*) malloc (sizeof(float)*size);
    float * res = (float*) malloc (sizeof(float)*size);

    float* d_a;
    float* d_b;
    float* d_res;

    cudaMalloc(&d_a, sizeof(float)*size);
    cudaMalloc(&d_b, sizeof(float)*size);
    cudaMalloc(&d_res, sizeof(float)*size);

    // vecInit (a, size);
    // vecInit (b, size);

    vecInitOnes(a, size);
    vecInitOnes(b, size);

    cudaMemcpy(d_a, a, sizeof(float)*size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float)*size , cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);
    cublasSdot(h, size, d_a, 1, d_b, 1, d_res);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cublasDestroy(h);

    printf("Calling Cublas took %fms\n", time);

    cudaMemcpy(res, d_res, sizeof(float)*size, cudaMemcpyDeviceToHost);

    // for(int i = 0; i< size; i++){
    //     printf("res[%d] = %f\n", i, res[i]);
    // }
    
    if (res[0] != (float)size){
        printf("The result is %f, but should be %f \n", res[0], (float) size);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);

    free(a);
    free(b);
    free(res);
}

void callNaiveGlobalMem(int size, int threads){
    cudaEvent_t start, stop;
    float time;

    cudaEventCreate (&start);
    cudaEventCreate(&stop);
    
    printf("Threads = %d, size = %d\n", threads, size );

    float n = size - 1.0;
    float expectedRes = (pow(n, 2.0) + n) / 2;          // This is the expected result for the gaussian sum

    float * a = (float*) malloc (sizeof(float)*size);
    float * b = (float*) malloc (sizeof(float)*size);
    float * res = (float*) malloc (sizeof(float)*size);

    float* d_a;
    float* d_b;
    float* d_res;

    cudaMalloc(&d_a, sizeof(float)*size);
    cudaMalloc(&d_b, sizeof(float)*size);
    cudaMalloc(&d_res, sizeof(float)*size);

    // Random Vector initializsation:
    // vecInit (a, size);
    // vecInit (b, size);

    // vecInitGauss(a, size);

    vecInitOnes(a, size);
    vecInitOnes(b, size);

    cudaMemcpy(d_a, a, sizeof(float)*size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float)*size , cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);
    // to support a size bigger than 32 we need to
    // basically do another addition for each one of the blocks!

    //LOOOOOOOOL this only applies to standard warp reduction, not using global or shared Memory!!

    // Every thread can add two numbers, thus we can add 2*threads many numbers in one iteration
    naiveGlobalMem <<<1, threads>>> (d_a, d_b, d_res, size);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Cuda Error after Kernel Call: %s\n", cudaGetErrorString(err));
        printf("nBlocks = %d threads = %d\n", 1, threads);
    }
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Naive with Global Memory took %fms\n", time);

    cudaMemcpy(res, d_res, sizeof(float)*size, cudaMemcpyDeviceToHost);
    
    // Just a sanity check such that we get a message, if the result is incorrect
    if (res[0] != (float)size){
        printf("The result is %f, but should be %f \n", res[0], (float) size);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);

    free(a);
    free(b);
    free(res);
}

void callNaiveSharedMem(int size, int threads){
    cudaEvent_t start, stop;
    float time;

    cudaEventCreate (&start);
    cudaEventCreate(&stop);

    float * a = (float*) malloc (sizeof(float)*size);
    float * b = (float*) malloc (sizeof(float)*size);
    float * res = (float*) malloc (sizeof(float)*size);

    float* d_a;
    float* d_b;
    float* d_res;

    cudaMalloc(&d_a, sizeof(float)*size);
    cudaMalloc(&d_b, sizeof(float)*size);
    cudaMalloc(&d_res, sizeof(float)*size);

    // vecInit (a, size);
    // vecInit (b, size);

    vecInitGauss(a, size);

    // vecInitOnes(a, size);
    vecInitOnes(b, size);

    cudaMemcpy(d_a, a, sizeof(float)*size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float)*size , cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);
    // to support a size bigger than 32 we need to
    // basically do another addition for each one of the blocks!
    //again. that is true for the reduction version only!
    
    naiveSharedMem<<<1, threads>>> (d_a, d_b, d_res, size);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Naive with Shared Memory took %fms\n", time);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Cuda Error after Kernel Call: %s\n", cudaGetErrorString(err));
        printf("threads = %d\n",threads);
    }

    // printArr <<<1, threads>>> (d_res);

    cudaMemcpy(res, d_res, sizeof(float)*size, cudaMemcpyDeviceToHost);

    float n = size - 1.0;
    float expectedRes = (pow(n, 2.0) + n) / 2;
    float diff = res[0] - expectedRes;

    if (diff > 1.0 || diff < -1.0){
        printf("The result is %f, but should be %f, difference is %f \n", res[0], expectedRes, diff);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);

    free(a);
    free(b);
}

void callNaiveWarpRed(int size, int threads){
    cudaEvent_t start, stop;
    float time;

    cudaEventCreate (&start);
    cudaEventCreate(&stop);

    float * a = (float*) malloc (sizeof(float)*size);
    float * b = (float*) malloc (sizeof(float)*size);
    float * res = (float*) malloc (sizeof(float)*size);

    float* d_a;
    float* d_b;
    float* d_res;

    cudaMalloc(&d_a, sizeof(float)*size);
    cudaMalloc(&d_b, sizeof(float)*size);
    cudaMalloc(&d_res, sizeof(float)*size);

    // vecInit (a, size);
    // vecInit (b, size);

    vecInitGauss (a, size);
    
    //vecInitOnes(a, size);
    vecInitOnes(b, size);

    cudaMemcpy(d_a, a, sizeof(float)*size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float)*size , cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);
    naiveWarpRed<<<1, threads>>> (d_a, d_b, d_res, size);
    cudaDeviceSynchronize();

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    
    cudaMemcpy(res, d_res, sizeof(float)*size, cudaMemcpyDeviceToHost);
    printf("WarpReduction Version took %fms\n", time);

    // for (int i = 0; i< size; i++){
    //     printf("res[%d] = %f\n", i, res[i]);
    // }
    
    float expRes = (pow((size-1), 2.0) + (size-1)) / 2;

    if (res[0] != (float)expRes){
        printf("The result is %f, but should be %f \n", res[0], (float) expRes);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);

    free(a);
    free(b);
    free(res);
}

int main(){

    int threads = 1 << 5;
    int size = 1 << 6;
    for (int i = 0; i < 5; i++){
        
        callCublas(size);
        callNaiveGlobalMem(size, threads);
        callNaiveSharedMem(size, threads);
        callNaiveWarpRed(size, threads); 
    }
    

    return 0;
}
