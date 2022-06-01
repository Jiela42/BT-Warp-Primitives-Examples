#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cublas_v2.h>

#define cudaCheckErr() {                                                                           \
    cudaError_t err = cudaGetLastError();                                                          \
    if(err != cudaSuccess){                                                                        \
    printf("Cuda Error in align: %s, %d %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
    exit(0);                                                                                   \
    }                                                                                              \
}

// try to do vector ops of sum of a*b (where a and b are vectors) So basically the inner product

// This is a function, that can be useful for debugging
__global__ void printArr(float * a){
    int id = threadIdx.x + (blockIdx.x * blockDim.x);
    printf("From PrintArr: res[%d] = %f\n", id, a[id]);
}

__global__ void align (float* a, float* res, int stride){
    int id = threadIdx.x + (blockIdx.x * blockDim.x);
    res[id] = a[id * stride];
}

__global__ void naiveGlobalMem(float * a, float * b, float* res, int size){
   
    /* since every thread adds up two numbers (in the first iteration)
    we need double the block-starting point to keep working on disjoint parts of the input */
    int id = threadIdx.x + 2*(blockIdx.x * blockDim.x);
    int stepSize = size/2;

    // first iteration: 
    res[id] = (a[id] * b[id]) + (a[id + stepSize] * b[id + stepSize]);

    __syncthreads();

    for(int i = stepSize / 2; i > 0; i /= 2){
        if (id - 2 * (blockDim.x * blockIdx.x) < i){
            if(id == 112 || id == 108){
                //printf("I am thread %d, with res[%d] = %f. I will add res[%d] = %f\n", id, id, res[id], id + stepSize, res[id + stepSize]);
            }
            res[id] += res[id + i];
            stepSize /= 2;
        }
        __syncthreads();
    }
}

__global__ void globalMemSum(float * res, int size){
    int id = threadIdx.x + 2 * (blockDim.x * blockIdx.x);

    for(int i = size / 2; i > 0; i /= 2){
        if ((id - 2 * (blockDim.x * blockIdx.x)) < i){
            res[id] += res[id + i];
        }
        __syncthreads();
    }
}

__global__ void naiveSharedMem(float * a, float * b, float* res, int size){

    extern __shared__ float rab[];
    
    // because we are computing this result blockwise "local" aka in shared memory,
    // the global vs. the blockId are different
    int bId = threadIdx.x;
    int gId = threadIdx.x + 2 * (blockDim.x * blockIdx.x);
    int stepSize = size / 2;
    
    int s_a_i = bId + size;
    int s_b_i = bId + (2 * size);

    //loading into shared Memory
    rab[s_a_i] = a[gId];
    rab[s_b_i] = b[gId];
    rab[s_a_i + stepSize] = a[gId + stepSize];
    rab[s_b_i + stepSize] = b[gId + stepSize];
   
    // First Iteration
    rab[bId] = rab[s_a_i] * rab[s_b_i] + (rab[s_a_i + stepSize] * rab[s_b_i + stepSize]);
    //this is just a variable for debugging purposes
    int it = 2;
    __syncthreads();
    for(int i = stepSize / 2; i > 0; i /= 2){
        if (bId < i){
            rab[bId] += rab[bId + i];
            it ++;
        }
        __syncthreads();
    }

    // Writing the result bacck to global Memory
    if (bId == 0){
      res[gId] = rab[bId];
    }
}

__global__ void sharedMemSum (float * res, int size){

    extern __shared__ float r[];
    int bId = threadIdx.x;
    int gId = threadIdx.x + 2 * (blockDim.x * blockIdx.x);

    // First iteration + loading:
    r[bId] = res[gId] + res[gId + size / 2];

    __syncthreads();
    for(int i = size / 4; i > 0; i /= 2){

        if(bId < i){
            r[bId] += r[bId + i];
        }
        __syncthreads();
    }

    // writing the result back to global Memory
    if(bId == 0){
        res[gId] = r[bId];
    }
}

__global__ void naiveWarpRed(float* a, float* b, float* res, int size){

    extern __shared__ float r[];

    int id = threadIdx.x + 2 * (blockDim.x * blockIdx.x);
    int blockId = threadIdx.x;
    int laneId = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;
    int nWarps = blockDim.x / 32;
    int stepSize = size / 2;

    // Loading and first iteration:
    float mySum = a[id] * b[id] + a[id + stepSize] * b[id + stepSize];

    // First warp Reduction
    for(int i = min (size / 4, 16); i > 0; i /= 2){

        // The threads that values need to be read from, must partition in the shuffle!
        int condition = laneId < (i * 2);
        unsigned mask = __ballot_sync(0xffffffff, condition);

        if(condition){
            mySum += __shfl_down_sync(mask, mySum, i);
        }
    }

    // loading warp results into shared Memory
    if (laneId == 0){
       r[warpId] = mySum;
    }

    __syncthreads();
    // Reducing results from first reduction
    if (nWarps > 1){
        for (int i = nWarps / 2; i > 0; i /= 2){
            if(blockId < i){
                r[blockId] += r[blockId + i];
            }
            __syncthreads();
        }
    }

    // Writing back to global Memory
    if(threadIdx.x == 0){
        res[blockIdx.x] = r[0];
    }
}

__global__ void warpRedSum(float* a, float* res, int size){

    extern __shared__ float r[];

    int id = threadIdx.x + 2 * (blockDim.x * blockIdx.x);
    int blockId = threadIdx.x;
    int laneId = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;
    int nWarps = blockDim.x / 32;
    int stepSize = size / 2;

    // First Iteration
    float mySum = a[id] + a[id + stepSize];

    // First warp Reduction
    // The number of iterations is adjusted to account for the possiblity of size == 2
    for(int i = min(16, size / 4); i > 0; i /= 2){
        // The threads that values need to be read from, must partition in the shuffle!
        int condition = laneId < (i * 2);
        unsigned mask = __ballot_sync(0xffffffff, condition);

        if(condition){
            mySum += __shfl_down_sync(mask, mySum, i);
        }
    }

    // loading warp results into shared Memory
    if (laneId == 0){
        r[warpId] = mySum;
    }

    __syncthreads();
    // Reducing results from first reduction
    if (nWarps > 1){
        for (int i = nWarps / 2; i > 0; i /= 2){
            if(blockId < i){
               r[blockId] += r[blockId + i];
            }
            __syncthreads();
        }
    }

    // Writing back to global Memory
    if(threadIdx.x == 0){
      res[blockIdx.x] = r[0];
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

void callCublas(){
    cublasHandle_t h;
    cublasCreate(&h);

    cudaEvent_t start, stop;
    float time;

    cudaEventCreate (&start);
    cudaEventCreate(&stop);

    int size = 32;
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

    // Every thread can add two numbers, thus we can add 256 number in one iteration
    int nBlocks = size / (2 * threads);

    
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

   vecInitGauss(a, size);

    //vecInitOnes(a, size);
    vecInitOnes(b, size);

    cudaMemcpy(d_a, a, sizeof(float)*size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float)*size , cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);
    // to support a size bigger than 32 we need to
    // basically do another addition for each one of the blocks!

    // Every thread can add two numbers, thus we can add 2*threads many numbers in one iteration
    naiveGlobalMem <<<nBlocks, threads>>> (d_a, d_b, d_res, threads * 2);
    
    
    // nBlocks is the number of elements that still need summing up
    while (nBlocks > threads){
        cudaDeviceSynchronize();
        int new_nBlocks = nBlocks / (threads * 2);
        align <<<2 * new_nBlocks, threads>>> (d_res, d_a, 2 * threads);
        cudaCheckErr();
        cudaDeviceSynchronize();

        // Align swaps around d_a and d_res, so we swap it back
        float * temp = d_res;
        d_res = d_a;
        d_a = temp;

        nBlocks = new_nBlocks;
        globalMemSum <<<nBlocks, threads>>> (d_res, threads * 2);
    }
    
    if (nBlocks > 1){
        
        cudaDeviceSynchronize();
        align <<<1, nBlocks>>> (d_res, d_a, 2 * threads);
        cudaDeviceSynchronize();
        cudaCheckErr();
        // Align swaps around d_a and d_res, so we swap it back
        float * temp = d_res;
        d_res = d_a;
        d_a = temp;
        
        globalMemSum <<<1, nBlocks / 2>>> (d_res, nBlocks);
    }
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    
    cudaMemcpy(res, d_res, sizeof(float)*size, cudaMemcpyDeviceToHost);

    float time_s = time/(float)1e3;
    float GB = (float) size * sizeof(float);
    float GBs = GB / time_s / (float)1e9;

    float expectedRes = ((size - 1) * (size - 1) + size - 1) / 2;
    
    if (res[0] != (float)expectedRes){
        printf("The result is %f, but should be %f \n", res[0], (float) expectedRes);
    } else {
        printf("Naive with Global Memory with  %d threads and size %d TIME: %fs GB/s: %f\n", threads, size, time_s, GBs);
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

    int nBlocks = size / (2 * threads);
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

    // There are 2 * threads elements to be added and there are three arrays of that many elements
    naiveSharedMem<<<nBlocks, threads, sizeof(float) * threads * 2 * 3>>> (d_a, d_b, d_res, threads * 2);

    cudaCheckErr();
    
    int new_nBlocks = nBlocks / (threads * 2);
    if (nBlocks > 1 && new_nBlocks == 0){
        new_nBlocks = 1;
    }
    
    cudaDeviceSynchronize();
    
    cudaCheckErr();

    while(nBlocks > 2 * threads){
       
        align<<<new_nBlocks * 2, threads>>> (d_res, d_a, 2 * threads);
        float * temp = d_res;
        d_res = d_a;
        d_a = temp;
        
        cudaDeviceSynchronize();

        cudaCheckErr();
        
        nBlocks = new_nBlocks;
        sharedMemSum<<<nBlocks, threads, sizeof(float) * threads>>> (d_res, threads * 2);

        cudaDeviceSynchronize();

        cudaCheckErr();

        new_nBlocks = nBlocks / (threads * 2);
        if (nBlocks > 1 && new_nBlocks == 0){
            new_nBlocks = 1;
        }
    }

    if (nBlocks > 1){
        align<<<new_nBlocks, nBlocks>>> (d_res, d_a, 2 * threads);
        
        float * temp = d_res;
        d_res = d_a;
        d_a = temp;
        
        cudaDeviceSynchronize();
        
       cudaCheckErr();

        sharedMemSum<<<1, (nBlocks / 2), sizeof(float) * (nBlocks / 2)>>> (d_res, nBlocks);
    }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(res, d_res, sizeof(float)*size, cudaMemcpyDeviceToHost);
    
    float expectedRes = ((size - 1) * (size - 1) + size - 1) / 2;           //Using the little Gaussian formula

    if (res[0] != (float) expectedRes){
        printf("Threads = %d, size = %d \nThe result is %f, but should be %f \n", threads, size, res[0], (float) expectedRes);
    } else {
        printf("Naive with Shared Memory with %d threads and size %d took %fms\n", threads, size, time);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);

    free(a);
    free(b);
    free(res);
}

void callNaiveWarpRed(int size, int threads){
    cudaEvent_t start, stop;
    float time;
    
    int nBlocks = size / (2 * threads);
    int warps_per_block = threads / 32 + 1;     // the +1 is to ensure we assign memory even if int division rounds down

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
    naiveWarpRed<<<nBlocks, threads, sizeof(float) * warps_per_block>>> (d_a, d_b, d_res, threads * 2);
    cudaDeviceSynchronize();
    
    cudaCheckErr();

    int new_nBlocks = nBlocks / (threads * 2);

    cudaDeviceSynchronize();

    while (nBlocks > threads * 2){

        nBlocks = new_nBlocks;
        warpRedSum<<<nBlocks, threads, sizeof(float) * warps_per_block>>>(d_res, d_a, threads * 2);
        cudaDeviceSynchronize();

        float * temp = d_res;
        d_res = d_a;
        d_a = temp;

        new_nBlocks = nBlocks / (threads * 2);
    }

    if (nBlocks > 1){
        
        nBlocks /= 2;
        warps_per_block = nBlocks / 32 + 1;         // the plus 1 is in case integer summation rounds down

        warpRedSum<<<1, nBlocks, sizeof(float) * warps_per_block>>> (d_res, d_a, nBlocks * 2);
        
        float * temp = d_res;
        d_res = d_a;
        d_a = temp;        
        cudaDeviceSynchronize();

        cudaCheckErr();
    }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(res, d_res, sizeof(float)*size, cudaMemcpyDeviceToHost);

    float expRes = ((size - 1) * (size - 1) + size - 1) / 2;
    
    if (res[0] != (float)expRes){
        printf("In Warp Reduction, the result is %f, but should be %f with %d threads and size %d \n", res[0], (float) expRes, threads, size);
    } else {
        printf("WarpReduction Version with %d Threads and Size %d took %fms\n",threads, size, time);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);

    free(a);
    free(b);
    free(res);
}

int main(){

    for (int it = 0; it < 1; it ++){
        
        for (int i = 0; i <= 10; i ++){
            for (int j = i + 1; j < 16; j++){
                
                int size = 1 << j;
                int threads = 1 << i;
                //printf("Threads = %d, size = %d\n", threads, size );
                callNaiveGlobalMem(size, threads);
                //callNaiveSharedMem(size, threads);
                //callNaiveWarpRed(size, threads);
       
            }
        }
    }

    int j = 3;
    int i = 0;


    int size = 1 << j;
    int threads = 1 << i;
    
    
    // callCublas();
    //callNaiveGlobalMem(size, threads);
    // callNaiveSharedMem(size, threads);
   //callNaiveWarpRed(size, threads); 
   //testSharedMemSum(2048, 1024);

    return 0;
}
