#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cublas_v2.h>

// try to do vector ops of sum of a*b (where a and b are vectors) So basically the inner product

// This is a function, that can be useful for debugging
__global__ void printArr(float * a){
    int id = threadIdx.x + (blockIdx.x * blockDim.x);
    printf("From PrintArr: res[%d] = %f\n", id, a[id]);
}

__global__ void copy(float * a, float* res){
    int id = threadIdx.x + (blockIdx.x * blockDim.x);
    // if(id == 0){
    //     printf( "Copy has been entered\n");
    // }
    // printf("I will put a[%d]= %f into res[%d]\n", id, a[id], id);
    res[id] = a[id];
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
            if(id == 112 || id == 108){
                //printf("I am thread %d, with res[%d] = %f. I will add res[%d] = %f\n", id, id, res[id], id + stepSize, res[id + stepSize]);
            }
            res[id] += res[id + stepSize];
            stepSize /= 2;
        }
        __syncthreads();
    }
}

__global__ void globalMemSum(float * res, int size){
    int id = threadIdx.x + 2 * (blockDim.x * blockIdx.x);
    int stepSize = size/2;
    
    int iterations = 1;
    int logSize = size;

    while(logSize > 1){
        logSize /= 2;
        iterations++;
    }

    if(id == 0){
        // printf("globalMemSum has been invoked\n");
    }
    for(int i = 0; i < iterations; i++){
        if ((id - 2 * (blockDim.x * blockIdx.x)) < stepSize){
            //printf("I am thread %d, with res[%d] = %f. I will add res[%d] = %f\n", id, id, res[id], id + stepSize, res[id + stepSize]);   
            res[id] += res[id + stepSize];
            stepSize /= 2;
        }
        __syncthreads();
    }
    if(id == 0){
        //printf("the result is res[0] = %f\n", res[0]);
    }
}

__global__ void naiveSharedMem(float * a, float * b, float* res, int size){

    int bSize = blockDim.x;
    extern __shared__ float rab[];
    
    // because we are computing this result blockwise "local" aka in shared memory,
    // the global vs. the blockId are different
    int gId = threadIdx.x + 2 * (blockDim.x * blockIdx.x);
    int bId = threadIdx.x;
    int stepSize = size/2;
    
    int s_a_i = bId + size;
    int s_b_i = bId + (2 * size);

    if(gId == 0){
        printf("Entered kernel\n");
    }
    
    //loading into shared Memory
    rab[s_a_i] = a[bId];
    rab[s_b_i] = b[bId];
    rab[s_a_i + stepSize] = a[bId + stepSize];
    rab[s_b_i + stepSize] = b[bId + stepSize];

    // First Iteration
    rab[bId] = rab[s_a_i] * rab[s_b_i] + (rab[s_a_i + stepSize] * rab[s_b_i + stepSize]);

    
    int iterations = 1;
    int logSize = size;

    while(logSize > 1){
        logSize /= 2;
        iterations++;
    }

    for(int i = 1; i < iterations; i++){
        if (bId < stepSize){
            // printf("I am Tread %d, this is iteration %d, and I will add cell %d\n", id, i, id+ stepSize);
            rab[bId] += rab[bId + stepSize];
            stepSize /= 2;
        }
        __syncthreads();
    }

    if (bId == 0){
        res[gId] = rab[bId];
    }
}

__global__ void sharedMemSum (float * res, int size){

}

__global__ void naiveWarpRed(float* a, float* b, float* res, int size){

    int id = threadIdx.x;
    int stepSize = size/2;
    
    int iterations = 1;
    int logSize = size;

    while(logSize > 1){
        logSize /= 2;
        iterations++;
    }

    float mySum = a[id] * b [id];

    for(int i = 0; i < iterations-1; i++){
        int condition = id < stepSize;
        unsigned mask = __ballot_sync(0xffffffff, condition);
        if(condition){
            mySum += __shfl_up_sync(mask, mySum, stepSize);
            stepSize /= 2;
        }
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
    int bSize = size / (2 * threads);

    float n = size - 1.0;
    float expectedRes = (pow(n, 2.0) + n) / 2;

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
    naiveGlobalMem <<<bSize, threads>>> (d_a, d_b, d_res, threads * 2);
    
    
    // bSize is the number of elements that still need summing up
    while (bSize > threads){
        
        cudaDeviceSynchronize();
        int newBSize = bSize / (threads * 2);
        align <<<2*newBSize, threads>>> (d_res, d_a, bSize, 2*threads);
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess){
            printf("Cuda Error in align: %s\n", cudaGetErrorString(err));
            printf("Config Args: newBsize = %d, threads = %d\n", newBSize, threads);
        }
        cudaDeviceSynchronize();

        // Align swaps around d_a and d_res, so we swap it back
        float * temp = d_res;
        d_res = d_a;
        d_a = temp;

        bSize = newBSize;
        globalMemSum <<<bSize, threads>>> (d_res, threads * 2);
    }
    
    if (bSize > 1){
        // printf("Entered bSize > 1 with bSize = %d\n", bSize);
        // printf("After alignment:\n");
        // printArr <<<1, bSize>>> (d_res);
        cudaDeviceSynchronize();
        align <<<1, bSize>>> (d_res, d_a, bSize, 2*threads);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess){
            printf("Cuda Error in align: %s\n", cudaGetErrorString(err));
            printf("Config Args: newBsize = %d, threads = %d\n", 1, bSize);
        }

        // Align swaps around d_a and d_res, so we swap it back
        float * temp = d_res;
        d_res = d_a;
        d_a = temp;

        globalMemSum <<<1, bSize>>> (d_res, bSize);
    }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Naive with Global Memory took %fms\n", time);

    cudaMemcpy(res, d_res, sizeof(float)*size, cudaMemcpyDeviceToHost);
    
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

    int bSize = size / (2 * threads);
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


    // There are 2 * threads elements to be added and there are three arrays of that many elements
    naiveSharedMem<<<bSize, threads, sizeof(float) * threads * 2 * 3>>> (d_a, d_b, d_res, threads * 2);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Cuda Error before printArr: %s\n", cudaGetErrorString(err));
        printf("Launch config: bsize = %d, threads = %d, sharedMem = %d\n", bSize, threads, threads * 2 * 3);
    }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Naive with Shared Memory took %fms\n", time);

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

    vecInitOnes(a, size);
    vecInitOnes(b, size);

    cudaMemcpy(d_a, a, sizeof(float)*size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float)*size , cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);
    naiveWarpRed<<<1,size>>> (d_a, d_b, d_res, size);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("WarpReduction Version took %fms\n", time);

    cudaMemcpy(res, d_res, sizeof(float)*size, cudaMemcpyDeviceToHost);

    // for (int i = 0; i< size; i++){
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

int main(){


    int size = 1 << 16;
    int threads = 1 << 5;

    printf("Threads = %d, size = %d\n", threads, size );

    for (int i = 0; i < 5; i++){
    }
    
    // callCublas();
    callNaiveGlobalMem(size, threads);
    callNaiveSharedMem(size, threads);
   // callNaiveWarpRed(size, threads); 

    return 0;
}
