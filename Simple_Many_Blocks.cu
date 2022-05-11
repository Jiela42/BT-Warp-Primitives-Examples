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

//another function useful for debugging:
__global__ void printVal(float * a){
    printf("The requested Value is %f\n", a[threadIdx.x]);
}

__global__ void copy(float * a, float* res){
    int id = threadIdx.x + (blockIdx.x * blockDim.x);
    // if(id == 0){
    //     printf( "Copy has been entered\n");
    // }
    // printf("I will put a[%d]= %f into res[%d]\n", id, a[id], id);
    res[id] = a[id];
}

//elaborated function to check which block and how is not doing alirght in Gaussian
__global__ void gaussCheck(float * a){

    int id = threadIdx.x + (blockIdx.x * blockDim.x);
    int low;
    id == 0 ? low = 0 : low = 4095 + (id - 1) * 4096;
    int top = 4095 + id * 4096;

    float subGauss = ((pow(top, 2.0) + top) / 2) - ((pow(low, 2.0) + low) / 2);
    if (blockDim.x == 4){
        printf("I am Thread %d, my subGauss is %f\n", id, subGauss);
    }
    if (subGauss =! a[id]){
        printf("In the previous iteration block %d failed, produced %f, should be %f\n", id, a[id], subGauss);
    }


}

__global__ void align (float* a, float* res, int stride){
    int id = threadIdx.x + (blockIdx.x * blockDim.x);
    res[id] = a[id * stride];
    //printf("I put a[%d] = %f into res[%d], the stride is %d\n", id* stride, a[id*stride], id, stride);
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
            //printf("I am thread %d, with res[%d] = %f. I will add res[%d] = %f\n", id, id, res[id], id + stepSize, res[id + stepSize]);   
            res[id] += res[id + i];
        }
        __syncthreads();
    }
    if(id == 0){
        //printf("the result is res[0] = %f\n", res[0]);
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
      //  printf("I am from block %d, I will write %f into cell %d\n", blockIdx.x, rab[bId], gId);
        res[gId] = rab[bId];
    }
}

__global__ void sharedMemSum (float * res, int size){

    extern __shared__ float r[];
    int bId = threadIdx.x;
    int gId = threadIdx.x + 2 * (blockDim.x * blockIdx.x);

    // if(bId== 0){

    //     printf("Blocksize is %d, size is %d\n", blockDim.x,size);
    // }

    // First iteration + loading:
    r[bId] = res[gId] + res[gId + size / 2];
    
    if(gridDim.x == 1){

        printf("I am thread %d, and after the first iteration my value is %f\n", bId, r[bId]);
    }

    __syncthreads();
    int its = 0;
    for(int i = size / 4; i > 0; i /= 2){
        
        if(bId < i){

            r[bId] += r[bId + i];
            if(gridDim.x == 1){
               printf("I am thread %d and after iteration %d my value is %f\n", bId, its, r[bId]);
            }
        }
        __syncthreads();
        its++;
    }

    // writing the result back to global Memory
    if(bId == 0){
        if(gridDim.x == 1){
         printf("I am thread 0 from block %d and my result is %f, my gid is %d\n", blockIdx.x, r[bId], gId);
        }
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
            if(blockIdx.x == 1){
                // printf("I am thread %d, mySum is %f before addition, i is %d\n", id, mySum, i);
            }
            mySum += __shfl_down_sync(mask, mySum, i);

            if(blockIdx.x == 1){
                // printf("I am thread %d, mySum is %f after addition\n", id, mySum);
            }
        }
    }

    // loading warp results into shared Memory
    if (laneId == 0){
        // printf("from first kernel call: I am thread %d, in lane 0 of warp %d and mySum is %f\n", id, warpId, mySum);
        // printf("from first kernel call: nWarps is %d\n", nWarps);
        r[warpId] = mySum;
    }

    __syncthreads();
    // Reducing results from first reduction
    if (nWarps > 1){
        for (int i = nWarps / 2; i > 0; i /= 2){
            if(blockId < i){
                //printf("from first kernel call: I am thread 0 from block %d, I will add %f to %f\n", blockId, r[blockId], r[blockId + i]);
                r[blockId] += r[blockId + i];
            }
            __syncthreads();
        }
    }

    // Writing back to global Memory
    if(threadIdx.x == 0){
     //   printf("Hit in writeback in first call, I am thread %d and I am writing back %f\n", id, mySum);
        res[blockIdx.x] = r[0];
    }
}
__global__ void warpRedSum(float* a, float* res, int size){

    extern __shared__ float r[];

    int id = threadIdx.x + (blockDim.x * blockIdx.x);
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
        // printf("I am thread %d, in lane 0 of warp %d and mySum is %f\n", id, warpId, mySum);
        // printf("nWarps is %d\n", nWarps);
        r[warpId] = mySum;
    }

    __syncthreads();
    // Reducing results from first reduction
    if (nWarps > 1){
        for (int i = nWarps / 2; i > 0; i /= 2){
            if(blockId < i){
                // printf("I am thread 0 from block %d, I will add %f to %f\n", blockId, r[blockId], r[blockId + i]);
                r[blockId] += r[blockId + i];
            }
            __syncthreads();
        }
    }

    // Writing back to global Memory
    if(threadIdx.x == 0){
      //printf("Hit in writeback, I am thread %d and I am writing back %f\n", id, mySum);
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

void testSharedMemSum(int size, int threads){

    int nBlocks = size / (2 * threads);
    float * a = (float*) malloc (sizeof(float)*size);
    float* d_a;

    cudaMalloc(&d_a, sizeof(float)*size);

    vecInitGauss (a, size);
    cudaMemcpy(d_a, a, sizeof(float)*size , cudaMemcpyHostToDevice);

    sharedMemSum <<<nBlocks, threads, sizeof(float) * threads>>> (d_a, size);

    cudaMemcpy(a, d_a, sizeof(float)*size, cudaMemcpyDeviceToHost);

    float n = size - 1;
    float expectedRes = (pow(n, 2.0) + n) / 2;
    if (a[0] != (float) expectedRes){
        printf("The result is %f, but should be %f \n", a[0], (float) expectedRes);
    }
}

float subGauss(int low, int top){
    
    return (((pow(top, 2.0) + top) / 2) - ((pow(low, 2.0) + low) / 2));
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

   vecInitGauss(a, size);

    //vecInitOnes(a, size);
    vecInitOnes(b, size);

    cudaMemcpy(d_a, a, sizeof(float)*size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float)*size , cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);
    // to support a size bigger than 32 we need to
    // basically do another addition for each one of the blocks!

    //LOOOOOOOOL this only applies to standard warp reduction, not using global or shared Memory!!

    // Every thread can add two numbers, thus we can add 2*threads many numbers in one iteration
    naiveGlobalMem <<<nBlocks, threads>>> (d_a, d_b, d_res, threads * 2);
    
    
    // nBlocks is the number of elements that still need summing up
    while (nBlocks > threads){
        //careful! There is no case that handles if the number of elements to add are somewhere inbetween threads and 2*threads
        cudaDeviceSynchronize();
        int new_nBlocks = nBlocks / (threads * 2);
        align <<<2 * new_nBlocks, threads>>> (d_res, d_a, 2 * threads);
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess){
            printf("Cuda Error in align: %s\n", cudaGetErrorString(err));
            printf("Config Args: new_nBlocks = %d, threads = %d\n", new_nBlocks, threads);
        }
        cudaDeviceSynchronize();

        // Align swaps around d_a and d_res, so we swap it back
        float * temp = d_res;
        d_res = d_a;
        d_a = temp;

        nBlocks = new_nBlocks;
        globalMemSum <<<nBlocks, threads>>> (d_res, threads * 2);
    }
    
    if (nBlocks > 1){
        // printf("Entered nBlocks > 1 with nBlocks = %d\n", nBlocks);
        // printf("After alignment:\n");
        // printArr <<<1, nBlocks>>> (d_res);
        cudaDeviceSynchronize();
        align <<<1, nBlocks>>> (d_res, d_a, 2 * threads);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess){
            printf("Cuda Error in align: %s\n", cudaGetErrorString(err));
            printf("Config Args: new_nBlocks = %d, threads = %d\n", 1, nBlocks);
        }

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

    // printf("Naive with Global Memory took %fms\n", time);

    cudaMemcpy(res, d_res, sizeof(float)*size, cudaMemcpyDeviceToHost);
    
    if (res[0] != (float)expectedRes){
        printf("The result is %f, but should be %f \n", res[0], (float) expectedRes);
    } else {
        printf("Result is correct\n");
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

    // printf("This is before first iteration: nBlocks = %d\n", nBlocks);

    // There are 2 * threads elements to be added and there are three arrays of that many elements
    naiveSharedMem<<<nBlocks, threads, sizeof(float) * threads * 2 * 3>>> (d_a, d_b, d_res, threads * 2);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Cuda Error after first iteration: %s\n", cudaGetErrorString(err));
        printf("Launch config: nBlocks = %d, threads = %d, sharedMem = %d\n", nBlocks, threads, threads * 2 * 3);
    }
    
    int new_nBlocks = nBlocks / (threads * 2);
    if (nBlocks > 1 && new_nBlocks == 0){
        new_nBlocks = 1;
    }
    
    cudaDeviceSynchronize();
    
    err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Cuda Error after align: %s\n", cudaGetErrorString(err));
        printf("Launch config: new_nBlocks = %d, threads = %d, sharedMem = %d\n", new_nBlocks, threads, threads * 2 * 3);
    }
    int its = 0;
    // printf("nBlocks before while: %d\n",nBlocks);
    while(nBlocks > 2 * threads){
        its++;
        //printf("This is while iteration %d, new_nBlocks = %d, nBlocks = %d\n", its, new_nBlocks, nBlocks);
        
        align<<<new_nBlocks * 2, threads>>> (d_res, d_a, 2 * threads);
        
        float * temp = d_res;
        d_res = d_a;
        d_a = temp;
        
        cudaDeviceSynchronize();
        gaussCheck <<<new_nBlocks, threads>>> (d_res);
        
       // printArr<<<new_nBlocks, threads>>> (d_res);
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if(err != cudaSuccess){
            printf("Cuda Error after Align in while: %s\n", cudaGetErrorString(err));
            printf("Launch config: #blocks = %d, threads = %d\n", new_nBlocks * 2, threads);
        }
        
        nBlocks = new_nBlocks;
        sharedMemSum<<<nBlocks, threads, sizeof(float) * threads>>> (d_res, threads * 2);

        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if(err != cudaSuccess){
            printf("Cuda Error after SharedMemSum in while: %s\n", cudaGetErrorString(err));
            printf("Launch config: nBlocks = %d, threads = %d, sharedMem = %d\n", nBlocks, threads, threads);
        }
        new_nBlocks = nBlocks / (threads * 2);
        if (nBlocks > 1 && new_nBlocks == 0){
            new_nBlocks = 1;
        }
    }

    if (nBlocks > 1){
        align<<<new_nBlocks, nBlocks>>> (d_res, d_a, 2 * threads);
        
        //  printf("In if new_nBlocks = %d, nBlocks is %d\n", new_nBlocks, nBlocks);
        
        float * temp = d_res;
        d_res = d_a;
        d_a = temp;
        
        cudaDeviceSynchronize();
        
        err = cudaGetLastError();
         if(err != cudaSuccess){
             printf("Cuda Error after align in if: %s\n", cudaGetErrorString(err));
             printf("Launch config: new_nBlocks = %d, threads = %d\n", new_nBlocks, nBlocks);
        }
        // printf("from if clause: nBlocks= %d\n", nBlocks);
         printArr <<<1, nBlocks>>> (d_res);


        gaussCheck <<<1, nBlocks>>> (d_res);

        sharedMemSum<<<1, (nBlocks / 2), sizeof(float) * (nBlocks / 2)>>> (d_res, nBlocks);
    }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //printf("Naive with Shared Memory took %fms\n", time);

    cudaMemcpy(res, d_res, sizeof(float)*size, cudaMemcpyDeviceToHost);
    
    float expectedRes = subGauss(0, size - 1);

    if (res[0] != (float) expectedRes){
        printf("Threads = %d, size = %d \nThe result is %f, but should be %f \n", threads, size, res[0], (float) expectedRes);
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
    
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Cuda Error after first iteration: %s\n", cudaGetErrorString(err));
        printf("Launch config: gridDim = %d, threads = %d, sharedMem = %d\n", 1, size / 2, warps_per_block);
    }
    int new_nBlocks = nBlocks / (threads * 2);

    //printVal<<<1,1>>>(d_res);
    cudaDeviceSynchronize();

    while (nBlocks > threads * 2){

        nBlocks = new_nBlocks;
        warpRedSum<<<nBlocks, threads, sizeof(float) * warps_per_block>>>(d_res, d_a, threads * 2);
        cudaDeviceSynchronize();
        //printVal<<<1,1>>>(d_res);

        float * temp = d_res;
        d_res = d_a;
        d_a = temp;


        new_nBlocks = nBlocks / (threads * 2);
     //   printf("In While:\n");
       // printArr<<<new_nBlocks * 2, threads>>>(d_res);
    }

   // printf("Between While and if\n");
    //printArr<<<new_nBlocks * 2, threads>>>(d_res);
    //printVal<<<1,1>>>(d_res);

    if (nBlocks > 1){
        
        nBlocks /= 2;
        warps_per_block = nBlocks / 32 + 1;         // the plus 1 is in case integer summation rounds down

        warpRedSum<<<1, nBlocks, sizeof(float) * warps_per_block>>> (d_res, d_a, nBlocks * 2);
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if(err != cudaSuccess){
            printf("Cuda Error after first iteration: %s\n", cudaGetErrorString(err));
            printf("Launch config: gridDim = %d, threads = %d, sharedMem = %d\n", 1, nBlocks, warps_per_block);
    }

        float * temp = d_res;
        d_res = d_a;
        d_a = temp;        
    }


    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(res, d_res, sizeof(float)*size, cudaMemcpyDeviceToHost);

    float expRes = subGauss(0, size-1);
    
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
            for (int j = i + 1; j < 27; j++){
                
                int size = 1 << j;
                int threads = 1 << i;
                //printf("Threads = %d, size = %d\n", threads, size );
                // callNaiveGlobalMem(size, threads);
                // callNaiveSharedMem(size, threads);
                callNaiveWarpRed(size, threads);
       
            }
        }
    }

    int j = 6;
    int i = 5;


    int size = 1 << j;
    int threads = 1 << i;
    
    
    // callCublas();
    //callNaiveGlobalMem(size, threads);
    // callNaiveSharedMem(size, threads);
   // callNaiveWarpRed(size, threads); 
   //testSharedMemSum(2048, 1024);

    return 0;
}
