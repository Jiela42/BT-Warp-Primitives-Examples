#pragma once

#define cudaCheckErr() {                                                                           \
    cudaError_t err = cudaGetLastError();                                                          \
    if(err != cudaSuccess){                                                                        \
    printf("Cuda Error in align: %s, %d %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
    exit(0);                                                                                   \
    }                                                                                              \
}

__global__ void naiveWarpRed(float* a, float* b, float* res, int size){

    extern __shared__ float r[];

    int id = threadIdx.x + 2 * (blockDim.x * blockIdx.x);
    int threadId = threadIdx.x;
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

    if (warpId == 0){
        mySum = (threadId < nWarps) ? r[threadId] : 0.0;
    }

    for(int i = nWarps / 2; i > 0; i /= 2){
        int condition = threadId < (2 * i) && warpId == 0;
        unsigned mask = __ballot_sync(0xffffffff, condition);
        if (condition){

            mySum += __shfl_down_sync(mask, mySum, i);
        }
    }

    /*__syncthreads();
    // Reducing results from first reduction
    if (nWarps > 1){
        for (int i = nWarps / 2; i > 0; i /= 2){
            if(blockId < i){
                r[blockId] += r[blockId + i];
            }
            __syncthreads();
        }
    }
*/
    // Writing back to global Memory
    if(threadIdx.x == 0){
        res[blockIdx.x] = mySum;
    }
}

__global__ void warpRedSum(float* a, float* res, int size){

    extern __shared__ float r[];

    int id = threadIdx.x + 2 * (blockDim.x * blockIdx.x);
    int threadId = threadIdx.x;
    int laneId = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;
    int nWarps = blockDim.x / 32;
    int stepSize = size / 2;

    nWarps = (nWarps == 0) ? 1 : nWarps;

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

    if(warpId == 0){
        mySum = (threadId < nWarps) ? r[threadId] : 0.0;

    }

    for(int i = nWarps / 2; i > 0; i /= 2){
        int condition = threadId < (2 * i) && warpId == 0;
        unsigned mask = __ballot_sync(0xffffffff, condition);

        if (condition){
            mySum += __shfl_down_sync(mask, mySum, i);    
        }
    }

    // Reducing results from first reduction
  /*  if (nWarps > 1){
        for (int i = nWarps / 2; i > 0; i /= 2){
            if(blockId < i){
               r[blockId] += r[blockId + i];
            }
            __syncthreads();
        }
    }
*/
    // Writing back to global Memory
    if(threadIdx.x == 0){
      res[blockIdx.x] = mySum;
    }
}

int multi_block_warps(float* a, float* b, float* res, int size, int threads){

    float * res_cpy = res;
    int nBlocks = size / (2 * threads);
    int warps_per_block = threads / 32 + 1;     // the +1 is to ensure we assign memory even if int division rounds down


    naiveWarpRed<<<nBlocks, threads, sizeof(float) * warps_per_block>>> (a, b, res, threads * 2);
    cudaDeviceSynchronize();

    cudaCheckErr();

    int new_nBlocks = nBlocks / (threads * 2);


    cudaDeviceSynchronize();

    while (nBlocks > threads * 2){
        
        nBlocks = new_nBlocks;
        warpRedSum<<<nBlocks, threads, sizeof(float) * warps_per_block>>>(res, a, threads * 2);
        cudaDeviceSynchronize();

        float * temp = res;
        res = a;
        a = temp;
        new_nBlocks = nBlocks / (threads * 2);
    }

    if (nBlocks > 1){
        
        nBlocks /= 2;
        warps_per_block = nBlocks / 32 + 1;         // the plus 1 is in case integer summation rounds down

        warpRedSum<<<1, nBlocks, sizeof(float) * warps_per_block>>> (res, a, nBlocks * 2);
        
        cudaDeviceSynchronize();

        cudaCheckErr();
        float * temp = res;
        res = a;
        a = temp;        
    }

    int swap = 0;
    if(res != res_cpy){
        swap = 1;
    }
    return swap;
}

