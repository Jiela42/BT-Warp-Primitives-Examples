#pragma once

#define cudaCheckErr() {                                                                           \
    cudaError_t err = cudaGetLastError();                                                          \
    if(err != cudaSuccess){                                                                        \
    printf("Cuda Error in align: %s, %d %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
    exit(0);                                                                                   \
    }                                                                                              \
}

// This is a function, that can be useful for debugging
__global__ void printArr(float * a){
    int id = threadIdx.x + (blockIdx.x * blockDim.x);
    printf("From PrintArr: res[%d] = %f\n", id, a[id]);
}

__global__ void align (float* a, float* res, int stride){
    int id = threadIdx.x + (blockIdx.x * blockDim.x);
    res[id] = a[id * stride];
}

__global__ void globalMemProdSumKernel(float * a, float * b, float* res, int size){
   
    /* since every thread adds up two numbers (in the first iteration)
    we need double the block-starting point to keep working on disjoint parts of the input */
    int id = threadIdx.x + 2 * (blockIdx.x * blockDim.x);
    int stepSize = size/2;

    // first iteration: 
    res[id] = (a[id] * b[id]) + (a[id + stepSize] * b[id + stepSize]);

    __syncthreads();

    for(int i = stepSize / 2; i > 0; i /= 2){
        if (id - 2 * (blockDim.x * blockIdx.x) < i){
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

int multi_block_globalMem(float* a, float* b, float* res, int size, int threads){
    // Every thread can add two numbers, thus we can add 256 number in one iteration
    int nBlocks = size / (2 * threads);

    float * res_ptr = res;

    // to support a size bigger than 32 we need to
    // basically do another addition for each one of the blocks!

    // Every thread can add two numbers, thus we can add 2*threads many numbers in one iteration
    globalMemProdSumKernel <<<nBlocks, threads>>> (a, b, res, threads * 2); 

    // nBlocks is the number of elements that still need summing up
    while (nBlocks > 2 * threads){
        cudaDeviceSynchronize();
        int new_nBlocks = nBlocks / (threads * 2);
        align <<<2 * new_nBlocks, threads>>> (res, a, 2 * threads);
        cudaCheckErr();
        cudaDeviceSynchronize();

        // Align swaps around a and res, so we swap it back
        float * temp = res;
        res = a;
        a = temp;

        nBlocks = new_nBlocks;
        globalMemSum <<<nBlocks, threads>>> (res, threads * 2);
    }
    
    if (nBlocks > 1){
        cudaDeviceSynchronize();
        align <<<1, nBlocks>>> (res, a, 2 * threads);
        cudaDeviceSynchronize();
        cudaCheckErr();
        // Align swaps around a and res, so we swap it back
        float * temp = res;
        res = a;
        a = temp;

        globalMemSum <<<1, nBlocks / 2>>> (res, nBlocks);
    }

    int swap = 0;
    // the res may now be a, the return value ensures that the right value gets copied
    if(res != res_ptr){
      swap = 1;
    }
    return swap;
}