#pragma once

#define cudaCheckErr() {                                                                           \
    cudaError_t err = cudaGetLastError();                                                          \
    if(err != cudaSuccess){                                                                        \
    printf("Cuda Error in align: %s, %d %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
    exit(0);                                                                                   \
    }                                                                                              \
}

__global__ void sharedMemProd(float * a, float * b, float* res, int size){

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

int multi_block_sharedMem(float* a, float* b, float* res, int size, int threads){

    int nBlocks = size / (2 * threads);
    float * res_cpy = res;

    // There are 2 * threads elements to be added and there are three arrays of that many elements
    sharedMemProd<<<nBlocks, threads, sizeof(float) * threads * 2 * 3>>> (a, b, res, threads * 2);

    cudaCheckErr();
    
    int new_nBlocks = nBlocks / (threads * 2);
    if (nBlocks > 1 && new_nBlocks == 0){
        new_nBlocks = 1;
    }
    
    cudaDeviceSynchronize();
    
    cudaCheckErr();

    while(nBlocks > threads){
       
        align<<<new_nBlocks * 2, threads>>> (res, a, 2 * threads);
        float * temp = res;
        res = a;
        a = temp;
        
        cudaDeviceSynchronize();

        cudaCheckErr();
        
        nBlocks = new_nBlocks;
        sharedMemSum<<<nBlocks, threads, sizeof(float) * threads>>> (res, threads * 2);

        cudaDeviceSynchronize();

        cudaCheckErr();

        new_nBlocks = nBlocks / (threads * 2);
        if (nBlocks > 1 && new_nBlocks == 0){
            new_nBlocks = 1;
        }
    }

    if (nBlocks > 1){
        align<<<new_nBlocks, nBlocks>>> (res, a, 2 * threads);
        
        float * temp = res;
        res = a;
        a = temp;
        
        cudaDeviceSynchronize();
        
       cudaCheckErr();

        sharedMemSum<<<1, (nBlocks / 2), sizeof(float) * (nBlocks / 2)>>> (res, nBlocks);
    }

    int swap = 0;

    if(res_cpy != res){
        swap = 1;
    }
    return swap;
}