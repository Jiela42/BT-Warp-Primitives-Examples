#include <cub/cub.cuh>
#include <iostream>

using namespace std;




__global__ void elmt_wise_mult(float * a, float * b, float * res){
    int id = threadIdx.x + (blockIdx.x * blockDim.x);
    res[id] = a[id] * b[id];
}

void init_Ones(float* a, int size){
    for(int i = 0; i < size; i++){
        a[i] = 1;
    }
}


int main(){

    int size = 1 << 6;

    float * a = (float*) malloc (sizeof(float)*size);
    float * b = (float*) malloc (sizeof(float)*size);
    float * res = (float*) malloc (sizeof(float));

    float *d_a, *d_b, *d_res, *d_temp;

    cudaMalloc(&d_a, sizeof(float) * size);
    cudaMalloc(&d_b, sizeof(float) * size);
    cudaMalloc(&d_res, sizeof(float));
    cudaMalloc(&d_temp, sizeof(float)*size);

    init_Ones(a, size);
    init_Ones(b, size);

    cudaMemcpy(d_a, a, sizeof(float)*size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float)*size , cudaMemcpyHostToDevice);

    elmt_wise_mult<<<1,size>>>(d_a, d_b, d_temp);

    cudaDeviceSynchronize();

    void *d_temp_storage= NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_temp, d_res, size);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_temp, d_res, size);

    cudaMemcpy(res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Result: " << *res << "\n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);

    free(a);
    free(b);
    free(res);

    return 0;
}
