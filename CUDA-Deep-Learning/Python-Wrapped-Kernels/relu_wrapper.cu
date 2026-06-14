#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <iostream>

//  CUDA Kernels 

// Elementwise ReLU: keeps positive values, zeroes out negatives
__global__ void relu_forward_kernel(const float *input, float *output, int size)
{
    // Each thread handles one element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Disable threads beyond array size
    if (idx < size)
    {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// ReLU derivative: gradient flows only where input > 0
__global__ void relu_backward_kernel(const float *grad_output, const float *input,
                                     float *grad_input, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}


//C wrappers for python ctypes 
extern "C"   // ensures function names aren't C++ mangled
{
    // Memory helpers (Python will call these through ctypes)

    // Allocate space on GPU for float array
    float *allocate_gpu_memory(size_t size)
    {
        float *ptr;
        cudaMalloc(&ptr, size * sizeof(float));
        return ptr;
    }

    // Free previously allocated GPU memory
    void free_gpu_memory(float *ptr)
    {
        cudaFree(ptr);
    }

    // copy from CPU to GPU
    void copy_to_gpu(float *gpu_ptr, const float *cpu_ptr, size_t size)
    {
        cudaMemcpy(gpu_ptr, cpu_ptr, size * sizeof(float), cudaMemcpyHostToDevice);
    }

    // copy from GPU to CPU
    void copy_from_gpu(float *cpu_ptr, const float *gpu_ptr, size_t size)
    {
        cudaMemcpy(cpu_ptr, gpu_ptr, size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    //functions your Python script calls
    // Runs ReLU forward kernel across entire tensor
    void relu_forward_wrapper(float *input, float *output, int size)
    {
        // Choose execution configuration
        dim3 threads_per_block(512); 
        dim3 number_of_blocks((size + threads_per_block.x - 1) / threads_per_block.x);

        // Launch kernel on GPU
        relu_forward_kernel<<<number_of_blocks, threads_per_block>>>(
            input, output, size);

        cudaDeviceSynchronize();  // ensure kernel finishes before returning
    }

    // Runs ReLU backward pass during training
    void relu_backward_wrapper(float *grad_output, float *input,
                               float *grad_input, int size)
    {
        dim3 threads_per_block(512);
        dim3 number_of_blocks((size + threads_per_block.x - 1) / threads_per_block.x);

        relu_backward_kernel<<<number_of_blocks, threads_per_block>>>(
            grad_output, input, grad_input, size);
        
        cudaDeviceSynchronize();
    }

    //optional helpers for debugging
    void cuda_synchronize()
    {
        cudaDeviceSynchronize();
    }

    int get_cuda_device_count()
    {
        int count;
        cudaGetDeviceCount(&count);
        return count;
    }
}
