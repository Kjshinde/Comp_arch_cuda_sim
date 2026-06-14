#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <iostream>

//Forward Pass Kernel
__global__ void linear_forward_kernel(const float *input, const float *weight, const float *bias,
                                      float *output, int batch_size, int input_size, int output_size)
{
    // Identify which sample + output neuron this thread processes
    int batch_idx = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure valid thread
    if (batch_idx < batch_size && out_idx < output_size)
    {
        float sum = 0.0f;

        // Dot product between input vector and weight row
        for (int i = 0; i < input_size; i++)
        {
            sum += input[batch_idx * input_size + i] * weight[out_idx * input_size + i];
        }

        // Store linear output with bias added
        output[batch_idx * output_size + out_idx] = sum + bias[out_idx];
    }
}

//Backward Pass Kernels
// Computes gradient wrt input
__global__ void linear_backward_input_kernel(const float *grad_output, const float *weight,
                                             float *grad_input, int batch_size, int input_size, int output_size)
{
    int batch_idx = blockIdx.y;
    int in_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && in_idx < input_size)
    {
        float sum = 0.0f;

        // sum over all output neurons
        for (int i = 0; i < output_size; i++)
        {
            sum += grad_output[batch_idx * output_size + i] * weight[i * input_size + in_idx];
        }

        grad_input[batch_idx * input_size + in_idx] = sum;
    }
}

// Computes gradient wrt weights
__global__ void linear_backward_weight_kernel(const float *grad_output, const float *input,
                                              float *grad_weight, int batch_size, int input_size, int output_size)
{
    int out_idx = blockIdx.y;
    int in_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx < output_size && in_idx < input_size)
    {
        float sum = 0.0f;

        //Sum across batch dimension
        for (int b = 0; b < batch_size; b++)
        {
            sum += grad_output[b * output_size + out_idx] * input[b * input_size + in_idx];
        }

        grad_weight[out_idx * input_size + in_idx] = sum;
    }
}

// Computes gradient wrt bias: sum of gradients over batch
__global__ void linear_backward_bias_kernel(const float *grad_output, float *grad_bias,
                                            int batch_size, int output_size)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx < output_size)
    {
        float sum = 0.0f;

        // Sum gradient contributions for each output neuron
        for (int b = 0; b < batch_size; b++)
        {
            sum += grad_output[b * output_size + out_idx];
        }

        grad_bias[out_idx] = sum;
    }
}

//ctypes-Accessible Wrapper Functions
extern "C"   
{
    // Allocate float buffer on GPU
    float *allocate_gpu_memory(size_t size)
    {
        float *ptr;
        cudaMalloc(&ptr, size * sizeof(float));
        return ptr;
    }

    // Free allocated GPU memory
    void free_gpu_memory(float *ptr)
    {
        cudaFree(ptr);
    }

    // copy from CPU to GPU
    void copy_to_gpu(float *gpu_ptr, const float *cpu_ptr, size_t size)
    {
        cudaMemcpy(gpu_ptr, cpu_ptr, size * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Copy from GPU to CPU
    void copy_from_gpu(float *cpu_ptr, const float *gpu_ptr, size_t size)
    {
        cudaMemcpy(cpu_ptr, gpu_ptr, size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Launch linear forward kernel
    void linear_forward_wrapper(float *input, float *weight, float *bias, float *output,
                                int batch_size, int input_size, int output_size)
    {
        dim3 threads_per_block(512);  // number of threads per block
        // Grid shaped so each batch + neuron gets its own thread
        dim3 blocks((output_size + threads_per_block.x - 1) / threads_per_block.x, batch_size);

        linear_forward_kernel<<<blocks, threads_per_block>>>(
            input, weight, bias, output, batch_size, input_size, output_size);
        
        cudaDeviceSynchronize();  // wait for GPU completion
    }

    // Launch all backward kernels
    void linear_backward_wrapper(float *grad_output, float *input, float *weight,
                                 float *grad_input, float *grad_weight, float *grad_bias,
                                 int batch_size, int input_size, int output_size)
    {
        dim3 threads_per_block(512);

        dim3 input_blocks((input_size + threads_per_block.x - 1) / threads_per_block.x, batch_size);
        linear_backward_input_kernel<<<input_blocks, threads_per_block>>>(
            grad_output, weight, grad_input, batch_size, input_size, output_size);

        dim3 weight_blocks((input_size + threads_per_block.x - 1) / threads_per_block.x, output_size);
        linear_backward_weight_kernel<<<weight_blocks, threads_per_block>>>(
            grad_output, input, grad_weight, batch_size, input_size, output_size);

        dim3 bias_blocks((output_size + threads_per_block.x - 1) / threads_per_block.x);
        linear_backward_bias_kernel<<<bias_blocks, threads_per_block>>>(
            grad_output, grad_bias, batch_size, output_size);

        cudaDeviceSynchronize();  // ensure all kernels finish
    }

    // Optional utility sync
    void cuda_synchronize()
    {
        cudaDeviceSynchronize();
    }

    // Returns number of available GPUs
    int get_cuda_device_count()
    {
        int count;
        cudaGetDeviceCount(&count);
        return count;
    }
}
