__global__ void linear_forward_kernel(const float* input, const float* weight, const float* bias, 
                                     float* output, int batch_size, int input_size, int output_size) {

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (batch_idx < batch_size && out_idx < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            sum += input[batch_idx * input_size + i] * weight[out_idx * input_size + i];
        }
        output[batch_idx * output_size + out_idx] = sum + bias[out_idx];
    }
}