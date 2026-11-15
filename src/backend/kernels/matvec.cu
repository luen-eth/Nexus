// Dense F32 matrix-vector product, GGUF row-major payload.
// Each output row is reduced by one 256-thread block. No fast math.
extern "C" __global__ void nexus_matvec(
    const float* weights, const float* input, float* output,
    unsigned int rows, unsigned int cols) {
    const unsigned int row = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (unsigned int col = lane; col < cols; col += blockDim.x)
        sum += weights[(unsigned long long)row * cols + col] * input[col];
    __shared__ float sums[256];
    sums[lane] = sum;
    __syncthreads();
    for (unsigned int stride = 128; stride; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        __syncthreads();
    }
    if (lane == 0) output[row] = sums[0];
}
