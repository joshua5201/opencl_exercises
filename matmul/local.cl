__kernel void matmul (
        int m, int n, int p,
        __global float *A,
        __global float *B,
        __global float *C,
        __local float *localA,
        __local float *localB,
        int wg_size)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    int local_row = get_local_id(0);
    int local_col = get_local_id(1);
    int wg_count = n / wg_size;

    float sum = 0.0f;

    for (int wg_ix = 0; wg_ix < wg_count; wg_ix++) {
        localA[local_row*wg_size + local_col] = A[row*n + local_col+wg_size*wg_ix];
        localB[local_row*wg_size + local_col] = B[(local_row+wg_size*wg_ix)*p + col];
        
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j = 0; j < wg_size; j++) {
            sum += localA[local_row*wg_size + j] * B[j*wg_size + local_col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[row*p + col] = sum;
}

