__kernel void matmul(
        int m, int n, int p, 
        __global float *A,
        __global float *B,
        __global float *C)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    float sum = 0.0f;
    
    for (int i = 0; i < n; i++) {
        sum += A[row*n + i] * B[i*p + col];
    }
    C[row*p + col] = sum;
}

