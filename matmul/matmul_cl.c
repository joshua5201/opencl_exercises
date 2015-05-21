#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include "CL/cl.h"
#include "data.h"

int load_kernel_file(char *name, char **buff)
{
    FILE *fp;
    if ((fp = fopen(name, "r")) == NULL) {
        printf("error opening file %s\n", name);
        return 1;
    }

    struct stat st;
    stat(name, &st);
    size_t code_size = st.st_size;
    *buff = malloc(code_size + 1);
    (*buff)[code_size] = '\0';
    fread(*buff, st.st_size, 1, fp);
    fclose(fp);
    return 0;
}
int run_file(const char *filename)
{
    cl_platform_id platform;
    cl_uint num_platforms = 0;

    cl_int err = 0;
    err = clGetPlatformIDs(1, &platform, &num_platforms);
    if (err) {
        printf("Unable to get platform id\n");
        return 1;
    }

    cl_device_id device;;
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &num_devices);
    if (err) {
        printf("unable to get device id\n");
        return 1;
    }
    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
    cl_context context = clCreateContext(properties, 1, &device, NULL, NULL, &err);
    if (err) {
        printf("error creating context\n");
        return 1;
    }

    cl_command_queue cmd_que = clCreateCommandQueue(context, device, 0, &err);
    if (err) {
        printf("error creating command queue\n");
        return 1;
    }

    extern float mat_A[SIZE_M*SIZE_N];
    extern float mat_B[SIZE_N*SIZE_P];
    float mat_C[SIZE_M*SIZE_P] = {0.0f};
    cl_mem buffer_A = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(mat_A), NULL, &err);
    if (err) {
        printf("error creating buffer for mat_A\n");
        return 1;
    }
    cl_mem buffer_B = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(mat_B), NULL, &err);
    if (err) {
        printf("error creating buffer for mat_B\n");
        return 1;
    }
    cl_mem buffer_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(mat_C), NULL, &err);
    if (err) {
        printf("error creating buffer for mat_C\n");
        return 1;
    }

    char *kernel_file;
    if (load_kernel_file(filename, &kernel_file)) {
        return 1;
    }
    
    const char *kernel_codes[] = {kernel_file, NULL};
    cl_program program = clCreateProgramWithSource(context, 1, kernel_codes, NULL, &err);
    if (err) {
        printf("error creating program\n");
        return 1;
    }

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err) {
        printf("error building program\n");
        int len;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, NULL, NULL, &len);
        char *log = malloc(len); 
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, log, NULL);
        printf("%s\n", log);
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "matmul", &err);
    if (err) {
        printf("error creating kernel\n");
        return 1;
    }

    err = clEnqueueWriteBuffer(cmd_que, buffer_A, CL_TRUE, 0, sizeof(mat_A), mat_A, 0, NULL, NULL);
    if (err) {
        printf("error writing buffer\n");
        return 1;
    }
    err = clEnqueueWriteBuffer(cmd_que, buffer_B, CL_TRUE, 0, sizeof(mat_B), mat_B, 0, NULL, NULL);
    if (err) {
        printf("error writing buffer\n");
        return 1;
        }
/*    err = clEnqueueWriteBuffer(cmd_que, buffer_C, CL_TRUE, 0, sizeof(mat_C), mat_C, 0, NULL, NULL);
    if (err) {
        printf("error writing buffer\n");
        return 1;
    }*/
    int m = SIZE_M;
    int n = SIZE_N;
    int p = SIZE_P;
    int wg_size = 16;
    clSetKernelArg(kernel, 0, sizeof(int), &m);
    clSetKernelArg(kernel, 1, sizeof(int), &n);
    clSetKernelArg(kernel, 2, sizeof(int), &p);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &buffer_A);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &buffer_B);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &buffer_C);
    if (strcmp("local.cl", filename) == 0) {
        clSetKernelArg(kernel, 6, wg_size * wg_size * sizeof(float), NULL);
        clSetKernelArg(kernel, 7, wg_size * wg_size * sizeof(float), NULL);
        clSetKernelArg(kernel, 8, sizeof(int), &wg_size);
    }

    const size_t global_work_size[2] = {m, p}; 

    if (strcmp("local.cl", filename) == 0) {
        const size_t local_work_size[2] = {wg_size, wg_size};
        clEnqueueNDRangeKernel(cmd_que, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    } else {
        clEnqueueNDRangeKernel(cmd_que, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    }
    clFinish(cmd_que);

    err = clEnqueueReadBuffer(cmd_que, buffer_C, CL_TRUE, 0, sizeof(mat_C), mat_C, 0, NULL, NULL);
    if (err) {
        printf("error reading buffer\n");
        return 1;
    }

    char output_filename[strlen(filename) + strlen(".out") + 1];
    strcat(output_filename, filename);
    strcat(output_filename, ".out");
    FILE *outfp = fopen(output_filename, "w");
    for (int i = 0; i < SIZE_M; i++) {
        for (int j = 0; j < SIZE_P; j++) {
            fprintf(outfp, "%.2f ", mat_C[i*SIZE_P + j]);
        }
        fprintf(outfp, "\n");
    }
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseMemObject(buffer_A);
    clReleaseMemObject(buffer_B);
    clReleaseMemObject(buffer_C);
    clReleaseCommandQueue(cmd_que);
    clReleaseContext(context);

    return 0;

}
int main()
{
    if (run_file("global.cl")) {
        return 1;
    }
    if (run_file("local.cl")) {
        return 1;
    }
    return 0;
}

