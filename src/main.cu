#include <iostream>

#include <cuda_runtime.h>
#include <cudnn.h>

#define CUDA_CHECK(err)                                                                                     \
    if (err != cudaSuccess)                                                                                 \
    {                                                                                                       \
        printf("cuda error: file: %s line: %d details: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::terminate();                                                                                   \
    }

#define CUDNN_CHECK(err)                                                                                      \
    if (err != CUDNN_STATUS_SUCCESS)                                                                          \
    {                                                                                                         \
        printf("cudnn error: file: %s line: %d details: %s\n", __FILE__, __LINE__, cudnnGetErrorString(err)); \
        std::terminate();                                                                                     \
    }

void ConvolutionBackwardFilter(int input_n, int input_c, int input_d, int input_h, int input_w,
                               int output_c, int kernel_d, int kernel_h, int kernel_w,
                               int stride_d, int stride_h, int stride_w,
                               int pad_d, int pad_h, int pad_w,
                               int dilation_d, int dilation_h, int dilation_w,
                               int group)
{
    int x_size = input_n * input_c * input_d * input_h * input_w;
    // malloc device
    float *d_x;
    CUDA_CHECK(cudaMalloc(&d_x, x_size * sizeof(float)));

    int w_size = output_c * input_c * kernel_d * kernel_h * kernel_w;
    float *d_w;
    CUDA_CHECK(cudaMalloc(&d_w, w_size * sizeof(float)));

    int kdd = (kernel_d - 1) * dilation_d + 1;
    int khd = (kernel_h - 1) * dilation_h + 1;
    int kwd = (kernel_w - 1) * dilation_w + 1;
    int output_d = (input_d - kdd + 2 * pad_d) / stride_d + 1;
    int output_h = (input_h - khd + 2 * pad_h) / stride_h + 1;
    int output_w = (input_w - kwd + 2 * pad_w) / stride_w + 1;

    int y_size = input_n * output_c * output_d * output_h * output_w;
    float *d_y;
    CUDA_CHECK(cudaMalloc(&d_y, y_size * sizeof(float)));

    // malloc host
    float *h_x = new float[x_size]{0};
    float *h_w = new float[w_size]{0};
    float *h_ref_w = new float[w_size]{0};
    float *h_y = new float[y_size]{0};

    // init x
    for (int i = 0; i < x_size; ++i)
    {
        h_x[i] = static_cast<float>(i % 4);
    }

    // init y
    for (int i = 0; i < y_size; ++i)
    {
        h_y[i] = static_cast<float>(i % 3);
    }

    // memcpy host -> device
    CUDA_CHECK(cudaMemcpy(d_x, h_x, x_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, y_size * sizeof(float), cudaMemcpyHostToDevice));

    // gpu
    float alpha = 1.f;
    float beta = 0.f;
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    cudnnFilterDescriptor_t wDesc;
    cudnnTensorDescriptor_t xDesc, yDesc;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&wDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));

    const int dim_w[5] = {output_c, input_c, kernel_d, kernel_h, kernel_w};

    CUDNN_CHECK(cudnnSetFilterNdDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, dim_w));

    const int dim_x[5] = {input_n, input_c, input_d, input_h, input_w};
    const int stride_x[5] = {input_c * input_d * input_h * input_w, input_d * input_h * input_w, input_h * input_w, input_w, 1};
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(xDesc, CUDNN_DATA_FLOAT, 5, dim_x, stride_x));

    const int dim_y[5] = {input_n, output_c, output_d, output_h, output_w};
    const int stride_y[5] = {output_c * output_d * output_h * output_w, output_d * output_h * output_w, output_h * output_w, output_w, 1};
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(yDesc, CUDNN_DATA_FLOAT, 5, dim_y, stride_y));

    cudnnConvolutionDescriptor_t convDesc;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));
    int pad[] = {0, 0, 0};
    int stride[] = {1, 1, 1};
    int upscale[] = {1, 1, 1};
    CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(convDesc, 3, pad, stride, upscale, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    cudnnConvolutionBwdFilterAlgo_t algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3;
    size_t workspace_size;
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, xDesc, yDesc, convDesc, wDesc, algo, &workspace_size));
    float *workSpace;
    CUDA_CHECK(cudaMalloc(&workSpace, workspace_size));
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(handle, &alpha, xDesc, d_x, yDesc, d_y, convDesc, algo, workSpace, workspace_size, &beta, wDesc, d_w));

    CUDNN_CHECK(cudnnDestroy(handle));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(wDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convDesc));

    CUDA_CHECK(cudaMemcpy(h_w, d_w, w_size * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "gpu(cudnn):" << std::endl;
    for (int n = 0; n < output_c; ++n)
    {
        for (int d = 0; d < kernel_d; ++d)
        {
            for (int i = 0; i < kernel_h; ++i)
            {
                for (int j = 0; j < input_c; ++j)
                {
                    for (int k = 0; k < kernel_w; ++k)
                    {
                        std::cout << h_w[n * kernel_h * kernel_w * input_c * kernel_d + d * kernel_h * kernel_w + j * kernel_d * kernel_h * kernel_w + i * kernel_w + k] << "\t";
                    }
                    std::cout << "\t\t";
                }
                std::cout << std::endl;
            }
        }
        std::cout << "\n";
    }

    // free
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_y));

    CUDA_CHECK(cudaFree(workSpace));

    delete[] h_x;
    delete[] h_w;
    delete[] h_ref_w;
    delete[] h_y;
}

int main()
{
    int input_n = 1;
    int input_c = 4;
    int input_d = 4;
    int input_h = 4;
    int input_w = 4;

    int output_c = 1;
    int kernel_d = 3;
    int kernel_h = 3;
    int kernel_w = 3;

    int stride_d = 1;
    int stride_h = 1;
    int stride_w = 1;

    int pad_d = 0;
    int pad_h = 0;
    int pad_w = 0;

    int dilation_d = 1;
    int dilation_h = 1;
    int dilation_w = 1;

    int group = 1;

    ConvolutionBackwardFilter(input_n, input_c, input_d, input_h, input_w,
                              output_c, kernel_d, kernel_h, kernel_w,
                              stride_d, stride_h, stride_w,
                              pad_d, pad_h, pad_w,
                              dilation_d, dilation_h, dilation_w,
                              group);

    return 0;
}