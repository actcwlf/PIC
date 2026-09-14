#include <vector>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/find.h>
#include <thrust/execution_policy.h>
#include <pybind11/pybind11.h>
#include "bitstream_codec.h"
#include "cuhd_codetable.h"
#include <latent_decoder_kernels.h>
#include "bilinear.h"
#include "reconstruct_latent.h"
#include "fast_dct.h"
#include <thrust/copy.h>
#include <c10/cuda/CUDAStream.h>
#include "cuda_matmul_prelude.cuh"


#define OUT_CHANNEL 3
#define WARP_SIZE 32
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

__global__ void reformat_weight_and_bias_inplace(float * src) {
    // 重新布局元素
    // m_w3.index_put_({torch::indexing::Ellipsis, m_slice}, torch::from_blob(param_buf + param_offset, { 16 * 3}, options).resize_({16, 3}));
    // param_offset += 16 * 3;
        
    // m_b3.index_put_({m_slice}, torch::from_blob(param_buf + param_offset, { 3 }, options));
    // param_offset += 3;

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >=  16 * 16 + 16) {
        return;
    }

    __shared__ float block[16 * 16 + 16]; // thread_per_block * BLOCK_SIZE
    block[tid] = src[tid];
    __syncthreads();

    int row = tid / 16;
    int col = tid % 16;
    if(row == 16) {
        // bias
        
        if(col >= OUT_CHANNEL) {
            src[tid] = 0;
        } else {
            src[tid] = block[16 * OUT_CHANNEL + col];
        }
        return;
    }

    if(row >= OUT_CHANNEL) {
        src[tid] = 0;
        return;
    }

    src[tid] = block[OUT_CHANNEL * col + row];

}

__global__ void permute_weight_inplace(float * src) {
    // 重新布局元素
    // m_w3.index_put_({torch::indexing::Ellipsis, m_slice}, torch::from_blob(param_buf + param_offset, { 16 * 3}, options).resize_({16, 3}));
    // param_offset += 16 * 3;
        
    // m_b3.index_put_({m_slice}, torch::from_blob(param_buf + param_offset, { 3 }, options));
    // param_offset += 3;

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >=  16 * 16) {
        return;
    }

    __shared__ float block[16 * 16]; // thread_per_block * BLOCK_SIZE
    block[tid] = src[tid];
    __syncthreads();

    int row = tid / 16;
    int col = tid % 16;


    src[tid] = block[16 * col + row];

}

__global__ void fused_mlp_inplace(
    float * src, float * dst, 
    float * w0, float * b0,
    float * w1, float * b1,
    float * w2, float * b2,
    float * w3, float * b3,
    int width, int height, 
    int in_channel, int out_channel

) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float input[ 32 * 8 * 16];
    __shared__ float output[ 32 * 8 * 16];
    float * local_input = input + (threadIdx.y * 32 + threadIdx.x) * 16;
    float * local_output = output + (threadIdx.y * 32 + threadIdx.x) * 16;
    for(int i = 0; i < in_channel; i++) {
        local_input[i] = src[(y * width + x) * in_channel + i ];
        // if(x == 0 && y == 0) {
        //     printf("%f ", local_input[i]);
        // }
    }
    // if(x == 0 && y == 0) {
    //     printf("\n");
    // }
    
    for(int i = in_channel; i < 16; i++) {
        local_input[i] = 0;
    }

    // __syncthreads();
    float * warp_input = input + threadIdx.y * 32 * 16;
    float * warp_output = output + threadIdx.y * 32 * 16;

    // layer 0
    wmma_inline_matmul<32, 16, 16, 16, 16 , 8>(warp_input, w0, warp_output);
    #pragma unroll
    for(int i = 0; i < 16; i++) {
        local_output[i] += b0[i];
        local_output[i] = max(0.0, local_output[i]); // relu
    }
    thrust::swap(warp_input, warp_output);
    thrust::swap(local_input, local_output);

    // layer 1
    wmma_inline_matmul<32, 16, 16, 16, 16 , 8>(warp_input, w1, warp_output);
    #pragma unroll
    for(int i = 0; i < 16; i++) {
        local_output[i] += b1[i];
        local_output[i] = max(0.0, local_output[i]); // relu
    }
    thrust::swap(warp_input, warp_output);
    thrust::swap(local_input, local_output);

    // layer 2
    wmma_inline_matmul<32, 16, 16, 16, 16 , 8>(warp_input, w2, warp_output);
    #pragma unroll
    for(int i = 0; i < 16; i++) {
        local_output[i] += b2[i];
        local_output[i] = max(0.0, local_output[i]); // relu
    }
    thrust::swap(warp_input, warp_output);
    thrust::swap(local_input, local_output);

    // layer 3
    wmma_inline_matmul<32, 16, 16, 16, 16 , 8>(warp_input, w3, warp_output);
    #pragma unroll
    for(int i = 0; i < 16; i++) {
        local_output[i] += b3[i];
        local_output[i] = max(0.0, local_output[i]); // relu
    }
    // thrust::swap(warp_input, warp_output);
    // thrust::swap(local_input, local_output);


    // #pragma unroll
    for(int i = 0; i < out_channel ; i++) {
        dst[(y * width + x) * out_channel + i ] = local_output[i];
    }



    // float * warp_input = src + y * 32 * 16;
    // float * warp_output = dst + y * 32 * 16;

    // wmma_inline_matmul<32, 16, 16, 16, 16 , 8>(warp_input, w0, warp_output);

}


__global__ void fused_mlp_for_pic(
    float * src, float * dst, 
    float * w0, float * b0,
    float * w1, float * b1,
    float * w2, float * b2,
    float * w3, float * b3,
    int width, int height, 
    int in_channel, int out_channel

) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float input[ 32 * 8 * 16];
    __shared__ float output[ 32 * 8 * 16];
    float * local_input = input + (threadIdx.y * 32 + threadIdx.x) * 16;
    float * local_output = output + (threadIdx.y * 32 + threadIdx.x) * 16;
    for(int i = 0; i < in_channel; i++) {
        // if(x>= width || y >= height) {
        //     local_input[i] = 0;
        // } else {
            local_input[i] = src[ i * width * height + (y * width + x) ];  // 超出边界的会出现错误，但在最后错误的部分会自动丢弃掉，所以不需要额外处理
        // }
        
        // if(x == 0 && y == 0) {
        //     printf("%f ", local_input[i]);
        // }
    }
    // if(x == 0 && y == 0) {
    //     printf("\n");
    // }
    
    for(int i = in_channel; i < 16; i++) {
        local_input[i] = 0;
    }

    // __syncthreads();
    float * warp_input = input + threadIdx.y * 32 * 16;
    float * warp_output = output + threadIdx.y * 32 * 16;

    // layer 0
    wmma_inline_matmul<32, 16, 16, 16, 16 , 8>(warp_input, w0, warp_output);
    #pragma unroll
    for(int i = 0; i < 16; i++) {
        local_output[i] += b0[i];
        local_output[i] = max(0.0, local_output[i]); // relu
    }
    thrust::swap(warp_input, warp_output);
    thrust::swap(local_input, local_output);

    // layer 1
    wmma_inline_matmul<32, 16, 16, 16, 16 , 8>(warp_input, w1, warp_output);
    #pragma unroll
    for(int i = 0; i < 16; i++) {
        local_output[i] += b1[i];
        local_output[i] = max(0.0, local_output[i]); // relu
    }
    thrust::swap(warp_input, warp_output);
    thrust::swap(local_input, local_output);

    // layer 2
    wmma_inline_matmul<32, 16, 16, 16, 16 , 8>(warp_input, w2, warp_output);
    #pragma unroll
    for(int i = 0; i < 16; i++) {
        local_output[i] += b2[i];
        local_output[i] = max(0.0, local_output[i]); // relu
    }
    thrust::swap(warp_input, warp_output);
    thrust::swap(local_input, local_output);

    // layer 3
    wmma_inline_matmul<32, 16, 16, 16, 16 , 8>(warp_input, w3, warp_output);
    #pragma unroll
    for(int i = 0; i < 16; i++) {
        local_output[i] += b3[i];
        local_output[i] = max(0.0, local_output[i]); // relu
    }
    // thrust::swap(warp_input, warp_output);
    // thrust::swap(local_input, local_output);


    // #pragma unroll
    if(x>= width || y >= height) {
        return;
    }
    for(int i = 0; i < out_channel ; i++) {
        dst[(y * width + x) * out_channel + i ] = min(local_output[i], 1.0);
    }



    // float * warp_input = src + y * 32 * 16;
    // float * warp_output = dst + y * 32 * 16;

    // wmma_inline_matmul<32, 16, 16, 16, 16 , 8>(warp_input, w0, warp_output);

}

torch::Tensor fused_mlp(
    torch::Tensor &x, 
    torch::Tensor &w0, torch::Tensor &b0, 
    torch::Tensor &w1, torch::Tensor &b1, 
    torch::Tensor &w2, torch::Tensor &b2, 
    torch::Tensor &w3, torch::Tensor &b3, 
    
    int width, int height, int in_channel, int out_channel) {
    dim3 threads(32, 8);
    dim3 grid(width / 32, height / 8);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor dst_tensor = torch::empty({width, height, out_channel}, options);
    fused_mlp_inplace<<<grid, threads>>>(
        x.data_ptr<float>(),
        dst_tensor.data_ptr<float>(),
        w0.data_ptr<float>(), b0.data_ptr<float>(),
        w1.data_ptr<float>(), b1.data_ptr<float>(),
        w2.data_ptr<float>(), b2.data_ptr<float>(),
        w3.data_ptr<float>(), b3.data_ptr<float>(),
        width, height, in_channel, out_channel
    );

    return dst_tensor;

}
void append_tensor(std::vector<int> & buf, torch::Tensor tensor) {
    buf.insert(
        buf.end(), 
        reinterpret_cast<int *>(tensor.data_ptr<float>()),
        reinterpret_cast<int *>(tensor.data_ptr<float>() + tensor.numel())
    );
}


py::bytes encode_bitstream(
    int h, int w, int n_latent, int total_symbols, int total_blocks,

    torch::Tensor compact_tensor_dec_table,
    torch::Tensor tensor_compressed,
    torch::Tensor w0,
    torch::Tensor b0,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor w2,
    torch::Tensor b2,
    torch::Tensor w3,
    torch::Tensor b3
) {
    std::vector<int> output_buffer;
    int num_entries = compact_tensor_dec_table.size(0);
    int compressed_size = tensor_compressed.size(0);
    output_buffer.push_back(h);
    output_buffer.push_back(w);
    output_buffer.push_back(n_latent);
    output_buffer.push_back(num_entries); // num_entries
    output_buffer.push_back(compressed_size); // compressed_size
    output_buffer.push_back(total_symbols);
    output_buffer.push_back(total_blocks);

    output_buffer.insert(
        output_buffer.end(), 
        compact_tensor_dec_table.data_ptr<int>(),
        compact_tensor_dec_table.data_ptr<int>() + num_entries
    );
    output_buffer.insert(
        output_buffer.end(), 
        tensor_compressed.data_ptr<int>(),
        tensor_compressed.data_ptr<int>() + compressed_size
    );

    append_tensor(output_buffer, w0);
    append_tensor(output_buffer, b0);
    append_tensor(output_buffer, w1);
    append_tensor(output_buffer, b1);
    append_tensor(output_buffer, w2);
    append_tensor(output_buffer, b2);
    append_tensor(output_buffer, w3);
    append_tensor(output_buffer, b3);


    const char* data = reinterpret_cast<const char*>(output_buffer.data());
    size_t byte_length = output_buffer.size() * sizeof(int);
    // std::cout << "encoding " << (int)(*data) << '\n';
    return py::bytes(data, byte_length);
}


std::tuple<int, int, int, int, int, int, int, torch::Tensor, torch::Tensor,
torch::Tensor, torch::Tensor,
torch::Tensor, torch::Tensor,
torch::Tensor, torch::Tensor,
torch::Tensor, torch::Tensor,
torch::Tensor
>
decode_bitstream(py::bytes bytes_data) {
        cudaEvent_t start1;
    cudaEvent_t stop1;
    cudaEventCreate(&start1);

    cudaEventCreate(&stop1);
    cudaEventRecord(start1, NULL);

    char* buffer;
    ssize_t byte_length;
    if (PyBytes_AsStringAndSize(bytes_data.ptr(), &buffer, &byte_length) == -1) {
        throw py::error_already_set(); // 处理 Python 异常
    }


    // std::cout << "decoding " << (int)(*buffer) << '\n';
    int * buf = reinterpret_cast<int *>(buffer);
    int h = buf[0];
    int w = buf[1];
    int n_latent = buf[2];
    int num_entries = buf[3];
    int compressed_size = buf[4];
    int total_symbols = buf[5];
    int total_blocks = buf[6];

    // std::cout << "h " << h << std::endl;

    int * buf_start = buf + 7;

    auto options = torch::TensorOptions().dtype(torch::kInt32);
    torch::Tensor compact_tensor_dec_table = torch::from_blob(buf_start, {num_entries}, options);
    buf_start += num_entries;



    torch::Tensor tensor_compressed = torch::from_blob(buf_start, {compressed_size}, options).to(torch::kCUDA, true);
    buf_start += compressed_size;

    // std::cout << "before weight\n";

    

    int total_numel = 16 * 16 * 3 + 3 * 16 + 16 * 3 + 3;
    options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor all_params = torch::from_blob(buf_start, {total_numel}, options).to(torch::kCUDA, true);

    // cudaEventRecord(stop1, NULL);
    // cudaEventSynchronize(stop1);
    // float msecTotal1 = 0.0f;
    // cudaEventElapsedTime(&msecTotal1, start1, stop1);
    //     printf("%f ", msecTotal1);

    // cudaEvent_t start1;
    // cudaEvent_t stop1;
    // cudaEventCreate(&start1);

    // cudaEventCreate(&stop1);
    // cudaEventRecord(start1, NULL);

    options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    int param_offset = 0;
    torch::Tensor w0 = torch::from_blob(all_params.data_ptr<float>() + param_offset, {16 * 16}, options).resize_({16, 16});
    param_offset += 16 * 16;

    torch::Tensor b0 = torch::from_blob(all_params.data_ptr<float>() + param_offset, { 16 }, options);
    param_offset += 16;

    torch::Tensor w1 = torch::from_blob(all_params.data_ptr<float>() + param_offset, { 16 * 16}, options).resize_({16, 16});
    param_offset += 16 * 16;

    torch::Tensor b1 = torch::from_blob(all_params.data_ptr<float>() + param_offset, { 16}, options);
    param_offset += 16;

    torch::Tensor w2 = torch::from_blob(all_params.data_ptr<float>() + param_offset, { 16 * 16}, options).resize_({16, 16});
    param_offset += 16 * 16;

    torch::Tensor b2 = torch::from_blob(all_params.data_ptr<float>() + param_offset, { 16}, options);
    param_offset += 16;

    torch::Tensor w3 = torch::from_blob(all_params.data_ptr<float>() + param_offset, { 16 * 3}, options).resize_({16, 3});
    param_offset += 16 * 3;

    torch::Tensor b3 = torch::from_blob(all_params.data_ptr<float>() + param_offset, { 3 }, options);
    param_offset += 3;

        
    cudaEventRecord(stop1, NULL);
    cudaEventSynchronize(stop1);
    float msecTotal1 = 0.0f;
    cudaEventElapsedTime(&msecTotal1, start1, stop1);
        printf("%f\n", msecTotal1);





    return std::make_tuple(
        h, w, n_latent, num_entries, compressed_size, total_symbols, total_blocks,
        compact_tensor_dec_table, tensor_compressed,
        w0, b0,
        w1, b1,
        w2, b2,
        w3, b3,
        all_params
    );
}


PICCodec::PICCodec(int max_codeword_length, int subseq_size=4): hcodec(HuffmanCodec(max_codeword_length, subseq_size)), m_total_blocks(0), m_subseq_size(subseq_size) {
    d_buffer = thrust::device_vector<int>(1024 * 10 / 4);
    // d_compact_dec_table = thrust::device_vector<int>(1024 * 10 / 4);
    d_compressed = thrust::device_vector<int>(1024 * 10 / 4);

    d_rec_buffer = thrust::device_vector<float>(1920 * 1080 * 2);
    d_feature = thrust::device_vector<float>(1920 * 1080 * 7);

    d_block_buf = thrust::device_vector<float>(1920 * 1080 * 2);
    d_symbols = thrust::device_vector<std::uint8_t>(1024);

    d_indices_vec = thrust::device_vector<int>(1024 * 16);

    d_output = thrust::device_vector<float>(1920 * 1080 * 3);

    cudaMalloc(&m_pre_allocate, 1024);

    // hcodec = HuffmanCodec(max_codeword_length);
    auto  options =  torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    // m_w0 = torch::empty({16, 16}, options);
    // m_w1 = torch::empty({16, 16}, options);
    // m_w2 = torch::empty({16, 16}, options);
    m_w3 = torch::empty({16, 16}, options).permute({1, 0});
    // m_b0 = torch::empty({16}, options);
    // m_b1 = torch::empty({16}, options);
    // m_b2 = torch::empty({16}, options);
    m_b3 = torch::empty({16}, options);

    // m_feature = torch::empty({1920 * 1080 }, options).resize_({1920 * 1080 * 7});
}

PICCodec::~PICCodec() {
    cudaFree(m_pre_allocate);
}


void PICCodec::decode_bitstream(py::bytes bytes_data) {
//     cudaEvent_t start1;
// cudaEvent_t stop1;
// float msecTotal1 = 0.0f;

// cudaEventCreate(&start1);
// cudaEventCreate(&stop1);
// cudaEventRecord(start1, NULL);

    char* buffer;
    ssize_t byte_length;
    if (PyBytes_AsStringAndSize(bytes_data.ptr(), &buffer, &byte_length) == -1) {
        throw py::error_already_set(); // 处理 Python 异常
    }


    // std::cout << "decoding " << (int)(*buffer) << '\n';
    int * buf = reinterpret_cast<int *>(buffer);
    m_h = buf[0];
    m_w = buf[1];
    m_n_latent = buf[2];
    int num_entries = buf[3];
    int compressed_size = buf[4];
    m_total_symbols = buf[5];
    
    m_total_blocks = buf[6];

// std::cout << "h " << h << std::endl;

    int * buf_start = buf + 7;

    // auto options = torch::TensorOptions().dtype(torch::kInt32);
    // torch::Tensor compact_tensor_dec_table = torch::from_blob(buf_start, {num_entries}, options);
    int * compressed_buf_start = buf_start + num_entries;


    // int total_numel = 16 * 16 * 3 + 3 * 16 + 16 * 3 + 3;

    int total_numel = 16 * 16 * 4 + 16 * 4;  // 最后一个权重矩阵中为0的部分也预留空间
    int last_w_and_b_padding = 16 * 17 - 16 * 3 - 3;

    if(d_compressed.size() < compressed_size + total_numel) {
        d_compressed.resize(compressed_size+ total_numel);
    }

    // printf("d_compressed/compressed/in_ptr %d\n", d_compressed.size());

    
    if(d_symbols.size() < m_total_symbols) {
        d_symbols.resize(m_total_symbols);
    }


    // cudaEventRecord(stop1, NULL);
    // cudaEventSynchronize(stop1);
    
    // cudaEventElapsedTime(&msecTotal1, start1, stop1);
    //     printf("%f ", msecTotal1);


    cudaMemcpy(
        d_compressed.data().get(), 
        compressed_buf_start, 
        (compressed_size + total_numel - last_w_and_b_padding) * sizeof(int), // 计算包含有意义数据的长度
        cudaMemcpyHostToDevice
    );


    // at::cuda::CUDAStream myStream = at::cuda::getStreamFromPool();
    // // set current CUDA stream from default stream to `myStream` on device 0
    // at::cuda::setCurrentCUDAStream(myStream);
    // sum() on tensor0 uses `myStream` as current CUDA stream
    // tensor0.sum();

    // // get the default CUDA stream on device 0
    // at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
    // // set current CUDA stream back to default CUDA stream on device 0
    // at::cuda::setCurrentCUDAStream(defaultStream);


    float * param_buf = reinterpret_cast<float *>(d_compressed.data().get() + compressed_size);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    int param_offset = 0;

    permute_weight_inplace<<<1, 16 * 16>>>(param_buf + param_offset);
    m_w0 = torch::from_blob(param_buf + param_offset, {16 * 16}, options); //.resize_({16, 16}).permute({1, 0}).contiguous();

    // std::cout << m_w0.reshape({16, 16});
    // m_w0 = torch::from_blob(param_buf + param_offset, {16 * 16}, options).resize_({16, 16}).permute({1, 0}).contiguous();

    // std::cout << m_w0;


    // std::cout << "loading weight "<< m_w0;
    param_offset += 16 * 16;
        
    m_b0 = torch::from_blob(param_buf + param_offset, { 16 }, options);
    param_offset += 16;
    
    permute_weight_inplace<<<1, 16 * 16>>>(param_buf + param_offset);
    m_w1 = torch::from_blob(param_buf + param_offset, { 16 * 16}, options); //.resize_({16, 16}).permute({1, 0}).contiguous();
    param_offset += 16 * 16;
        
    m_b1 = torch::from_blob(param_buf + param_offset, { 16}, options);
    param_offset += 16;
    
    permute_weight_inplace<<<1, 16 * 16>>>(param_buf + param_offset);
    m_w2 = torch::from_blob(param_buf + param_offset, { 16 * 16}, options); //.resize_({16, 16}).permute({1, 0}).contiguous();
    param_offset += 16 * 16;
        
    m_b2 = torch::from_blob(param_buf + param_offset, { 16}, options);
    param_offset += 16;
        
    // m_w3.index_put_({torch::indexing::Ellipsis, m_slice}, torch::from_blob(param_buf + param_offset, { 16 * 3}, options).resize_({16, 3}));
    // param_offset += 16 * 3;
        
    // m_b3.index_put_({m_slice}, torch::from_blob(param_buf + param_offset, { 3 }, options));
    // param_offset += 3;

    // float * src = reinterpret_cast<float*>(compressed_buf_start + (compressed_size + param_offset - 16 * 3 - 3) );

    // for(int i = 0; i < 16 * 16; i++) {
    //     std::cout << *(src + i) << " ";
    //     if(i%16 == 15) {
    //         std::cout << "\n";
    //     }
    // }
    // std::cout << "\n";

    // auto h_w3 = m_w3.to(torch::kCPU);
    // for(int i = 0; i < 16 * 16; i++) {
    //     std::cout << *(h_w3.data_ptr<float>() + i) << " ";
    //     if(i%16 == 15) {
    //         std::cout << "\n";
    //     }
    // }
    // std::cout << "\n";

    // std::cout << m_w3 << "\n";
    // std::cout << m_b3 << "\n";

    // param_offset  = param_offset - 16 * 3 - 3;

    // 手工布局内存元素
    reformat_weight_and_bias_inplace<<<1, 512>>>(param_buf + param_offset);

    m_w3 = torch::from_blob(param_buf + param_offset, { 16 * 16}, options); //.resize_({16, 16}).permute({1, 0}).contiguous();
    param_offset += 16 * 16;

    m_b3 = torch::from_blob(param_buf + param_offset, { 16 }, options);
    param_offset += 16;

    // std::cout << m_w3 << "\n";
    // std::cout << m_b3 << "\n";






    hcodec.huffman_decode_raw(
        reinterpret_cast<cuhd::CUHDCompactCodeTableItem*>(buf_start), 
        num_entries,
        compressed_size, 
        reinterpret_cast<UNIT_TYPE*>(d_compressed.data().get()),
        m_total_symbols,
        d_symbols,
        m_subseq_size
    );

    // for (int i = m_total_symbols- 5; i < m_total_symbols; i++) {
    //     std::cout << (int)d_symbols[i] << " ";
    // }
    // std::cout<<std::endl;

    // torch::Tensor compact_tensor_dec_table = torch::from_blob(buf_start, {num_entries}, options);
    // if(m_feature.numel() != h * w * 7) {
    //     m_feature.resize_({h * w * 7});
    // }
    // m_feature = torch::empty({h * w * 7}, options);
    // d_feature.resize(h * w * 7);

    if(d_feature.size() < m_h * m_w * m_n_latent) {
        d_feature.resize(m_h * m_w * m_n_latent);
    }

    if(d_output.size() < m_h * m_w * 3) {
        d_output.resize(m_h * m_w * 3);
    }

    d_block_buf.resize(64 * m_total_blocks );
    d_rec_buffer.resize(64 * m_total_blocks );
    // thrust::device_vector<int> v(4);
    thrust::fill(d_block_buf.begin(), d_block_buf.end(), 0);

        // return m_symbols;


        // cudaEventCreate(&start1);
        // cudaEventCreate(&stop1);
        // cudaEventRecord(start1, NULL);


    // return m_symbols;
    //     cudaEventRecord(stop1, NULL);
    // cudaEventSynchronize(stop1);
    // msecTotal1 = 0.0f;
    // cudaEventElapsedTime(&msecTotal1, start1, stop1);
    //     printf("%f\n", msecTotal1);
}

struct is_boundary
{
    int b;
    is_boundary(int v): b(v) {}

  __host__ __device__
  bool operator()(const int x)
  {
    return x == b;
  }
};


torch::Tensor PICCodec::reconstruct_feature(int bound_value, int fill_value, float q_scale) {
    // m_symbols[m_symbols.size(0) - 1] = bound_value; // bug fix for gpuhd
    // auto tmp_v = (m_symbols == bound_value);
    // auto indices = tmp_v.nonzero().to(torch::kInt32);

    // d_symbols[m_total_symbols - 1] = bound_value;



    // storage for the nonzero indices
    d_indices_vec.resize(m_total_blocks);
    // thrust::device_vector<std::uint8_t> src(m_symbols.data_ptr<std::uint8_t>(), m_symbols.data_ptr<std::uint8_t>() + m_symbols.size(0));

    // // compute indices of nonzero elements
    // thrust::device_vector::iterator IndexIterator;

    // use make_counting_iterator to define the sequence [0, 8)
    thrust::copy_if(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>( m_total_symbols),  // 由于gpuhd的bug，symbols多了一个占位符，此处丢弃
        d_symbols.begin(),
        d_indices_vec.begin(),
        is_boundary(bound_value)
    );

    // std::cout << "m_total_blocks " << m_total_blocks;
    // std::cout << " m_total_symbols " << m_total_symbols;
    // std::cout << " d_block_buf.size() " << d_block_buf.size();
    // std::cout << " d_indices_vec.size() " << d_indices_vec.size();
    // std::cout << " d_indices_vec[-2] = " << d_indices_vec[d_indices_vec.size()-2];
    // std::cout << " d_indices_vec[-1] = " << d_indices_vec[d_indices_vec.size()-1];
    // std::cout << std::endl;



    // std::cout.flush();
    // sleep(1);


    expand_blocks_zigzag_inplace(d_symbols, d_indices_vec, d_block_buf, fill_value, m_total_blocks, m_total_symbols);
    // auto merged_blocks = m_block_buf.reshape({-1, 8, 8});
    // auto merged_blocks = bd  / q_scale;

    // std::cout << "before idct\n";

    fast_idct_inplace(d_block_buf, m_total_blocks, q_scale);
    // std::cout << "after idct\n";
        // # print(merged_blocks.shape, merged_blocks[0])
    split_reshape_crop_interpolate_inplace(d_block_buf, d_rec_buffer, m_w, m_h, m_n_latent, m_total_blocks);
    crop_and_interpolate_inplace(d_rec_buffer, d_feature, m_w, m_h, m_n_latent);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    // // torch::Tensor compact_tensor_dec_table = torch::from_blob(buf_start, {num_entries}, options);
    // // m_feature = torch::empty({h * w * 7}, options);
    //     // # print('after idct')
    m_output = torch::from_blob(d_feature.data().get(), {m_h * m_w * m_n_latent}, options).resize_({m_n_latent, m_h, m_w});
    // auto x = m_feature.reshape({7, 768, 512}).permute({1, 2, 0});
    return m_output;
    
}


torch::Tensor PICCodec::fused_mlp_render() {
    int thread_x = WARP_SIZE;
    int thread_y = CEIL_DIV(256, WARP_SIZE);
    dim3 threads(thread_x, thread_y);
    dim3 grid(
        CEIL_DIV(m_w, thread_x),
        CEIL_DIV(m_h, thread_y)
    );

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    // torch::Tensor dst_tensor = torch::empty({width, height, out_channel}, options);
    fused_mlp_for_pic<<<grid, threads>>>(
        d_feature.data().get(),
        d_output.data().get(),
        m_w0.data_ptr<float>(), m_b0.data_ptr<float>(),
        m_w1.data_ptr<float>(), m_b1.data_ptr<float>(),
        m_w2.data_ptr<float>(), m_b2.data_ptr<float>(),
        m_w3.data_ptr<float>(), m_b3.data_ptr<float>(),
        m_w, m_h, m_n_latent, 3
    );

    m_output = torch::from_blob(d_output.data().get(), {m_h * m_w * 3}, options).resize_({m_h, m_w, 3});
    return m_output;

}