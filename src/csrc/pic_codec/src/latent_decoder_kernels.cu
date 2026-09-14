#include <cstdint>
#include <torch/extension.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/find.h>
#include <thrust/execution_policy.h>


__device__ constexpr int ZIGZAG_ORDER[] = {0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,35,42,
    49,56,57,50,43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63};

__global__ void expand_symbols_cuda(
    std::uint8_t * symbols,
    int * indices,
    std::uint8_t * output,
    int total_symbols,
    int total_threads
) {

    const std::uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid >= total_threads) {
        return;
    }

    int offset_start = 0;
    if (gid != 0) {
        offset_start = indices[gid -1] + 1;
    }
    int offset_end = indices[gid];

    std::uint8_t * output_symbols = output + gid * 64;

    int offset = 0;
    // printf("total symbols %d \n", total_symbols);
    for(int i = offset_start ; i < offset_end; i++) {

        output_symbols[offset] = symbols[i];
        offset += 1;
    }


}


torch::Tensor expand_blocks(torch::Tensor & symbols, torch::Tensor & indices, int fill_value, int total_blocks) {

    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kUInt8);
    const int thread_per_block = 256;
    int total_cuda_blocks = (indices.size(0) + thread_per_block - 1) / thread_per_block;

    torch::Tensor output = torch::empty({64 * total_blocks}, options.device(device)).fill_(fill_value);
    expand_symbols_cuda<<<total_cuda_blocks, thread_per_block>>>(
        reinterpret_cast<std::uint8_t *>(symbols.data_ptr()),
        indices.data_ptr<int>(), 
        reinterpret_cast<std::uint8_t *>(output.data_ptr()), 
        symbols.size(0),
        indices.size(0)
    );

    return output;
}


__global__ void expand_symbols_zigzag_cuda(
    std::uint8_t * symbols,
    int * indices,
    float * output,
    int total_symbols,
    int total_blocks,
    float fill_value
    // int max_bound
) {
    const std::uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid >= total_blocks) {
        return;
    }

    int offset_start = 0;
    if (gid != 0) {
        offset_start = indices[gid -1] + 1;
    }
    int offset_end = indices[gid];

    int max_bound = static_cast<int>(symbols[offset_end]); // 直接从符号流中获取max_bound的值

    float  * output_symbols = output + gid * 64;

    int offset = 0;
    // printf("total symbols %d \n", total_symbols);
    for(int i = offset_start ; i < offset_end; i++) {

        // printf("%d zig_off %d  symbol[%d] %d\n", gid, offset, i, (int)symbols[i]);

        int s = static_cast<int>(symbols[i]);
        //if(s == max_bound) {
            //printf("total_block %d\n", total_blocks);
            //printf("indices[%d]=%d indices[%d]=%d indices[%d]=%d indices[%d]=%d\n",
                    //gid-1, indices[gid -1], gid, indices[gid], gid+1, indices[gid+1]);
            //printf("gid %d offset %d ZIGZAG_ORDER %p i %d s %d max_bound %d offset_start %d offset_end %d\n",
                    //gid,   offset,   ZIGZAG_ORDER,   i,   s,   max_bound, offset_start, offset_end);
        //}
        if(s >= max_bound - 17 + 1) {
            //if(s == max_bound) {
            //    while(offset < 64) {
            //        output_symbols[ZIGZAG_ORDER[offset]] = 0;
            //    }
            //} else {
                for(int j = 0; j < s + 17 - max_bound; j++) {
                    //if(offset >63) {
                        //for(int k = offset_start ; k <= offset_end; k++) {
                            //printf("gid %d symbols[%d] %d\n", gid, k, (int)symbols[k]);
                        //}
                         //printf("gid %d offset %d ZIGZAG_ORDER %p i %d j %d s %d max_bound %d\n",
                                 //gid,   offset,   ZIGZAG_ORDER,   i,   j,   s,   max_bound);
                    //}
                    output_symbols[ZIGZAG_ORDER[offset]] = 0;
                    offset++;

                }
            //}
        } else {
                //if(offset >63) {
                    //for(int k = offset_start ; k <= offset_end; k++) {
                        //printf("gid %d symbols[%d] %d\n", gid, k, (int)symbols[k]);
                    //}
                    //printf("gid %d offset %d ZIGZAG_ORDER %p %p i %d s %d max_bound %d\n", gid, offset,  ZIGZAG_ORDER, i, s, max_bound);
                //}
            output_symbols[ZIGZAG_ORDER[offset]] = static_cast<float>(symbols[i]) - fill_value;
            offset += 1;
        }


    }


}


torch::Tensor expand_blocks_zigzag(torch::Tensor & symbols, torch::Tensor & indices, int fill_value, int total_blocks) {

    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kFloat32);
    const int thread_per_block = 256;
    int total_cuda_blocks = (indices.size(0) + thread_per_block - 1) / thread_per_block;

    torch::Tensor output = torch::zeros({64 * total_blocks}, options.device(device));

    // std::cout << " output.shape " << output.size(0);
    // std::cout << " total_symbols " << symbols.size(0);
    // std::cout << " indices.size(0) " << indices.size(0);
    // std::cout <<std::endl;
    // sleep(1);
    expand_symbols_zigzag_cuda<<<total_cuda_blocks, thread_per_block>>>(
        reinterpret_cast<std::uint8_t *>(symbols.data_ptr()),
        indices.data_ptr<int>(), 
        output.data_ptr<float>(), 
        symbols.size(0),
        indices.size(0),
        fill_value

    );

    return output;
}


void expand_blocks_zigzag_inplace(
    thrust::device_vector<std::uint8_t>  & symbols,  
    thrust::device_vector<int>  & indices, 
    thrust::device_vector<float> & output, int fill_value, int total_blocks, int total_symbols) {

    // torch::Device device(torch::kCUDA);
    // torch::TensorOptions options(torch::kFloat32);
    const int thread_per_block = 256;
    int total_cuda_blocks = (total_blocks + thread_per_block - 1) / thread_per_block;

    // torch::Tensor output = torch::zeros({64 * total_blocks}, options.device(device));
    expand_symbols_zigzag_cuda<<<total_cuda_blocks, thread_per_block>>>(
        symbols.data().get(),
        indices.data().get(), 
        output.data().get(), 
        total_symbols,
        total_blocks,
        fill_value
        // max_bound
    );

    // return output;
}


__global__ void zigzag_and_trim_blocks_kernel(
    int * symbols,
    int * trimmed_symbols,
    int * patch_length,
    int total_symbols,
    int total_threads,
    int max_bound
) {
    const std::uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid >= total_threads) {
        return;
    }
    int * src_block_start = symbols + gid * 64;
    int * dst_block_start = trimmed_symbols + gid * 65; // 尾部可能包含终止符，因此块大小为65
    #pragma unroll
    for(int i = 0; i < 64; i ++) {
        dst_block_start[i] = src_block_start[ZIGZAG_ORDER[i]];
    }
    int i = 63;
    while(( i>= 0 )&&( dst_block_start[i] == 0)) {
        i--;
    }
    int max_bnound_pos = i + 1;
    int zero_count = 0;
    int cursor = 0;
    // 注意前面的循环已经确保最后一个符号一定不是0
    // 对0执行rlc
    for(int i = 0; i < max_bnound_pos; i++) {
        int s = dst_block_start[i];

        if (s != 0) {
            if (zero_count == 0) {
                dst_block_start[cursor] = s;
                cursor++;
            } else {
                int rlc_code = max_bound - 17 + zero_count;
                dst_block_start[cursor] = rlc_code;
                cursor++;
                dst_block_start[cursor] = s;
                cursor++;
                zero_count = 0;
            }
            
        } else {
            zero_count++;
            if(zero_count == 16) {
                int rlc_code = max_bound - 17 + zero_count;
                dst_block_start[cursor] = rlc_code;
                cursor++;
                zero_count = 0;
            }
        }
    }
    dst_block_start[cursor] = max_bound;
    patch_length[gid] = cursor + 1;
}


__global__ void reduce_trimmed_blocks(
    int * symbols,
    const int * cum_patch_length,
    int * output,
    int total_threads
) {

    const std::uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid >= total_threads) {
        return;
    }

    
    int offset_start = 0;
    if (gid != 0) {
        offset_start = cum_patch_length[gid -1];
    }
    int block_patch_length = cum_patch_length[gid] - offset_start;
    // int offset_end = indices[gid] + 1;

    int * input_block_start = symbols + gid * 65;
    int * output_start = output + offset_start;

    // int offset = 0;
    // printf("total symbols %d \n", total_symbols);
    for(int i = 0 ; i < block_patch_length ; i++) {

        output_start[i] = input_block_start[i];
    }

    // indices[gid] = offset_end - offset_start;


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


torch::Tensor zigzag_trim_and_reduce_blocks(torch::Tensor & blocks, int bound_value) {
    int total_block_num = blocks.size(0);
    thrust::device_vector<int> buf1(total_block_num * 65);
    thrust::device_vector<int> buf2(total_block_num * 65);
    thrust::device_vector<int> patch_length(total_block_num);

    int thread_per_block = 256;
    int grid = (total_block_num + thread_per_block- 1) / thread_per_block;

    zigzag_and_trim_blocks_kernel<<<grid, thread_per_block>>>(
        blocks.data_ptr<int>(),
        buf1.data().get(),
        patch_length.data().get(),
        blocks.numel(),
        total_block_num,
        bound_value
    );

    thrust::inclusive_scan(patch_length.begin(), patch_length.end(), patch_length.begin(), thrust::plus<int>());
    // thrust::copy_if(
    //     thrust::counting_iterator<int>(0),
    //     thrust::counting_iterator<int>(total_block_num * 65),
    //     buf1.begin(),
    //     indices.begin(),
    //     is_boundary(bound_value)
    // );
    // for(auto v: patch_length) {
    //     std::cout << v << ' ';
    // }
    // std::cout << '\n';
    reduce_trimmed_blocks<<<grid, thread_per_block>>>(
        buf1.data().get(),
        patch_length.data().get(),
        buf2.data().get(),
        total_block_num
    );

    // for(auto v: patch_length) {
    //     std::cout << v << ' ';
    // }
    // std::cout << '\n';

    // int total_symbols = thrust::reduce(indices.begin(), indices.end(), 0, thrust::plus<int>());

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    return torch::from_blob(buf2.data().get(), {patch_length.back()}, options).clone();


}