#include <cuda.h>
#include <torch/extension.h>
 #include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/find.h>
#include <thrust/execution_policy.h>


#define VEC_SIZE 8
#define BLOCK_SIZE 8
#define BLOCK_SIZE2 64

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))


__global__ void split_and_reshape_kernel(float *src, float *dst, int width, int height, int n_latent, int total_vec_block) {
    // 每个线程处理长度为8的一个向量
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_vec_block) {
        return;
    }


    int cum_blocks = 0;   // 记录当前累计的block数量，注意这里是8x8的block
    int cum_vec_blocks = 0; // 记录当前累计的vec数量，注意这里是8的vec
    int current_blocks = 0;   // 记录当前latent中的block数量，注意这里是8x8的block
    int current_vec_blocks = 0;  // 记录当前latent中的vec数量，注意这里是8x8的block
    int current_padded_width = 0;  // 按8向上取整的宽度
    int current_padded_height = 0; // 按8向上取整的高

    for (int i = 0; i < n_latent; i++) {
        
        current_padded_width =  (width + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        current_padded_height = (height + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        current_blocks = current_padded_width * current_padded_height / BLOCK_SIZE2;

        cum_blocks += current_blocks;
        cum_vec_blocks = cum_blocks * BLOCK_SIZE; // 每个block中有BLOCK_SIZE个vec
        if(gid < cum_vec_blocks) {
            break;
        }
        width = (width + 1) >> 1;
        height = (height + 1) >> 1;
    }
    float *src_start = src + gid * 8; // 当前线程处理的vec在src中的地址
    float *latent_start = dst + (cum_blocks - current_blocks) * BLOCK_SIZE2;  // 当前latent在dst中的起始地址


    // 确认vec在latent_start中的地址

    int vec_in_latent = gid - (cum_blocks - current_blocks) * BLOCK_SIZE;  // vec在当前latent中的id（未重排）

    // if(vec_in_latent == 0) {
    //     printf("padded w %d padded h %d\n", current_padded_width, current_padded_height);
    // }

    int block_per_row = current_padded_width / BLOCK_SIZE;
    int vec_block_id = vec_in_latent / 8; // vec所在block的id
    int macro_row = vec_block_id / block_per_row;
    int row = macro_row * 8 + gid % 8;
    int col = vec_block_id % block_per_row * 8;
    float * dst_start = latent_start + row * current_padded_width + col ;

    // printf("gid %d cum_blocks %d current_blocks %d vec_in_latent %d row %d col %d\n", 
    //         gid,   cum_blocks,   current_blocks,   vec_in_latent,   row,   col);
    #pragma unroll
    for(int i = 0; i < 8; i++) {
        dst_start[i] = src_start[i];
    }
}



__device__ void  pad_for_patch_embedding(int w, int h, int & pu, int & pr, int & pb, int &pl) {

    int patch_size = 8;
    int n_h = CEIL_DIV(h, patch_size);
    int padded_h = n_h * patch_size;

    int n_w = CEIL_DIV(w, patch_size);

    int padded_w = n_w * patch_size;

    // padding = (
    //     (padded_w - w) // 2,
    //     padded_w - w - (padded_w - w) // 2,
    //     (padded_h - h) // 2,
    //     padded_h - h - (padded_h - h) // 2
    // )

    // padded_x = F.pad(x, padding)

    pl = (padded_w - w) / 2;                // left padding
    // pr = padded_w - w - (padded_w - w) / 2; // right padding
    pu = (padded_h - h) / 2;                // upper padding
    // pb = padded_h - h - (padded_h - h) / 2; // bottom padding
}


#define PROC_BLOCK_SIZE 1
__global__ void crop_and_bilinear_interpolate(float *src, float *dst, int width, int height, int n_latent, const int * meta_info) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int current_latent = blockIdx.z;



    // int cum_blocks = 0;   // 记录当前累计的block数量，注意这里是8x8的block
    // // int cum_vec_blocks = 0; // 记录当前累计的vec数量，注意这里是8的vec
    // int current_blocks = 0;   // 记录当前latent中的block数量，注意这里是8x8的block
    // // int current_vec_blocks = 0;  // 记录当前latent中的vec数量，注意这里是8x8的block
    // int current_padded_width = 0;  // 按8向上取整的宽度
    // int current_padded_height = 0; // 按8向上取整的高
    // int current_width = width;
    // int current_height = height;
    int output_width = width;
    int output_height = height;

    if( x >= width || y >= height) {
        return;
    }

    // for (int i = 0; i <= current_latent; i++) {
        
    //     current_width = width;
    //     current_height = height;

    //     current_padded_width =  (width + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    //     current_padded_height = (height + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    //     current_blocks = current_padded_width * current_padded_height / BLOCK_SIZE2;

    //     cum_blocks += current_blocks;
        
    //     width = width >> 1;
    //     height = height >> 1;
    // }

    int current_width = meta_info[current_latent * 5 ];
    int current_height = meta_info[current_latent * 5 +1];
    int current_padded_width = meta_info[current_latent * 5 +2] ;
    int current_padded_height = meta_info[current_latent * 5 +3] ;
    int offset = meta_info[current_latent * 5 +4];
    

    int pu, pl; // 上右下左的padding

    pl = (current_padded_width - current_width) / 2;                // left padding
    // pr = padded_w - w - (padded_w - w) / 2; // right padding
    pu = (current_padded_height - current_height) / 2;                // upper padding

    
    // pad_for_patch_embedding(current_width, current_height, pu,  pr,  pb, pl);

    if(current_latent == 0) {
        float * src_start = src + pu * current_padded_width;
        // int x = base_x * PROC_BLOCK_SIZE;
        // while (x < output_width && x < (base_x + 1) * PROC_BLOCK_SIZE ) {
            *(dst + y * output_width + x ) = *(src_start + y * current_padded_width + pl + x);
            // x += 1;
        // }

        // if((x == 0)&&(y==0)) {
        //     printf("latent %d val %f\n", current_latent, *src_start);
        // }
        
        return;
    }

    float *src_start = src + offset * 64; // 当前latent在src中的起始地址


    // if((x == 0)&&(y==0)) {
    //     printf("latent %d val %f\n", current_latent, *src_start);
    // }
    float *dst_start = dst + current_latent * output_width *  output_height;

    float cell_size_x = (float)(output_width -1) / (current_width - 1);
    float cell_size_y = (float)(output_height -1) / (current_height - 1);
    // float x_ratio = (float)x / output_width;
    // float y_ratio = (float)y / output_height;

    // int base_x = x * 8;
    // int x = base_x * PROC_BLOCK_SIZE;
    // while (x < output_width && x < (base_x + 1) * PROC_BLOCK_SIZE ) {

    
        // 无padding情况下的cell id
        int cell_id_x = min((int)(x / cell_size_x), current_width - 2);
        int cell_id_y = min((int)(y  / cell_size_y), current_height - 2);
        float weight_x = x / cell_size_x - cell_id_x;
        float weight_y = y / cell_size_y - cell_id_y;

        // 考虑padding
        cell_id_x += pl;
        cell_id_y += pu;

        // 计算四个近邻元素的值
        // float vxx = *(src_start + (cell_id_y * current_padded_width) + cell_id_x);
        // float vxy = *(src_start + (cell_id_y * current_padded_width) + cell_id_x + 1);
        // float vyx = *(src_start + ((cell_id_y + 1) * current_padded_width) + cell_id_x);
        // float vyy = *(src_start + ((cell_id_y + 1) * current_padded_width) + cell_id_x + 1);

        float *pvxx = src_start + (cell_id_y * current_padded_width) + cell_id_x;
        float vxx = *pvxx;
        float vxy = *(pvxx + 1);
        float vyx = *(pvxx + current_padded_width);
        float vyy = *(pvxx + current_padded_width + 1);



        float val = (1 - weight_x) * (1 - weight_y) * vxx;
        val += (weight_x) * (1 - weight_y) * vxy;
        val += (1 - weight_x) * (weight_y) * vyx;
        val += (weight_x) * (weight_y) * vyy;


        float * dst_pixel = dst_start + y * output_width + x;

    // if ((x == 0) && (y == 511) && (current_latent == 3)) {
        // printf("base_x %d x %d y %d current_latent %d pu %d pl %d cell_id_x %d cell_id_y %d weight_x %f weight_y %f val %f vxx %f vxy %f vyx %f vyy %f\n", 
        //         base_x,   x,   y,   current_latent,   pu,   pl,   cell_id_x,   cell_id_y,   weight_x,   weight_y,  val,   vxx,   vxy,   vyx,   vyy);

    //     printf("current_width %d current_height %d\n",
    //         current_width, current_height
    //     );
    // }

        // if(isnan(val)) {
        //     printf("x %d y %d l %d val %f cell_id_x %d cell_id_y %d weight_x %f weight_y %f vxx %f, vxy %f vyx %f vyy %f\n", 
        //             x, y, current_latent, val, cell_id_x, cell_id_y, weight_x, weight_y, vxx, vxy, vyx, vyy);
        // }
        (*dst_pixel) = val;
    //     x += 1;
    // }

}




torch::Tensor split_and_reshape(
    torch::Tensor merged_blocks,
    int width,
    int height,
    int n_latent) {
    int thread_per_block = 256;
    int total_block = merged_blocks.size(0);
    int blocks = (total_block * 8 + thread_per_block - 1) / thread_per_block;

    // thrust::host_vector<int> h_meta_info(5 * n_latent);
    // int cum_blocks = 0;   // 记录当前累计的block数量，注意这里是8x8的block
    // // int cum_vec_blocks = 0; // 记录当前累计的vec数量，注意这里是8的vec
    // int current_blocks = 0;   // 记录当前latent中的block数量，注意这里是8x8的block
    // // int current_vec_blocks = 0;  // 记录当前latent中的vec数量，注意这里是8x8的block
    // int current_padded_width = 0;  // 按8向上取整的宽度
    // int current_padded_height = 0; // 按8向上取整的高
    // int current_width = width;
    // int current_height = height;
    // int local_width = width;
    // int local_height = height;



    // for (int i = 0; i < n_latent; i++) {
        
    //     current_width = local_width;
    //     current_height = local_height;

    //     current_padded_width =  CEIL_DIV(local_width, BLOCK_SIZE) * BLOCK_SIZE;
    //     current_padded_height = CEIL_DIV(local_height, BLOCK_SIZE) * BLOCK_SIZE;
    //     current_blocks = current_padded_width * current_padded_height / BLOCK_SIZE2;

    //     cum_blocks += current_blocks;
        
    //     local_width = local_width >> 1;
    //     local_height = local_height >> 1;

    //     h_meta_info[i * 5] = current_width;
    //     h_meta_info[i * 5+1] = current_height;
    //     h_meta_info[i * 5+2] = current_padded_width;
    //     h_meta_info[i * 5+3] = current_padded_height;
    //     h_meta_info[i * 5+4] = cum_blocks - current_blocks;


    //     // meta_info[i+5] = current_height;
    // }

    // thrust::device_vector<int> d_meta_info = h_meta_info;



    // printf("total blocks %d blocks %d\n", total_block, blocks);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor reshaped_buf = torch::empty({merged_blocks.numel()}, options);
    split_and_reshape_kernel<<<blocks, thread_per_block>>>(
        merged_blocks.data_ptr<float>(),
        reshaped_buf.data_ptr<float>(),
        width, height, n_latent, total_block * 8
    );
    return reshaped_buf;

}


void split_reshape_crop_interpolate_inplace(thrust::device_vector<float> & merged_blocks, thrust::device_vector<float> & dst_buf, int width, int height, int n_latent, int total_block) {
    int thread_per_block = 256;
    // int total_block = merged_blocks.size(0);
    int blocks = (total_block * 8 + thread_per_block - 1) / thread_per_block;

    // printf("total blocks %d blocks %d\n", total_block, blocks);
    // auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    // torch::Tensor reshaped_buf = torch::empty({merged_blocks.numel()}, options);

    if(dst_buf.size() < merged_blocks.size()) {
        dst_buf.resize(merged_blocks.size());
    }
    split_and_reshape_kernel<<<blocks, thread_per_block>>>(
        merged_blocks.data().get(),
        dst_buf.data().get(),
        width, height, n_latent, total_block * 8
    );
    // return reshaped_buf;

}


torch::Tensor crop_and_interpolate(torch::Tensor & reshaped_buf, int width, int height, int n_latent) {
    int thread_per_block = 256;
    // int total_block = merged_blocks.size(0);
    // int blocks = (total_block * 8 + thread_per_block - 1) / thread_per_block;

    // printf("total blocks %d blocks %d\n", total_block, blocks);
    // width, height, padded_width, padded_height, src_offset
    thrust::host_vector<int> h_meta_info(5 * n_latent);
    int cum_blocks = 0;   // 记录当前累计的block数量，注意这里是8x8的block
    // int cum_vec_blocks = 0; // 记录当前累计的vec数量，注意这里是8的vec
    int current_blocks = 0;   // 记录当前latent中的block数量，注意这里是8x8的block
    // int current_vec_blocks = 0;  // 记录当前latent中的vec数量，注意这里是8x8的block
    int current_padded_width = 0;  // 按8向上取整的宽度
    int current_padded_height = 0; // 按8向上取整的高
    int current_width = width;
    int current_height = height;
    int local_width = width;
    int local_height = height;



    for (int i = 0; i < n_latent; i++) {
        
        current_width = local_width;
        current_height = local_height;

        current_padded_width =  (local_width + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        current_padded_height = (local_height + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        current_blocks = current_padded_width * current_padded_height / BLOCK_SIZE2;

        cum_blocks += current_blocks;
        
        local_width = (local_width +1) >> 1;
        local_height = (local_height +1) >> 1;

        h_meta_info[i * 5] = current_width;
        h_meta_info[i * 5+1] = current_height;
        h_meta_info[i * 5+2] = current_padded_width;
        h_meta_info[i * 5+3] = current_padded_height;
        h_meta_info[i * 5+4] = cum_blocks - current_blocks;


        // meta_info[i+5] = current_height;
    }

    // for (int elem : h_meta_info) {
    //     std::cout << elem << " ";
    // }
    

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor result_buf = torch::empty({width * height * n_latent}, options);

    dim3 threads(16, 16, 1);
    dim3 grid(CEIL_DIV(width, 16), CEIL_DIV(height, 16), n_latent);

    thrust::device_vector<int> d_meta_info = h_meta_info;

    crop_and_bilinear_interpolate<<<grid, threads>>>(
        reshaped_buf.data_ptr<float>(),
        result_buf.data_ptr<float>(),
        width, height, n_latent, d_meta_info.data().get()
    );
    return result_buf;

}


void crop_and_interpolate_inplace(thrust::device_vector<float> & reshaped_buf, thrust::device_vector<float>  & result_buf, int width, int height, int n_latent) {
    int thread_per_block = 256;
    // int total_block = merged_blocks.size(0);
    // int blocks = (total_block * 8 + thread_per_block - 1) / thread_per_block;

    // printf("total blocks %d blocks %d\n", total_block, blocks);
    // width, height, padded_width, padded_height, src_offset
    thrust::host_vector<int> h_meta_info(5 * n_latent);
    int cum_blocks = 0;   // 记录当前累计的block数量，注意这里是8x8的block
    // int cum_vec_blocks = 0; // 记录当前累计的vec数量，注意这里是8的vec
    int current_blocks = 0;   // 记录当前latent中的block数量，注意这里是8x8的block
    // int current_vec_blocks = 0;  // 记录当前latent中的vec数量，注意这里是8x8的block
    int current_padded_width = 0;  // 按8向上取整的宽度
    int current_padded_height = 0; // 按8向上取整的高
    int current_width = width;
    int current_height = height;
    int local_width = width;
    int local_height = height;



    for (int i = 0; i < n_latent; i++) {
        
        current_width = local_width;
        current_height = local_height;

        current_padded_width =  (local_width + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        current_padded_height = (local_height + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        current_blocks = current_padded_width * current_padded_height / BLOCK_SIZE2;

        cum_blocks += current_blocks;
        
        local_width = (local_width +1) >> 1;
        local_height = (local_height+1) >> 1;

        h_meta_info[i * 5] = current_width;
        h_meta_info[i * 5+1] = current_height;
        h_meta_info[i * 5+2] = current_padded_width;
        h_meta_info[i * 5+3] = current_padded_height;
        h_meta_info[i * 5+4] = cum_blocks - current_blocks;


        // meta_info[i+5] = current_height;
    }

    // for (int elem : h_meta_info) {
    //     std::cout << elem << " ";
    // }
    

    // auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    // torch::Tensor result_buf = torch::empty({width * height * n_latent}, options);


    dim3 threads(16, 16, 1);
    dim3 grid(CEIL_DIV(width, 16), CEIL_DIV(height, 16), n_latent);

    thrust::device_vector<int> d_meta_info = h_meta_info;

    crop_and_bilinear_interpolate<<<grid, threads>>>(
        reshaped_buf.data().get(),
        result_buf.data().get(),
        width, height, n_latent, d_meta_info.data().get()
    );
    // return result_buf;

}