#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/find.h>
#include <thrust/execution_policy.h>
#include <torch/extension.h>


torch::Tensor split_and_reshape(torch::Tensor merged_blocks, int width, int height, int n_latent);
torch::Tensor crop_and_interpolate(torch::Tensor & reshaped_buf, int width, int height, int n_latent);

void split_reshape_crop_interpolate_inplace(thrust::device_vector<float>  & merged_blocks, thrust::device_vector<float> & dst_buf, int width, int height, int n_latent, int total_block);
void crop_and_interpolate_inplace(thrust::device_vector<float> & reshaped_buf, thrust::device_vector<float> &  result_buf, int width, int height, int n_latent);
