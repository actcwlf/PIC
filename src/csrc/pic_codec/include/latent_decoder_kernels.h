#pragma once
#include <torch/extension.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/find.h>
#include <thrust/execution_policy.h>


torch::Tensor expand_blocks(torch::Tensor & symbols, torch::Tensor & indices, int fill_value, int total_blocks);
torch::Tensor expand_blocks_zigzag(torch::Tensor & symbols, torch::Tensor & indices, int fill_value, int total_blocks);

void expand_blocks_zigzag_inplace(
    thrust::device_vector<std::uint8_t> & symbols, 
    thrust::device_vector<int>   & indices, 
    thrust::device_vector<float> & output, int fill_value, int total_blocks, int total_symbols);

torch::Tensor zigzag_trim_and_reduce_blocks(torch::Tensor & blocks, int bound_value);