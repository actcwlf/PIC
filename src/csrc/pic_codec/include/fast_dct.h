#pragma once

#include "Common.h"
#include <torch/extension.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/find.h>
#include <thrust/execution_policy.h>


void fast_idct(torch::Tensor & input, float scale);

void fast_idct_inplace(thrust::device_vector<float> & input, int total_blocks, float scale);