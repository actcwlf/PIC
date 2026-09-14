#pragma once
#include <torch/extension.h>


torch::Tensor bilinear_interpolate(torch::Tensor x, int target_height, int target_width);