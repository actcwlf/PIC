#pragma once
#include <torch/extension.h>
#include "Common.h"
#include "BmpUtil.h"

// typedef unsigned char byte;


// std::tuple<byte *, int, ROI> build_img_buffer(const char *pSampleImageFpath);
// int release_img_buffer(byte *ImgSrc);

std::tuple<torch::Tensor, int, int> read_image(const char *s);

torch::Tensor dct_type1(torch::Tensor img_buf, int width, int height);