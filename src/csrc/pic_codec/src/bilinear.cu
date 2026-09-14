#include <torch/extension.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#define N 1024


#define CHECK_CUDA_ERROR(val)   checkCudaErrors(val)


// texture object is a kernel argument
__global__ void bilinear_kernel(float * output, cudaTextureObject_t texObj, 
  float u00, float v00, float base_u, float  base_v,
  int target_width, int target_height
) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= target_width) {
    return;
  }

  if (y >= target_height ) {
    return;
  }


  float u = x / (float)(target_width - 1) * base_u + u00;
  float v = y / (float)(target_height - 1) * base_v + v00;

  // float u = x / (float)target_width  + u00;
  // float v = y / (float)target_height + v00;

  output[y * target_width + x] = tex2D<float>(texObj, u, v);

  // printf("x %d y %d u %f v %f val %f\n", x, y, u, v, tex2D<float>(texObj, u, v));

}

// void call_bilinear_kernel(cudaTextureObject_t tex, float * dst_buffer, int target_height, int target_width) {
//   dim3 block(128,1,1);
//   dim3 grid(target_height/block.x,1,1);
//   bilinear_kernel <<<grid, block>>>(tex, dst_buffer);
// }


__global__ void accessTextureKernel(float *out, cudaTextureObject_t texObj, int width, int height)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    float u = idx + 0.5f;
    float v = idy + 0.5f;
    out[idy * width + idx] = tex2D<float>(texObj, u, v);

    
}

// Simple transformation kernel
__global__ void transformKernel(float* output,
  cudaTextureObject_t texObj,
  int width, int height,
  float theta)
{
// Calculate normalized texture coordinates
unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

float u = x / (float)width;
float v = y / (float)height;

// Transform coordinates
u -= 0.5f;
v -= 0.5f;
float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

// Read from texture and write to global memory
output[y * width + x] = tex2D<float>(texObj, tu, tv);
}


torch::Tensor bilinear_interpolate(torch::Tensor x, int target_height, int target_width) {

  int height = x.size(0);
  int width = x.size(1);
  float angle = 0.5;

  // Allocate and set some host data
  // float *h_data = (float *)std::malloc(sizeof(float) * target_height * target_width);
  // for (int i = 0; i < height * width; ++i)
  //     printf("%f\n", x.data_ptr<float>()[i]);

  // Allocate CUDA array in device memory
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaArray_t cuArray;
  cudaMallocArray(&cuArray, &channelDesc, width, height);


  // float *xh_data = (float *)std::malloc(sizeof(float) * height * width);
  // for (int i = 0; i < height * width; ++i) {
  //   xh_data[i] = x.data_ptr<float>()[i];
  // }


  // Set pitch of the source (the width in memory in bytes of the 2D array pointed
  // to by src, including padding), we dont have any padding
  size_t spitch = width * sizeof(float);
  // Copy data located at address h_data in host memory to device memory
  cudaMemcpy2DToArray(cuArray, 0, 0, x.data_ptr<float>(), spitch, width * sizeof(float),
                      height, cudaMemcpyDeviceToDevice);

  // Specify texture
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  // Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeBorder;
  texDesc.addressMode[1] = cudaAddressModeBorder;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 1;

  // Create texture object
  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

  // Allocate result of transformation in device memory
  float *output;
  cudaMalloc(&output, target_width * target_height * sizeof(float));

  // Invoke kernel
  dim3 threadsperBlock(16, 16);
  dim3 numBlocks((target_width + threadsperBlock.x - 1) / threadsperBlock.x,
                  (target_height + threadsperBlock.y - 1) / threadsperBlock.y);
  
  float u00 = 0.5 / (float)width;
  float v00 = 0.5 / (float)height ;
  float u11 = ((float)width - 0.5) / (float)width ;
  float v11 = ((float)height - 0.5) / (float)height;
  
  float base_u = u11 - u00;
  float base_v = v11 - v00;
  bilinear_kernel<<<numBlocks, threadsperBlock>>>(output, texObj, u00, v00, base_u, base_v, target_width, target_height);
  // // Copy data from device back to host
  // cudaMemcpy(h_data, output, target_width * target_height * sizeof(float),
  //             cudaMemcpyDeviceToHost);

  // Destroy texture object
  cudaDestroyTextureObject(texObj);
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  torch::Tensor bilinear_tensor = torch::from_blob(output, {target_width * target_height}, options).clone();
  // Free device memory
  cudaFreeArray(cuArray);
  cudaFree(output);

  // Free host memory
  // free(h_data);
  // free(xh_data);

  return bilinear_tensor;

}