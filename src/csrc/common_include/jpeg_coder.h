#pragma once

#include <sys/types.h>
#include <malloc.h>
#include <memory.h>
#include <iostream>
#include <exception>
#include <nvjpeg.h>
#include <torch/extension.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/find.h>
#include <thrust/execution_policy.h>
 #include <torch/extension.h>

class JpegCoderError: public std::runtime_error{
protected:
    int _code;
public:
    JpegCoderError(int code, const std::string& str):std::runtime_error(str){
        this->_code = code;
    }
    JpegCoderError(int code, const char* str):std::runtime_error(str){
        this->_code = code;
    }
    int code(){
        return this->_code;
    }
};

// typedef enum
// {
//     JPEGCODER_CSS_444 = 0,
//     JPEGCODER_CSS_422 = 1,
//     JPEGCODER_CSS_420 = 2,
//     JPEGCODER_CSS_440 = 3,
//     JPEGCODER_CSS_411 = 4,
//     JPEGCODER_CSS_410 = 5,
//     JPEGCODER_CSS_GRAY = 6,
//     JPEGCODER_CSS_UNKNOWN = -1
// } JpegCoderChromaSubsampling;

// typedef enum{
//     JPEGCODER_PIXFMT_RGB         = 3,
//     JPEGCODER_PIXFMT_BGR         = 4, 
//     JPEGCODER_PIXFMT_RGBI        = 5, 
//     JPEGCODER_PIXFMT_BGRI        = 6,
// }JpegCoderColorFormat;

class JpegCoderImage{
public:
    void* img;

    nvjpegChromaSubsampling_t subsampling;
    size_t height;
    size_t width;
    short nChannel;
    
    JpegCoderImage(size_t width, size_t height, short nChannel, nvjpegChromaSubsampling_t subsampling,   thrust::device_vector<std::uint8_t> &buf);
    ~JpegCoderImage();
    void fill(const unsigned char* data);
    unsigned char* buffer();
};

// class JpegCoderBytes{
// public:
//     size_t size;
//     unsigned char* data;
//     JpegCoderBytes(size_t size){
//         this->data = (unsigned char*)malloc(size);
//         this->size = size;
//     }

//     JpegCoderBytes(unsigned char* data, size_t size){
//         this->data = data;
//         this->size = size;
//     }

//     ~JpegCoderBytes(){
//         if(this->data!=nullptr){
//             free(this->data);
//         }
//     }
// };

inline const char* error_string(nvjpegStatus_t code) {
    switch(code) {
      case NVJPEG_STATUS_SUCCESS: return "success";
      case NVJPEG_STATUS_NOT_INITIALIZED: return "not initialized";
      case NVJPEG_STATUS_INVALID_PARAMETER: return "invalid parameter";
      case NVJPEG_STATUS_BAD_JPEG: return "bad jpeg";
      case NVJPEG_STATUS_JPEG_NOT_SUPPORTED: return "not supported";
      case NVJPEG_STATUS_ALLOCATOR_FAILURE: return "allocation failed";
      case NVJPEG_STATUS_EXECUTION_FAILED: return "execution failed";
      case NVJPEG_STATUS_ARCH_MISMATCH: return "arch mismatch";
      case NVJPEG_STATUS_INTERNAL_ERROR: return "internal error";
      default: return "unknown";
    }
}



class JpegException : public std::exception {
    nvjpegStatus_t code;
    std::string context;
  
    public:
      JpegException(std::string const& _context, nvjpegStatus_t _code) :
        code(_code), context(_context)
      { }
          
      const char * what () const throw () {
        std::stringstream ss;
        ss << context << ", nvjpeg error " << code << ": " << error_string(code);
        return ss.str().c_str();
  
      }
  };

  
class JpegCoder{
protected:
//     static void* _global_context;
//     void* _local_context;
  torch::Tensor t;
public:
nvjpegHandle_t nv_handle;
nvjpegJpegState_t nv_statue;
nvjpegEncoderState_t enc_state;
torch::Tensor m_output;
thrust::device_vector<std::uint8_t> m_buf;
    JpegCoder();
    ~JpegCoder();
    // void ensureThread(long threadIdent);
    nvjpegEncoderParams_t createParams(int quality, nvjpegChromaSubsampling_t subsampling, cudaStream_t stream);
    nvjpegImage_t createImage(torch::Tensor const& data, nvjpegInputFormat_t input_format, size_t &width, size_t &height);
    torch::Tensor  decode(py::bytes bytes_data);
    py::bytes encode(torch::Tensor const& data, int quality, nvjpegInputFormat_t input_format, nvjpegChromaSubsampling_t subsampling);
    // static void cleanUpEnv();
};