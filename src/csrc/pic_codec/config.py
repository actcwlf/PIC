
import os.path as osp
import os

module_path = os.path.dirname(os.path.abspath(__file__))
include_dirs = [osp.join(module_path, "include")]
include_dirs += [osp.join(module_path, "llhuff/include")]

NAME = '_pic_codec'
CXX_FLAGS = []
NVCC_FLAGS = ['-g', '-G']
SOURCES = [
    module_path + '/pic_codec.cpp',
    module_path + '/src/bilinear.cu',
    module_path + '/src/bitstream_codec.cu',
    module_path + '/src/reconstruct_latent.cu',
    module_path + '/src/fast_dct.cu',
    module_path + '/src/latent_decoder_kernels.cu',
    module_path + '/src/cuhd_codetable.cc',
    module_path + '/src/cuhd_gpu_input_buffer.cc',
    module_path + '/src/cuhd_input_buffer.cc',
    module_path + '/src/cuhd_util.cc',
    module_path + '/src/cuhd_gpu_codetable.cc',
    module_path + '/src/cuhd_gpu_decoder_memory.cc',
    module_path + '/src/cuhd_gpu_output_buffer.cc',
    module_path + '/src/cuhd_output_buffer.cc',
    module_path + '/src/cuhd_codec.cu',
    module_path + '/src/cuhd_gpu_decoder.cu',
    module_path + '/llhuff/src/llhuffman_encoder.cc'
]
EXTRA_INCLUDE_PATHS = include_dirs
LD_DIRS = []
LIBRARIES = []

