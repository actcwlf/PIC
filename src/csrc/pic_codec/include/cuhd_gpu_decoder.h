/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#ifndef CUHD_GPU_DECODER_
#define CUHD_GPU_DECODER_
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/find.h>
#include <thrust/execution_policy.h>
#include "cuhd_constants.h"
#include "cuhd_gpu_input_buffer.h"
#include "cuhd_gpu_output_buffer.h"
#include "cuhd_gpu_codetable.h"
#include "cuhd_gpu_decoder_memory.h"
#include "cuhd_subsequence_sync_point.h"

namespace cuhd {
    class CUHDGPUDecoder {
        public:
            static void decode(
                
                // std::shared_ptr<cuhd::CUHDGPUInputBuffer> input,
                UNIT_TYPE* input_ptr,
                size_t input_size,
                    // std::shared_ptr<cuhd::CUHDGPUOutputBuffer> output,
                SYMBOL_TYPE * output_ptr,
                size_t output_size,
                // std::shared_ptr<cuhd::CUHDGPUCodetable> table,
                cuhd::CUHDCodetableItemSingle* table_ptr,
                std::shared_ptr<cuhd::CUHDGPUDecoderMemory> aux,
                uint4* sync_info,
                std::uint32_t* output_sizes,
                thrust::device_vector<std::uint8_t> &sequence_synced_device_vec,
                size_t max_codeword_length,
                size_t preferred_subsequence_size,
                size_t threads_per_block);
    };
}

#endif /* CUHD_GPU_DECODER */

