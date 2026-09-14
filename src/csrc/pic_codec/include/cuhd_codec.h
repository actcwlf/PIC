#pragma once

#include <iostream>
#include <algorithm>
#include <random>
#include <torch/extension.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/find.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <llhuff.h>     // encoder
#include <cuhd.h>       // decoder



class HuffmanCodec {
    private:
    int m_max_codeword_length;
    int m_subseq_size;
    public:
    torch::Tensor m_output;

    thrust::device_vector<int> m_sync_info_vec;
    thrust::device_vector<int> m_output_sizes_vec;
    thrust::device_vector<std::uint8_t> m_sequence_synced_device_vec;
    thrust::host_vector<cuhd::CUHDCodetableItemSingle> m_table_vec;  // sizeof(SYMBOL_TYPE) + sizeof(SYMBOL_TYPE) == 2
    thrust::device_vector<cuhd::CUHDCodetableItemSingle> m_d_dec_table;
    thrust::device_vector<UNIT_TYPE> compressed_buf;
    
    HuffmanCodec(int max_codeword_length, int subseq_size);
    ~HuffmanCodec();

    std::tuple<torch::Tensor, size_t, size_t, torch::Tensor> huffman_encode(const torch::Tensor & symbols);

    torch::Tensor huffman_decode(
        torch::Tensor tensor_dec_table, size_t num_entries,   
        size_t compressed_size, torch::Tensor & tensor_compressed, size_t total_symbols
    );

    void huffman_decode_raw(
        cuhd::CUHDCompactCodeTableItem* dec_table, 
        size_t num_entries,
        size_t compressed_size, 
        UNIT_TYPE* compressed,
        size_t total_symbols,
        thrust::device_vector<std::uint8_t> & output,
        int subseq_size
    );
};


