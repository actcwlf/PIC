#pragma once

#include <vector>
#include <torch/extension.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/find.h>
#include <thrust/execution_policy.h>
#include <pybind11/pybind11.h>
#include "cuhd_codec.h"


py::bytes encode_bitstream(
    int h, int w, int n_latent, int total_symbols, int total_blocks,

    torch::Tensor compact_tensor_dec_table,
    torch::Tensor tensor_compressed,
    torch::Tensor w0,
    torch::Tensor b0,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor w2,
    torch::Tensor b2,
    torch::Tensor w3,
    torch::Tensor b3
);

std::tuple<int, int, int, int, int, int, int, torch::Tensor, torch::Tensor,
torch::Tensor, torch::Tensor,
torch::Tensor, torch::Tensor,
torch::Tensor, torch::Tensor,
torch::Tensor, torch::Tensor,
torch::Tensor
>
decode_bitstream(py::bytes buf);

torch::Tensor fused_mlp(torch::Tensor &x,
    torch::Tensor &w0, torch::Tensor &b0,
    torch::Tensor &w1, torch::Tensor &b1,
    torch::Tensor &w2, torch::Tensor &b2,
    torch::Tensor &w3, torch::Tensor &b3,
    
    int width, int height, int in_channel, int out_channel
);


class PICCodec {
    public:
    thrust::device_vector<int> d_buffer;
    thrust::device_vector<float> d_rec_buffer;
    torch::Tensor m_feature;
    thrust::device_vector<float> d_feature;
    thrust::device_vector<float> d_output;
    thrust::device_vector<float> d_block_buf;
    thrust::device_vector<std::uint8_t> d_symbols;
    thrust::device_vector<int> d_indices_vec;
    torch::Tensor m_w0;
    torch::Tensor m_w1;
    torch::Tensor m_w2;
    torch::Tensor m_w3;
    torch::Tensor m_b0;
    torch::Tensor m_b1;
    torch::Tensor m_b2;
    torch::Tensor m_b3;

    torch::Tensor m_output;

    torch::indexing::Slice m_slice = torch::indexing::Slice(0, 3);

    torch::Tensor w0() const {return m_w0; }
    torch::Tensor w1() const {return m_w1; }
    torch::Tensor w2() const {return m_w2; }
    torch::Tensor w3() const {return m_w3; }
    torch::Tensor b0() const {return m_b0; }
    torch::Tensor b1() const {return m_b1; }
    torch::Tensor b2() const {return m_b2; }
    torch::Tensor b3() const {return m_b3; }

    int * m_pre_allocate;
    int m_h;
    int m_w;
    int m_n_latent;
    int m_subseq_size = 4;


    // int h;
    // int w;
    // int num_entries;
    // int compressed_size; 
    int m_total_symbols;
    // int total_blocks;
    int m_total_blocks;
    thrust::device_vector<int> d_compact_dec_table;
    thrust::device_vector<int> d_compressed;
    HuffmanCodec hcodec;
    torch::Tensor m_symbols;

    PICCodec(int max_codeword_length, int subseq_size);
    ~PICCodec();
    void decode_bitstream(py::bytes buf);
    torch::Tensor reconstruct_feature(int bound_value, int fill_value, float q_scale);
    torch::Tensor fused_mlp_render();

};
