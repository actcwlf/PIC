/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

 #include <iostream>
 #include <algorithm>
 #include <random>
 #include <torch/extension.h>
//  #include <thrust/device_ptr.h>
// #include <thrust/scan.h>
// #include <thrust/device_vector.h>
// #include <thrust/find.h>
// #include <thrust/execution_policy.h>

 #include <cuda_runtime.h>
 #include <cuda_runtime_api.h>
 #include <cuda.h>
 #include <vector>
 #include <llhuff.h>     // encoder
 #include <cuhd.h>       // decoder
 #include "cuhd_subsequence_sync_point.h"
 #include "cuhd_codec.h"

 // subsequence size
 #define SUBSEQ_SIZE 4
 
 // threads per block
 #define NUM_THREADS 256
 
 // performance parameter, high = high performance, low = low memory consumption
 #define DEVICE_PREF 9
 


HuffmanCodec::HuffmanCodec(int max_codeword_length, int subseq_size): m_max_codeword_length(max_codeword_length), m_subseq_size(subseq_size) {
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    m_output = torch::empty({64}, options);
    m_table_vec.resize(1<< max_codeword_length);
    m_d_dec_table.resize(1<< max_codeword_length);
    compressed_buf.reserve( 10 * 1024 / 4); // 预留10MB
}


HuffmanCodec::~HuffmanCodec() {

}


std::tuple<torch::Tensor, size_t, size_t, torch::Tensor>
HuffmanCodec::huffman_encode(const torch::Tensor & symbols) {
    // std::cerr << "before get_encoder_table" << std::endl ;
    // determine optimal lengths for the codewords to be generated
    auto lengths = llhuff::LLHuffmanEncoder::get_symbol_lengths(
            symbols.data_ptr<SYMBOL_TYPE>(), symbols.size(0), m_max_codeword_length
    );
    
            // std::cout << "before get_encoder_table\n" ;
        // generate encoder table
        auto enc_table = llhuff::LLHuffmanEncoder::get_encoder_table(lengths);

        // std::cout << "before get_decoder_table\n" ;
        // generate decoder table
        auto  dec_table = llhuff::LLHuffmanEncoder::get_decoder_table(enc_table, m_max_codeword_length);
    // TIMER_STOP

    
    // std::cout << "compressed_size " << enc_table->compressed_size << '\n';
    // buffer for compressed data
    // std::cerr << "before " << enc_table->compressed_size << std::endl;

    // std::unique_ptr<UNIT_TYPE[]> compressed
    //     = std::make_unique<UNIT_TYPE[]>(enc_table->compressed_size);

    std::vector<UNIT_TYPE> compressed(enc_table->compressed_size);
    
    // if (!compressed) {
    //         std::cerr << "内存分配失败" << std::endl;
    // } else {
    //     compressed = std::make_unique<UNIT_TYPE[]>(enc_table->compressed_size);
    // }
    // compress
    // TIMER_START(timings, "encoding")
        llhuff::LLHuffmanEncoder::encode_memory(compressed.data(),
            enc_table->compressed_size, symbols.data_ptr<SYMBOL_TYPE>(), symbols.size(0), enc_table);
    // TIMER_STOP
    // std::cout << "raw compressed " << compressed.get()[0] << std::endl;
    // std::cout << "raw compressed re " << reinterpret_cast<int *>(compressed.get())[0] << std::endl;
    // std::cout << "addr " << compressed.get() << std::endl;

    // for (int i = 0; i < enc_table->compressed_size; i++) {
    //     std::cout << "bs idx " << i << " " <<  std::bitset<sizeof(int) * 8>(compressed.get()[i]) << '\n';
    // }
    // std::cout.flush();


    torch::Tensor tensor_dec_table = torch::from_blob(dec_table->get(), {static_cast<long>(dec_table->get_size() / 2)}, torch::kInt32).clone();

    torch::Tensor compact_tensor_dec_table = torch::from_blob(
        dec_table->get_compact_table(), {static_cast<long>(dec_table->get_num_entries())}, torch::kInt32
    ).clone();



    // from_blob 使用的是引用
    torch::Tensor encoded_symbols = torch::from_blob(compressed.data(), {static_cast<long>(enc_table->compressed_size)}, torch::kInt32).clone();

    // torch::cuda::synchronize();
    // std::cerr << "after" << std::endl;
    return std::make_tuple(
        compact_tensor_dec_table,
        dec_table->get_num_entries(), enc_table->compressed_size, 
        encoded_symbols
    );

}


void expand_compact_code_table(
    cuhd::CUHDCompactCodeTableItem* compact_code_table_ptr, // on host
    cuhd::CUHDCodetableItemSingle* table_ptr, // on host
    const size_t num_entries, 
    const int max_codeword_length) {
    int table_size = 1 << max_codeword_length;
    auto current_compact_item = compact_code_table_ptr[0];
    int item_idx = 0;
    for(int i = 0; i < table_size; i++) {
        if(i < current_compact_item.cum_run_length) {
            table_ptr[i] = current_compact_item.item;
        } else {
            item_idx++;
            current_compact_item = compact_code_table_ptr[item_idx];
            table_ptr[i] = current_compact_item.item;
        }
    }
}


torch::Tensor HuffmanCodec::huffman_decode(
    torch::Tensor tensor_dec_table, 
    size_t num_entries,
    size_t compressed_size, 
    torch::Tensor & tensor_compressed,
    size_t total_symbols
) {


    //      cudaEvent_t start1;
    //  cudaEvent_t stop1;
    //  cudaEventCreate(&start1);

    //  cudaEventCreate(&stop1);
    //  cudaEventRecord(start1, NULL);


    expand_compact_code_table(
        reinterpret_cast<cuhd::CUHDCompactCodeTableItem*>(tensor_dec_table.data_ptr<int>()),
        m_table_vec.data(), 
        num_entries, 
        m_max_codeword_length
    );


    // auto t = m_table_vec.data();

    m_d_dec_table = m_table_vec; // to device
    
    // cudaEventRecord(stop1, NULL);
    // cudaEventSynchronize(stop1);
    // float msecTotal1 = 0.0f;
    // cudaEventElapsedTime(&msecTotal1, start1, stop1);
    //     printf("%f\n", msecTotal1);

    //printf("total_symbols %d\n", total_symbols);
    torch::Tensor tensor_out = m_output.resize_({(signed long)total_symbols});



    size_t in_buf_param_size = compressed_size * sizeof(UNIT_TYPE);
    // auto compressed_size_ = in_buf_param_size;
         
         // input buffer 会在尾部做padding
         // calculate number of units
    auto compressed_size_units_ = in_buf_param_size % sizeof(UNIT_TYPE) == 0 ?
         in_buf_param_size / sizeof(UNIT_TYPE) : in_buf_param_size / sizeof(UNIT_TYPE) + 1;
     
    // avoid invalid read at end of input during decoding
    ++compressed_size_units_;



     // auxiliary memory for decoding
     auto gpu_decoder_memory = std::make_shared<cuhd::CUHDGPUDecoderMemory>(
        //  in_buf->get_compressed_size_units(),
         compressed_size_units_,
         m_subseq_size, NUM_THREADS);

        //  size_t num_units, size_t subsequence_size, size_t num_threads)

    auto num_units = compressed_size_units_;
    auto subsequence_size = m_subseq_size;
    auto num_threads = NUM_THREADS;
    auto num_subseq_ = SDIV(num_units, subsequence_size);
    // auto num_blocks_ = SDIV(num_subseq_, num_threads);


    m_sync_info_vec.resize(gpu_decoder_memory->num_subseq_ * 4);
    uint4* sync_info = reinterpret_cast<uint4*>(m_sync_info_vec.data().get());


    m_output_sizes_vec.resize(gpu_decoder_memory->num_subseq_);
    std::uint32_t* output_sizes =  reinterpret_cast<std::uint32_t*>(m_output_sizes_vec.data().get());

    m_sequence_synced_device_vec.resize(gpu_decoder_memory->num_blocks_);
 
    
        
        // std::cout << "before decode  m_max_codeword_length " <<  m_max_codeword_length << "\n";
     cuhd::CUHDGPUDecoder::decode(
        // compressed_buf.data().get(), 
        reinterpret_cast<UNIT_TYPE*>(tensor_compressed.data_ptr<int>()),

        compressed_size_units_,
             reinterpret_cast<SYMBOL_TYPE*>(tensor_out.data_ptr<uint8_t>()), total_symbols,
             m_d_dec_table.data().get(), gpu_decoder_memory,
            //  reinterpret_cast<uint4*>(sync_info.data().get()),
            //  output_sizes.data().get(),
            sync_info, output_sizes,
            m_sequence_synced_device_vec,
            m_max_codeword_length, m_subseq_size, NUM_THREADS);

    // cudaEventRecord(stop1, NULL);
    // cudaEventSynchronize(stop1);
    // float msecTotal1 = 0.0f;
    // cudaEventElapsedTime(&msecTotal1, start1, stop1);
    //     printf("%f\n", msecTotal1);
 
     // float msecTotal1 = 0.0f;
     // cudaEventElapsedTime(&msecTotal1, start1, stop1);
     // printf("%f\n", msecTotal1);
     
     // copy decoded data back to host
    //  TIMER_START(timings, "GPU memcpy DtH")
        //  gpu_out_buf->cpy_device_to_host();
    //  TIMER_STOP;
 
    //  // print timings
    //  for(auto &i: timings) {
    //      std::cout << i.first << ".. " << i.second << "µs" << std::endl;
    //  }
     
    //  // compare decompressed output to uncompressed input
    //  cuhd::CUHDUtil::equals(buffer.data(),
    //      out_buf->get_decompressed_data().get(), size) ? std::cout << std::endl
    //          : std::cout << std::endl << "mismatch" << std::endl;
     
    // torch::Tensor decoded_symbols = torch::from_blob(out_buf->get_decompressed_data().get(), {static_cast<long>(total_symbols)}, torch::kUInt8).clone();
    //printf("tensor_out %d\n", tensor_out.size(0));
    return tensor_out;
}


void HuffmanCodec::huffman_decode_raw(
    cuhd::CUHDCompactCodeTableItem* dec_table, 
    size_t num_entries,
    size_t compressed_size, 
    UNIT_TYPE* compressed,
    size_t total_symbols,
    thrust::device_vector<std::uint8_t> & output,
    int subseq_size = SUBSEQ_SIZE
) {


    //      cudaEvent_t start1;
    //  cudaEvent_t stop1;
    //  cudaEventCreate(&start1);

    //  cudaEventCreate(&stop1);
    //  cudaEventRecord(start1, NULL);


    expand_compact_code_table(
        dec_table,
        m_table_vec.data(), 
        num_entries, 
        m_max_codeword_length
    );


    // auto t = m_table_vec.data();

    m_d_dec_table = m_table_vec; // to device
    
    // cudaEventRecord(stop1, NULL);
    // cudaEventSynchronize(stop1);
    // float msecTotal1 = 0.0f;
    // cudaEventElapsedTime(&msecTotal1, start1, stop1);
    //     printf("%f\n", msecTotal1);


    // m_output.resize_({(signed long)total_symbols});



    size_t in_buf_param_size = compressed_size * sizeof(UNIT_TYPE);
    // auto compressed_size_ = in_buf_param_size;
         
         // input buffer 会在尾部做padding
         // calculate number of units
    auto compressed_size_units_ = in_buf_param_size % sizeof(UNIT_TYPE) == 0 ?
         in_buf_param_size / sizeof(UNIT_TYPE) : in_buf_param_size / sizeof(UNIT_TYPE) + 1;
     
    // avoid invalid read at end of input during decoding
    ++compressed_size_units_;



     // auxiliary memory for decoding
     auto gpu_decoder_memory = std::make_shared<cuhd::CUHDGPUDecoderMemory>(
        //  in_buf->get_compressed_size_units(),
         compressed_size_units_,
         subseq_size, NUM_THREADS);

        //  size_t num_units, size_t subsequence_size, size_t num_threads)

    auto num_units = compressed_size_units_;
    // auto subsequence_size = SUBSEQ_SIZE;
    auto num_threads = NUM_THREADS;
    auto num_subseq_ = SDIV(num_units, subseq_size);
    // auto num_blocks_ = SDIV(num_subseq_, num_threads);


    m_sync_info_vec.resize(gpu_decoder_memory->num_subseq_ * 4);
    uint4* sync_info = reinterpret_cast<uint4*>(m_sync_info_vec.data().get());
//     printf("m_sync_info_vec %d %p\n", m_sync_info_vec.size(), m_sync_info_vec.data().get());

    m_output_sizes_vec.resize(gpu_decoder_memory->num_subseq_);
    std::uint32_t* output_sizes =  reinterpret_cast<std::uint32_t*>(m_output_sizes_vec.data().get());

//     printf("m_output_sizes_vec %d %p\n", m_output_sizes_vec.size(), m_output_sizes_vec.data().get());
    m_sequence_synced_device_vec.resize(gpu_decoder_memory->num_blocks_);
 
//     printf("code table %d %p\n", m_d_dec_table.size(), m_d_dec_table.data().get());
//     printf("output_sizes %d %p\n", m_output_sizes_vec.size(), m_output_sizes_vec.data().get());
//     printf("m_sequence_synced_device_vec %d %p\n", m_sequence_synced_device_vec.size(), m_sequence_synced_device_vec.data().get());
        
        // std::cout << "before decode  m_max_codeword_length " <<  m_max_codeword_length << "\n";
     cuhd::CUHDGPUDecoder::decode(
        // compressed_buf.data().get(), 
        compressed,

        compressed_size_units_,
             reinterpret_cast<SYMBOL_TYPE*>(output.data().get()), total_symbols,
             m_d_dec_table.data().get(), gpu_decoder_memory,
            //  reinterpret_cast<uint4*>(sync_info.data().get()),
            //  output_sizes.data().get(),
            sync_info, output_sizes,
            m_sequence_synced_device_vec,
            m_max_codeword_length, subseq_size, NUM_THREADS);

    // return m_output;
}