/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#include "cuhd_codetable.h"

cuhd::CUHDCodetable::CUHDCodetable(size_t num_entries, int max_codeword_length)
    : size_(1 << max_codeword_length),
      num_entries_(num_entries),
      table_(std::make_unique<CUHDCodetableItemSingle[]>(get_size())),
      compact_table_(std::make_unique<CUHDCompactCodeTableItem[]>(get_num_entries())),
      m_max_codeword_length(max_codeword_length) {
    
}

size_t cuhd::CUHDCodetable::get_size() {
    return size_;
}

size_t cuhd::CUHDCodetable::get_num_entries() {
    return num_entries_;
}

size_t cuhd::CUHDCodetable::get_max_codeword_length() {
    return m_max_codeword_length;
}

cuhd::CUHDCodetableItemSingle* cuhd::CUHDCodetable::get() {
    return table_.get();
}

cuhd::CUHDCompactCodeTableItem* cuhd::CUHDCodetable::get_compact_table() {
    return compact_table_.get();
}


