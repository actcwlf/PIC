#include <torch/extension.h>
// #include "dct8x8.h"
#include "bilinear.h"
#include "reconstruct_latent.h"
#include "bitstream_codec.h"
#include "fast_dct.h"
#include "cuhd_codec.h"
#include "latent_decoder_kernels.h"



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("bilinear_interpolate", &bilinear_interpolate);
  m.def("split_reshape_crop_interpolate", &split_and_reshape);
  m.def("crop_and_interpolate", &crop_and_interpolate);
  m.def("encode_bitstream", &encode_bitstream);
  m.def("decode_bitstream", &decode_bitstream);
  m.def("fused_mlp", &fused_mlp);
  m.def("fast_idct", &fast_idct);

  m.def("expand_blocks", &expand_blocks);
  m.def("expand_blocks_zigzag", &expand_blocks_zigzag);
  m.def("zigzag_trim_and_reduce_blocks", &zigzag_trim_and_reduce_blocks);

  auto huffman = py::class_<HuffmanCodec>(m, "HuffmanCodec");
  huffman.def(py::init<int, int>())
    .def("encode", &HuffmanCodec::huffman_encode)
    .def("decode", &HuffmanCodec::huffman_decode)
    .def("__repr__", [](const HuffmanCodec &a) { return "HuffmanCodec"; });


  auto pic_codec = py::class_<PICCodec>(m, "PICCodec");
  pic_codec.def(py::init<int, int>())
    .def("decode_bitstream", &PICCodec::decode_bitstream, py::return_value_policy::reference)
    .def("reconstruct_feature", &PICCodec::reconstruct_feature, py::return_value_policy::reference)
    .def("fused_mlp_render", &PICCodec::fused_mlp_render, py::return_value_policy::reference)
    .def_property_readonly("w0", &PICCodec::w0,  py::return_value_policy::reference)
    .def_property_readonly("w1", &PICCodec::w1,  py::return_value_policy::reference)
    .def_property_readonly("w2", &PICCodec::w2,  py::return_value_policy::reference)
    .def_property_readonly("w3", &PICCodec::w3,  py::return_value_policy::reference)
    .def_property_readonly("b0", &PICCodec::b0,  py::return_value_policy::reference)
    .def_property_readonly("b1", &PICCodec::b1,  py::return_value_policy::reference)
    .def_property_readonly("b2", &PICCodec::b2,  py::return_value_policy::reference)
    .def_property_readonly("b3", &PICCodec::b3,  py::return_value_policy::reference)
    .def("__repr__", [](const PICCodec &a) { return "PICCodec"; });

}