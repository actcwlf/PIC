try:
    from . import _pic_codec
except ImportError:
    from ._backend import _C as _pic_codec



PICCodec = _pic_codec.PICCodec
HuffmanCodec = _pic_codec.HuffmanCodec


bilinear_interpolate = _pic_codec.bilinear_interpolate
split_reshape_crop_interpolate = _pic_codec.split_reshape_crop_interpolate
crop_and_interpolate = _pic_codec.crop_and_interpolate
encode_bitstream = _pic_codec.encode_bitstream
decode_bitstream = _pic_codec.decode_bitstream
fused_mlp = _pic_codec.fused_mlp
fast_idct = _pic_codec.fast_idct
expand_blocks = _pic_codec.expand_blocks
expand_blocks_zigzag = _pic_codec.expand_blocks_zigzag
zigzag_trim_and_reduce_blocks = _pic_codec.zigzag_trim_and_reduce_blocks
