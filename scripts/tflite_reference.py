#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
Face Landmark Model — pure NumPy reference implementation.

Loads weights from the JSON+binary format (face_landmarks_weights.json + .bin)
and runs the full forward pass in Python/NumPy, exactly matching the WebGPU
WGSL shader semantics described in landmark_model.ts / landmark_shaders.ts.

Architecture:
  Input: [3, 256, 256] CHW float32, normalized to [0,1]
  Output: { landmarks: [1434] (478 x 3), score: sigmoid(raw_score) }

  Stem: Conv 3x3 s=2, BN (fused bias), PReLU -> [16, 128, 128]
  Stage 1: 4 bottleneck blocks at 128x128, 16ch (bottleneck 8), all with residual ADD
  Down 1: Conv 2x2 s=2 (16->16) + MaxPool + ChannelPad(16->32) + ADD -> BN+PReLU -> [16, 64, 64]
  Stage 2: first block (no narrow, no skip), 4 regular blocks at 64x64, 32ch (bottleneck 16)
  Down 2: Conv 2x2 s=2 (32->32) + MaxPool + ChannelPad(32->64) + ADD -> BN+PReLU -> [32, 32, 32]
  Stage 3: same pattern, 64ch (bottleneck 32) at 32x32
  Down 3: Conv 2x2 s=2 (64->64) + MaxPool + ChannelPad(64->128) + ADD -> BN+PReLU -> [64, 16, 16]
  Stage 4: 128ch (bottleneck 64) at 16x16
  Down 4: Conv 2x2 s=2 (128->64) + MaxPool (truncated to 64ch) + ADD -> BN+PReLU -> [64, 8, 8]
  Stage 5: 128ch at 8x8
  Down 5: same pattern -> [64, 4, 4]
  Stage 6: 128ch at 4x4
  Down 6: -> [64, 2, 2]
  Stage 7: 128ch at 2x2
  Landmarks head: Conv 2x2 s=2 -> [1434, 1, 1] + bias -> flatten
  Presence head: Conv 2x2 s=2 -> [1] + bias -> sigmoid

Weight format (JSON+bin, TF.js-style):
  BN entries: shape [C] — fused additive bias (mean/variance/scale/offset already merged)
  PReLU entries: shape [1, 1, C] — per-channel negative slope
  Conv entries: shape [outCh, kH, kW, inCh] — TFLite/NHWC kernel layout
  DW entries: shape [1, kH, kW, C] — transposed by loader to [C, kH*kW]
"""

import json
import os
import sys
import numpy as np

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'weights')
JSON_PATH = os.path.join(WEIGHTS_DIR, 'face_landmarks_weights.json')
BIN_PATH = os.path.join(WEIGHTS_DIR, 'face_landmarks_weights.bin')


# ============================================================
# Weight loading
# ============================================================

def load_weights():
    with open(JSON_PATH) as f:
        meta = json.load(f)
    with open(BIN_PATH, 'rb') as f:
        buf = f.read()

    weights = {}
    for i in range(len(meta['keys'])):
        key = meta['keys'][i]
        shape = meta['shapes'][i]
        offset = meta['offsets'][i]
        size = 1
        for s in shape:
            size *= s
        data = np.frombuffer(buf, dtype=np.float32, count=size, offset=offset).copy()
        data = data.reshape(shape)
        weights[key] = data
    return weights, meta['keys']


def fw(weights, key_list, *substrings):
    """Find a weight tensor by substrings (all must match)."""
    for k in key_list:
        if all(s in k for s in substrings):
            return weights[k]
    raise KeyError(f"Weight not found for substrings: {substrings}")


# ============================================================
# Individual operations (matching WGSL shaders exactly)
# ============================================================

def prelu(x, alpha):
    """PReLU: x if x >= 0 else alpha * x (per-channel, CHW input)."""
    # alpha shape: [1,1,C] or [C], x shape: [C, H, W]
    a = alpha.flatten()  # [C]
    a = a[:, None, None]  # [C, 1, 1]
    return np.where(x >= 0, x, x * a)


def conv3x3_s2_prelu(inp, weight, bn_bias, prelu_alpha):
    """
    Stem: Conv 3x3 stride-2, SAME padding (TFLite asymmetric: pad_bottom=1, pad_right=1),
    then fused BN bias, then PReLU.

    inp: [3, 256, 256]
    weight: [16, 3, 3, 3] — TFLite layout [outCh, kH, kW, inCh]
    bn_bias: [16] — fused additive bias
    prelu_alpha: [1, 1, 16]
    -> [16, 128, 128]

    TFLite SAME padding for 3x3 stride-2 on 256:
      total_pad = max(0, (128-1)*2 + 3 - 256) = 1
      pad_top = floor(1/2) = 0, pad_bottom = 1
      pad_left = floor(1/2) = 0, pad_right = 1
    WGSL shader uses no explicit padding (just bounds check) which is equivalent
    to pad_top=0, pad_left=0 (drops pad_bottom/right as they'd be out of bounds anyway).
    """
    in_c, in_h, in_w = inp.shape  # [3, 256, 256]
    out_c = weight.shape[0]
    out_h = (in_h + 1) // 2  # 128
    out_w = (in_w + 1) // 2  # 128

    # Weight: [outCh, kH, kW, inCh] -> for conv: [outCh, inCh, kH, kW]
    w = weight.transpose(0, 3, 1, 2)  # [outCh, inCh, 3, 3]

    # Build output
    out = np.zeros((out_c, out_h, out_w), dtype=np.float32)

    # Matching WGSL: for each (out_x, out_y, oc):
    #   sum over ic, ky, kx: input[ic, out_y*2+ky, out_x*2+kx] * weight[oc, ky, kx, ic]
    #   bounds: iy in [0, in_h), ix in [0, in_w)  (no padding needed for top-left corner)
    for ky in range(3):
        for kx in range(3):
            # Compute valid output positions
            iy_all = np.arange(out_h) * 2 + ky  # [out_h]
            ix_all = np.arange(out_w) * 2 + kx  # [out_w]
            iy_valid = iy_all < in_h  # bool [out_h]
            ix_valid = ix_all < in_w  # bool [out_w]

            if not iy_valid.any() or not ix_valid.any():
                continue

            # inp_patch: [in_c, out_h, out_w] with zeros where out of bounds
            patch = np.zeros((in_c, out_h, out_w), dtype=np.float32)
            iy_idx = np.where(iy_valid, iy_all, 0)
            ix_idx = np.where(ix_valid, ix_all, 0)
            p = inp[:, iy_idx[:, None], ix_idx[None, :]]  # [in_c, out_h, out_w]
            patch = p * iy_valid[:, None] * ix_valid[None, :]

            # w_patch: [out_c, in_c] for this ky, kx
            w_patch = w[:, :, ky, kx]  # [out_c, in_c]

            # out[out_c, out_h, out_w] += w_patch @ patch.reshape(in_c, out_h*out_w)
            out += np.tensordot(w_patch, patch, axes=([1], [0]))  # [out_c, out_h, out_w]

    out += bn_bias[:, None, None]
    out = prelu(out, prelu_alpha)
    return out


def conv1x1_prelu(inp, weight, bn_bias, prelu_alpha):
    """
    Pointwise 1x1 conv + fused BN bias + PReLU.
    inp: [inCh, H, W]
    weight: [outCh, 1, 1, inCh]
    -> [outCh, H, W]
    """
    in_c, h, w = inp.shape
    out_c = weight.shape[0]
    w2 = weight.reshape(out_c, in_c)  # [outCh, inCh]
    # Reshape inp to [inCh, H*W], matmul -> [outCh, H*W]
    out = w2 @ inp.reshape(in_c, h * w)  # [outCh, H*W]
    out = out.reshape(out_c, h, w)
    out += bn_bias[:, None, None]
    out = prelu(out, prelu_alpha)
    return out


def conv1x1(inp, weight, bn_bias):
    """
    Pointwise 1x1 conv + fused BN bias (NO activation).
    inp: [inCh, H, W]
    weight: [outCh, 1, 1, inCh]
    -> [outCh, H, W]
    """
    in_c, h, w = inp.shape
    out_c = weight.shape[0]
    w2 = weight.reshape(out_c, in_c)
    out = w2 @ inp.reshape(in_c, h * w)
    out = out.reshape(out_c, h, w)
    out += bn_bias[:, None, None]
    return out


def depthwise3x3(inp, weight_raw, bias):
    """
    Depthwise 3x3 conv, stride=1, SAME padding (pad=1 on each side).
    inp: [C, H, W]
    weight_raw: [1, 3, 3, C] — raw TFLite layout, transposed here to [C, 3, 3]
    bias: [C] (zeros for bottleneck blocks)
    -> [C, H, W]

    Matches WGSL: pad=1, stride=1.
    """
    # Transpose [1, 3, 3, C] -> [C, 3, 3]
    # weight_raw shape: [1, kH, kW, C]
    w = weight_raw[0].transpose(2, 0, 1)  # [C, 3, 3]
    c, h, w_in = inp.shape
    out = np.zeros_like(inp)

    # Pad input: pad=1 on all sides
    inp_padded = np.pad(inp, ((0, 0), (1, 1), (1, 1)), mode='constant')

    for ky in range(3):
        for kx in range(3):
            out += inp_padded[:, ky:ky + h, kx:kx + w_in] * w[:, ky, kx][:, None, None]

    out += bias[:, None, None]
    return out


def conv2x2_s2(inp, weight, bn_bias):
    """
    Conv 2x2 stride-2, NO padding, NO activation.
    Used for downsampling path.
    inp: [inCh, H, W]
    weight: [outCh, 2, 2, inCh]
    -> [outCh, H//2, W//2]

    Matches WGSL: iy=out_y*2+ky, ix=out_x*2+kx, bounds check iy<in_h, ix<in_w.
    """
    in_c, in_h, in_w = inp.shape
    out_c = weight.shape[0]
    out_h = in_h // 2
    out_w = in_w // 2
    w = weight.transpose(0, 3, 1, 2)  # [outCh, inCh, 2, 2]

    # Use strided extraction
    # For ky=0,1 and kx=0,1: always valid since out_y*2+1 < in_h for out_y < out_h
    out = np.zeros((out_c, out_h, out_w), dtype=np.float32)
    for ky in range(2):
        for kx in range(2):
            patch = inp[:, ky::2, kx::2][:, :out_h, :out_w]  # [inCh, out_h, out_w]
            w_patch = w[:, :, ky, kx]  # [outCh, inCh]
            out += np.tensordot(w_patch, patch, axes=([1], [0]))

    out += bn_bias[:, None, None]
    return out


def maxpool2x2(inp):
    """
    MaxPool 2x2 stride-2. inp: [C, H, W] -> [C, H//2, W//2]
    Matches WGSL: max over 2x2 blocks.
    """
    c, h, w = inp.shape
    out_h = h // 2
    out_w = w // 2
    inp_view = inp[:, :out_h * 2, :out_w * 2].reshape(c, out_h, 2, out_w, 2)
    return inp_view.max(axis=(2, 4))


def channel_pad(inp, out_ch):
    """
    Pad channel dimension with zeros: [inCh, H, W] -> [outCh, H, W].
    Matching WGSL: output channels beyond inCh are zero.
    """
    c, h, w = inp.shape
    if out_ch <= c:
        return inp[:out_ch]
    out = np.zeros((out_ch, h, w), dtype=np.float32)
    out[:c] = inp
    return out


def output_conv2x2(inp, weight, bias):
    """
    Output head: Conv 2x2 stride-2 on [inCh, 2, 2] -> [outCh, 1, 1] -> flatten.
    Weight: [outCh, 2, 2, inCh]

    Matches WGSL: for each oc, sum over ic, ky, kx:
      input[ic*4 + ky*2 + kx] * weight[oc*4*inCh + ky*2*inCh + kx*inCh + ic]
    This is just a matmul: reshape input to [inCh*4] and weight to [outCh, 4*inCh].
    """
    in_c = inp.shape[0]
    out_c = weight.shape[0]
    # inp is [inCh, 2, 2], flatten to [inCh*4] in C order (channel-major = CHW)
    inp_flat = inp.flatten()  # [inCh * 4] in CHW order: ch0[0,0], ch0[0,1], ch0[1,0], ch0[1,1], ch1...

    # WGSL weight index: oc*4*inCh + ky*2*inCh + kx*inCh + ic
    # This is NOT the same as a simple reshape of [outCh, 2, 2, inCh]!
    # Weight raw layout: [outCh, kH, kW, inCh] i.e. index = oc*(2*2*inCh) + ky*(2*inCh) + kx*inCh + ic
    # Input index: ic*4 + ky*2 + kx = ic*(2*2) + ky*2 + kx  (CHW layout)
    #
    # So we need weight[oc, ky, kx, ic] * input[ic, ky, kx]
    # = einsum('okyxi, ikyx -> o', weight, inp.reshape(inCh, 2, 2))
    # But careful about index order:
    #   weight WGSL idx = oc * (4*inCh) + ky * (2*inCh) + kx * inCh + ic
    #   weight numpy shape [outCh, 2, 2, inCh]: w[oc, ky, kx, ic]
    #   input WGSL idx = ic * 4 + ky * 2 + kx
    #   input numpy shape [inCh, 2, 2]: inp[ic, ky, kx]
    # So: out[oc] = sum_{ic, ky, kx} w[oc, ky, kx, ic] * inp[ic, ky, kx]
    # w[oc, ky, kx, ic] shaped as [outCh, 2, 2, inCh]
    # x[ic, ky, kx] shaped as [inCh, 2, 2]
    # out[oc] = sum_{ic, ky, kx} w[oc, ky, kx, ic] * x[ic, ky, kx]
    # Reshape w to [outCh, 4, inCh] and x to [inCh, 4] (C order = same ky,kx ordering)
    out = np.einsum('oki,ik->o', weight.reshape(out_c, 4, in_c), inp.reshape(in_c, 4))
    out += bias
    return out


# ============================================================
# Block builders (matching landmark_model.ts exactly)
# ============================================================

class WeightReader:
    """Sequential weight reader matching the counter logic in landmark_model.ts."""

    def __init__(self, weights, key_list):
        self.weights = weights
        self.key_list = key_list
        self.bn_num = 132
        self.prelu_num = 126
        self.conv_num = 82
        self.dw_num = 60

    def next_bn(self):
        w = fw(self.weights, self.key_list, f'batch_normalization_{self.bn_num}')
        self.bn_num += 1
        return w.flatten()  # [C]

    def next_prelu(self):
        w = fw(self.weights, self.key_list, f'p_re_lu_{self.prelu_num}')
        self.prelu_num += 1
        return w.flatten()  # [C]

    def next_conv(self):
        w = fw(self.weights, self.key_list, f'conv2d_{self.conv_num}', 'Conv2D')
        self.conv_num += 1
        return w

    def next_dw(self):
        # Search for exact suffix '/depthwise' (not '/depthwise1')
        num = self.dw_num
        self.dw_num += 1
        suffix_exact = f'depthwise_conv2d_{num}/depthwise'
        suffix_alt = f'depthwise_conv2d_{num}/depthwise1'
        for k in self.key_list:
            if suffix_exact in k and 'depthwise1' not in k.split('/')[-1]:
                return self.weights[k]  # [1, 3, 3, C]
        for k in self.key_list:
            if suffix_alt in k:
                return self.weights[k]  # [1, 3, 3, C]
        raise KeyError(f"DW weight not found for depthwise_conv2d_{num}")


def run_res_block(x, reader, ch, bn_ch, h, has_skip):
    """
    Standard bottleneck residual block.
    ch: full channels (e.g. 16)
    bn_ch: bottleneck channels (e.g. 8)
    h: spatial size
    has_skip: whether to add skip connection
    """
    # narrow: ch -> bn_ch
    narrow_w = reader.next_conv()
    narrow_bn = reader.next_bn()
    narrow_prelu = reader.next_prelu()
    t1 = conv1x1_prelu(x, narrow_w, narrow_bn, narrow_prelu)

    # depthwise 3x3
    dw_w = reader.next_dw()
    dw_bias = np.zeros(bn_ch, dtype=np.float32)
    t2 = depthwise3x3(t1, dw_w, dw_bias)

    # project: bn_ch -> ch (no PReLU yet)
    proj_w = reader.next_conv()
    proj_bn = reader.next_bn()
    proj_prelu = reader.next_prelu()
    t1 = conv1x1(t2, proj_w, proj_bn)  # note: conv1x1 (no PReLU)

    if has_skip:
        # ADD residual + PReLU
        t2 = t1 + x
        out = prelu(t2, proj_prelu)
    else:
        # No skip: just PReLU
        out = prelu(t1, proj_prelu)

    return out


def run_first_block(x, reader, out_ch, h):
    """
    First block of a new stage: no narrow conv, no skip.
    x has curCh channels (= bottleneck ch from prev downsample output).
    """
    cur_ch = x.shape[0]

    # DW on x directly (curCh channels)
    dw_w = reader.next_dw()
    dw_bias = np.zeros(cur_ch, dtype=np.float32)
    t1 = depthwise3x3(x, dw_w, dw_bias)

    # project: curCh -> out_ch
    proj_w = reader.next_conv()
    proj_bn = reader.next_bn()
    proj_prelu = reader.next_prelu()
    t2 = conv1x1(t1, proj_w, proj_bn)

    # No skip, just PReLU
    out = prelu(t2, proj_prelu)
    return out


def run_downsample(x, reader, out_ch, pad_ch):
    """
    Downsample transition:
      1. Conv 2x2 s=2: cur -> [out_ch, H/2, W/2]
      2. MaxPool 2x2 s=2: cur -> [in_ch, H/2, W/2]
      3. Channel PAD (if pad_ch > in_ch): maxpool -> [pad_ch, H/2, W/2]
      4. ADD: conv + pool[:out_ch] -> [out_ch, H/2, W/2]
      5. PReLU -> output

    pad_ch: target channel count for maxpool after padding, or None (no pad)
    """
    in_ch = x.shape[0]

    # Conv 2x2 s=2
    down_w = reader.next_conv()
    down_bn = reader.next_bn()
    down_prelu = reader.next_prelu()
    t1 = conv2x2_s2(x, down_w, down_bn)  # [out_ch, H/2, W/2]

    # MaxPool
    t2 = maxpool2x2(x)  # [in_ch, H/2, W/2]

    # Channel PAD
    if pad_ch is not None and pad_ch > in_ch:
        t3 = channel_pad(t2, pad_ch)  # [pad_ch, H/2, W/2]
        pool_buf = t3
    else:
        pool_buf = t2

    # ADD: t1 (out_ch ch) + pool_buf[:out_ch] ch
    # For pad_ch cases: pool_buf has pad_ch >= out_ch channels? No:
    #   Stage 1-3 pads: out_ch=inCh, pad_ch=2*inCh. ADD uses out_ch elements from each.
    #   Actually: t1 has out_ch ch, pool_buf has pad_ch ch. ADD should be out_ch elements.
    #   But pad_ch > out_ch in stages 1-3. So we take first out_ch ch of pool_buf.
    #   In stages 4-6: pad_ch=None, pool_buf=t2 has in_ch=128 ch, out_ch=64.
    #   ADD only uses out_ch=64 channels (first 64 of 128).
    t4 = t1 + pool_buf[:out_ch]  # element-wise on [out_ch, H/2, W/2]

    # PReLU
    out = prelu(t4, down_prelu)
    return out


# ============================================================
# Full forward pass
# ============================================================

def forward(inp_chw, weights, key_list):
    """
    Run the full face landmark model.
    inp_chw: numpy array [3, 256, 256] float32, values in [0,1]
    Returns: (landmarks [1434], score float)
    """
    reader = WeightReader(weights, key_list)

    # ============ STEM ============
    # conv2d_81 [16, 3, 3, 3], BN_132 [16], PReLU_126 [1,1,16]
    stem_w = fw(weights, key_list, 'conv2d_81', 'Conv2D')
    stem_bn = reader.next_bn()    # BN_132
    stem_prelu = reader.next_prelu()  # PReLU_126
    x = conv3x3_s2_prelu(inp_chw, stem_w, stem_bn, stem_prelu)
    # x: [16, 128, 128]

    # ============ STAGE 1: 4 blocks, 16ch, bottleneck 8, 128x128 ============
    for _ in range(4):
        x = run_res_block(x, reader, 16, 8, 128, has_skip=True)

    # ============ DOWN 1: 16->16, pad maxpool 16->32, output 16ch ============
    x = run_downsample(x, reader, 16, 32)
    # x: [16, 64, 64]

    # ============ STAGE 2: 5 blocks, 32ch, bottleneck 16, 64x64 ============
    x = run_first_block(x, reader, 32, 64)  # block 1: no narrow, no skip, 16->32ch
    for _ in range(4):
        x = run_res_block(x, reader, 32, 16, 64, has_skip=True)

    # ============ DOWN 2: 32->32, pad maxpool 32->64 ============
    x = run_downsample(x, reader, 32, 64)
    # x: [32, 32, 32]

    # ============ STAGE 3: 5 blocks, 64ch, bottleneck 32, 32x32 ============
    x = run_first_block(x, reader, 64, 32)
    for _ in range(4):
        x = run_res_block(x, reader, 64, 32, 32, has_skip=True)

    # ============ DOWN 3: 64->64, pad maxpool 64->128 ============
    x = run_downsample(x, reader, 64, 128)
    # x: [64, 16, 16]

    # ============ STAGE 4: 5 blocks, 128ch, bottleneck 64, 16x16 ============
    x = run_first_block(x, reader, 128, 16)
    for _ in range(4):
        x = run_res_block(x, reader, 128, 64, 16, has_skip=True)

    # ============ DOWN 4: 128->64, no pad (truncate maxpool to 64ch) ============
    x = run_downsample(x, reader, 64, None)
    # x: [64, 8, 8]

    # ============ STAGE 5: 5 blocks, 128ch, bottleneck 64, 8x8 ============
    x = run_first_block(x, reader, 128, 8)
    for _ in range(4):
        x = run_res_block(x, reader, 128, 64, 8, has_skip=True)

    # ============ DOWN 5: 128->64, no pad ============
    x = run_downsample(x, reader, 64, None)
    # x: [64, 4, 4]

    # ============ STAGE 6: 5 blocks, 128ch, bottleneck 64, 4x4 ============
    x = run_first_block(x, reader, 128, 4)
    for _ in range(4):
        x = run_res_block(x, reader, 128, 64, 4, has_skip=True)

    # ============ DOWN 6: 128->64, no pad ============
    x = run_downsample(x, reader, 64, None)
    # x: [64, 2, 2]

    # ============ STAGE 7: 5 blocks, 128ch, bottleneck 64, 2x2 ============
    x = run_first_block(x, reader, 128, 2)
    for _ in range(4):
        x = run_res_block(x, reader, 128, 64, 2, has_skip=True)
    # x: [128, 2, 2]

    # ============ OUTPUT HEADS ============
    # Landmarks: conv2d_150 [1434, 2, 2, 128] + bias [1434]
    lm_conv_w = fw(weights, key_list, 'conv2d_150', 'Conv2D')   # [1434, 2, 2, 128]
    lm_bias = fw(weights, key_list, 'conv2d_150', 'BiasAdd').flatten()  # [1434]
    landmarks_raw = output_conv2x2(x, lm_conv_w, lm_bias)  # [1434]

    # Presence: conv2d_151 [1, 2, 2, 128] + bias [1]
    # NOTE: conv2d_152 (outer wrapper) gives wrong results; conv2d_151 is the actual presence head
    pres_conv_w = fw(weights, key_list, 'conv2d_151', 'Conv2D')  # [1, 2, 2, 128]
    pres_bias = fw(weights, key_list, 'conv2d_151', 'BiasAdd').flatten()  # [1]
    pres_raw = output_conv2x2(x, pres_conv_w, pres_bias)  # [1]
    score = 1.0 / (1.0 + np.exp(-pres_raw[0]))
    return landmarks_raw, score, pres_raw[0]

def print_results(label, landmarks, score, raw_logit=None):
    print(f"\n{'='*60}")
    print(f"INPUT: {label}")
    print(f"{'='*60}")
    if raw_logit is not None:
        print(f"Face presence raw logit: {raw_logit:.4f}")
    print(f"Face presence score (sigmoid): {score:.6f}")
    print(f"Landmarks shape: {landmarks.shape}")
    print(f"Landmarks min: {landmarks.min():.6f}")
    print(f"Landmarks max: {landmarks.max():.6f}")
    print(f"Landmarks mean: {landmarks.mean():.6f}")
    print(f"\nFirst 10 landmark values (raw, before /256 normalization):")
    for i in range(10):
        print(f"  landmarks[{i:3d}] = {landmarks[i]:.6f}")
    print(f"\nFirst 5 landmarks as (x, y, z) in pixel coords:")
    lm = landmarks.reshape(478, 3)
    for i in range(5):
        x, y, z = lm[i]
        print(f"  landmark[{i}]: x={x:.4f}, y={y:.4f}, z={z:.4f}")
    print(f"\nFirst 5 landmarks normalized to [0,1] (divided by 256):")
    for i in range(5):
        x, y, z = lm[i] / 256.0
        print(f"  landmark[{i}]: x={x:.6f}, y={y:.6f}, z={z:.6f}")


def main():
    print("Loading face landmark weights...")
    weights, key_list = load_weights()
    print(f"Loaded {len(weights)} weight tensors")

    # Print model info
    print(f"\nModel summary:")
    print(f"  Input: [3, 256, 256] CHW float32")
    print(f"  Output landmarks: [1434] (478 landmarks * 3)")
    print(f"  Output score: sigmoid(raw) -> [0,1]")

    # ============ ZERO INPUT ============
    inp_zero = np.zeros((3, 256, 256), dtype=np.float32)
    print(f"\nRunning inference on zero input...")
    lm_zero, score_zero, logit_zero = forward(inp_zero, weights, key_list)
    print_results("zeros [1,256,256,3] (CHW: [3,256,256])", lm_zero, score_zero, logit_zero)

    # ============ GRAY INPUT (0.5) ============
    inp_gray = np.full((3, 256, 256), 0.5, dtype=np.float32)
    print(f"\nRunning inference on gray (0.5) input...")
    lm_gray, score_gray, logit_gray = forward(inp_gray, weights, key_list)
    print_results("gray 0.5 [3,256,256]", lm_gray, score_gray, logit_gray)

    # ============ ONES INPUT (1.0) ============
    inp_ones = np.ones((3, 256, 256), dtype=np.float32)
    print(f"\nRunning inference on ones (1.0) input...")
    lm_ones, score_ones, logit_ones = forward(inp_ones, weights, key_list)
    print_results("ones 1.0 [3,256,256]", lm_ones, score_ones, logit_ones)

    # ============ SUMMARY ============
    print(f"\n{'='*60}")
    print("SUMMARY (for WebGPU comparison)")
    print(f"{'='*60}")
    print(f"{'Input':<12} {'RawLogit':>12}  {'Score':>10}  {'LM[0]':>12}  {'LM[1]':>12}  {'LM[2]':>12}  {'LM min':>10}  {'LM max':>10}")
    print(f"{'zeros':<12} {logit_zero:>12.4f}  {score_zero:>10.6f}  {lm_zero[0]:>12.6f}  {lm_zero[1]:>12.6f}  {lm_zero[2]:>12.6f}  {lm_zero.min():>10.4f}  {lm_zero.max():>10.4f}")
    print(f"{'gray(0.5)':<12} {logit_gray:>12.4f}  {score_gray:>10.6f}  {lm_gray[0]:>12.6f}  {lm_gray[1]:>12.6f}  {lm_gray[2]:>12.6f}  {lm_gray.min():>10.4f}  {lm_gray.max():>10.4f}")
    print(f"{'ones(1.0)':<12} {logit_ones:>12.4f}  {score_ones:>10.6f}  {lm_ones[0]:>12.6f}  {lm_ones[1]:>12.6f}  {lm_ones[2]:>12.6f}  {lm_ones.min():>10.4f}  {lm_ones.max():>10.4f}")


if __name__ == '__main__':
    main()
