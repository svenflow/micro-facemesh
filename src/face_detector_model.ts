/**
 * BlazeFace Short-Range Face Detection WebGPU Model
 *
 * BlazeNet backbone with ReLU activations and 2-scale SSD output heads.
 *
 * Architecture (face_detection_short_range):
 * 1. Initial conv 5x5 stride-2 + ReLU: 128x128x3 -> 64x64x24
 * 2. BlazeBlocks (DW 3x3 + PW 1x1 + residual ADD with channel PAD):
 *    Block 0:  24->24, stride 1, 64x64  (same channels, identity skip)
 *    Block 1:  24->28, stride 1, 64x64  (channel pad skip)
 *    Block 2:  28->32, stride 2, 64->32  (stride-2 transition with maxpool skip)
 *    Block 3:  32->36, stride 1, 32x32
 *    Block 4:  36->42, stride 1, 32x32
 *    Block 5:  42->48, stride 2, 32->16  (stride-2 transition)
 *    Block 6:  48->56, stride 1, 16x16
 *    Block 7:  56->64, stride 1, 16x16   [SSD head 1: 16x16, 88ch after block 8]
 *    Block 8:  64->72, stride 1, 16x16
 *    Block 9:  72->80, stride 1, 16x16
 *    Block 10: 80->88, stride 1, 16x16   -> 16x16 SSD head (88ch, 2 anchors)
 *    Block 11: 88->96, stride 2, 16->8   (stride-2 transition)
 *    -> 8x8 SSD head (96ch, 6 anchors)
 *
 * SSD outputs:
 *   16x16: 2 classifiers + 32 regressors (2 anchors x 16 values) = 512 anchors
 *   8x8: 6 classifiers + 96 regressors (6 anchors x 16 values) = 384 anchors
 *   Total: 896 anchors, each with 1 score + 16 regression values (4 box + 6 kp x 2)
 *
 * Weight format: JSON metadata (keys, shapes, offsets) + binary float32 file
 */

import {
  FACE_CONV5X5_STRIDE2_RELU_SHADER,
  FACE_DEPTHWISE_3X3_SHADER,
  FACE_POINTWISE_SKIP_RELU_SHADER,
  FACE_CONV1X1_SHADER,
  FACE_CANVAS_INPUT_SHADER,
  FACE_LETTERBOX_RESIZE_SHADER,
  QUANTIZE_F16_SHADER,
} from './face_detector_shaders.js';

export interface Tensor {
  data: Float32Array;
  shape: number[];
}

export interface FaceDetectionOutput {
  scores: Float32Array;      // [896] raw classifier logits
  regressors: Float32Array;  // [896 * 16] raw regressor outputs
}

export interface CompiledFaceDetectorModel {
  device: GPUDevice;
  run: (source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap) => Promise<FaceDetectionOutput>;
  /** Run with GPU-based letterbox resize matching MediaPipe's bilinear interpolation exactly */
  runWithResize: (source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap | HTMLVideoElement | HTMLImageElement, srcW: number, srcH: number) => Promise<{ output: FaceDetectionOutput; lbPadX: number; lbPadY: number }>;
}

/**
 * Load face detector weights from a URL directory.
 * Expects: {url}/face_detector.json (metadata) and {url}/face_detector.bin (binary)
 */
export async function loadFaceDetectorWeights(
  weightsUrl: string,
): Promise<Map<string, Tensor>> {
  const [metaResponse, binResponse] = await Promise.all([
    fetch(`${weightsUrl}/face_detector.json`),
    fetch(`${weightsUrl}/face_detector.bin`),
  ]);

  if (!metaResponse.ok) throw new Error(`Failed to load face detector metadata: ${metaResponse.status}`);
  if (!binResponse.ok) throw new Error(`Failed to load face detector weights: ${binResponse.status}`);

  const meta = await metaResponse.json() as Array<{ key: string; shape: number[]; offset: number; size: number }>;
  const bin = await binResponse.arrayBuffer();

  const weights = new Map<string, Tensor>();
  for (const entry of meta) {
    const data = new Float32Array(bin, entry.offset, entry.size / 4);
    weights.set(entry.key, { data, shape: entry.shape });
  }

  return weights;
}

// Block specification for the backbone
interface BackboneBlock {
  dwWeightBuf: GPUBuffer;
  dwBiasBuf: GPUBuffer;
  pwWeightBuf: GPUBuffer;
  pwBiasBuf: GPUBuffer;
  inCh: number;
  outCh: number;
  stride: 1 | 2;
  inH: number;
}

export async function compileFaceDetectorModel(
  weights: Map<string, Tensor>,
  existingDevice?: GPUDevice,
): Promise<CompiledFaceDetectorModel> {
  // Use existing device or create new one
  let device: GPUDevice;
  if (existingDevice) {
    device = existingDevice;
  } else {
    if (!navigator.gpu) throw new Error('WebGPU not supported');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No GPU adapter found');
    device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBuffersPerShaderStage: Math.min(adapter.limits.maxStorageBuffersPerShaderStage, 8),
      },
    });
  }

  // Helpers
  const BT: Record<string, GPUBufferBindingType> = { r: 'read-only-storage', s: 'storage', u: 'uniform' };
  function makeLayout(types: string[]): GPUBindGroupLayout {
    return device.createBindGroupLayout({
      entries: types.map((t, i) => {
        if (t === 't') return { binding: i, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' as const } };
        if (t === 'sm') return { binding: i, visibility: GPUShaderStage.COMPUTE, sampler: {} };
        return { binding: i, visibility: GPUShaderStage.COMPUTE, buffer: { type: BT[t]! } };
      }),
    });
  }

  const linearSampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });
  const SC = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const SO = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
  const SOC = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
  const UC = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

  function makeBuf(size: number, usage: number): GPUBuffer {
    return device.createBuffer({ size: Math.max(size, 4), usage });
  }
  function writeBuf(buf: GPUBuffer, offset: number, data: ArrayBufferView | ArrayBuffer): void {
    device.queue.writeBuffer(buf, offset, data as unknown as ArrayBuffer);
  }
  function uploadWeights(tensor: Tensor): GPUBuffer {
    const buf = makeBuf(tensor.data.byteLength, SC);
    writeBuf(buf, 0, tensor.data);
    return buf;
  }

  // Build a key index for quick lookup
  const keyList = Array.from(weights.keys());

  function getWeight(key: string): Tensor {
    const t = weights.get(key);
    if (!t) throw new Error(`Weight not found: ${key}`);
    return t;
  }

  function findWeight(...substrings: string[]): Tensor {
    const key = keyList.find(k => substrings.every(s => k.includes(s)));
    if (!key) throw new Error(`Weight not found for: ${substrings.join(', ')}`);
    return getWeight(key);
  }

  // ============ Weight transposition helpers ============
  // TFLite stores conv weights as [outCh, kH, kW, inCh] — already correct for our shaders
  // TFLite stores depthwise weights as [1, kH, kW, channels]
  // We need depthwise as [channels, 9] (channels groups of 9 weights for 3x3)

  function transposeDW3x3(tensor: Tensor): Float32Array {
    // Input: [1, 3, 3, channels], Output: [channels, 9]
    const [, kH, kW, ch] = tensor.shape;
    const result = new Float32Array(ch * 9);
    for (let c = 0; c < ch; c++) {
      for (let ky = 0; ky < kH; ky++) {
        for (let kx = 0; kx < kW; kx++) {
          result[c * 9 + ky * 3 + kx] = tensor.data[ky * kW * ch + kx * ch + c];
        }
      }
    }
    return result;
  }

  function transposePW(tensor: Tensor): Float32Array {
    // Input: [outCh, 1, 1, inCh], Output: [outCh, inCh] — already in correct layout
    const [outCh, , , inCh] = tensor.shape;
    const result = new Float32Array(outCh * inCh);
    for (let oc = 0; oc < outCh; oc++) {
      for (let ic = 0; ic < inCh; ic++) {
        result[oc * inCh + ic] = tensor.data[oc * inCh + ic];
      }
    }
    return result;
  }

  // ============ Create Shader Modules ============
  const inputConvMod = device.createShaderModule({ code: FACE_CONV5X5_STRIDE2_RELU_SHADER });
  const dwMod = device.createShaderModule({ code: FACE_DEPTHWISE_3X3_SHADER });
  const pwReluMod = device.createShaderModule({ code: FACE_POINTWISE_SKIP_RELU_SHADER });
  const conv1x1Mod = device.createShaderModule({ code: FACE_CONV1X1_SHADER });
  const canvasInputMod = device.createShaderModule({ code: FACE_CANVAS_INPUT_SHADER });
  const letterboxResizeMod = device.createShaderModule({ code: FACE_LETTERBOX_RESIZE_SHADER });
  const quantizeF16Mod = device.createShaderModule({ code: QUANTIZE_F16_SHADER });

  // ============ Create Layouts ============
  const inputConvLayout = makeLayout(['r', 'r', 'r', 's', 'u']);         // input, weight, bias, output, params
  const dwLayout = makeLayout(['r', 'r', 'r', 's', 'u']);               // input, weight, bias, output, params
  const pwReluLayout = makeLayout(['r', 'r', 'r', 'r', 's', 'u']);      // dw_out, skip, pw_w, pw_b, output, params
  const conv1x1Layout = makeLayout(['r', 'r', 'r', 's', 'u']);          // input, weight, bias, output, params
  const canvasInputLayout = makeLayout(['t', 's', 'u']);
  const letterboxResizeLayout = makeLayout(['t', 's', 'u', 'sm']);
  const quantizeF16Layout = makeLayout(['s', 'u']);

  // ============ Create Pipelines ============
  function makePipe(layout: GPUBindGroupLayout, mod: GPUShaderModule): GPUComputePipeline {
    return device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module: mod, entryPoint: 'main' },
    });
  }

  const inputConvPipe = makePipe(inputConvLayout, inputConvMod);
  const dwPipe = makePipe(dwLayout, dwMod);
  const pwReluPipe = makePipe(pwReluLayout, pwReluMod);
  const conv1x1Pipe = makePipe(conv1x1Layout, conv1x1Mod);
  const canvasInputPipe = makePipe(canvasInputLayout, canvasInputMod);
  const letterboxResizePipe = makePipe(letterboxResizeLayout, letterboxResizeMod);
  const _quantizeF16Pipe = makePipe(quantizeF16Layout, quantizeF16Mod);

  // ============ Load and upload weights ============

  // Initial conv: 5x5 stride-2, 3->24 channels
  const initConvW = findWeight('conv2d/Conv2D');
  const initConvB = findWeight('batch_normalization/', 'conv2d/Conv2D');
  const initConvWeightBuf = uploadWeights(initConvW);
  const initConvBiasBuf = uploadWeights(initConvB);

  // ============ Define backbone blocks ============
  // BlazeFace short-range backbone: 12 BlazeBlocks
  // Each block: DW 3x3 + PW 1x1 + channel-pad residual + ReLU
  //
  // Channel progression: 24->24->28->32->36->42->48->56->64->72->80->88->96
  // Downsampling at blocks 2 (64->32), 5 (32->16), 11 (16->8)
  //
  // Weight naming: depthwise_conv2d_N, conv2d_N+1, batch_normalization_N+1

  const blockDefs: Array<{
    dwKey: string;
    pwKey: string;
    bnKey: string;
    inCh: number;
    outCh: number;
    stride: 1 | 2;
    inH: number;
  }> = [
    // Block 0: 24->24, stride 1, 64x64
    { dwKey: 'depthwise_conv2d/', pwKey: 'conv2d_1/', bnKey: 'batch_normalization_1/', inCh: 24, outCh: 24, stride: 1, inH: 64 },
    // Block 1: 24->28, stride 1, 64x64
    { dwKey: 'depthwise_conv2d_1/', pwKey: 'conv2d_2/', bnKey: 'batch_normalization_2/', inCh: 24, outCh: 28, stride: 1, inH: 64 },
    // Block 2: 28->32, stride 2, 64->32
    { dwKey: 'depthwise_conv2d_2/', pwKey: 'conv2d_3/', bnKey: 'batch_normalization_3/', inCh: 28, outCh: 32, stride: 2, inH: 64 },
    // Block 3: 32->36, stride 1, 32x32
    { dwKey: 'depthwise_conv2d_3/', pwKey: 'conv2d_4/', bnKey: 'batch_normalization_4/', inCh: 32, outCh: 36, stride: 1, inH: 32 },
    // Block 4: 36->42, stride 1, 32x32
    { dwKey: 'depthwise_conv2d_4/', pwKey: 'conv2d_5/', bnKey: 'batch_normalization_5/', inCh: 36, outCh: 42, stride: 1, inH: 32 },
    // Block 5: 42->48, stride 2, 32->16
    { dwKey: 'depthwise_conv2d_5/', pwKey: 'conv2d_6/', bnKey: 'batch_normalization_6/', inCh: 42, outCh: 48, stride: 2, inH: 32 },
    // Block 6: 48->56, stride 1, 16x16
    { dwKey: 'depthwise_conv2d_6/', pwKey: 'conv2d_7/', bnKey: 'batch_normalization_7/', inCh: 48, outCh: 56, stride: 1, inH: 16 },
    // Block 7: 56->64, stride 1, 16x16
    { dwKey: 'depthwise_conv2d_7/', pwKey: 'conv2d_8/', bnKey: 'batch_normalization_8/', inCh: 56, outCh: 64, stride: 1, inH: 16 },
    // Block 8: 64->72, stride 1, 16x16
    { dwKey: 'depthwise_conv2d_8/', pwKey: 'conv2d_9/', bnKey: 'batch_normalization_9/', inCh: 64, outCh: 72, stride: 1, inH: 16 },
    // Block 9: 72->80, stride 1, 16x16
    { dwKey: 'depthwise_conv2d_9/', pwKey: 'conv2d_10/', bnKey: 'batch_normalization_10/', inCh: 72, outCh: 80, stride: 1, inH: 16 },
    // Block 10: 80->88, stride 1, 16x16 -> SSD head 1
    { dwKey: 'depthwise_conv2d_10/', pwKey: 'conv2d_11/', bnKey: 'batch_normalization_11/', inCh: 80, outCh: 88, stride: 1, inH: 16 },
    // Block 11: 88->96, stride 2, 16->8 -> SSD head 2
    { dwKey: 'depthwise_conv2d_11/', pwKey: 'conv2d_12/', bnKey: 'batch_normalization_12/', inCh: 88, outCh: 96, stride: 2, inH: 16 },
  ];

  const blocks: BackboneBlock[] = blockDefs.map(def => {
    const dwTensor = findWeight(def.dwKey);
    const pwTensor = findWeight(def.pwKey);
    const bnTensor = findWeight(def.bnKey);

    // Transpose DW weights from [1,3,3,ch] to [ch, 9]
    const dwTransposed = transposeDW3x3(dwTensor);
    const dwBuf = makeBuf(dwTransposed.byteLength, SC);
    writeBuf(dwBuf, 0, dwTransposed);

    // Zero bias for DW (bias is folded into weights in TFLite export)
    const dwBias = new Float32Array(def.inCh);
    const dwBiasBuf = makeBuf(dwBias.byteLength, SC);
    writeBuf(dwBiasBuf, 0, dwBias);

    // PW weights: [outCh, 1, 1, inCh] = [outCh, inCh]
    const pwTransposed = transposePW(pwTensor);
    const pwBuf = makeBuf(pwTransposed.byteLength, SC);
    writeBuf(pwBuf, 0, pwTransposed);

    // PW bias from fused batchnorm
    const pwBiasBuf = uploadWeights(bnTensor);

    return {
      dwWeightBuf: dwBuf,
      dwBiasBuf: dwBiasBuf,
      pwWeightBuf: pwBuf,
      pwBiasBuf: pwBiasBuf,
      inCh: def.inCh,
      outCh: def.outCh,
      stride: def.stride,
      inH: def.inH,
    };
  });

  // ============ SSD Head weights ============
  // 16x16 head (2 anchors per cell): after block 10 (88ch output)
  const cls16W = transposePW(findWeight('classificator_8', 'Conv2D'));
  const cls16WBuf = makeBuf(cls16W.byteLength, SC);
  writeBuf(cls16WBuf, 0, cls16W);
  const cls16BBuf = uploadWeights(findWeight('classificator_8', 'BiasAdd'));

  const reg16W = transposePW(findWeight('regressor_8', 'Conv2D'));
  const reg16WBuf = makeBuf(reg16W.byteLength, SC);
  writeBuf(reg16WBuf, 0, reg16W);
  const reg16BBuf = uploadWeights(findWeight('regressor_8', 'BiasAdd'));

  // 8x8 head (6 anchors per cell): after block 11 (96ch output)
  const cls8W = transposePW(findWeight('classificator_16', 'Conv2D'));
  const cls8WBuf = makeBuf(cls8W.byteLength, SC);
  writeBuf(cls8WBuf, 0, cls8W);
  const cls8BBuf = uploadWeights(findWeight('classificator_16', 'BiasAdd'));

  const reg8W = transposePW(findWeight('regressor_16', 'Conv2D'));
  const reg8WBuf = makeBuf(reg8W.byteLength, SC);
  writeBuf(reg8WBuf, 0, reg8W);
  const reg8BBuf = uploadWeights(findWeight('regressor_16', 'BiasAdd'));

  // ============ Activation buffers ============
  // Max buffer size needed across all layers
  const maxBufSize = Math.max(
    128 * 128 * 3,   // input
    64 * 64 * 96,    // largest possible feature map
    32 * 32 * 96,
    16 * 16 * 96,
    8 * 8 * 96,
  ) * 4;

  const inputBuf = makeBuf(128 * 128 * 3 * 4, SC);
  const actBufA = makeBuf(maxBufSize, SO);
  const actBufB = makeBuf(maxBufSize, SO);
  const dwOutBuf = makeBuf(maxBufSize, SO);

  // SSD output buffers
  // 16x16 head: 2 classifiers + 32 regressors (2 anchors x 16 values)
  const cls16Buf = makeBuf(16 * 16 * 2 * 4, SOC);
  const reg16Buf = makeBuf(16 * 16 * 32 * 4, SOC);
  // 8x8 head: 6 classifiers + 96 regressors (6 anchors x 16 values)
  const cls8Buf = makeBuf(8 * 8 * 6 * 4, SOC);
  const reg8Buf = makeBuf(8 * 8 * 96 * 4, SOC);

  // Readback buffers
  const cls16ReadBuf = makeBuf(16 * 16 * 2 * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  const reg16ReadBuf = makeBuf(16 * 16 * 32 * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  const cls8ReadBuf = makeBuf(8 * 8 * 6 * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  const reg8ReadBuf = makeBuf(8 * 8 * 96 * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);

  // Canvas input texture
  const canvasInputTexture = device.createTexture({
    size: [128, 128, 1],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // ============ Build the command encoder function ============
  function ceil(a: number, b: number): number {
    return Math.ceil(a / b);
  }

  function makeUniform(data: Uint32Array): GPUBuffer {
    const buf = makeBuf(data.byteLength, UC);
    writeBuf(buf, 0, data);
    return buf;
  }

  // Pre-create uniform buffers for each layer

  // Input conv: 128x128x3 -> 64x64x24
  const inputConvUniform = makeUniform(new Uint32Array([1, 3, 24, 128, 128, 64, 64]));

  // Pre-create uniform buffers for backbone blocks
  const blockUniforms = blocks.map(block => {
    const outH = block.stride === 2 ? block.inH / 2 : block.inH;
    const outW = outH;
    const pad = block.stride === 2 ? 0 : 1;  // 3x3 kernel: SAME pad = 1 for stride 1, 0 for stride 2 (TFLite)
    const channelPad = block.inCh;
    return {
      dw: makeUniform(new Uint32Array([1, block.inCh, block.inH, block.inH, outH, outW, block.stride, pad])),
      pw: makeUniform(new Uint32Array([1, block.inCh, block.outCh, outH, outW, channelPad, block.stride, block.inH, block.inH])),
      outH, outW,
    };
  });

  // SSD head uniforms
  // 16x16 head: input is 88ch from block 10 output
  const ssdCls16Uniform = makeUniform(new Uint32Array([1, 88, 2, 16, 16]));
  const ssdReg16Uniform = makeUniform(new Uint32Array([1, 88, 32, 16, 16]));
  // 8x8 head: input is 96ch from block 11 output
  const ssdCls8Uniform = makeUniform(new Uint32Array([1, 96, 6, 8, 8]));
  const ssdReg8Uniform = makeUniform(new Uint32Array([1, 96, 96, 8, 8]));

  // Pre-create canvas input bind group (texture view is stable)
  const canvasInputUniform = makeUniform(new Uint32Array([128, 128, 128]));
  const canvasInputBG = device.createBindGroup({
    layout: canvasInputLayout,
    entries: [
      { binding: 0, resource: canvasInputTexture.createView() },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: canvasInputUniform } },
    ],
  });

  // Letterbox resize: dynamic texture + uniform for full-res -> 128x128
  let lbSrcTexture: GPUTexture | null = null;
  let lbSrcW = 0;
  let lbSrcH = 0;
  const lbParamsUniform = makeBuf(32, UC);

  function ensureLetterboxTexture(srcW: number, srcH: number): GPUTexture {
    if (lbSrcTexture && lbSrcW === srcW && lbSrcH === srcH) {
      return lbSrcTexture;
    }
    if (lbSrcTexture) lbSrcTexture.destroy();
    lbSrcTexture = device.createTexture({
      size: [srcW, srcH, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    lbSrcW = srcW;
    lbSrcH = srcH;
    return lbSrcTexture;
  }

  // Pre-create initial conv bind group
  const initConvBG = device.createBindGroup({
    layout: inputConvLayout,
    entries: [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: initConvWeightBuf } },
      { binding: 2, resource: { buffer: initConvBiasBuf } },
      { binding: 3, resource: { buffer: actBufA } },
      { binding: 4, resource: { buffer: inputConvUniform } },
    ],
  });

  // f16 quantization disabled (matches desktop behavior)
  function encodeQuantizeF16(_encoder: GPUCommandEncoder, _buffer: GPUBuffer, _count: number): void {
    // no-op — f16 quantization disabled
  }

  // Helper to encode a DW+PW block into the command encoder
  function encodeDwPwBlock(
    encoder: GPUCommandEncoder,
    block: BackboneBlock,
    inputBuffer: GPUBuffer,
    outputBuffer: GPUBuffer,
    skipBuffer: GPUBuffer,
    uniforms: { dw: GPUBuffer; pw: GPUBuffer; outH: number },
  ): void {
    const outW = uniforms.outH;

    const dwBG = device.createBindGroup({
      layout: dwLayout,
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: block.dwWeightBuf } },
        { binding: 2, resource: { buffer: block.dwBiasBuf } },
        { binding: 3, resource: { buffer: dwOutBuf } },
        { binding: 4, resource: { buffer: uniforms.dw } },
      ],
    });

    const pass1 = encoder.beginComputePass();
    pass1.setPipeline(dwPipe);
    pass1.setBindGroup(0, dwBG);
    pass1.dispatchWorkgroups(ceil(outW, 8), ceil(uniforms.outH, 8), block.inCh);
    pass1.end();

    encodeQuantizeF16(encoder, dwOutBuf, block.inCh * uniforms.outH * outW);

    const pwBG = device.createBindGroup({
      layout: pwReluLayout,
      entries: [
        { binding: 0, resource: { buffer: dwOutBuf } },
        { binding: 1, resource: { buffer: skipBuffer } },
        { binding: 2, resource: { buffer: block.pwWeightBuf } },
        { binding: 3, resource: { buffer: block.pwBiasBuf } },
        { binding: 4, resource: { buffer: outputBuffer } },
        { binding: 5, resource: { buffer: uniforms.pw } },
      ],
    });

    const pass2 = encoder.beginComputePass();
    pass2.setPipeline(pwReluPipe);
    pass2.setBindGroup(0, pwBG);
    pass2.dispatchWorkgroups(ceil(outW, 8), ceil(uniforms.outH, 8), block.outCh);
    pass2.end();

    encodeQuantizeF16(encoder, outputBuffer, block.outCh * uniforms.outH * outW);
  }

  // Encode a conv1x1 (no activation) for SSD heads
  function encodeConv1x1(
    encoder: GPUCommandEncoder,
    inputBuffer: GPUBuffer,
    weightBuf: GPUBuffer,
    biasBuf: GPUBuffer,
    outputBuffer: GPUBuffer,
    uniform: GPUBuffer,
    outCh: number, h: number, w: number,
  ): void {
    const bg = device.createBindGroup({
      layout: conv1x1Layout,
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: weightBuf } },
        { binding: 2, resource: { buffer: biasBuf } },
        { binding: 3, resource: { buffer: outputBuffer } },
        { binding: 4, resource: { buffer: uniform } },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(conv1x1Pipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(ceil(w, 8), ceil(h, 8), outCh);
    pass.end();
  }

  /** Encode backbone + SSD heads + readback. Assumes inputBuf already has 128x128 CHW data. */
  async function encodeAndReadback(encoder: GPUCommandEncoder): Promise<FaceDetectionOutput> {
    // Quantize input to f16 (matches MediaPipe)
    encodeQuantizeF16(encoder, inputBuf, 128 * 128 * 3);

    // Initial conv 5x5 stride 2 + ReLU: 128x128x3 -> 64x64x24
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(inputConvPipe);
      pass.setBindGroup(0, initConvBG);
      pass.dispatchWorkgroups(ceil(64, 8), ceil(64, 8), 24);
      pass.end();
    }

    encodeQuantizeF16(encoder, actBufA, 64 * 64 * 24);

    // Backbone blocks 0-11 with double buffering
    let curBuf = actBufA;
    let altBuf = actBufB;

    // We need to save the 16x16 feature map (after block 10) for the SSD head
    // and the 8x8 feature map (after block 11) for the second SSD head
    let ssd16SourceBuf: GPUBuffer = curBuf; // placeholder

    for (let i = 0; i < blocks.length; i++) {
      const block = blocks[i];
      encodeDwPwBlock(encoder, block, curBuf, altBuf, curBuf, blockUniforms[i]);

      // Swap buffers
      const tmp = curBuf;
      curBuf = altBuf;
      altBuf = tmp;

      // After block 10: 16x16x88 — save for SSD head 1
      if (i === 10) {
        ssd16SourceBuf = curBuf;
      }
    }

    // After block 11: curBuf has 8x8x96

    // ============ SSD Heads ============
    // 16x16 head (2 anchors per cell): on block 10 output (16x16x88)
    encodeConv1x1(encoder, ssd16SourceBuf, cls16WBuf, cls16BBuf, cls16Buf, ssdCls16Uniform, 2, 16, 16);
    encodeQuantizeF16(encoder, cls16Buf, 16 * 16 * 2);
    encodeConv1x1(encoder, ssd16SourceBuf, reg16WBuf, reg16BBuf, reg16Buf, ssdReg16Uniform, 32, 16, 16);
    encodeQuantizeF16(encoder, reg16Buf, 16 * 16 * 32);

    // 8x8 head (6 anchors per cell): on block 11 output (8x8x96)
    encodeConv1x1(encoder, curBuf, cls8WBuf, cls8BBuf, cls8Buf, ssdCls8Uniform, 6, 8, 8);
    encodeQuantizeF16(encoder, cls8Buf, 8 * 8 * 6);
    encodeConv1x1(encoder, curBuf, reg8WBuf, reg8BBuf, reg8Buf, ssdReg8Uniform, 96, 8, 8);
    encodeQuantizeF16(encoder, reg8Buf, 8 * 8 * 96);

    // Submit compute passes
    device.queue.submit([encoder.finish()]);

    // Copy outputs to readback buffers
    const copyEncoder = device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(cls16Buf, 0, cls16ReadBuf, 0, 16 * 16 * 2 * 4);
    copyEncoder.copyBufferToBuffer(reg16Buf, 0, reg16ReadBuf, 0, 16 * 16 * 32 * 4);
    copyEncoder.copyBufferToBuffer(cls8Buf, 0, cls8ReadBuf, 0, 8 * 8 * 6 * 4);
    copyEncoder.copyBufferToBuffer(reg8Buf, 0, reg8ReadBuf, 0, 8 * 8 * 96 * 4);
    device.queue.submit([copyEncoder.finish()]);

    // Wait and read back
    await Promise.all([
      cls16ReadBuf.mapAsync(GPUMapMode.READ),
      reg16ReadBuf.mapAsync(GPUMapMode.READ),
      cls8ReadBuf.mapAsync(GPUMapMode.READ),
      reg8ReadBuf.mapAsync(GPUMapMode.READ),
    ]);

    const cls16Data = new Float32Array(cls16ReadBuf.getMappedRange()).slice();
    const reg16Data = new Float32Array(reg16ReadBuf.getMappedRange()).slice();
    const cls8Data = new Float32Array(cls8ReadBuf.getMappedRange()).slice();
    const reg8Data = new Float32Array(reg8ReadBuf.getMappedRange()).slice();

    cls16ReadBuf.unmap();
    reg16ReadBuf.unmap();
    cls8ReadBuf.unmap();
    reg8ReadBuf.unmap();

    // Combine outputs: reorder from CHW (GPU) to HWC (SSD anchor ordering)
    // 16x16 grid, 2 anchors per cell: cls is [2, 16, 16], reg is [32, 16, 16]
    // 32 = 2 anchors x 16 values
    // 8x8 grid, 6 anchors per cell: cls is [6, 8, 8], reg is [96, 8, 8]
    // 96 = 6 anchors x 16 values

    const totalAnchors = 512 + 384; // 896
    const scores = new Float32Array(totalAnchors);
    const regressors = new Float32Array(totalAnchors * 16);

    // Reorder 16x16 head: CHW -> HWC
    let anchorIdx = 0;
    for (let y = 0; y < 16; y++) {
      for (let x = 0; x < 16; x++) {
        for (let a = 0; a < 2; a++) {
          // Score: cls16Data layout is [2, 16, 16] (CHW)
          scores[anchorIdx] = cls16Data[a * 256 + y * 16 + x];
          // Regressors: reg16Data layout is [32, 16, 16] (CHW)
          // 32 = 2 anchors x 16 values
          for (let v = 0; v < 16; v++) {
            const ch = a * 16 + v;
            regressors[anchorIdx * 16 + v] = reg16Data[ch * 256 + y * 16 + x];
          }
          anchorIdx++;
        }
      }
    }

    // Reorder 8x8 head: CHW -> HWC
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        for (let a = 0; a < 6; a++) {
          scores[anchorIdx] = cls8Data[a * 64 + y * 8 + x];
          for (let v = 0; v < 16; v++) {
            const ch = a * 16 + v;
            regressors[anchorIdx * 16 + v] = reg8Data[ch * 64 + y * 8 + x];
          }
          anchorIdx++;
        }
      }
    }

    return { scores, regressors };
  }

  /** Legacy: accepts pre-resized 128x128 canvas (uses browser's canvas interpolation) */
  async function run(source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap): Promise<FaceDetectionOutput> {
    device.queue.copyExternalImageToTexture(
      { source },
      { texture: canvasInputTexture },
      [128, 128],
    );
    const encoder = device.createCommandEncoder();
    // Canvas texture -> CHW buffer
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(canvasInputPipe);
      pass.setBindGroup(0, canvasInputBG);
      pass.dispatchWorkgroups(ceil(128, 16), ceil(128, 16), 1);
      pass.end();
    }
    return encodeAndReadback(encoder);
  }

  /** GPU letterbox resize matching MediaPipe's exact bilinear interpolation */
  async function runWithResize(
    source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap | HTMLVideoElement | HTMLImageElement,
    srcW: number, srcH: number,
  ): Promise<{ output: FaceDetectionOutput; lbPadX: number; lbPadY: number }> {
    const scale = Math.min(128 / srcW, 128 / srcH);
    const scaledW = Math.round(srcW * scale);
    const scaledH = Math.round(srcH * scale);
    const offsetX = Math.floor((128 - scaledW) / 2);
    const offsetY = Math.floor((128 - scaledH) / 2);

    const lbPadX = offsetX / 128;
    const lbPadY = offsetY / 128;

    const tex = ensureLetterboxTexture(srcW, srcH);

    let uploadSource: HTMLCanvasElement | OffscreenCanvas | ImageBitmap;
    if (source instanceof HTMLVideoElement) {
      uploadSource = await createImageBitmap(source, { colorSpaceConversion: 'none' });
    } else if (source instanceof HTMLImageElement) {
      uploadSource = await createImageBitmap(source, { colorSpaceConversion: 'none' });
    } else {
      uploadSource = source;
    }
    device.queue.copyExternalImageToTexture(
      { source: uploadSource },
      { texture: tex },
      [srcW, srcH],
    );

    // Write letterbox params uniform
    const paramsData = new ArrayBuffer(32);
    const u32View = new Uint32Array(paramsData);
    const f32View = new Float32Array(paramsData);
    u32View[0] = srcW;
    u32View[1] = srcH;
    u32View[2] = 128;
    u32View[3] = 0;
    f32View[4] = srcW / scaledW;
    f32View[5] = srcH / scaledH;
    f32View[6] = offsetX;
    f32View[7] = offsetY;
    device.queue.writeBuffer(lbParamsUniform, 0, paramsData);

    const lbBG = device.createBindGroup({
      layout: letterboxResizeLayout,
      entries: [
        { binding: 0, resource: tex.createView() },
        { binding: 1, resource: { buffer: inputBuf } },
        { binding: 2, resource: { buffer: lbParamsUniform } },
        { binding: 3, resource: linearSampler },
      ],
    });

    const encoder = device.createCommandEncoder();
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(letterboxResizePipe);
      pass.setBindGroup(0, lbBG);
      pass.dispatchWorkgroups(ceil(128, 16), ceil(128, 16), 1);
      pass.end();
    }
    const output = await encodeAndReadback(encoder);
    return { output, lbPadX, lbPadY };
  }

  return { device, run, runWithResize };
}
