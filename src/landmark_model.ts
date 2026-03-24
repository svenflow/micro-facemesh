/**
 * Face Landmark Model — deep bottleneck network with PReLU activations.
 *
 * Input: [1, 3, 256, 256] float32 CHW, normalized to [0,1] (from crop shader)
 * Output: { landmarks: Float32Array(1434), score: number }
 *
 * Architecture:
 *   Stem: Conv 3x3 s=2 + BN + PReLU -> [16, 128, 128]
 *   Stage 1: 4 bottleneck blocks at 128x128, 16ch (bottleneck 8) — all have residual ADD
 *   Down 1: Conv 2x2 s=2 (16->16) + MaxPool + ChannelPad(16->32) + ADD -> BN+PReLU -> [16, 64, 64]
 *   Stage 2: 5 blocks at 64x64, 32ch (bottleneck 16) — block1 no skip, blocks 2-5 have ADD
 *   Down 2: Conv 2x2 s=2 (32->32) + MaxPool + ChannelPad(32->64) + ADD -> BN+PReLU -> [32, 32, 32]
 *   Stage 3: 5 blocks at 32x32, 64ch (bottleneck 32)
 *   Down 3: Conv 2x2 s=2 (64->64) + MaxPool + ChannelPad(64->128) + ADD -> BN+PReLU -> [64, 16, 16]
 *   Stage 4: 5 blocks at 16x16, 128ch (bottleneck 64)
 *   Down 4: Conv 2x2 s=2 (128->64) + MaxPool(128->64 truncate?) + ADD -> BN+PReLU -> [64, 8, 8]
 *   Stage 5: 5 blocks at 8x8, 128ch (bottleneck 64)
 *   Down 5: Conv 2x2 s=2 (128->64) + MaxPool + ADD -> BN+PReLU -> [64, 4, 4]
 *   Stage 6: 5 blocks at 4x4, 128ch (bottleneck 64)
 *   Down 6: Conv 2x2 s=2 (128->64) + MaxPool + ADD -> BN+PReLU -> [64, 2, 2]
 *   Stage 7: 5 blocks at 2x2, 128ch (bottleneck 64)
 *   Landmarks head: Conv 2x2 s=2 -> [1434] (478 * 3)
 *   Presence head: Conv 2x2 s=2 -> [1] -> sigmoid
 *
 * PReLU activations:
 *   69 total = 1 stem + 34 block-internal (after narrow) + 34 after-project/ADD
 *   Wait, actually: 69 = 1 stem + 6 downsample + 28+34 block-internal
 *
 *   The exact breakdown: each conv (except output heads and DW) has a BN+PReLU pair.
 *   72 total CONV_2D - 3 output heads = 69 convs with BN+PReLU.
 *   34 DW convs have zero bias and no PReLU.
 *
 * BN/PReLU pair ordering (indices 132..200 / 126..194):
 *   0: stem conv
 *   1: stage1 block1 narrow 1x1
 *   2: stage1 block1 project 1x1
 *   3: stage1 block2 narrow
 *   4: stage1 block2 project
 *   5,6: stage1 block3
 *   7,8: stage1 block4
 *   9: down1 conv
 *   10: stage2 block1 project (no narrow for first block)
 *   11,12: stage2 block2 narrow, project
 *   13,14: stage2 block3
 *   15,16: stage2 block4
 *   17,18: stage2 block5
 *   19: down2 conv
 *   ... etc
 *
 * Conv ordering (conv2d_81..152):
 *   81: stem
 *   82,83: stage1 block1 narrow, project
 *   84,85: stage1 block2
 *   86,87: stage1 block3
 *   88,89: stage1 block4
 *   90: down1
 *   91: stage2 block1 project (no narrow)
 *   92,93: stage2 block2 narrow, project
 *   ... etc
 *   150: landmarks output
 *   151: presence output (inner)
 *   152: presence output (outer)
 *
 * DW ordering (depthwise_conv2d_60..93):
 *   60-63: stage1 blocks 1-4
 *   64-68: stage2 blocks 1-5
 *   ... etc
 */

import {
  LM_CONV3X3_S2_PRELU_SHADER,
  LM_CONV1X1_PRELU_SHADER,
  LM_DEPTHWISE_3X3_SHADER,
  LM_CONV1X1_SHADER,
  LM_ADD_SHADER,
  LM_CONV2X2_S2_SHADER,
  LM_MAXPOOL_2X2_SHADER,
  LM_CHANNEL_PAD_SHADER,
  LM_OUTPUT_CONV2X2_SHADER,
  LM_SIGMOID_SHADER,
  LM_PRELU_SHADER,
} from './landmark_shaders.js';

// ============ Types ============

export interface Tensor {
  data: Float32Array;
  shape: number[];
}

export interface LandmarkWeightsMetadata {
  keys: string[];
  shapes: number[][];
  offsets: number[];
  dtype?: 'float32' | 'float16';
}

export interface LandmarkOutput {
  /** 1434 values: 478 landmarks * 3 (x, y, z) */
  landmarks: Float32Array;
  /** Face presence score after sigmoid, 0-1 */
  score: number;
}

export interface CompiledLandmarkModel {
  device: GPUDevice;
  /** Run the landmark model. Input is a GPUBuffer with CHW float32 [3,256,256]. */
  run: (inputBuffer: GPUBuffer, encoder: GPUCommandEncoder) => void;
  /** Read back results after GPU submission */
  readback: () => Promise<LandmarkOutput>;
  /** The input buffer size expected (3 * 256 * 256 * 4 bytes) */
  inputBufferSize: number;
}

// ============ Weight Loading ============

export async function loadLandmarkWeights(
  weightsUrl: string,
): Promise<Map<string, Tensor>> {
  const [metaRes, binRes] = await Promise.all([
    fetch(`${weightsUrl}/face_landmarks_weights.json`),
    fetch(`${weightsUrl}/face_landmarks_weights.bin`),
  ]);

  if (!metaRes.ok) throw new Error(`Failed to load landmark metadata: ${metaRes.status}`);
  if (!binRes.ok) throw new Error(`Failed to load landmark weights: ${binRes.status}`);

  const meta = await metaRes.json() as LandmarkWeightsMetadata;
  const bin = await binRes.arrayBuffer();

  return loadLandmarkWeightsFromBuffer(meta, bin);
}

export function loadLandmarkWeightsFromBuffer(
  meta: LandmarkWeightsMetadata,
  buffer: ArrayBuffer,
): Map<string, Tensor> {
  const weights = new Map<string, Tensor>();
  const dtype = meta.dtype ?? 'float32';

  for (let i = 0; i < meta.keys.length; i++) {
    const key = meta.keys[i]!;
    const shape = meta.shapes[i]!;
    const offset = meta.offsets[i]!;
    const size = shape.reduce((a, b) => a * b, 1);

    let data: Float32Array;
    if (dtype === 'float32') {
      data = new Float32Array(buffer, offset, size);
    } else {
      const view = new DataView(buffer);
      data = new Float32Array(size);
      for (let j = 0; j < size; j++) {
        data[j] = float16ToFloat32(view.getUint16(offset + j * 2, true));
      }
    }

    weights.set(key, { data, shape });
  }

  return weights;
}

function float16ToFloat32(h: number): number {
  const sign = (h >> 15) & 0x1;
  const exponent = (h >> 10) & 0x1f;
  const mantissa = h & 0x3ff;
  if (exponent === 0) {
    if (mantissa === 0) return sign ? -0 : 0;
    return (sign ? -1 : 1) * Math.pow(2, -14) * (mantissa / 1024);
  }
  if (exponent === 0x1f) {
    if (mantissa === 0) return sign ? -Infinity : Infinity;
    return NaN;
  }
  return (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
}

// ============ Weight Helpers ============

function findWeight(
  weights: Map<string, Tensor>,
  keyList: string[],
  ...substrings: string[]
): Tensor {
  const key = keyList.find(k => substrings.every(s => k.includes(s)));
  if (!key) throw new Error(`Weight not found for: ${substrings.join(', ')}`);
  return weights.get(key)!;
}

/** Transpose DW weights from TFLite [1, kH, kW, ch] to [ch, kH*kW] */
function transposeDW3x3(tensor: Tensor): Float32Array {
  const [, kH, kW, ch] = tensor.shape;
  const kk = kH * kW;
  const result = new Float32Array(ch * kk);
  for (let c = 0; c < ch; c++) {
    for (let ky = 0; ky < kH; ky++) {
      for (let kx = 0; kx < kW; kx++) {
        result[c * kk + ky * kW + kx] = tensor.data[ky * kW * ch + kx * ch + c];
      }
    }
  }
  return result;
}

// ============ Compile Model ============

export async function compileLandmarkModel(
  device: GPUDevice,
  weights: Map<string, Tensor>,
): Promise<CompiledLandmarkModel> {
  const keyList = Array.from(weights.keys());

  function fw(...substrings: string[]): Tensor {
    return findWeight(weights, keyList, ...substrings);
  }

  // ============ GPU Helpers ============
  const BT: Record<string, GPUBufferBindingType> = { r: 'read-only-storage', s: 'storage', u: 'uniform' };
  function makeLayout(types: string[]): GPUBindGroupLayout {
    return device.createBindGroupLayout({
      entries: types.map((t, i) => ({
        binding: i,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: BT[t]! },
      })),
    });
  }

  const SC = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const SOCD = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const SOC = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
  const UC = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

  function makeBuf(size: number, usage: number): GPUBuffer {
    return device.createBuffer({ size: Math.max(size, 4), usage });
  }
  function writeBuf(buf: GPUBuffer, data: ArrayBufferView): void {
    device.queue.writeBuffer(buf, 0, data as unknown as ArrayBuffer);
  }
  function uploadF32(data: Float32Array): GPUBuffer {
    const buf = makeBuf(data.byteLength, SC);
    writeBuf(buf, data);
    return buf;
  }
  function makePipe(layout: GPUBindGroupLayout, code: string): GPUComputePipeline {
    return device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
    });
  }
  function bind(layout: GPUBindGroupLayout, bufs: GPUBuffer[]): GPUBindGroup {
    return device.createBindGroup({
      layout,
      entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });
  }

  // ============ Layouts ============
  const L6 = makeLayout(['r', 'r', 'r', 'r', 's', 'u']); // stem, conv1x1_prelu
  const L5 = makeLayout(['r', 'r', 'r', 's', 'u']); // dw3x3, conv1x1, conv2x2, output_conv
  const L4add = makeLayout(['r', 'r', 's', 'u']); // add
  const L3pool = makeLayout(['r', 's', 'u']); // maxpool
  const L3pad = makeLayout(['r', 's', 'u']); // channel_pad
  const L4prelu = makeLayout(['r', 'r', 's', 'u']); // prelu
  const L3sig = makeLayout(['r', 's', 'u']); // sigmoid

  // ============ Pipelines ============
  const pipeStem = makePipe(L6, LM_CONV3X3_S2_PRELU_SHADER);
  const pipeConv1x1PReLU = makePipe(L6, LM_CONV1X1_PRELU_SHADER);
  const pipeDW = makePipe(L5, LM_DEPTHWISE_3X3_SHADER);
  const pipeConv1x1 = makePipe(L5, LM_CONV1X1_SHADER);
  const pipeAdd = makePipe(L4add, LM_ADD_SHADER);
  const pipeConv2x2 = makePipe(L5, LM_CONV2X2_S2_SHADER);
  const pipeMaxPool = makePipe(L3pool, LM_MAXPOOL_2X2_SHADER);
  const pipePad = makePipe(L3pad, LM_CHANNEL_PAD_SHADER);
  const pipeOutConv = makePipe(L5, LM_OUTPUT_CONV2X2_SHADER);
  const pipeSigmoid = makePipe(L3sig, LM_SIGMOID_SHADER);
  const pipePReLU = makePipe(L4prelu, LM_PRELU_SHADER);

  // ============ Activation Buffers ============
  // Max intermediate: 128 * 128 * 128 = 2M floats = 8MB (generous)
  const MAX = 128 * 128 * 128 * 4;
  const bufA = makeBuf(MAX, SOCD); // input goes here
  const bufB = makeBuf(MAX, SOCD);
  const bufC = makeBuf(MAX, SOCD);
  const bufD = makeBuf(MAX, SOCD);
  const bufE = makeBuf(MAX, SOCD);

  // Output
  const bufLM = makeBuf(1434 * 4, SOC);
  const bufPres = makeBuf(4, SOC);
  const bufPresSig = makeBuf(4, SOC);
  const readbackBuf = makeBuf((1434 + 1) * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);

  // ============ Weight Index Tracking ============
  let bnNum = 132;
  let preluNum = 126;
  let convNum = 82; // after stem (81)
  let dwNum = 60;

  function nextBN(): GPUBuffer {
    const t = fw(`batch_normalization_${bnNum}`);
    bnNum++;
    return uploadF32(t.data);
  }
  function nextPReLU(): GPUBuffer {
    const t = fw(`p_re_lu_${preluNum}`);
    preluNum++;
    return uploadF32(t.data);
  }
  function nextConv(): { buf: GPUBuffer; shape: number[] } {
    const t = fw(`conv2d_${convNum}`, 'Conv2D');
    convNum++;
    return { buf: uploadF32(t.data), shape: t.shape };
  }
  function nextDW(): GPUBuffer {
    let t: Tensor;
    try {
      t = fw(`depthwise_conv2d_${dwNum}/depthwise`);
    } catch {
      t = fw(`depthwise_conv2d_${dwNum}/depthwise1`);
    }
    dwNum++;
    return uploadF32(transposeDW3x3(t));
  }
  /** BN + PReLU pair consumed together */
  function nextBNPReLU(): { bn: GPUBuffer; prelu: GPUBuffer } {
    return { bn: nextBN(), prelu: nextPReLU() };
  }

  // ============ Dispatch Command List ============
  interface Cmd {
    pipe: GPUComputePipeline;
    bg: GPUBindGroup;
    wg: [number, number, number];
  }
  const cmds: Cmd[] = [];
  function emit(pipe: GPUComputePipeline, bg: GPUBindGroup, wg: [number, number, number]) {
    cmds.push({ pipe, bg, wg });
  }

  // Helpers for workgroup calculation
  const C8 = (n: number) => Math.ceil(n / 8);
  const C64 = (n: number) => Math.ceil(n / 64);
  const C256 = (n: number) => Math.ceil(n / 256);

  function makeU(data: number[]): GPUBuffer {
    const buf = makeBuf(data.length * 4, UC);
    writeBuf(buf, new Uint32Array(data));
    return buf;
  }

  // ============ STEM ============
  // conv2d_81 [16,3,3,3], BN_132 [16], PReLU_126 [1,1,16]
  const stemW = uploadF32(fw('conv2d_81', 'Conv2D').data);
  const stemBN = nextBN();   // BN_132
  const stemA = nextPReLU(); // PReLU_126
  const stemU = makeU([3, 16, 256, 256, 128, 128]);

  // Input in bufA, output to bufB
  emit(pipeStem, bind(L6, [bufA, stemW, stemBN, stemA, bufB, stemU]),
    [C8(128), C8(128), 16]);

  // State: curBuf=bufB, curCh=16, curH=128
  let cur = bufB;
  let curCh = 16;
  let curH = 128;

  // Free buffers rotate: we use cur and 4 temps
  // tmp1=bufC, tmp2=bufD, tmp3=bufE, tmp4=bufA (input done)
  let [t1, t2, t3, t4] = [bufC, bufD, bufE, bufA];

  // ============ STAGE 1: 4 blocks at 128x128, 16ch, bottleneck 8 ============
  // All blocks in stage 1 have: narrow(16->8) + DW(8) + project(8->16) + ADD(skip)
  // Each block consumes 2 BN+PReLU pairs (narrow, project) + 2 convs + 1 DW

  function emitResBlock(ch: number, bn: number, h: number, hasSkip: boolean) {
    // narrow 1x1 (ch -> bn) + BN + PReLU
    const narrowConv = nextConv();
    const narrowBP = nextBNPReLU();
    const narrowU = makeU([ch, bn, h, h]);

    // cur -> t1 (narrow + BN + PReLU)
    emit(pipeConv1x1PReLU, bind(L6, [cur, narrowConv.buf, narrowBP.bn, narrowBP.prelu, t1, narrowU]),
      [C8(h), C8(h), bn]);

    // DW 3x3 (zero bias, stride 1, pad 1)
    const dwBuf = nextDW();
    const dwZeroBias = uploadF32(new Float32Array(bn));
    const dwU = makeU([bn, h, h, h, h, 1, 1]);

    // t1 -> t2 (DW)
    emit(pipeDW, bind(L5, [t1, dwBuf, dwZeroBias, t2, dwU]),
      [C8(h), C8(h), bn]);

    // project 1x1 (bn -> ch) + BN
    const projConv = nextConv();
    const projBP = nextBNPReLU();
    const projU = makeU([bn, ch, h, h]);

    // t2 -> t1 (project + BN, no PReLU yet)
    emit(pipeConv1x1, bind(L5, [t2, projConv.buf, projBP.bn, t1, projU]),
      [C8(h), C8(h), ch]);

    if (hasSkip) {
      // ADD: t1 + cur -> t2
      const addN = ch * h * h;
      const addU = makeU([addN]);
      emit(pipeAdd, bind(L4add, [t1, cur, t2, addU]),
        [C256(addN), 1, 1]);

      // PReLU: t2 -> t3
      const preluU = makeU([ch, h, h]);
      emit(pipePReLU, bind(L4prelu, [t2, projBP.prelu, t3, preluU]),
        [C8(h), C8(h), ch]);

      // Rotate: cur = t3, old cur and t1/t2 become temps
      const oldCur = cur;
      cur = t3;
      t3 = oldCur;
    } else {
      // No skip: just PReLU on project output
      // t1 -> t2 (PReLU)
      const preluU = makeU([ch, h, h]);
      emit(pipePReLU, bind(L4prelu, [t1, projBP.prelu, t2, preluU]),
        [C8(h), C8(h), ch]);

      const oldCur = cur;
      cur = t2;
      t2 = oldCur;
    }
    curCh = ch;
  }

  /** Emit first block of a new stage (no narrow conv, no skip) */
  function emitFirstBlock(ch: number, bn: number, h: number) {
    // DW 3x3 on cur (which has bn channels = bottleneck from downsample)
    const dwBuf = nextDW();
    const dwZeroBias = uploadF32(new Float32Array(curCh));
    const dwU = makeU([curCh, h, h, h, h, 1, 1]);

    // cur -> t1 (DW)
    emit(pipeDW, bind(L5, [cur, dwBuf, dwZeroBias, t1, dwU]),
      [C8(h), C8(h), curCh]);

    // project 1x1 (curCh -> ch) + BN
    const projConv = nextConv();
    const projBP = nextBNPReLU();
    const projU = makeU([curCh, ch, h, h]);

    // t1 -> t2 (project + BN)
    emit(pipeConv1x1, bind(L5, [t1, projConv.buf, projBP.bn, t2, projU]),
      [C8(h), C8(h), ch]);

    // No skip ADD for first block.
    // PReLU: t2 -> t1
    const preluU = makeU([ch, h, h]);
    emit(pipePReLU, bind(L4prelu, [t2, projBP.prelu, t1, preluU]),
      [C8(h), C8(h), ch]);

    const oldCur = cur;
    cur = t1;
    t1 = oldCur;
    curCh = ch;
  }

  /** Emit downsample transition */
  function emitDownsample(outCh: number, padCh: number | null) {
    const newH = curH / 2;
    const inCh = curCh;

    // 1. Conv 2x2 s=2: cur -> t1
    const downConv = nextConv();
    const downBP = nextBNPReLU();
    const downU = makeU([inCh, outCh, curH, curH, newH, newH]);

    emit(pipeConv2x2, bind(L5, [cur, downConv.buf, downBP.bn, t1, downU]),
      [C8(newH), C8(newH), outCh]);

    // 2. MaxPool 2x2 s=2: cur -> t2
    const mpU = makeU([inCh, curH, curH, newH, newH]);
    emit(pipeMaxPool, bind(L3pool, [cur, t2, mpU]),
      [C8(newH), C8(newH), inCh]);

    // 3. Channel PAD (if needed): t2 -> t3
    let poolBuf: GPUBuffer;
    if (padCh !== null && padCh > inCh) {
      const padU = makeU([inCh, padCh, newH, newH]);
      emit(pipePad, bind(L3pad, [t2, t3, padU]),
        [C8(newH), C8(newH), padCh]);
      poolBuf = t3;
    } else {
      poolBuf = t2;
    }

    // 4. ADD: t1 + poolBuf -> t4
    const addN = outCh * newH * newH;
    const addU = makeU([addN]);
    emit(pipeAdd, bind(L4add, [t1, poolBuf, t4, addU]),
      [C256(addN), 1, 1]);

    // 5. PReLU: t4 -> cur
    const preluU = makeU([outCh, newH, newH]);
    emit(pipePReLU, bind(L4prelu, [t4, downBP.prelu, cur, preluU]),
      [C8(newH), C8(newH), outCh]);

    // After downsample, cur still points to the same buffer but has new content
    curCh = outCh;
    curH = newH;
  }

  // Stage 1: 4 blocks, all have residual skip (channels stay 16)
  for (let i = 0; i < 4; i++) {
    emitResBlock(16, 8, 128, true);
  }

  // Down 1: 16->16, pad maxpool 16->32
  emitDownsample(16, 32);
  // After: curCh=16, curH=64

  // Stage 2: 5 blocks, 32ch, bottleneck 16
  // Block 1: no narrow, no skip (transforms 16->32)
  emitFirstBlock(32, 16, 64);
  // Blocks 2-5: regular with skip
  for (let i = 0; i < 4; i++) {
    emitResBlock(32, 16, 64, true);
  }

  // Down 2: 32->32, pad maxpool 32->64
  emitDownsample(32, 64);
  // After: curCh=32, curH=32

  // Stage 3: 5 blocks, 64ch, bottleneck 32
  emitFirstBlock(64, 32, 32);
  for (let i = 0; i < 4; i++) {
    emitResBlock(64, 32, 32, true);
  }

  // Down 3: 64->64, pad maxpool 64->128
  emitDownsample(64, 128);
  // After: curCh=64, curH=16

  // Stage 4: 5 blocks, 128ch, bottleneck 64
  emitFirstBlock(128, 64, 16);
  for (let i = 0; i < 4; i++) {
    emitResBlock(128, 64, 16, true);
  }

  // Down 4: 128->64, NO pad (conv outputs 64ch, maxpool outputs 128ch)
  // Wait, ADD requires same size. Conv outputs [64, 8, 8] and maxpool outputs [128, 8, 8].
  // They can't be added. Let me check: maybe only the first 64ch of maxpool are used?
  // Or maybe there's no ADD and it's just the conv output?
  //
  // Actually, looking at the TFLite op structure more carefully:
  // Down 4 conv: [64, 2, 2, 128] -> takes 128ch input, outputs 64ch
  // MaxPool: takes 128ch, outputs 128ch
  // To make ADD work: maxpool 128ch is somehow reduced to 64ch.
  //
  // One possibility: the "channel padding" for these transitions pads the CONV
  // output up to match maxpool (128ch). But there are only 3 channel_padding ops.
  //
  // Another possibility: these downsamples don't have ADD at all. The conv output
  // is just used directly. This would remove 3 ADDs from the count:
  // 34 block ADDs + 3 padded-down ADDs + 3 non-padded-down = 40 != 34.
  // 34 block ADDs + 3 padded-down ADDs = 37 != 34.
  //
  // Hmm. Let me reconsider: 28 block ADDs (stage1=4, stages2-7 first block no ADD = 4 each * 6 = 24,
  // total = 4+24 = 28). Plus 6 downsample ADDs = 34. YES!
  //
  // So: 28 block ADDs + 6 downsample ADDs = 34 total ADDs.
  // This means ALL downsamples have ADD, and only blocks 2-N of each stage (not block 1) have ADD.
  //
  // For down 4-6: conv outputs 64ch, maxpool outputs 128ch. ADD requires same size.
  // So we need to truncate the maxpool to 64ch, or something else.
  //
  // Let me check: maybe the MaxPool for these transitions isn't on the full 128ch.
  // The 6 MAX_POOL_2D ops (one per downsample) - they pool whatever comes in.
  //
  // Actually, I think for down 4-6, the pattern is:
  //   Conv 2x2 s=2 (128->64): outputs [64, newH, newH]
  //   MaxPool 2x2 s=2 on input [128, H, H]: outputs [128, newH, newH]
  //   The maxpool output is then truncated/sliced to 64ch for ADD.
  //   Or alternatively, the maxpool is on a different tensor.
  //
  // Wait, a simpler explanation: maybe the MaxPool for these transitions pools
  // to [128, newH, newH], then the ADD is:
  //   ADD(conv_64ch_padded_to_128ch, maxpool_128ch)
  //
  // But there are only 3 channel_padding ops total, all used for stages 1-3.
  // For stages 4-6 (where conv outputs 64ch), there must be different handling.
  //
  // I think the most likely explanation given the TFLite analysis is:
  // For down 4-6, the maxpool output is taken as only the first 64 channels
  // (a slice/gather op that shows up as something else), OR the conv output
  // is padded to 128ch (but we don't have padding ops for these).
  //
  // Let me just trust the weight shapes and architecture description:
  // Down 4: conv2d_120 [64, 2, 2, 128] outputs 64ch. MaxPool gives 128ch.
  // ADD(64ch, 64ch) -> 64ch. So maxpool must be reduced to 64ch somehow.
  //
  // The simplest explanation: the maxpool output is sliced to the first 64 channels.
  // In TFLite, this might be done via a strided slice or gather op that I'm not
  // counting in the op list (DEQUANTIZE ops perhaps, or it's implicit).
  //
  // For implementation, I'll maxpool the full input, then take only the first
  // outCh channels for the ADD. Since our maxpool shader can output all channels,
  // and the ADD shader just adds element-wise, I need to make sure I only ADD
  // the right number of elements.
  //
  // Actually, we can just maxpool only the first `outCh` channels. Let's do that.

  // Down 4: conv 128->64, maxpool first 64ch only, ADD
  emitDownsample(64, null); // null means no padding, maxpool channels = outCh
  // Wait, this is wrong. Our maxpool shader pools ALL channels. And the ADD
  // works on outCh * H * W elements. If maxpool wrote 128ch but we only ADD
  // 64ch * H * W, we'd be fine because the ADD only reads that many elements.
  // But the maxpool output buffer has channels 0..127 contiguous in CHW layout,
  // so the first 64ch are at indices [0, 64*H*W). The ADD would correctly add
  // these with the conv output's 64ch.
  //
  // Actually yes! Since both buffers are in CHW layout:
  //   conv output [64, 8, 8]: channels 0..63, each 8*8 = 64 values
  //   maxpool output [128, 8, 8]: channels 0..127, each 8*8 = 64 values
  //   ADD of first 64*8*8 = 4096 elements picks channels 0..63 from both.
  // So the ADD naturally truncates the maxpool to the first 64 channels!
  //
  // Perfect. So emitDownsample with padCh=null means: maxpool all channels,
  // ADD only the first outCh * H * H elements. Since the conv output has outCh
  // channels and the ADD count is outCh * H * H, this works correctly.
  //
  // But wait, we need to ensure the maxpool shader runs on the full 128ch input.
  // Currently emitDownsample passes inCh as the channel count to maxpool.
  // And ADD uses outCh * H * H elements. This should work!

  // Stage 5: 5 blocks, 128ch, bottleneck 64
  emitFirstBlock(128, 64, 8);
  for (let i = 0; i < 4; i++) {
    emitResBlock(128, 64, 8, true);
  }

  // Down 5: 128->64, no pad
  emitDownsample(64, null);

  // Stage 6: 5 blocks, 128ch, bottleneck 64
  emitFirstBlock(128, 64, 4);
  for (let i = 0; i < 4; i++) {
    emitResBlock(128, 64, 4, true);
  }

  // Down 6: 128->64, no pad
  emitDownsample(64, null);

  // Stage 7: 5 blocks, 128ch, bottleneck 64
  emitFirstBlock(128, 64, 2);
  for (let i = 0; i < 4; i++) {
    emitResBlock(128, 64, 2, true);
  }

  // ============ OUTPUT HEADS ============
  // After stage 7: cur has [128, 2, 2]

  // Landmarks: conv2d_150 [1434, 2, 2, 128] + bias [1434]
  const lmConvBuf = uploadF32(fw('conv2d_150', 'Conv2D').data);
  const lmBiasBuf = uploadF32(fw('conv2d_150', 'BiasAdd').data);
  const lmU = makeU([128, 1434]);
  emit(pipeOutConv, bind(L5, [cur, lmConvBuf, lmBiasBuf, bufLM, lmU]),
    [C64(1434), 1, 1]);

  // Presence: conv2d_152 [1, 2, 2, 128] + bias [1]
  const presConvBuf = uploadF32(fw('conv2d_152', 'Conv2D').data);
  const presBiasBuf = uploadF32(fw('conv2d_152', 'BiasAdd').data);
  const presU = makeU([128, 1]);
  emit(pipeOutConv, bind(L5, [cur, presConvBuf, presBiasBuf, bufPres, presU]),
    [1, 1, 1]);

  // Sigmoid on presence
  const sigU = makeU([1]);
  emit(pipeSigmoid, bind(L3sig, [bufPres, bufPresSig, sigU]),
    [1, 1, 1]);

  // ============ Run & Readback ============
  const INPUT_BYTES = 3 * 256 * 256 * 4;

  function run(inputBuffer: GPUBuffer, encoder: GPUCommandEncoder): void {
    // Copy input CHW data into bufA
    encoder.copyBufferToBuffer(inputBuffer, 0, bufA, 0, INPUT_BYTES);

    const pass = encoder.beginComputePass();
    for (const c of cmds) {
      pass.setPipeline(c.pipe);
      pass.setBindGroup(0, c.bg);
      pass.dispatchWorkgroups(...c.wg);
    }
    pass.end();

    // Copy results to readback
    encoder.copyBufferToBuffer(bufLM, 0, readbackBuf, 0, 1434 * 4);
    encoder.copyBufferToBuffer(bufPresSig, 0, readbackBuf, 1434 * 4, 4);
  }

  async function readback(): Promise<LandmarkOutput> {
    await readbackBuf.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(readbackBuf.getMappedRange().slice(0));
    readbackBuf.unmap();

    return {
      landmarks: data.subarray(0, 1434),
      score: data[1434]!,
    };
  }

  return {
    device,
    run,
    readback,
    inputBufferSize: INPUT_BYTES,
  };
}
