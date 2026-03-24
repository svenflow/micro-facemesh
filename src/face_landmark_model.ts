/**
 * Face Landmark Model interface and weight loading.
 *
 * The face landmark model takes a 256x256 cropped face image and outputs:
 * - 478 landmarks (x, y, z) = 1434 values
 * - 1 face presence score (sigmoid)
 *
 * Model architecture: MediaPipe FaceMesh V2 (details in the compiled model).
 * This file defines the interface consumed by facemesh.ts.
 *
 * Weight files:
 * - face_landmarks_weights.json + face_landmarks_weights.bin (f32)
 * - face_landmarks_weights_f16.json + face_landmarks_weights_f16.bin (f16)
 */

export interface Tensor {
  data: Float32Array;
  shape: number[];
}

export interface WeightsMetadata {
  keys: string[];
  shapes: number[][];
  offsets: number[];
  dtype?: 'float32' | 'float16';
}

export interface FaceLandmarksOutput {
  /** 478 landmarks x 3 (x, y, z) = 1434 values, normalized to [0,1] in crop space */
  landmarks: Float32Array;
  /** Face presence score (1 value, raw — apply sigmoid externally) */
  facePresence: Float32Array;
}

export interface CompiledFaceLandmarkModel {
  device: GPUDevice;
  run: (input: Float32Array) => Promise<FaceLandmarksOutput>;
  runFromCanvas: (source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap) => Promise<FaceLandmarksOutput>;
  runFromGPUBuffer: (inputBuffer: GPUBuffer) => Promise<FaceLandmarksOutput>;
}

/**
 * Load face landmark weights from JSON metadata + binary buffer.
 * Handles duplicate keys by appending shape suffix for disambiguation.
 */
export function loadFaceLandmarkWeights(
  metadata: WeightsMetadata,
  buffer: ArrayBuffer,
): Map<string, Tensor> {
  const weights = new Map<string, Tensor>();
  const dtype = metadata.dtype ?? 'float32';
  const keyCounts = new Map<string, number>();

  for (let i = 0; i < metadata.keys.length; i++) {
    const baseKey = metadata.keys[i]!;
    const shape = metadata.shapes[i]!;
    const offset = metadata.offsets[i]!;
    const size = shape.reduce((a, b) => a * b, 1);

    let data: Float32Array;
    if (dtype === 'float32') {
      data = new Float32Array(buffer, offset, size);
    } else {
      // float16 → float32 conversion
      const view = new DataView(buffer);
      data = new Float32Array(size);
      for (let j = 0; j < size; j++) {
        data[j] = float16ToFloat32(view.getUint16(offset + j * 2, true));
      }
    }

    // Handle duplicate keys by appending shape suffix
    const count = keyCounts.get(baseKey) ?? 0;
    keyCounts.set(baseKey, count + 1);
    const key = count === 0 ? baseKey : `${baseKey}_${shape.join('x')}`;
    weights.set(key, { data, shape });
  }

  return weights;
}

/** Convert a float16 (stored as uint16) to float32 */
function float16ToFloat32(h: number): number {
  const sign = (h >> 15) & 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;

  if (exp === 0) {
    if (frac === 0) return sign ? -0 : 0;
    // Subnormal
    const val = frac / 1024;
    return (sign ? -1 : 1) * val * Math.pow(2, -14);
  }
  if (exp === 0x1f) {
    return frac === 0 ? (sign ? -Infinity : Infinity) : NaN;
  }

  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

/**
 * Compile the face landmark model for WebGPU inference.
 *
 * Wraps the low-level landmark_model.ts implementation with the
 * CompiledFaceLandmarkModel interface expected by facemesh.ts.
 */
export async function compileFaceLandmarkModel(
  weights: Map<string, Tensor>,
  existingDevice?: GPUDevice,
): Promise<CompiledFaceLandmarkModel> {
  const { compileLandmarkModel } = await import('./landmark_model.js');
  type LMTensor = import('./landmark_model.js').Tensor;

  // Convert weights from our Tensor type to landmark_model's Tensor type
  // (they're structurally identical but TypeScript needs explicit mapping)
  const lmWeights = new Map<string, LMTensor>();
  for (const [k, v] of weights) {
    lmWeights.set(k, { data: v.data, shape: v.shape });
  }

  const device = existingDevice ?? (await (async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('WebGPU adapter not available');
    return adapter.requestDevice();
  })());

  const compiled = await compileLandmarkModel(device, lmWeights);

  async function runFromFloat32(input: Float32Array): Promise<FaceLandmarksOutput> {
    const inputBuf = device.createBuffer({
      size: input.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(inputBuf.getMappedRange()).set(input);
    inputBuf.unmap();

    const encoder = device.createCommandEncoder();
    compiled.run(inputBuf, encoder);
    device.queue.submit([encoder.finish()]);

    const result = await compiled.readback();
    inputBuf.destroy();

    return {
      landmarks: result.landmarks,
      facePresence: new Float32Array([result.score]),
    };
  }

  async function runFromGPUBuffer(inputBuffer: GPUBuffer): Promise<FaceLandmarksOutput> {
    const encoder = device.createCommandEncoder();
    compiled.run(inputBuffer, encoder);
    device.queue.submit([encoder.finish()]);

    const result = await compiled.readback();
    return {
      landmarks: result.landmarks,
      facePresence: new Float32Array([result.score]),
    };
  }

  async function runFromCanvas(source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap): Promise<FaceLandmarksOutput> {
    const w = 256, h = 256;
    const canvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(w, h) : document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d')!;
    (ctx as CanvasRenderingContext2D).drawImage(source as CanvasImageSource, 0, 0, w, h);
    const imageData = (ctx as CanvasRenderingContext2D).getImageData(0, 0, w, h);

    const chw = new Float32Array(3 * w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * 4;
        chw[0 * w * h + y * w + x] = imageData.data[idx]! / 255;
        chw[1 * w * h + y * w + x] = imageData.data[idx + 1]! / 255;
        chw[2 * w * h + y * w + x] = imageData.data[idx + 2]! / 255;
      }
    }

    return runFromFloat32(chw);
  }

  return {
    device: compiled.device,
    run: runFromFloat32,
    runFromCanvas,
    runFromGPUBuffer,
  };
}
