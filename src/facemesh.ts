/**
 * Main facemesh pipeline: face detector + landmark model + ROI tracking.
 *
 * Pipeline per frame:
 * 1. First frame: Run face detector (128x128) → get detections with 6 keypoints → compute ROI
 * 2. Affine crop face region → 256x256 (via crop_shader)
 * 3. Run landmark model → get 478 landmarks + presence score
 * 4. If presence > threshold: convert landmarks to original coords, use for next ROI
 * 5. Subsequent frames: Use previous landmarks → compute ROI → skip detector
 * 6. If landmarks lost (score < threshold): fall back to detector
 *
 * Follows the exact same pattern as micro-handpose/src/handpose.ts.
 */

import { compileFaceDetectorModel, loadFaceDetectorWeights } from './face_detector_model.js';
import type { CompiledFaceDetectorModel } from './face_detector_model.js';
import { createFaceDetector, detectionToROI } from './face_detector.js';
import type { FaceDetection, FaceROI } from './face_detector.js';
import type { CompiledFaceLandmarkModel, FaceLandmarksOutput, WeightsMetadata } from './face_landmark_model.js';
import { loadFaceLandmarkWeights } from './face_landmark_model.js';
import { createCropPipeline } from './crop_shader.js';
import type { CropPipeline } from './crop_shader.js';
import type { Facemesh, FacemeshInput, FacemeshOptions, FacemeshResult, Landmark } from './types.js';
import { toFaceKeypoints } from './types.js';

// Default: jsdelivr CDN (auto-mirrors npm packages)
const DEFAULT_WEIGHTS_BASE = 'https://cdn.jsdelivr.net/npm/@svenflow/micro-facemesh@0.1.0/weights';

// Face landmark model input size
const LANDMARK_SIZE = 256;

// Number of landmarks output by the face mesh model
const NUM_LANDMARKS = 478;

// Landmark indices for ROI computation from landmarks
const LEFT_EYE_OUTER = 33;
const RIGHT_EYE_OUTER = 263;

/**
 * Create a face mesh detector.
 *
 * Downloads model weights and compiles the WebGPU pipelines.
 * Call this once, then use `detect()` repeatedly.
 *
 * @example
 * ```typescript
 * const facemesh = await createFacemesh()
 * const faces = await facemesh.detect(videoElement)
 * for (const face of faces) {
 *   console.log(face.keypoints.leftEye) // {x, y, z}
 *   console.log(face.landmarks.length)  // 478
 * }
 * ```
 */
export async function createFacemesh(options: FacemeshOptions = {}): Promise<Facemesh> {
  const {
    weightsUrl,
    scoreThreshold = 0.5,
    faceScoreThreshold = 0.5,
    maxFaces = 1,
  } = options;

  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new Error('micro-facemesh requires WebGPU. Check browser support at https://webgpureport.org');
  }

  // Load all weights in parallel from CDN (or custom URL)
  const base = (weightsUrl ?? DEFAULT_WEIGHTS_BASE).replace(/\/$/, '') + '/';

  const [landmarkMetaRes, landmarkBinRes, detectorMetaRes, detectorBinRes] = await Promise.all([
    fetch(`${base}face_landmarks_weights_f16.json`),
    fetch(`${base}face_landmarks_weights_f16.bin`),
    fetch(`${base}face_detector_weights.json`),
    fetch(`${base}face_detector_weights.bin`),
  ]);

  if (!landmarkMetaRes.ok) throw new Error(`Failed to fetch landmark weights: ${landmarkMetaRes.status}`);
  if (!landmarkBinRes.ok) throw new Error(`Failed to fetch landmark weights: ${landmarkBinRes.status}`);
  if (!detectorMetaRes.ok) throw new Error(`Failed to fetch face detector weights: ${detectorMetaRes.status}`);
  if (!detectorBinRes.ok) throw new Error(`Failed to fetch face detector weights: ${detectorBinRes.status}`);

  const [landmarkMeta, landmarkBuf, detectorMeta, detectorBuf] = await Promise.all([
    landmarkMetaRes.json() as Promise<WeightsMetadata>,
    landmarkBinRes.arrayBuffer(),
    detectorMetaRes.json() as Promise<WeightsMetadata>,
    detectorBinRes.arrayBuffer(),
  ]);

  const landmarkWeights = loadFaceLandmarkWeights(landmarkMeta, landmarkBuf);

  // Load detector weights using same keys/shapes/offsets format
  const detectorWeights = new Map<string, { data: Float32Array; shape: number[] }>();
  for (let i = 0; i < detectorMeta.keys.length; i++) {
    const key = detectorMeta.keys[i]!;
    const shape = detectorMeta.shapes[i]!;
    const offset = detectorMeta.offsets[i]!;
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(detectorBuf, offset, size);
    detectorWeights.set(key, { data, shape });
  }

  // Compile face detector model
  const detectorModel = await compileFaceDetectorModel(detectorWeights);
  const faceDetector = createFaceDetector(detectorModel, {
    scoreThreshold: faceScoreThreshold,
    maxFaces,
  });

  // Compile face landmark model (shares the same GPU device as detector)
  // TODO: compileFaceLandmarkModel will be implemented in face_landmark_model.ts
  // For now we import it dynamically to allow the interface to compile
  const { compileFaceLandmarkModel } = await import('./face_landmark_model.js') as any;
  const landmarkModel: CompiledFaceLandmarkModel = await compileFaceLandmarkModel(
    landmarkWeights,
    detectorModel.device,
  );

  // ROI tracking state: previous frame's landmarks per face (for tracking mode)
  // When tracking, we compute the next-frame crop ROI from previous landmarks
  // instead of re-running face detection (matches MediaPipe's approach)
  let trackedFaces: Array<{ landmarks: Landmark[] }> = [];

  /**
   * Compute ROI from previous frame's landmarks (MediaPipe's tracking path).
   *
   * 1. Use landmarks 33 (left eye outer) and 263 (right eye outer) for rotation angle
   * 2. Compute tight bounding box from all 478 landmarks
   * 3. Scale by 1.5x and square it
   */
  function landmarksToROI(
    landmarks: Landmark[], imgW: number, imgH: number,
  ): { centerXpx: number; centerYpx: number; sizePx: number; rotation: number } {
    // Step 1: Compute rotation from eye landmarks
    const leftEyeOuter = landmarks[LEFT_EYE_OUTER]!;
    const rightEyeOuter = landmarks[RIGHT_EYE_OUTER]!;

    const dxPx = (rightEyeOuter.x - leftEyeOuter.x) * imgW;
    const dyPx = (rightEyeOuter.y - leftEyeOuter.y) * imgH;

    // Angle of the line from left eye to right eye
    // For an upright face, eyes are roughly horizontal → angle ≈ 0
    const rotation = Math.atan2(dyPx, dxPx);

    // Step 2: Compute tight bounding box in rotated space
    const cosR = Math.cos(rotation);
    const sinR = Math.sin(rotation);

    let minRx = Infinity, maxRx = -Infinity;
    let minRy = Infinity, maxRy = -Infinity;

    for (let i = 0; i < NUM_LANDMARKS; i++) {
      const lm = landmarks[i]!;
      const px = lm.x * imgW;
      const py = lm.y * imgH;
      // Rotate to aligned space: R(-rotation)
      const rx = cosR * px + sinR * py;
      const ry = -sinR * px + cosR * py;
      minRx = Math.min(minRx, rx);
      maxRx = Math.max(maxRx, rx);
      minRy = Math.min(minRy, ry);
      maxRy = Math.max(maxRy, ry);
    }

    // Bounding box center in rotated space
    const rcx = (minRx + maxRx) / 2;
    const rcy = (minRy + maxRy) / 2;
    const boxW = maxRx - minRx;
    const boxH = maxRy - minRy;

    // Rotate center back to original pixel space
    const centerXpx = cosR * rcx - sinR * rcy;
    const centerYpx = sinR * rcx + cosR * rcy;

    // Step 3: Square the bounding box (use max dimension in pixels)
    const longSidePx = Math.max(boxW, boxH);

    // Scale by 1.5x for padding
    const scale = 1.5;
    const sizePx = longSidePx * scale;

    return { centerXpx, centerYpx, sizePx, rotation };
  }

  /**
   * Compute ROI from a face detection (detector path).
   *
   * Uses keypoints 0 (right eye) and 1 (left eye) for rotation.
   * Scale by 1.5x and square.
   *
   * Returns pixel-based ROI for cropping (center in pixels, size in pixels).
   */
  function detectionToPixelROI(
    det: FaceDetection, imgW: number, imgH: number,
  ): { centerXpx: number; centerYpx: number; sizePx: number; rotation: number } {
    const [cx, cy, w, h] = det.box;

    // Compute rotation from eye keypoints
    const rightEye = det.keypoints[0]!;
    const leftEye = det.keypoints[1]!;

    // Eye midpoint
    const eyeMidX = (rightEye[0] + leftEye[0]) / 2;
    const eyeMidY = (rightEye[1] + leftEye[1]) / 2;

    // Angle from eye midpoint to nose (kp2) — determines face orientation
    const noseTip = det.keypoints[2]!;
    const dx = noseTip[0] - eyeMidX;
    const dy = noseTip[1] - eyeMidY;

    // In image coords: +X = right, +Y = down
    // A face looking straight has nose below eye midpoint → angle ≈ π/2
    const angle = Math.atan2(dy, dx);
    const targetAngle = Math.PI / 2;
    const rawRotation = targetAngle - angle;
    // Normalize to [-PI, PI]
    const rotation = rawRotation - 2 * Math.PI * Math.floor((rawRotation + Math.PI) / (2 * Math.PI));

    // Square the bounding box using max dimension in pixels
    const longSidePx = Math.max(w * imgW, h * imgH);

    // Scale by 1.5x
    const scale = 1.5;
    const sizePx = longSidePx * scale;

    return {
      centerXpx: cx * imgW,
      centerYpx: cy * imgH,
      sizePx,
      rotation,
    };
  }

  // GPU crop resources (lazy-initialized, reused across frames)
  const cropDevice = landmarkModel.device;
  let gpuCropPipeline: CropPipeline | null = null;
  let gpuCropOutputBuffer: GPUBuffer | null = null;
  let gpuCropSourceTexture: GPUTexture | null = null;
  let gpuCropSourceWidth = 0;
  let gpuCropSourceHeight = 0;

  function ensureCropPipeline(): CropPipeline {
    if (!gpuCropPipeline) {
      gpuCropPipeline = createCropPipeline(cropDevice);
    }
    return gpuCropPipeline;
  }

  function ensureCropOutputBuffer(): GPUBuffer {
    if (!gpuCropOutputBuffer) {
      gpuCropOutputBuffer = cropDevice.createBuffer({
        size: 3 * LANDMARK_SIZE * LANDMARK_SIZE * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
    }
    return gpuCropOutputBuffer;
  }

  function ensureCropSourceTexture(width: number, height: number): GPUTexture {
    if (!gpuCropSourceTexture || gpuCropSourceWidth !== width || gpuCropSourceHeight !== height) {
      if (gpuCropSourceTexture) gpuCropSourceTexture.destroy();
      gpuCropSourceTexture = cropDevice.createTexture({
        size: [width, height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      gpuCropSourceWidth = width;
      gpuCropSourceHeight = height;
    }
    return gpuCropSourceTexture;
  }

  // Letterbox padding for removing letterbox from detections
  let lbPadX = 0; // normalized padding on left (and right)
  let lbPadY = 0; // normalized padding on top (and bottom)

  /**
   * Remove letterbox from a face detection.
   * Matches MediaPipe's DetectionLetterboxRemovalCalculator.
   * Converts from letterbox [0,1] coords to image [0,1] coords.
   */
  function removeLetterbox(det: FaceDetection): FaceDetection {
    const sx = 1 / (1 - 2 * lbPadX);
    const sy = 1 / (1 - 2 * lbPadY);
    return {
      score: det.score,
      box: [
        (det.box[0] - lbPadX) * sx,
        (det.box[1] - lbPadY) * sy,
        det.box[2] * sx,
        det.box[3] * sy,
      ],
      keypoints: det.keypoints.map(([kx, ky]) => [
        (kx - lbPadX) * sx,
        (ky - lbPadY) * sy,
      ]),
    };
  }

  function getSourceDimensions(source: FacemeshInput): [number, number] {
    if (source instanceof HTMLCanvasElement || source instanceof OffscreenCanvas) {
      return [source.width, source.height];
    }
    if (typeof ImageBitmap !== 'undefined' && source instanceof ImageBitmap) {
      return [source.width, source.height];
    }
    if (source instanceof ImageData) {
      return [source.width, source.height];
    }
    if (source instanceof HTMLVideoElement) {
      return [source.videoWidth, source.videoHeight];
    }
    if (source instanceof HTMLImageElement) {
      return [source.naturalWidth, source.naturalHeight];
    }
    return [LANDMARK_SIZE, LANDMARK_SIZE];
  }

  /**
   * Sigmoid activation function.
   */
  function sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  // ---- Shared helpers for landmark processing ----

  /** Encode crop + landmark inference into an encoder (no submit). */
  function encodeCropAndLandmark(
    pxROI: { centerXpx: number; centerYpx: number; sizePx: number; rotation: number },
    srcTexture: GPUTexture, srcWidth: number, srcHeight: number,
    cropPipeline: CropPipeline, cropOutputBuf: GPUBuffer,
    encoder: GPUCommandEncoder,
  ): void {
    const cosR = Math.cos(pxROI.rotation);
    const sinR = Math.sin(pxROI.rotation);
    const s = pxROI.sizePx / LANDMARK_SIZE;
    const halfLM = LANDMARK_SIZE / 2;
    const a = cosR * s / srcWidth;
    const b = -sinR * s / srcWidth;
    const tx = pxROI.centerXpx / srcWidth - halfLM * (a + b);
    const c = sinR * s / srcHeight;
    const d = cosR * s / srcHeight;
    const ty = pxROI.centerYpx / srcHeight - halfLM * (c + d);

    cropPipeline.crop(
      encoder, srcTexture, cropOutputBuf,
      [a, b, tx, c, d, ty],
      srcWidth, srcHeight, LANDMARK_SIZE,
    );
    landmarkModel.encodeFromGPUBuffer(cropOutputBuf, encoder);
  }

  /** Convert raw landmark output to Landmark[] in original image coords. */
  function convertLandmarks(
    output: FaceLandmarksOutput,
    pxROI: { centerXpx: number; centerYpx: number; sizePx: number; rotation: number },
    srcWidth: number, srcHeight: number,
  ): Landmark[] {
    const cosR = Math.cos(pxROI.rotation);
    const sinR = Math.sin(pxROI.rotation);
    const landmarks: Landmark[] = [];
    for (let i = 0; i < NUM_LANDMARKS; i++) {
      const lx = output.landmarks[i * 3]!;
      const ly = output.landmarks[i * 3 + 1]!;
      const lz = output.landmarks[i * 3 + 2]!;
      const dx = (lx - 0.5) * pxROI.sizePx;
      const dy = (ly - 0.5) * pxROI.sizePx;
      const origXpx = cosR * dx - sinR * dy + pxROI.centerXpx;
      const origYpx = sinR * dx + cosR * dy + pxROI.centerYpx;
      landmarks.push({ x: origXpx / srcWidth, y: origYpx / srcHeight, z: lz });
    }
    return landmarks;
  }

  /** Build FacemeshResult from landmarks + score. */
  function buildResult(landmarks: Landmark[], score: number): FacemeshResult {
    const detectorLandmarkIndices = [133, 362, 1, 13, 234, 454];
    const faceKps = detectorLandmarkIndices.map(idx => landmarks[idx]!);
    return { score, landmarks, keypoints: toFaceKeypoints(faceKps) };
  }

  /**
   * Run landmark inference for a single ROI (synchronous path).
   * Used by detection path and first tracking frame.
   */
  async function runLandmarkForROI(
    pxROI: { centerXpx: number; centerYpx: number; sizePx: number; rotation: number },
    srcTexture: GPUTexture,
    srcWidth: number, srcHeight: number,
    cropPipeline: CropPipeline, cropOutputBuf: GPUBuffer,
    isTracking = false,
  ): Promise<{ landmarks: Landmark[]; score: number } | null> {
    const encoder = cropDevice.createCommandEncoder();
    encodeCropAndLandmark(pxROI, srcTexture, srcWidth, srcHeight, cropPipeline, cropOutputBuf, encoder);
    cropDevice.queue.submit([encoder.finish()]);

    const output: FaceLandmarksOutput = await landmarkModel.readbackLandmarks();
    const presenceScore = output.facePresence[0]!;

    const effectiveThreshold = isTracking ? Math.min(scoreThreshold, 0.1) : scoreThreshold;
    if (presenceScore < effectiveThreshold) return null;

    return { landmarks: convertLandmarks(output, pxROI, srcWidth, srcHeight), score: presenceScore };
  }

  // ---- Double-buffered pipeline state (persists across detect() calls) ----
  // When pipelined, GPU processes current frame while we return previous frame's results.
  // This hides the mapAsync latency (~3-5ms on mobile) behind the inter-frame gap.
  let pipelinedReadback: Promise<FaceLandmarksOutput> | null = null;
  let pipelinedROI: { centerXpx: number; centerYpx: number; sizePx: number; rotation: number } | null = null;
  let pipelinedSrcDims: [number, number] | null = null;
  let pipelinedResults: FacemeshResult[] | null = null; // cached results from previous frame

  /** Reset pipeline state (on tracking loss or detection). */
  function resetPipeline(): void {
    pipelinedReadback = null;
    pipelinedROI = null;
    pipelinedSrcDims = null;
    pipelinedResults = null;
  }

  async function detect(source: FacemeshInput): Promise<FacemeshResult[]> {
    // Get source dimensions. For video/image elements, start createImageBitmap
    // as a promise (don't await yet) so it can overlap with pipelined readback.
    // iOS Safari WebGPU requires ImageBitmap for copyExternalImageToTexture.
    let srcWidth: number;
    let srcHeight: number;
    let uploadSourceOrPromise: HTMLCanvasElement | OffscreenCanvas | ImageBitmap | Promise<ImageBitmap>;

    if (source instanceof HTMLVideoElement) {
      srcWidth = source.videoWidth;
      srcHeight = source.videoHeight;
      // Start bitmap creation NOW — will overlap with readback await below
      uploadSourceOrPromise = createImageBitmap(source, { colorSpaceConversion: 'none' });
    } else if (source instanceof HTMLImageElement) {
      srcWidth = source.naturalWidth;
      srcHeight = source.naturalHeight;
      uploadSourceOrPromise = createImageBitmap(source, { colorSpaceConversion: 'none' });
    } else if (source instanceof ImageData) {
      uploadSourceOrPromise = createImageBitmap(source, { colorSpaceConversion: 'none' });
      // Need dimensions from source directly
      srcWidth = source.width;
      srcHeight = source.height;
    } else {
      [srcWidth, srcHeight] = getSourceDimensions(source);
      uploadSourceOrPromise = source as HTMLCanvasElement | OffscreenCanvas | ImageBitmap;
    }

    // ---- PIPELINED TRACKING PATH ----
    // If we have a pending readback from the previous frame, await it (should be
    // near-instant since GPU had the full inter-frame gap to finish), process
    // previous results, and submit current frame's work non-blocking.
    // Key: createImageBitmap runs in parallel with this readback await!
    if (trackedFaces.length > 0 && pipelinedReadback) {
      const prevOutput = await pipelinedReadback;
      pipelinedReadback = null;

      const prevROI = pipelinedROI!;
      const [prevW, prevH] = pipelinedSrcDims!;
      const presenceScore = prevOutput.facePresence[0]!;
      const effectiveThreshold = Math.min(scoreThreshold, 0.1);

      if (presenceScore >= effectiveThreshold) {
        // Previous frame succeeded — process its landmarks
        const prevLandmarks = convertLandmarks(prevOutput, prevROI, prevW, prevH);
        const result = buildResult(prevLandmarks, presenceScore);

        // Now await the bitmap (should be ready or nearly ready — ran in parallel)
        const uploadSource = uploadSourceOrPromise instanceof Promise
          ? await uploadSourceOrPromise : uploadSourceOrPromise;

        // Upload and submit CURRENT frame's work
        const cropPipeline = ensureCropPipeline();
        const cropOutputBuf = ensureCropOutputBuffer();
        const srcTexture = ensureCropSourceTexture(srcWidth, srcHeight);
        cropDevice.queue.copyExternalImageToTexture(
          { source: uploadSource }, { texture: srcTexture }, [srcWidth, srcHeight],
        );

        const pxROI = landmarksToROI(prevLandmarks, srcWidth, srcHeight);
        landmarkModel.flipReadbackBuffer();
        const encoder = cropDevice.createCommandEncoder();
        encodeCropAndLandmark(pxROI, srcTexture, srcWidth, srcHeight, cropPipeline, cropOutputBuf, encoder);
        cropDevice.queue.submit([encoder.finish()]);

        // Start non-blocking readback for current frame
        pipelinedReadback = landmarkModel.beginReadbackLandmarks();
        pipelinedROI = pxROI;
        pipelinedSrcDims = [srcWidth, srcHeight];
        pipelinedResults = [result];

        // Update tracking with previous frame's landmarks
        trackedFaces = [{ landmarks: prevLandmarks }];
        return [result];
      }

      // Tracking lost on previous frame — reset and fall through to detection
      resetPipeline();
      trackedFaces = [];
    }

    // For non-pipelined paths, resolve the bitmap now
    const uploadSource = uploadSourceOrPromise instanceof Promise
      ? await uploadSourceOrPromise : uploadSourceOrPromise;

    // Upload source to GPU texture (shared by tracking bootstrap and detection paths)
    const cropPipeline = ensureCropPipeline();
    const cropOutputBuf = ensureCropOutputBuffer();
    const srcTexture = ensureCropSourceTexture(srcWidth, srcHeight);
    cropDevice.queue.copyExternalImageToTexture(
      { source: uploadSource },
      { texture: srcTexture },
      [srcWidth, srcHeight],
    );

    // ---- BOOTSTRAP TRACKING (first tracking frame) ----
    // Submit work and start non-blocking readback. Return cached results
    // from the detection/previous frame (1 frame of latency to start pipeline).
    if (trackedFaces.length > 0 && !pipelinedReadback) {
      const tracked = trackedFaces[0]!;
      const pxROI = landmarksToROI(tracked.landmarks, srcWidth, srcHeight);

      landmarkModel.flipReadbackBuffer();
      const encoder = cropDevice.createCommandEncoder();
      encodeCropAndLandmark(pxROI, srcTexture, srcWidth, srcHeight, cropPipeline, cropOutputBuf, encoder);
      cropDevice.queue.submit([encoder.finish()]);

      // Start non-blocking readback — will be awaited on NEXT detect() call
      pipelinedReadback = landmarkModel.beginReadbackLandmarks();
      pipelinedROI = pxROI;
      pipelinedSrcDims = [srcWidth, srcHeight];

      // Return previous frame's cached results (detection results or last tracking results)
      if (pipelinedResults) {
        return pipelinedResults;
      }

      // No cached results yet (first frame after detection) — await synchronously
      const output = await pipelinedReadback;
      pipelinedReadback = null;
      const score = output.facePresence[0]!;
      const effectiveThreshold = Math.min(scoreThreshold, 0.1);

      if (score >= effectiveThreshold) {
        const landmarks = convertLandmarks(output, pxROI, srcWidth, srcHeight);
        const result = buildResult(landmarks, score);
        trackedFaces = [{ landmarks }];
        pipelinedResults = [result];
        return [result];
      }

      // Tracking failed on first frame — fall through to detection
      resetPipeline();
      trackedFaces = [];
    }

    // ---- DETECTION PATH ----
    // No tracked faces (first frame or tracking lost) — run face detection
    resetPipeline(); // ensure clean state
    const { detections: rawDetections, lbPadX: gpuLbPadX, lbPadY: gpuLbPadY } =
      await faceDetector.detectRawWithResize(uploadSource, srcWidth, srcHeight);
    lbPadX = gpuLbPadX;
    lbPadY = gpuLbPadY;

    if (rawDetections.length === 0) {
      trackedFaces = [];
      return [];
    }

    const results: FacemeshResult[] = [];

    for (const rawDet of rawDetections) {
      const det = removeLetterbox(rawDet);
      const pxROI = detectionToPixelROI(det, srcWidth, srcHeight);

      const result = await runLandmarkForROI(
        pxROI, srcTexture, srcWidth, srcHeight, cropPipeline, cropOutputBuf,
      );

      if (result) {
        results.push(buildResult(result.landmarks, result.score));
      }
    }

    // Store for tracking on next frame
    trackedFaces = results.map(r => ({ landmarks: r.landmarks }));
    pipelinedResults = results; // cache for pipeline bootstrap

    return results;
  }

  function dispose(): void {
    if (gpuCropSourceTexture) gpuCropSourceTexture.destroy();
    if (gpuCropOutputBuffer) gpuCropOutputBuffer.destroy();
    gpuCropSourceTexture = null;
    gpuCropOutputBuffer = null;
    gpuCropPipeline = null;
    landmarkModel.device.destroy();
    detectorModel.device.destroy();
  }

  /** Reset tracking state (call between unrelated images or when face is lost) */
  function reset(): void {
    trackedFaces = [];
  }

  return { detect, dispose, reset };
}
