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
    detectorMetaRes.json() as Promise<Array<{ key: string; shape: number[]; offset: number; size: number }>>,
    detectorBinRes.arrayBuffer(),
  ]);

  const landmarkWeights = loadFaceLandmarkWeights(landmarkMeta, landmarkBuf);

  // Load detector weights using its own format
  const detectorWeights = new Map<string, { data: Float32Array; shape: number[] }>();
  for (const entry of detectorMeta) {
    const data = new Float32Array(detectorBuf, entry.offset, entry.size / 4);
    detectorWeights.set(entry.key, { data, shape: entry.shape });
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

  /**
   * Run landmark inference for a single ROI. Shared between detection and tracking paths.
   * Returns landmarks + score, or null if face presence score is below threshold.
   */
  async function runLandmarkForROI(
    pxROI: { centerXpx: number; centerYpx: number; sizePx: number; rotation: number },
    srcTexture: GPUTexture,
    srcWidth: number, srcHeight: number,
    cropPipeline: CropPipeline, cropOutputBuf: GPUBuffer,
    isTracking = false,
  ): Promise<{ landmarks: Landmark[]; score: number } | null> {
    // Compute affine transform: crop pixel → source normalized [0,1]
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

    const encoder = cropDevice.createCommandEncoder();
    cropPipeline.crop(
      encoder, srcTexture, cropOutputBuf,
      [a, b, tx, c, d, ty],
      srcWidth, srcHeight, LANDMARK_SIZE,
    );
    cropDevice.queue.submit([encoder.finish()]);

    // Run landmark model on the cropped face
    const output: FaceLandmarksOutput = await landmarkModel.runFromGPUBuffer(cropOutputBuf);
    const presenceScore = sigmoid(output.facePresence[0]!);

    // Use a lower threshold for tracking frames to avoid dropping tracked faces
    // unnecessarily. Face detection will re-acquire if truly lost.
    const effectiveThreshold = isTracking ? Math.min(scoreThreshold, 0.1) : scoreThreshold;
    if (presenceScore < effectiveThreshold) return null;

    // Convert 478 landmarks from crop space to original image coordinates
    const landmarks: Landmark[] = [];
    for (let i = 0; i < NUM_LANDMARKS; i++) {
      const lx = output.landmarks[i * 3]!;
      const ly = output.landmarks[i * 3 + 1]!;
      const lz = output.landmarks[i * 3 + 2]!;

      // Transform from crop-normalized [0,1] to pixel offset from center
      const dx = (lx - 0.5) * pxROI.sizePx;
      const dy = (ly - 0.5) * pxROI.sizePx;

      // Rotate back to original image space
      const origXpx = cosR * dx - sinR * dy + pxROI.centerXpx;
      const origYpx = sinR * dx + cosR * dy + pxROI.centerYpx;

      landmarks.push({
        x: origXpx / srcWidth,
        y: origYpx / srcHeight,
        z: lz,
      });
    }

    return { landmarks, score: presenceScore };
  }

  async function detect(source: FacemeshInput): Promise<FacemeshResult[]> {
    // Get source dimensions and prepare upload source.
    // HTMLVideoElement and HTMLImageElement can be passed directly to
    // copyExternalImageToTexture — the browser handles orientation internally.
    // Only fall back to createImageBitmap for ImageData (which can't be copied directly).
    let srcWidth: number;
    let srcHeight: number;
    let uploadSource: HTMLCanvasElement | OffscreenCanvas | ImageBitmap | HTMLVideoElement | HTMLImageElement;

    if (source instanceof HTMLVideoElement) {
      srcWidth = source.videoWidth;
      srcHeight = source.videoHeight;
      uploadSource = source;
    } else if (source instanceof HTMLImageElement) {
      srcWidth = source.naturalWidth;
      srcHeight = source.naturalHeight;
      uploadSource = source;
    } else if (source instanceof ImageData) {
      const bmp = await createImageBitmap(source, { colorSpaceConversion: 'none' });
      [srcWidth, srcHeight] = [bmp.width, bmp.height];
      uploadSource = bmp;
    } else {
      [srcWidth, srcHeight] = getSourceDimensions(source);
      uploadSource = source as HTMLCanvasElement | OffscreenCanvas | ImageBitmap;
    }

    // Upload source to GPU texture (shared by both tracking and detection paths)
    const cropPipeline = ensureCropPipeline();
    const cropOutputBuf = ensureCropOutputBuffer();
    const srcTexture = ensureCropSourceTexture(srcWidth, srcHeight);
    cropDevice.queue.copyExternalImageToTexture(
      { source: uploadSource },
      { texture: srcTexture },
      [srcWidth, srcHeight],
    );

    // ---- TRACKING PATH ----
    // If we have previous landmarks, try to track using landmark-derived ROI
    // (skip face detection — matches MediaPipe's approach for smooth, fast tracking)
    if (trackedFaces.length > 0) {
      const results: FacemeshResult[] = [];

      for (const tracked of trackedFaces) {
        // Compute ROI from previous landmarks (MediaPipe's landmark-to-ROI path)
        const pxROI = landmarksToROI(tracked.landmarks, srcWidth, srcHeight);

        const result = await runLandmarkForROI(
          pxROI, srcTexture, srcWidth, srcHeight, cropPipeline, cropOutputBuf, true,
        );

        if (result) {
          // Use first 6 keypoints from landmarks corresponding to detector keypoints:
          // rightEye(0→lm133), leftEye(1→lm362), noseTip(2→lm1),
          // mouthCenter(3→lm13), rightEarTragion(4→lm234), leftEarTragion(5→lm454)
          const detectorLandmarkIndices = [133, 362, 1, 13, 234, 454];
          const faceKps = detectorLandmarkIndices.map(idx => result.landmarks[idx]!);

          results.push({
            score: result.score,
            landmarks: result.landmarks,
            keypoints: toFaceKeypoints(faceKps),
          });
        }
        // If face presence < threshold, this tracked face is lost — don't add to results
      }

      if (results.length > 0) {
        // Tracking succeeded — update tracked faces for next frame
        trackedFaces = results.map(r => ({ landmarks: r.landmarks }));
        return results;
      }

      // All tracked faces lost — fall through to face detection
      trackedFaces = [];
    }

    // ---- DETECTION PATH ----
    // No tracked faces (first frame or tracking lost) — run face detection
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
        // Map detector keypoint indices to landmark indices for keypoints
        const detectorLandmarkIndices = [133, 362, 1, 13, 234, 454];
        const faceKps = detectorLandmarkIndices.map(idx => result.landmarks[idx]!);

        results.push({
          score: result.score,
          landmarks: result.landmarks,
          keypoints: toFaceKeypoints(faceKps),
        });
      }
    }

    // Store for tracking on next frame
    trackedFaces = results.map(r => ({ landmarks: r.landmarks }));

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
