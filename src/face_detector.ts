/**
 * Face detection post-processing: anchor generation, decode, NMS, and crop ROI.
 *
 * BlazeFace short-range detector outputs:
 * - 896 anchors total (512 from 16x16 + 384 from 8x8)
 * - Per anchor: 1 score + 16 regression values (4 box + 6 keypoints x 2)
 *
 * All operations run on CPU (JavaScript) — fast enough for 896 anchors.
 */

import type { CompiledFaceDetectorModel, FaceDetectionOutput } from './face_detector_model.js';

export interface FaceDetection {
  /** Confidence score (sigmoid of raw logit) */
  score: number;
  /** Bounding box in normalized [0,1] coords: [center_x, center_y, width, height] */
  box: [number, number, number, number];
  /** 6 keypoints in normalized [0,1] coords: [[x,y], ...]
   * 0: right eye, 1: left eye, 2: nose tip,
   * 3: mouth center, 4: right ear tragion, 5: left ear tragion */
  keypoints: [number, number][];
}

export interface FaceROI {
  /** Center of crop region in original image coords [0,1] */
  centerX: number;
  centerY: number;
  /** Size of crop region in original image coords [0,1] */
  width: number;
  height: number;
  /** Rotation angle in radians (from eye midpoint to nose, aligned to 90 degrees) */
  rotation: number;
}

// ============ Anchor Generation ============

interface Anchor {
  x: number;  // center x in [0,1]
  y: number;  // center y in [0,1]
}

/**
 * Generate SSD anchors for BlazeFace short-range detector.
 *
 * Layer 0: 16x16 grid, 2 anchors per cell (stride 8, relative to 128)
 * Layer 1: 8x8 grid, 6 anchors per cell (stride 16, relative to 128)
 *
 * Anchor positions are at grid cell centers. No anchor size needed since
 * the model regresses offsets from centers directly.
 *
 * Total: 16*16*2 + 8*8*6 = 512 + 384 = 896 anchors
 */
function generateAnchors(): Anchor[] {
  const anchors: Anchor[] = [];

  // Layer 0: 16x16, 2 anchors per cell
  for (let y = 0; y < 16; y++) {
    for (let x = 0; x < 16; x++) {
      const cx = (x + 0.5) / 16;
      const cy = (y + 0.5) / 16;
      for (let a = 0; a < 2; a++) {
        anchors.push({ x: cx, y: cy });
      }
    }
  }

  // Layer 1: 8x8, 6 anchors per cell
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      const cx = (x + 0.5) / 8;
      const cy = (y + 0.5) / 8;
      for (let a = 0; a < 6; a++) {
        anchors.push({ x: cx, y: cy });
      }
    }
  }

  return anchors;
}

// Pre-generate anchors (same for every inference)
const ANCHORS = generateAnchors();

// ============ Decode + NMS ============

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Decode raw model output into face detections.
 *
 * Each anchor has:
 * - 1 score (logit -> sigmoid)
 * - 16 regression values:
 *   [0-3]: center_x_offset, center_y_offset, width, height (in pixels relative to 128)
 *   [4-15]: 6 keypoints x 2 (x_offset, y_offset) (in pixels relative to 128)
 *
 * Offsets are relative to anchor centers, scaled by 128 (input size).
 */
function decodeDetections(
  output: FaceDetectionOutput,
  scoreThreshold: number,
): FaceDetection[] {
  const detections: FaceDetection[] = [];
  const { scores, regressors } = output;
  const inputSize = 128;

  for (let i = 0; i < ANCHORS.length; i++) {
    const score = sigmoid(scores[i]);
    if (score < scoreThreshold) continue;

    const anchor = ANCHORS[i];
    const regBase = i * 16;

    // Decode box: offsets are in pixel space relative to anchor center
    const cx = anchor.x + regressors[regBase + 0] / inputSize;
    const cy = anchor.y + regressors[regBase + 1] / inputSize;
    const w = regressors[regBase + 2] / inputSize;
    const h = regressors[regBase + 3] / inputSize;

    // Decode 6 keypoints
    const keypoints: [number, number][] = [];
    for (let k = 0; k < 6; k++) {
      const kx = anchor.x + regressors[regBase + 4 + k * 2] / inputSize;
      const ky = anchor.y + regressors[regBase + 4 + k * 2 + 1] / inputSize;
      keypoints.push([kx, ky]);
    }

    detections.push({
      score,
      box: [cx, cy, w, h],
      keypoints,
    });
  }

  return detections;
}

/**
 * Weighted non-maximum suppression matching MediaPipe's WEIGHTED algorithm.
 *
 * For each remaining top-scoring detection, finds all overlapping detections
 * above the IoU threshold and computes a score-weighted average of their
 * bounding box coordinates and keypoints. This produces smoother, more
 * accurate detections than greedy NMS.
 *
 * Reference: mediapipe/calculators/util/non_max_suppression_calculator.cc
 */
function nms(detections: FaceDetection[], iouThreshold: number): FaceDetection[] {
  if (detections.length === 0) return [];

  // Sort by score descending
  const sorted = [...detections].sort((a, b) => b.score - a.score);
  const kept: FaceDetection[] = [];
  const suppressed = new Set<number>();

  for (let i = 0; i < sorted.length; i++) {
    if (suppressed.has(i)) continue;

    // Collect all overlapping detections (including self)
    const cluster: number[] = [i];
    for (let j = i + 1; j < sorted.length; j++) {
      if (suppressed.has(j)) continue;
      if (computeIoU(sorted[i]!, sorted[j]!) > iouThreshold) {
        cluster.push(j);
        suppressed.add(j);
      }
    }

    // Compute score-weighted average of box and keypoints across cluster
    let totalWeight = 0;
    let avgCx = 0, avgCy = 0, avgW = 0, avgH = 0;
    const avgKps: [number, number][] = [];
    for (let k = 0; k < 6; k++) avgKps.push([0, 0]);

    for (const idx of cluster) {
      const det = sorted[idx]!;
      const w = det.score;
      totalWeight += w;
      avgCx += det.box[0] * w;
      avgCy += det.box[1] * w;
      avgW += det.box[2] * w;
      avgH += det.box[3] * w;
      for (let k = 0; k < 6; k++) {
        avgKps[k]![0] += det.keypoints[k]![0] * w;
        avgKps[k]![1] += det.keypoints[k]![1] * w;
      }
    }

    const invW = 1 / totalWeight;
    kept.push({
      score: sorted[i]!.score, // Keep top score (not averaged)
      box: [avgCx * invW, avgCy * invW, avgW * invW, avgH * invW],
      keypoints: avgKps.map(([x, y]) => [x * invW, y * invW] as [number, number]),
    });
  }

  return kept;
}

function computeIoU(a: FaceDetection, b: FaceDetection): number {
  // Convert center format to corner format
  const ax1 = a.box[0] - a.box[2] / 2;
  const ay1 = a.box[1] - a.box[3] / 2;
  const ax2 = a.box[0] + a.box[2] / 2;
  const ay2 = a.box[1] + a.box[3] / 2;

  const bx1 = b.box[0] - b.box[2] / 2;
  const by1 = b.box[1] - b.box[3] / 2;
  const bx2 = b.box[0] + b.box[2] / 2;
  const by2 = b.box[1] + b.box[3] / 2;

  const ix1 = Math.max(ax1, bx1);
  const iy1 = Math.max(ay1, by1);
  const ix2 = Math.min(ax2, bx2);
  const iy2 = Math.min(ay2, by2);

  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const intersection = iw * ih;

  const aArea = (ax2 - ax1) * (ay2 - ay1);
  const bArea = (bx2 - bx1) * (by2 - by1);
  const union = aArea + bArea - intersection;

  return union > 0 ? intersection / union : 0;
}

// ============ ROI Computation ============

/**
 * Convert a face detection to a face crop ROI.
 *
 * Uses the eye keypoints to determine face orientation. The crop is rotated
 * so the face is upright, and scaled by 1.5x to include the full head.
 *
 * Keypoint indices in face detection:
 * 0: right eye
 * 1: left eye
 * 2: nose tip
 * 3: mouth center
 * 4: right ear tragion
 * 5: left ear tragion
 */
export function detectionToROI(detection: FaceDetection): FaceROI {
  const [cx, cy, w, h] = detection.box;

  // Compute rotation from eye midpoint to image vertical
  const rightEye = detection.keypoints[0];
  const leftEye = detection.keypoints[1];

  // Eye midpoint
  const eyeMidX = (rightEye[0] + leftEye[0]) / 2;
  const eyeMidY = (rightEye[1] + leftEye[1]) / 2;

  // Angle from eye midpoint to nose (kp2) — determines face orientation
  const noseTip = detection.keypoints[2];
  const dx = noseTip[0] - eyeMidX;
  const dy = noseTip[1] - eyeMidY;

  // In image coords: +X = right, +Y = down
  // A face looking straight has nose below eye midpoint → angle ≈ π/2
  const angle = Math.atan2(dy, dx);

  // Target angle: nose should point down in the crop = π/2 in image coords
  const targetAngle = Math.PI / 2;
  const rotation = targetAngle - angle;

  // MediaPipe's RectTransformationCalculator:
  // 1. long_side = max(w, h) of the UNSCALED face box
  // 2. Then scale: final_size = long_side * scale
  const longSide = Math.max(w, h);
  const scale = 1.5;
  const size = longSide * scale;

  // Shift in rotated frame toward face center (slight downward shift to include chin)
  const shiftAmount = 0.0 * longSide;
  const cosR = Math.cos(rotation);
  const sinR = Math.sin(rotation);
  const shiftX = shiftAmount * sinR;
  const shiftY = shiftAmount * cosR;

  return {
    centerX: cx + shiftX,
    centerY: cy + shiftY,
    width: size,
    height: size,
    rotation,
  };
}

// ============ Public API ============

export interface FaceDetector {
  /** Run face detection and return ROIs for detected faces */
  detect: (source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap) => Promise<FaceROI[]>;
  /** Run face detection and return raw detections (before ROI conversion) */
  detectRaw: (source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap) => Promise<FaceDetection[]>;
  /** Run face detection with GPU letterbox resize (matches MediaPipe's bilinear exactly) */
  detectRawWithResize: (source: any, srcW: number, srcH: number) => Promise<{ detections: FaceDetection[]; lbPadX: number; lbPadY: number }>;
  /** Run face detection and return raw SSD output tensors (scores + regressors) for debugging */
  detectRawSSD: (source: any, srcW: number, srcH: number) => Promise<{ scores: Float32Array; regressors: Float32Array; lbPadX: number; lbPadY: number }>;
  /** Get the compiled face detector model (for resource sharing) */
  model: CompiledFaceDetectorModel;
}

export interface FaceDetectorOptions {
  /** Minimum confidence score (0-1). Default: 0.5 */
  scoreThreshold?: number;
  /** IoU threshold for NMS. Default: 0.3 */
  nmsThreshold?: number;
  /** Maximum number of faces to detect. Default: 1 */
  maxFaces?: number;
}

/**
 * Create a face detector from a compiled model.
 */
export function createFaceDetector(
  model: CompiledFaceDetectorModel,
  options: FaceDetectorOptions = {},
): FaceDetector {
  const {
    scoreThreshold = 0.5,
    nmsThreshold = 0.3,
    maxFaces = 1,
  } = options;

  async function detect(source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap): Promise<FaceROI[]> {
    const output = await model.run(source);
    const detections = decodeDetections(output, scoreThreshold);
    const filtered = nms(detections, nmsThreshold);
    const limited = filtered.slice(0, maxFaces);
    return limited.map(detectionToROI);
  }

  async function detectRaw(source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap): Promise<FaceDetection[]> {
    const output = await model.run(source);
    const detections = decodeDetections(output, scoreThreshold);
    return nms(detections, nmsThreshold).slice(0, maxFaces);
  }

  async function detectRawWithResize(source: any, srcW: number, srcH: number): Promise<{ detections: FaceDetection[]; lbPadX: number; lbPadY: number }> {
    const { output, lbPadX, lbPadY } = await model.runWithResize(source, srcW, srcH);
    const detections = decodeDetections(output, scoreThreshold);
    return { detections: nms(detections, nmsThreshold).slice(0, maxFaces), lbPadX, lbPadY };
  }

  async function detectRawSSD(source: any, srcW: number, srcH: number): Promise<{ scores: Float32Array; regressors: Float32Array; lbPadX: number; lbPadY: number }> {
    const { output, lbPadX, lbPadY } = await model.runWithResize(source, srcW, srcH);
    return { scores: output.scores, regressors: output.regressors, lbPadX, lbPadY };
  }

  return { detect, detectRaw, detectRawWithResize, detectRawSSD, model };
}

/**
 * Compute the affine transform matrix for cropping a face region.
 *
 * Returns a 2x3 matrix [a, b, tx, c, d, ty] that maps from crop space [0,256]
 * to original image space [0,1] (normalized).
 *
 * Usage: originalX = a * cropX + b * cropY + tx
 *        originalY = c * cropX + d * cropY + ty
 */
export function computeCropTransform(roi: FaceROI, cropSize: number = 256): {
  forward: [number, number, number, number, number, number];  // crop -> original
  inverse: [number, number, number, number, number, number];  // original -> crop
} {
  const cos = Math.cos(roi.rotation);
  const sin = Math.sin(roi.rotation);
  const sx = roi.width / cropSize;
  const sy = roi.height / cropSize;

  const a = sx * cos;
  const b = sx * sin;
  const c = -sy * sin;
  const d = sy * cos;
  const tx = roi.centerX - (a * cropSize / 2 + b * cropSize / 2);
  const ty = roi.centerY - (c * cropSize / 2 + d * cropSize / 2);

  // Inverse: original [0,1] -> crop [0,cropSize]
  const det = a * d - b * c;
  const ia = d / det;
  const ib = -b / det;
  const ic = -c / det;
  const id = a / det;
  const itx = -(ia * tx + ib * ty);
  const ity = -(ic * tx + id * ty);

  return {
    forward: [a, b, tx, c, d, ty],
    inverse: [ia, ib, itx, ic, id, ity],
  };
}

/**
 * Project landmarks from crop space back to original image coordinates.
 *
 * @param landmarks Array of {x, y, z} in crop space [0, 1] (from 256x256 crop)
 * @param roi The face ROI used for cropping
 * @param srcWidth Original image width in pixels
 * @param srcHeight Original image height in pixels
 * @returns Array of {x, y, z} in original image space [0, 1]
 */
export function projectLandmarksToOriginal(
  landmarks: Array<{ x: number; y: number; z: number }>,
  roi: FaceROI,
  srcWidth: number,
  srcHeight: number,
): Array<{ x: number; y: number; z: number }> {
  const cos = Math.cos(roi.rotation);
  const sin = Math.sin(roi.rotation);
  const refDim = Math.min(srcWidth, srcHeight);
  const physicalSize = roi.width * refDim;
  const wx = physicalSize / srcWidth;
  const wy = physicalSize / srcHeight;

  return landmarks.map(lm => {
    const dx = lm.x - 0.5;
    const dy = lm.y - 0.5;
    const rx = cos * dx + sin * dy;
    const ry = -sin * dx + cos * dy;
    return {
      x: rx * wx + roi.centerX,
      y: ry * wy + roi.centerY,
      z: lm.z,
    };
  });
}
