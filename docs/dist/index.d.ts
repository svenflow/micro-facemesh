/** A 3D landmark point (x, y in [0,1] image coords, z is relative depth) */
interface Landmark {
    x: number;
    y: number;
    z: number;
}
/** Face landmark region names for ergonomic access */
declare const FACE_KEYPOINT_NAMES: readonly ["rightEye", "leftEye", "noseTip", "mouthCenter", "rightEarTragion", "leftEarTragion"];
/** Named face keypoints from the face detector (6 keypoints) */
type FaceKeypoints = {
    [K in (typeof FACE_KEYPOINT_NAMES)[number]]: Landmark;
};
/** Detection result for a single face (full 478-landmark mesh) */
interface FacemeshResult {
    /** Confidence score (0-1) that a face is present */
    score: number;
    /** 478 face landmarks in original image coordinates [0,1] */
    landmarks: Landmark[];
    /** Named detector keypoints for ergonomic access: result.keypoints.leftEye */
    keypoints: FaceKeypoints;
}
/** Options for creating a facemesh detector */
interface FacemeshOptions {
    /** URL to fetch weights from. Defaults to CDN. All weight files must be in this directory. */
    weightsUrl?: string;
    /** Minimum landmark confidence score (0-1). Default: 0.5 */
    scoreThreshold?: number;
    /** Minimum face detection score (0-1). Default: 0.5 */
    faceScoreThreshold?: number;
    /** Maximum number of faces to detect. Default: 1 */
    maxFaces?: number;
}
/** A facemesh detector instance */
interface Facemesh {
    /**
     * Detect face landmarks from a camera frame or image.
     *
     * Accepts: HTMLCanvasElement, OffscreenCanvas, ImageBitmap, HTMLImageElement,
     * HTMLVideoElement, or ImageData.
     *
     * Returns array of detected faces (empty if none found).
     */
    detect: (source: FacemeshInput) => Promise<FacemeshResult[]>;
    /** Dispose GPU resources */
    dispose: () => void;
    /** Reset temporal smoothing state (call between unrelated images/scenes) */
    reset: () => void;
}
/** Accepted input types for detection */
type FacemeshInput = HTMLCanvasElement | OffscreenCanvas | ImageBitmap | HTMLImageElement | HTMLVideoElement | ImageData;
/**
 * Notable landmark indices in the 478-point face mesh.
 *
 * The first 468 landmarks follow MediaPipe's face mesh topology.
 * Landmarks 468-477 are iris landmarks (5 per eye).
 */
declare const FACE_LANDMARK_INDICES: {
    readonly silhouetteTop: 10;
    readonly silhouetteBottom: 152;
    readonly leftEyeInner: 133;
    readonly leftEyeOuter: 33;
    readonly rightEyeInner: 362;
    readonly rightEyeOuter: 263;
    readonly leftEyebrowUpper: 66;
    readonly rightEyebrowUpper: 296;
    readonly noseTip: 1;
    readonly noseBottom: 2;
    readonly noseBridgeTop: 6;
    readonly upperLipTop: 13;
    readonly lowerLipBottom: 14;
    readonly mouthLeft: 61;
    readonly mouthRight: 291;
    readonly chin: 152;
    readonly leftEarTragion: 234;
    readonly rightEarTragion: 454;
    readonly leftIrisCenter: 468;
    readonly leftIrisRight: 469;
    readonly leftIrisTop: 470;
    readonly leftIrisLeft: 471;
    readonly leftIrisBottom: 472;
    readonly rightIrisCenter: 473;
    readonly rightIrisRight: 474;
    readonly rightIrisTop: 475;
    readonly rightIrisLeft: 476;
    readonly rightIrisBottom: 477;
};
/** Type for FACE_LANDMARK_INDICES keys */
type FaceLandmarkName = keyof typeof FACE_LANDMARK_INDICES;

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
 *    Block 7:  56->64, stride 1, 16x16
 *    Block 8:  64->72, stride 1, 16x16
 *    Block 9:  72->80, stride 1, 16x16
 *    Block 10: 80->88, stride 1, 16x16   -> 16x16 SSD head (88ch, 2 anchors)
 *    Block 11: 88->96, stride 2, 16->8   (stride-2 transition)
 *    Block 12: 96->96, stride 1, 8x8
 *    Block 13: 96->96, stride 1, 8x8
 *    Block 14: 96->96, stride 1, 8x8
 *    Block 15: 96->96, stride 1, 8x8     -> 8x8 SSD head (96ch, 6 anchors)
 *
 * SSD outputs:
 *   16x16: 2 classifiers + 32 regressors (2 anchors x 16 values) = 512 anchors
 *   8x8: 6 classifiers + 96 regressors (6 anchors x 16 values) = 384 anchors
 *   Total: 896 anchors, each with 1 score + 16 regression values (4 box + 6 kp x 2)
 *
 * Weight format: JSON metadata (keys, shapes, offsets) + binary float32 file
 */
interface Tensor$1 {
    data: Float32Array;
    shape: number[];
}
interface FaceDetectionOutput {
    scores: Float32Array;
    regressors: Float32Array;
}
interface CompiledFaceDetectorModel {
    device: GPUDevice;
    run: (source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap) => Promise<FaceDetectionOutput>;
    /** Run with GPU-based letterbox resize matching MediaPipe's bilinear interpolation exactly */
    runWithResize: (source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap | HTMLVideoElement | HTMLImageElement, srcW: number, srcH: number) => Promise<{
        output: FaceDetectionOutput;
        lbPadX: number;
        lbPadY: number;
    }>;
}
/**
 * Load face detector weights from a URL directory.
 * Expects: {url}/face_detector.json (metadata) and {url}/face_detector.bin (binary)
 */
declare function loadFaceDetectorWeights(weightsUrl: string): Promise<Map<string, Tensor$1>>;
declare function compileFaceDetectorModel(weights: Map<string, Tensor$1>, existingDevice?: GPUDevice): Promise<CompiledFaceDetectorModel>;

/**
 * Face detection post-processing: anchor generation, decode, NMS, and crop ROI.
 *
 * BlazeFace short-range detector outputs:
 * - 896 anchors total (512 from 16x16 + 384 from 8x8)
 * - Per anchor: 1 score + 16 regression values (4 box + 6 keypoints x 2)
 *
 * All operations run on CPU (JavaScript) — fast enough for 896 anchors.
 */

interface FaceDetection {
    /** Confidence score (sigmoid of raw logit) */
    score: number;
    /** Bounding box in normalized [0,1] coords: [center_x, center_y, width, height] */
    box: [number, number, number, number];
    /** 6 keypoints in normalized [0,1] coords: [[x,y], ...]
     * 0: right eye, 1: left eye, 2: nose tip,
     * 3: mouth center, 4: right ear tragion, 5: left ear tragion */
    keypoints: [number, number][];
}
interface FaceROI {
    /** Center of crop region in original image coords [0,1] */
    centerX: number;
    centerY: number;
    /** Size of crop region in original image coords [0,1] */
    width: number;
    height: number;
    /** Rotation angle in radians (from eye midpoint to nose, aligned to 90 degrees) */
    rotation: number;
}
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
declare function detectionToROI(detection: FaceDetection): FaceROI;
interface FaceDetector {
    /** Run face detection and return ROIs for detected faces */
    detect: (source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap) => Promise<FaceROI[]>;
    /** Run face detection and return raw detections (before ROI conversion) */
    detectRaw: (source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap) => Promise<FaceDetection[]>;
    /** Run face detection with GPU letterbox resize (matches MediaPipe's bilinear exactly) */
    detectRawWithResize: (source: any, srcW: number, srcH: number) => Promise<{
        detections: FaceDetection[];
        lbPadX: number;
        lbPadY: number;
    }>;
    /** Run face detection and return raw SSD output tensors (scores + regressors) for debugging */
    detectRawSSD: (source: any, srcW: number, srcH: number) => Promise<{
        scores: Float32Array;
        regressors: Float32Array;
        lbPadX: number;
        lbPadY: number;
    }>;
    /** Get the compiled face detector model (for resource sharing) */
    model: CompiledFaceDetectorModel;
}
interface FaceDetectorOptions {
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
declare function createFaceDetector(model: CompiledFaceDetectorModel, options?: FaceDetectorOptions): FaceDetector;
/**
 * Compute the affine transform matrix for cropping a face region.
 *
 * Returns a 2x3 matrix [a, b, tx, c, d, ty] that maps from crop space [0,256]
 * to original image space [0,1] (normalized).
 *
 * Usage: originalX = a * cropX + b * cropY + tx
 *        originalY = c * cropX + d * cropY + ty
 */
declare function computeCropTransform(roi: FaceROI, cropSize?: number): {
    forward: [number, number, number, number, number, number];
    inverse: [number, number, number, number, number, number];
};
/**
 * Project landmarks from crop space back to original image coordinates.
 *
 * @param landmarks Array of {x, y, z} in crop space [0, 1] (from 256x256 crop)
 * @param roi The face ROI used for cropping
 * @param srcWidth Original image width in pixels
 * @param srcHeight Original image height in pixels
 * @returns Array of {x, y, z} in original image space [0, 1]
 */
declare function projectLandmarksToOriginal(landmarks: Array<{
    x: number;
    y: number;
    z: number;
}>, roi: FaceROI, srcWidth: number, srcHeight: number): Array<{
    x: number;
    y: number;
    z: number;
}>;

interface CropPipeline {
    /** Execute the crop transform and write output to the given buffer */
    crop: (encoder: GPUCommandEncoder, sourceTexture: GPUTexture, outputBuffer: GPUBuffer, transform: [number, number, number, number, number, number], srcWidth: number, srcHeight: number, dstSize: number) => void;
}
/**
 * Create a reusable crop pipeline on the given device.
 * Default output size is 256x256 (face landmark model input).
 */
declare function createCropPipeline(device: GPUDevice): CropPipeline;

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
interface Tensor {
    data: Float32Array;
    shape: number[];
}
interface WeightsMetadata {
    keys: string[];
    shapes: number[][];
    offsets: number[];
    dtype?: 'float32' | 'float16';
}
interface FaceLandmarksOutput {
    /** 478 landmarks x 3 (x, y, z) = 1434 values, normalized to [0,1] in crop space */
    landmarks: Float32Array;
    /** Face presence score (1 value, already sigmoid'd, 0-1 range) */
    facePresence: Float32Array;
}
interface CompiledFaceLandmarkModel {
    device: GPUDevice;
    run: (input: Float32Array) => Promise<FaceLandmarksOutput>;
    runFromCanvas: (source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap) => Promise<FaceLandmarksOutput>;
    runFromGPUBuffer: (inputBuffer: GPUBuffer) => Promise<FaceLandmarksOutput>;
    /** Encode landmark inference into an existing command encoder (no submit). */
    encodeFromGPUBuffer: (inputBuffer: GPUBuffer, encoder: GPUCommandEncoder) => void;
    /** Read back results after submitting an encoder that called encodeFromGPUBuffer (blocking). */
    readbackLandmarks: () => Promise<FaceLandmarksOutput>;
    /** Begin non-blocking readback. Returns promise that resolves when GPU finishes. */
    beginReadbackLandmarks: () => Promise<FaceLandmarksOutput>;
    /** Flip to the other readback buffer (for double-buffered pipelining). */
    flipReadbackBuffer: () => void;
}
/**
 * Load face landmark weights from JSON metadata + binary buffer.
 * Handles duplicate keys by appending shape suffix for disambiguation.
 */
declare function loadFaceLandmarkWeights(metadata: WeightsMetadata, buffer: ArrayBuffer): Map<string, Tensor>;
/**
 * Compile the face landmark model for WebGPU inference.
 *
 * Wraps the low-level landmark_model.ts implementation with the
 * CompiledFaceLandmarkModel interface expected by facemesh.ts.
 */
declare function compileFaceLandmarkModel(weights: Map<string, Tensor>, existingDevice?: GPUDevice): Promise<CompiledFaceLandmarkModel>;

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
declare function createFacemesh(options?: FacemeshOptions): Promise<Facemesh>;

export { type CompiledFaceDetectorModel, type CompiledFaceLandmarkModel, type CropPipeline, FACE_KEYPOINT_NAMES, FACE_LANDMARK_INDICES, type FaceDetection, type FaceDetectionOutput, type FaceDetector, type FaceDetectorOptions, type FaceKeypoints, type FaceLandmarkName, type FaceLandmarksOutput, type FaceROI, type Facemesh, type FacemeshInput, type FacemeshOptions, type FacemeshResult, type Landmark, type Tensor$1 as Tensor, type WeightsMetadata, compileFaceDetectorModel, compileFaceLandmarkModel, computeCropTransform, createCropPipeline, createFaceDetector, createFacemesh, detectionToROI, loadFaceDetectorWeights, loadFaceLandmarkWeights, projectLandmarksToOriginal };
