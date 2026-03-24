/**
 * @svenflow/micro-facemesh
 *
 * WebGPU face landmark detection. Faster than MediaPipe, zero dependencies.
 *
 * @example
 * ```typescript
 * import { createFacemesh } from '@svenflow/micro-facemesh'
 *
 * const facemesh = await createFacemesh()
 * const faces = await facemesh.detect(videoElement)
 *
 * for (const face of faces) {
 *   console.log(face.keypoints.leftEye)  // {x, y, z}
 *   console.log(face.landmarks.length)   // 478
 * }
 * ```
 */

export { FACE_KEYPOINT_NAMES, FACE_LANDMARK_INDICES } from './types.js';
export type {
  Facemesh,
  FacemeshResult,
  FacemeshOptions,
  FacemeshInput,
  Landmark,
  FaceKeypoints,
  FaceLandmarkName,
} from './types.js';

export type {
  FaceDetection,
  FaceROI,
  FaceDetector,
  FaceDetectorOptions,
} from './face_detector.js';

export {
  createFaceDetector,
  detectionToROI,
  computeCropTransform,
  projectLandmarksToOriginal,
} from './face_detector.js';

export type {
  CompiledFaceDetectorModel,
  FaceDetectionOutput,
  Tensor,
} from './face_detector_model.js';

export {
  compileFaceDetectorModel,
  loadFaceDetectorWeights,
} from './face_detector_model.js';

export { createCropPipeline } from './crop_shader.js';
export type { CropPipeline } from './crop_shader.js';

export type {
  CompiledFaceLandmarkModel,
  FaceLandmarksOutput,
  WeightsMetadata,
} from './face_landmark_model.js';

export { loadFaceLandmarkWeights, compileFaceLandmarkModel } from './face_landmark_model.js';

export { createFacemesh } from './facemesh.js';
