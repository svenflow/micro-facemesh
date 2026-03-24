/** A 3D landmark point (x, y in [0,1] image coords, z is relative depth) */
export interface Landmark {
  x: number;
  y: number;
  z: number;
}

/** Face landmark region names for ergonomic access */
export const FACE_KEYPOINT_NAMES = [
  'rightEye',
  'leftEye',
  'noseTip',
  'mouthCenter',
  'rightEarTragion',
  'leftEarTragion',
] as const;

/** Named face keypoints from the face detector (6 keypoints) */
export type FaceKeypoints = {
  [K in (typeof FACE_KEYPOINT_NAMES)[number]]: Landmark;
};

/** Build a FaceKeypoints object from an array of 6 landmarks */
export function toFaceKeypoints(keypoints: Landmark[]): FaceKeypoints {
  const kp = {} as Record<string, Landmark>;
  for (let i = 0; i < FACE_KEYPOINT_NAMES.length; i++) {
    kp[FACE_KEYPOINT_NAMES[i]] = keypoints[i]!;
  }
  return kp as FaceKeypoints;
}

/** Detection result for a single face (full 478-landmark mesh) */
export interface FacemeshResult {
  /** Confidence score (0-1) that a face is present */
  score: number;
  /** 478 face landmarks in original image coordinates [0,1] */
  landmarks: Landmark[];
  /** Named detector keypoints for ergonomic access: result.keypoints.leftEye */
  keypoints: FaceKeypoints;
}

/** Options for creating a facemesh detector */
export interface FacemeshOptions {
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
export interface Facemesh {
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
export type FacemeshInput =
  | HTMLCanvasElement
  | OffscreenCanvas
  | ImageBitmap
  | HTMLImageElement
  | HTMLVideoElement
  | ImageData;

/**
 * Notable landmark indices in the 478-point face mesh.
 *
 * The first 468 landmarks follow MediaPipe's face mesh topology.
 * Landmarks 468-477 are iris landmarks (5 per eye).
 */
export const FACE_LANDMARK_INDICES = {
  // Silhouette / face oval
  silhouetteTop: 10,
  silhouetteBottom: 152,

  // Eyes (outer/inner corners)
  leftEyeInner: 133,
  leftEyeOuter: 33,
  rightEyeInner: 362,
  rightEyeOuter: 263,

  // Eyebrows
  leftEyebrowUpper: 66,
  rightEyebrowUpper: 296,

  // Nose
  noseTip: 1,
  noseBottom: 2,
  noseBridgeTop: 6,

  // Lips
  upperLipTop: 13,
  lowerLipBottom: 14,
  mouthLeft: 61,
  mouthRight: 291,

  // Chin
  chin: 152,

  // Ears
  leftEarTragion: 234,
  rightEarTragion: 454,

  // Iris landmarks (468-477)
  // Left iris (5 points: center + 4 cardinal)
  leftIrisCenter: 468,
  leftIrisRight: 469,
  leftIrisTop: 470,
  leftIrisLeft: 471,
  leftIrisBottom: 472,

  // Right iris (5 points: center + 4 cardinal)
  rightIrisCenter: 473,
  rightIrisRight: 474,
  rightIrisTop: 475,
  rightIrisLeft: 476,
  rightIrisBottom: 477,
} as const;

/** Type for FACE_LANDMARK_INDICES keys */
export type FaceLandmarkName = keyof typeof FACE_LANDMARK_INDICES;
