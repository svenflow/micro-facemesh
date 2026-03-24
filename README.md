# micro-facemesh

[![npm](https://img.shields.io/npm/v/@svenflow/micro-facemesh)](https://www.npmjs.com/package/@svenflow/micro-facemesh)
[![license](https://img.shields.io/npm/l/@svenflow/micro-facemesh)](./LICENSE)

**WebGPU face mesh for the browser. 478-landmark face tracking with iris detection and ROI tracking. No WASM, no ONNX Runtime — just 22 compute shaders. 52KB JS + model weights downloaded at runtime.**

[**Live Demo**](https://svenflow.github.io/micro-facemesh/) | [npm](https://www.npmjs.com/package/@svenflow/micro-facemesh)

---

## Quick Start

```bash
npm install @svenflow/micro-facemesh
```

```typescript
import { createFacemesh } from '@svenflow/micro-facemesh'

const facemesh = await createFacemesh()
const faces = await facemesh.detect(videoElement)

for (const face of faces) {
  console.log(face.score)               // 0.99
  console.log(face.keypoints.noseTip)   // { x, y, z }
  console.log(face.landmarks.length)    // 478
}
```

Create once, detect per frame. Weights download on first call from CDN and are cached by the browser. Full TypeScript types included.

## Benchmarks

### iPhone 16 Pro — Safari (iOS 18, WebGPU)

| | Inference | FPS | Backend |
|---|---|---|---|
| **micro-facemesh** | **5.0ms** | **60** | WebGPU |
| MediaPipe | 9.0ms | 60 | WebGPU |

### Mac Mini M4 Pro — Chrome 134

| | Median | p99 | Backend |
|---|---|---|---|
| **micro-facemesh** | **3.3ms** | **4.1ms** | WebGPU |
| MediaPipe | 5.2ms | 7.8ms | WebGPU |
| MediaPipe | 6.1ms | 9.4ms | WASM |

[**Run this benchmark on your device →**](https://svenflow.github.io/micro-facemesh/)

**~1.8x faster than MediaPipe GPU on iPhone Safari.** On desktop, ~1.6x faster. With ROI tracking, most frames skip face detection entirely — only landmark inference runs. Double-buffered GPU readback hides `mapAsync` latency behind the inter-frame gap.

## Features

- **52KB** minified JS (14KB gzipped) + 2.6MB weights (f16, served via CDN)
- **~1.8x faster** than MediaPipe on iPhone, ~1.6x on desktop
- **478 landmarks** — full face mesh with iris tracking (MediaPipe FaceMesh V2)
- **ROI tracking** — uses previous landmarks to track between frames, skipping face detection for smoother, faster results
- **Named keypoints** — `face.keypoints.leftEye`, `face.keypoints.noseTip`, etc.
- **Landmark indices** — `FACE_LANDMARK_INDICES.leftIrisCenter`, `FACE_LANDMARK_INDICES.noseTip`, etc.
- **Pipelined readback** — double-buffered GPU→CPU transfer overlaps with the next frame's inference
- **Zero dependencies** — pure WebGPU compute shaders, no WASM or ONNX Runtime

## API

### `createFacemesh(options?)`

Creates and initializes the detector. Downloads weights and compiles WebGPU pipelines.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `weightsUrl` | `string` | jsdelivr CDN | Base URL for weight files |
| `scoreThreshold` | `number` | `0.5` | Minimum landmark confidence (0-1) |
| `faceScoreThreshold` | `number` | `0.5` | Minimum face detection score (0-1) |
| `maxFaces` | `number` | `1` | Maximum faces to detect |

### `facemesh.detect(source)`

Detect face landmarks from a camera frame or image. Returns `FacemeshResult[]` (empty if no face found).

Accepts: `HTMLVideoElement`, `HTMLCanvasElement`, `OffscreenCanvas`, `ImageBitmap`, `HTMLImageElement`, `ImageData`

```typescript
interface FacemeshResult {
  score: number              // Confidence (0-1)
  landmarks: Landmark[]      // 478 points, normalized [0,1]
  keypoints: FaceKeypoints   // Named access: .noseTip, .leftEye, etc.
}
```

### Keypoints

6 named keypoints from the face detector landmarks:

```typescript
face.keypoints.rightEye         // { x, y, z }
face.keypoints.leftEye
face.keypoints.noseTip
face.keypoints.mouthCenter
face.keypoints.rightEarTragion
face.keypoints.leftEarTragion
```

### Landmark Indices

Named indices into the 478-point mesh for common features:

```typescript
import { FACE_LANDMARK_INDICES } from '@svenflow/micro-facemesh'

const nose = face.landmarks[FACE_LANDMARK_INDICES.noseTip]
const leftIris = face.landmarks[FACE_LANDMARK_INDICES.leftIrisCenter]
```

### `facemesh.reset()`

Reset tracking state. Call when switching between unrelated images to force face re-detection.

### `facemesh.dispose()`

Release GPU resources.

## How It Works

```
Video frame → Face Detection (128×128, 8 shaders)
           → ROI crop (affine warp, 1 shader)
           → Landmark model (256×256, 13 shaders)
           → 478 landmarks + face presence score
           → ROI tracking (landmarks → next frame's crop region)
```

On the first frame, face detection finds bounding boxes with 6 keypoints. On subsequent frames, landmarks from the previous frame compute the crop region directly — face detection is skipped entirely. Double-buffered pipelined readback means the GPU→CPU transfer for frame N happens while frame N+1's inference is already running.

## Self-Hosting Weights

```typescript
const facemesh = await createFacemesh({
  weightsUrl: '/models/facemesh'
})
```

Copy the `weights/` directory from the npm package to your server. The f16 weights are used by default (2.6MB total).

## Browser Support

| Browser | Status |
|---------|--------|
| Chrome 113+ | ✅ |
| Edge 113+ | ✅ |
| Safari 18+ (macOS/iOS) | ✅ |
| Firefox Nightly | Experimental |

## Error Handling

Check for WebGPU support before initializing:

```typescript
if (!navigator.gpu) {
  console.error('WebGPU is not supported in this browser')
  // Fall back to a non-WebGPU solution or show a message
}
```

Wrap initialization in a try/catch to handle GPU adapter or device failures:

```typescript
try {
  const facemesh = await createFacemesh()
  const faces = await facemesh.detect(videoElement)
} catch (err) {
  console.error('Failed to initialize face tracking:', err)
}
```

## SSR / Server-Side Rendering

`micro-facemesh` requires WebGPU and browser APIs (`navigator.gpu`, `OffscreenCanvas`, etc.) that are not available in server environments. If you use a framework with server-side rendering (Next.js, Nuxt, SvelteKit, etc.), make sure to only import and initialize it on the client:

```typescript
// Next.js example (app router)
'use client'

import { useEffect, useState } from 'react'
import type { Facemesh } from '@svenflow/micro-facemesh'

export default function FaceTracker() {
  const [facemesh, setFacemesh] = useState<Facemesh | null>(null)

  useEffect(() => {
    import('@svenflow/micro-facemesh').then(({ createFacemesh }) => {
      createFacemesh().then(setFacemesh)
    })
  }, [])

  // ...
}
```

## FAQ

**Does it work on mobile?**
Yes. WebGPU is supported in Safari on iOS 18+ and Chrome on Android. On iPhone 16 Pro (Safari, iOS 18) we measured 5.0ms inference at 60 FPS — 1.8x faster than MediaPipe GPU on the same device.

**How many faces can it track?**
One by default. Set `maxFaces` in the options to change this.

**Does it include iris tracking?**
Yes. Landmarks 468–477 are iris points (5 per eye: center + 4 cardinal directions). Access them via `FACE_LANDMARK_INDICES.leftIrisCenter`, etc.

**Does it work offline?**
Model weights are downloaded on first use and cached by the browser. After that, it works offline. You can also self-host the weights (see [Self-Hosting Weights](#self-hosting-weights)).

**What license is the model under?**
The model architecture and weights are derived from MediaPipe's FaceMesh V2 model, which is published under the Apache 2.0 license.

## Development

```bash
git clone https://github.com/svenflow/micro-facemesh.git
cd micro-facemesh
npm install
npm run dev    # Watch mode with hot reload
npm run build  # Production build
```

## Credits

- Face landmark model architecture and weights adapted from [MediaPipe FaceMesh](https://github.com/google-ai-edge/mediapipe) (Apache 2.0 license).
- Face detection model from MediaPipe BlazeFace short range.
- ROI tracking approach follows MediaPipe's pipeline design (face detection + landmark tracking with re-detection on loss).

## License

MIT
