/**
 * WebGPU compute shader for affine crop + rotate + resize.
 *
 * Takes an original camera frame texture and applies an affine transform
 * to produce a 256x256 cropped face image suitable for the face landmark model.
 *
 * Uses bilinear interpolation for smooth resampling.
 */

function S(s: string): string {
  return s.replace(/\/\/[^\n]*/g, '').replace(/\s+/g, ' ').replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g, '$1').trim();
}

/**
 * Affine crop shader.
 *
 * Params uniform:
 * - src_width, src_height: source texture dimensions
 * - dst_size: output size (256)
 * - _pad: padding for alignment
 *
 * Transform uniform (6 floats packed as vec4 + vec2):
 * - [a, b, tx, c, d, ty]: affine matrix mapping crop pixel -> source normalized [0,1]
 *   source_x = a * crop_x + b * crop_y + tx
 *   source_y = c * crop_x + d * crop_y + ty
 *
 * Output: CHW float32 buffer of size [3, dst_size, dst_size] normalized to [0,1]
 */
export const CROP_AFFINE_SHADER = S(`
struct CropParams { src_width:u32, src_height:u32, dst_size:u32, _pad:u32, }
struct AffineTransform { a:f32, b:f32, tx:f32, c:f32, d:f32, ty:f32, }

@group(0)@binding(0) var src_tex:texture_2d<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> params:CropParams;
@group(0)@binding(3) var<uniform> transform:AffineTransform;
@group(0)@binding(4) var src_sampler:sampler;

@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let dst_x=gid.x; let dst_y=gid.y;
  if(dst_x>=params.dst_size||dst_y>=params.dst_size){return;}

  // Map crop pixel to source normalized coordinates [0,1]
  let fx=f32(dst_x)+0.5;
  let fy=f32(dst_y)+0.5;
  let src_nx=transform.a*fx+transform.b*fy+transform.tx;
  let src_ny=transform.c*fx+transform.d*fy+transform.ty;

  let out_stride=params.dst_size*params.dst_size;

  // Hardware bilinear sampling via textureSampleLevel with clamp-to-edge sampler.
  // Clamp-to-edge matches MediaPipe's BORDER_REPLICATE default.
  let pixel = textureSampleLevel(src_tex, src_sampler, vec2<f32>(src_nx, src_ny), 0.0);

  // Write CHW format
  output[0u*out_stride+dst_y*params.dst_size+dst_x]=pixel.r;
  output[1u*out_stride+dst_y*params.dst_size+dst_x]=pixel.g;
  output[2u*out_stride+dst_y*params.dst_size+dst_x]=pixel.b;
}
`);

export interface CropPipeline {
  /** Execute the crop transform and write output to the given buffer */
  crop: (
    encoder: GPUCommandEncoder,
    sourceTexture: GPUTexture,
    outputBuffer: GPUBuffer,
    transform: [number, number, number, number, number, number],
    srcWidth: number,
    srcHeight: number,
    dstSize: number,
  ) => void;
}

/**
 * Create a reusable crop pipeline on the given device.
 * Default output size is 256x256 (face landmark model input).
 */
export function createCropPipeline(device: GPUDevice): CropPipeline {
  const shaderModule = device.createShaderModule({ code: CROP_AFFINE_SHADER });

  const layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' as const } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' as GPUBufferBindingType } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' as GPUBufferBindingType } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' as GPUBufferBindingType } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, sampler: {} },
    ],
  });

  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module: shaderModule, entryPoint: 'main' },
  });

  // Linear sampler for hardware bilinear interpolation (matches MediaPipe's GL_LINEAR)
  const linearSampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });

  // Pre-create reusable uniform buffers (avoid per-frame GPU buffer allocation leak)
  const paramsBuf = device.createBuffer({
    size: 16, // 4 x uint32
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const transformBuf = device.createBuffer({
    size: 32, // 8 x float32 (6 values + 2 padding)
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const transformPadded = new Float32Array(8);

  function crop(
    encoder: GPUCommandEncoder,
    sourceTexture: GPUTexture,
    outputBuffer: GPUBuffer,
    transform: [number, number, number, number, number, number],
    srcWidth: number,
    srcHeight: number,
    dstSize: number,
  ): void {
    device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([srcWidth, srcHeight, dstSize, 0]));

    transformPadded.set(transform);
    device.queue.writeBuffer(transformBuf, 0, transformPadded);

    const bindGroup = device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: sourceTexture.createView() },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: paramsBuf } },
        { binding: 3, resource: { buffer: transformBuf } },
        { binding: 4, resource: linearSampler },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(dstSize / 16), Math.ceil(dstSize / 16), 1);
    pass.end();
  }

  return { crop };
}
