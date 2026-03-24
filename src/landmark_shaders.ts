/**
 * WGSL compute shaders for the face landmark model.
 *
 * Architecture: Deep bottleneck network with PReLU activations.
 * Input: [1, 3, 256, 256] float32 (CHW, [0,1])
 * Output: 478 landmarks (x,y,z) + face presence score
 *
 * All data is in NCHW / CHW layout:
 *   buffer index = channel * H * W + y * W + x
 *
 * Weight layouts (from TFLite):
 *   Conv2D: [outCh, kH, kW, inCh]
 *   Depthwise: [1, kH, kW, channels]
 *   Pointwise 1x1: [outCh, 1, 1, inCh]
 *   PReLU alpha: [1, 1, channels]
 *   BN bias: [channels] (fused into single bias term)
 */

function S(s: string): string {
  return s.replace(/\/\/[^\n]*/g, '').replace(/\s+/g, ' ').replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g, '$1').trim();
}

/**
 * Stem: Conv 3x3 stride-2 + PReLU.
 * Input: [3, 256, 256], Output: [16, 128, 128]
 * Weight: [outCh, 3, 3, inCh] = [16, 3, 3, 3]
 * Bias: [outCh]
 * Alpha: [1, 1, outCh] (PReLU per-channel)
 *
 * TFLite SAME padding for 3x3 stride-2 on 256:
 *   out = ceil(256/2) = 128
 *   total_pad = (128-1)*2 + 3 - 256 = 1
 *   pad_top = 0, pad_left = 0 (asymmetric)
 *
 * Params: in_channels, out_channels, in_h, in_w, out_h, out_w
 */
export const LM_CONV3X3_S2_PRELU_SHADER = S(`
struct Params { in_channels:u32, out_channels:u32, in_h:u32, in_w:u32, out_h:u32, out_w:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read> alpha:array<f32>;
@group(0)@binding(4) var<storage,read_write> output:array<f32>;
@group(0)@binding(5) var<uniform> params:Params;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let oc=gid.z;
  if(out_x>=params.out_w||out_y>=params.out_h||oc>=params.out_channels){return;}
  var sum:f32=0.0;
  let in_h=i32(params.in_h); let in_w=i32(params.in_w);
  for(var ic:u32=0u;ic<params.in_channels;ic++){
    for(var ky:u32=0u;ky<3u;ky++){
      for(var kx:u32=0u;kx<3u;kx++){
        let iy=i32(out_y*2u+ky);
        let ix=i32(out_x*2u+kx);
        if(iy>=0 && iy<in_h && ix>=0 && ix<in_w){
          let in_idx=ic*params.in_h*params.in_w+u32(iy)*params.in_w+u32(ix);
          let w_idx=oc*params.in_channels*9u+ic*9u+ky*3u+kx;
          sum+=input[in_idx]*weight[w_idx];
        }
      }
    }
  }
  sum+=bias[oc];
  // PReLU: max(0,x) + alpha * min(0,x)
  if(sum<0.0){ sum=sum*alpha[oc]; }
  let out_idx=oc*params.out_h*params.out_w+out_y*params.out_w+out_x;
  output[out_idx]=sum;
}
`);

/**
 * Pointwise 1x1 conv + bias + PReLU.
 * Used for the first conv in each bottleneck (N -> N/2).
 * Weight: [outCh, 1, 1, inCh]
 * Bias: [outCh]
 * Alpha: [1, 1, outCh]
 *
 * Params: in_channels, out_channels, height, width
 */
export const LM_CONV1X1_PRELU_SHADER = S(`
struct Params { in_channels:u32, out_channels:u32, height:u32, width:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read> alpha:array<f32>;
@group(0)@binding(4) var<storage,read_write> output:array<f32>;
@group(0)@binding(5) var<uniform> params:Params;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let oc=gid.z;
  if(out_x>=params.width||out_y>=params.height||oc>=params.out_channels){return;}
  var sum:f32=0.0;
  let spatial=params.height*params.width;
  let pix=out_y*params.width+out_x;
  for(var ic:u32=0u;ic<params.in_channels;ic++){
    sum+=input[ic*spatial+pix]*weight[oc*params.in_channels+ic];
  }
  sum+=bias[oc];
  if(sum<0.0){ sum=sum*alpha[oc]; }
  output[oc*spatial+pix]=sum;
}
`);

/**
 * Depthwise 3x3 conv + bias (no activation).
 * Used in the middle of bottleneck blocks.
 * Weight: [1, 3, 3, channels] -> transposed to [channels, 9]
 * Bias: [channels]
 *
 * Params: channels, in_h, in_w, out_h, out_w, stride, pad
 */
export const LM_DEPTHWISE_3X3_SHADER = S(`
struct Params { channels:u32, in_h:u32, in_w:u32, out_h:u32, out_w:u32, stride:u32, pad:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:Params;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let c=gid.z;
  if(out_x>=params.out_w||out_y>=params.out_h||c>=params.channels){return;}
  let in_h=i32(params.in_h); let in_w=i32(params.in_w);
  let w_base=c*9u;
  let base_y=i32(out_y*params.stride)-i32(params.pad);
  let base_x=i32(out_x*params.stride)-i32(params.pad);
  var sum:f32=0.0;
  for(var ky:u32=0u;ky<3u;ky++){
    let iy=base_y+i32(ky);
    if(iy>=0 && iy<in_h){
      let row=c*params.in_h*params.in_w+u32(iy)*params.in_w;
      for(var kx:u32=0u;kx<3u;kx++){
        let ix=base_x+i32(kx);
        if(ix>=0 && ix<in_w){
          sum+=input[row+u32(ix)]*weight[w_base+ky*3u+kx];
        }
      }
    }
  }
  sum+=bias[c];
  output[c*params.out_h*params.out_w+out_y*params.out_w+out_x]=sum;
}
`);

/**
 * Pointwise 1x1 conv + bias (NO activation).
 * Used as the last conv in each bottleneck block before the residual ADD.
 * Weight: [outCh, 1, 1, inCh]
 * Bias: [outCh]
 *
 * Params: in_channels, out_channels, height, width
 */
export const LM_CONV1X1_SHADER = S(`
struct Params { in_channels:u32, out_channels:u32, height:u32, width:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:Params;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let oc=gid.z;
  if(out_x>=params.width||out_y>=params.height||oc>=params.out_channels){return;}
  var sum:f32=0.0;
  let spatial=params.height*params.width;
  let pix=out_y*params.width+out_x;
  for(var ic:u32=0u;ic<params.in_channels;ic++){
    sum+=input[ic*spatial+pix]*weight[oc*params.in_channels+ic];
  }
  sum+=bias[oc];
  output[oc*spatial+pix]=sum;
}
`);

/**
 * Element-wise ADD for residual connections.
 * output[i] = a[i] + b[i]
 *
 * Params: total element count
 */
export const LM_ADD_SHADER = S(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> count:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x;
  if(idx>=count){return;}
  output[idx]=a[idx]+b[idx];
}
`);

/**
 * Conv 2x2 stride-2 (no activation) for downsampling path.
 * Weight: [outCh, 2, 2, inCh]
 * Bias: [outCh]
 *
 * Params: in_channels, out_channels, in_h, in_w, out_h, out_w
 */
export const LM_CONV2X2_S2_SHADER = S(`
struct Params { in_channels:u32, out_channels:u32, in_h:u32, in_w:u32, out_h:u32, out_w:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:Params;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let oc=gid.z;
  if(out_x>=params.out_w||out_y>=params.out_h||oc>=params.out_channels){return;}
  var sum:f32=0.0;
  let in_h=params.in_h; let in_w=params.in_w;
  for(var ic:u32=0u;ic<params.in_channels;ic++){
    for(var ky:u32=0u;ky<2u;ky++){
      for(var kx:u32=0u;kx<2u;kx++){
        let iy=out_y*2u+ky;
        let ix=out_x*2u+kx;
        if(iy<in_h && ix<in_w){
          let in_idx=ic*in_h*in_w+iy*in_w+ix;
          let w_idx=oc*params.in_channels*4u+ic*4u+ky*2u+kx;
          sum+=input[in_idx]*weight[w_idx];
        }
      }
    }
  }
  sum+=bias[oc];
  output[oc*params.out_h*params.out_w+out_y*params.out_w+out_x]=sum;
}
`);

/**
 * MaxPool 2x2 stride-2.
 * Takes max over each 2x2 spatial block.
 *
 * Params: channels, in_h, in_w, out_h, out_w
 */
export const LM_MAXPOOL_2X2_SHADER = S(`
struct Params { channels:u32, in_h:u32, in_w:u32, out_h:u32, out_w:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> params:Params;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let c=gid.z;
  if(out_x>=params.out_w||out_y>=params.out_h||c>=params.channels){return;}
  var mx:f32=-1e38;
  let in_base=c*params.in_h*params.in_w;
  for(var dy:u32=0u;dy<2u;dy++){
    let iy=out_y*2u+dy;
    if(iy<params.in_h){
      for(var dx:u32=0u;dx<2u;dx++){
        let ix=out_x*2u+dx;
        if(ix<params.in_w){
          mx=max(mx,input[in_base+iy*params.in_w+ix]);
        }
      }
    }
  }
  output[c*params.out_h*params.out_w+out_y*params.out_w+out_x]=mx;
}
`);

/**
 * Channel PAD for residual connections when channels differ.
 * Pads channel dimension by appending zeros.
 * Input: [inCh, H, W], Output: [outCh, H, W] where outCh > inCh
 *
 * Params: in_channels, out_channels, height, width
 */
export const LM_CHANNEL_PAD_SHADER = S(`
struct Params { in_channels:u32, out_channels:u32, height:u32, width:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> params:Params;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let oc=gid.z;
  if(out_x>=params.width||out_y>=params.height||oc>=params.out_channels){return;}
  let spatial=params.height*params.width;
  let pix=out_y*params.width+out_x;
  var val:f32=0.0;
  if(oc<params.in_channels){
    val=input[oc*spatial+pix];
  }
  output[oc*spatial+pix]=val;
}
`);

/**
 * Output head: Conv 2x2 stride-2, no activation.
 * Input: [channels, 2, 2], Output: [outCh, 1, 1]
 * Weight: [outCh, 2, 2, inCh]
 * Bias: [outCh]
 *
 * Since output is 1x1, this is basically a dot product per output channel.
 * Params: in_channels, out_channels
 */
export const LM_OUTPUT_CONV2X2_SHADER = S(`
struct Params { in_channels:u32, out_channels:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let oc=gid.x;
  if(oc>=params.out_channels){return;}
  var sum:f32=0.0;
  for(var ic:u32=0u;ic<params.in_channels;ic++){
    for(var ky:u32=0u;ky<2u;ky++){
      for(var kx:u32=0u;kx<2u;kx++){
        let in_idx=ic*4u+ky*2u+kx;
        let w_idx=oc*params.in_channels*4u+ic*4u+ky*2u+kx;
        sum+=input[in_idx]*weight[w_idx];
      }
    }
  }
  sum+=bias[oc];
  output[oc]=sum;
}
`);

/**
 * Sigmoid activation (element-wise).
 * output[i] = 1 / (1 + exp(-input[i]))
 *
 * Params: count
 */
export const LM_SIGMOID_SHADER = S(`
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> count:u32;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x;
  if(idx>=count){return;}
  output[idx]=1.0/(1.0+exp(-input[idx]));
}
`);

/**
 * PReLU activation (per-channel, applied element-wise).
 * output[idx] = x >= 0 ? x : alpha[channel] * x
 *
 * Params: channels, height, width
 */
export const LM_PRELU_SHADER = S(`
struct Params { channels:u32, height:u32, width:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> alpha:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> params:Params;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let x=gid.x; let y=gid.y; let c=gid.z;
  if(x>=params.width||y>=params.height||c>=params.channels){return;}
  let idx=c*params.height*params.width+y*params.width+x;
  let val=input[idx];
  if(val<0.0){
    output[idx]=val*alpha[c];
  } else {
    output[idx]=val;
  }
}
`);
