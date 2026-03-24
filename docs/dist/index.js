import{a as He,b as vt}from"./chunk-WOYQSGEU.js";var ze=["rightEye","leftEye","noseTip","mouthCenter","rightEarTragion","leftEarTragion"];function We(a){let o={};for(let t=0;t<ze.length;t++)o[ze[t]]=a[t];return o}var Ct={silhouetteTop:10,silhouetteBottom:152,leftEyeInner:133,leftEyeOuter:33,rightEyeInner:362,rightEyeOuter:263,leftEyebrowUpper:66,rightEyebrowUpper:296,noseTip:1,noseBottom:2,noseBridgeTop:6,upperLipTop:13,lowerLipBottom:14,mouthLeft:61,mouthRight:291,chin:152,leftEarTragion:234,rightEarTragion:454,leftIrisCenter:468,leftIrisRight:469,leftIrisTop:470,leftIrisLeft:471,leftIrisBottom:472,rightIrisCenter:473,rightIrisRight:474,rightIrisTop:475,rightIrisLeft:476,rightIrisBottom:477};function Ut(){let a=[];for(let o=0;o<16;o++)for(let t=0;t<16;t++){let y=(t+.5)/16,d=(o+.5)/16;for(let u=0;u<2;u++)a.push({x:y,y:d})}for(let o=0;o<8;o++)for(let t=0;t<8;t++){let y=(t+.5)/8,d=(o+.5)/8;for(let u=0;u<6;u++)a.push({x:y,y:d})}return a}var tt=Ut();function Et(a){return 1/(1+Math.exp(-a))}function Ye(a,o){let t=[],{scores:y,regressors:d}=a,u=128;for(let c=0;c<tt.length;c++){let h=Et(y[c]);if(h<o)continue;let _=tt[c],m=c*16,s=_.x+d[m+0]/u,l=_.y+d[m+1]/u,f=d[m+2]/u,P=d[m+3]/u,i=[];for(let g=0;g<6;g++){let T=_.x+d[m+4+g*2]/u,w=_.y+d[m+4+g*2+1]/u;i.push([T,w])}t.push({score:h,box:[s,l,f,P],keypoints:i})}return t}function Ne(a,o){if(a.length===0)return[];let t=[...a].sort((u,c)=>c.score-u.score),y=[],d=new Set;for(let u=0;u<t.length;u++){if(d.has(u))continue;let c=[u];for(let i=u+1;i<t.length;i++)d.has(i)||Tt(t[u],t[i])>o&&(c.push(i),d.add(i));let h=0,_=0,m=0,s=0,l=0,f=[];for(let i=0;i<6;i++)f.push([0,0]);for(let i of c){let g=t[i],T=g.score;h+=T,_+=g.box[0]*T,m+=g.box[1]*T,s+=g.box[2]*T,l+=g.box[3]*T;for(let w=0;w<6;w++)f[w][0]+=g.keypoints[w][0]*T,f[w][1]+=g.keypoints[w][1]*T}let P=1/h;y.push({score:t[u].score,box:[_*P,m*P,s*P,l*P],keypoints:f.map(([i,g])=>[i*P,g*P])})}return y}function Tt(a,o){let t=a.box[0]-a.box[2]/2,y=a.box[1]-a.box[3]/2,d=a.box[0]+a.box[2]/2,u=a.box[1]+a.box[3]/2,c=o.box[0]-o.box[2]/2,h=o.box[1]-o.box[3]/2,_=o.box[0]+o.box[2]/2,m=o.box[1]+o.box[3]/2,s=Math.max(t,c),l=Math.max(y,h),f=Math.min(d,_),P=Math.min(u,m),i=Math.max(0,f-s),g=Math.max(0,P-l),T=i*g,w=(d-t)*(u-y),ne=(_-c)*(m-h),ae=w+ne-T;return ae>0?T/ae:0}function rt(a){let[o,t,y,d]=a.box,u=a.keypoints[0],c=a.keypoints[1],h=(u[0]+c[0])/2,_=(u[1]+c[1])/2,m=a.keypoints[2],s=m[0]-h,l=m[1]-_,f=Math.atan2(l,s),i=Math.PI/2-f,g=Math.max(y,d),w=g*1.5,ne=0*g,ae=Math.cos(i),Q=Math.sin(i),Ce=ne*Q,Ue=ne*ae;return{centerX:o+Ce,centerY:t+Ue,width:w,height:w,rotation:i}}function Xe(a,o={}){let{scoreThreshold:t=.5,nmsThreshold:y=.3,maxFaces:d=1}=o;async function u(m){let s=await a.run(m),l=Ye(s,t);return Ne(l,y).slice(0,d).map(rt)}async function c(m){let s=await a.run(m),l=Ye(s,t);return Ne(l,y).slice(0,d)}async function h(m,s,l){let{output:f,lbPadX:P,lbPadY:i}=await a.runWithResize(m,s,l),g=Ye(f,t);return{detections:Ne(g,y).slice(0,d),lbPadX:P,lbPadY:i}}async function _(m,s,l){let{output:f,lbPadX:P,lbPadY:i}=await a.runWithResize(m,s,l);return{scores:f.scores,regressors:f.regressors,lbPadX:P,lbPadY:i}}return{detect:u,detectRaw:c,detectRawWithResize:h,detectRawSSD:_,model:a}}function Ft(a,o=256){let t=Math.cos(a.rotation),y=Math.sin(a.rotation),d=a.width/o,u=a.height/o,c=d*t,h=d*y,_=-u*y,m=u*t,s=a.centerX-(c*o/2+h*o/2),l=a.centerY-(_*o/2+m*o/2),f=c*m-h*_,P=m/f,i=-h/f,g=-_/f,T=c/f,w=-(P*s+i*l),ne=-(g*s+T*l);return{forward:[c,h,s,_,m,l],inverse:[P,i,w,g,T,ne]}}function Mt(a,o,t,y){let d=Math.cos(o.rotation),u=Math.sin(o.rotation),c=Math.min(t,y),h=o.width*c,_=h/t,m=h/y;return a.map(s=>{let l=s.x-.5,f=s.y-.5,P=d*l+u*f,i=-u*l+d*f;return{x:P*_+o.centerX,y:i*m+o.centerY,z:s.z}})}function be(a){return a.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var nt=be(`
struct ConvParams { batch:u32, in_channels:u32, out_channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:ConvParams;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let oc_batch=gid.z;
  let oc=oc_batch%params.out_channels; let batch=oc_batch/params.out_channels;
  if(out_x>=params.out_width||out_y>=params.out_height||batch>=params.batch){return;}
  var sum:f32=0.0;
  let in_h=i32(params.in_height); let in_w=i32(params.in_width);
  let in_stride=params.in_height*params.in_width;
  let in_batch_base=batch*params.in_channels*in_stride;
  for(var ky:u32=0u;ky<5u;ky=ky+1u){
    let in_y=i32(out_y*2u+ky)-1;
    if(in_y<0 || in_y>=in_h){continue;}
    for(var kx:u32=0u;kx<5u;kx=kx+1u){
      let in_x=i32(out_x*2u+kx)-1;
      if(in_x<0 || in_x>=in_w){continue;}
      let pix_off=u32(in_y)*params.in_width+u32(in_x);
      let inp=vec3<f32>(
        input[in_batch_base+pix_off],
        input[in_batch_base+in_stride+pix_off],
        input[in_batch_base+2u*in_stride+pix_off]
      );
      let w_off=oc*75u+ky*15u+kx*3u;
      let w=vec3<f32>(weight[w_off],weight[w_off+1u],weight[w_off+2u]);
      sum+=dot(inp,w);
    }
  }
  sum=sum+bias[oc];
  // ReLU
  let result=max(0.0,sum);
  let out_idx=batch*params.out_channels*params.out_height*params.out_width+oc*params.out_height*params.out_width+out_y*params.out_width+out_x;
  output[out_idx]=result;
}
`),at=be(`
struct DepthwiseParams { batch:u32, channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, stride:u32, pad:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:DepthwiseParams;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let c_batch=gid.z;
  let c=c_batch%params.channels; let batch=c_batch/params.channels;
  if(out_x>=params.out_width||out_y>=params.out_height||batch>=params.batch){return;}
  let in_base=batch*params.channels*params.in_height*params.in_width+c*params.in_height*params.in_width;
  let w_base=c*9u; let in_h=i32(params.in_height); let in_w=i32(params.in_width); let pad=i32(params.pad);
  let base_in_y=i32(out_y*params.stride)-pad; let base_in_x=i32(out_x*params.stride)-pad;
  var sum:f32=0.0;
  for(var ky:u32=0u;ky<3u;ky=ky+1u){
    let in_y=base_in_y+i32(ky);
    if(in_y>=0 && in_y<in_h){
      let row_base=in_base+u32(in_y)*params.in_width;
      for(var kx:u32=0u;kx<3u;kx=kx+1u){
        let in_x=base_in_x+i32(kx);
        if(in_x>=0 && in_x<in_w){
          sum+=input[row_base+u32(in_x)]*weight[w_base+ky*3u+kx];
        }
      }
    }
  }
  sum+=bias[c];
  let out_idx=batch*params.channels*params.out_height*params.out_width+c*params.out_height*params.out_width+out_y*params.out_width+out_x;
  output[out_idx]=sum;
}
`),ot=be(`
struct PointwiseParams { batch:u32, in_channels:u32, out_channels:u32, height:u32, width:u32, channel_pad:u32, stride:u32, in_height:u32, in_width:u32, }
@group(0)@binding(0) var<storage,read> dw_output:array<f32>;
@group(0)@binding(1) var<storage,read> skip_input:array<f32>;
@group(0)@binding(2) var<storage,read> pw_weight:array<f32>;
@group(0)@binding(3) var<storage,read> pw_bias:array<f32>;
@group(0)@binding(4) var<storage,read_write> output:array<f32>;
@group(0)@binding(5) var<uniform> params:PointwiseParams;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let oc_batch=gid.z;
  let oc=oc_batch%params.out_channels; let batch=oc_batch/params.out_channels;
  if(out_x>=params.width||out_y>=params.height||batch>=params.batch){return;}
  var sum:f32=0.0;
  let dw_base=batch*params.in_channels*params.height*params.width+out_y*params.width+out_x;
  let w_base=oc*params.in_channels; let spatial_stride=params.height*params.width;
  let ic4=params.in_channels/4u;
  for(var i:u32=0u;i<ic4;i=i+1u){
    let ic=i*4u;
    let inp=vec4<f32>(
      dw_output[dw_base+ic*spatial_stride],
      dw_output[dw_base+(ic+1u)*spatial_stride],
      dw_output[dw_base+(ic+2u)*spatial_stride],
      dw_output[dw_base+(ic+3u)*spatial_stride]
    );
    let w=vec4<f32>(
      pw_weight[w_base+ic],
      pw_weight[w_base+ic+1u],
      pw_weight[w_base+ic+2u],
      pw_weight[w_base+ic+3u]
    );
    sum+=dot(inp,w);
  }
  sum+=pw_bias[oc];
  // Skip connection: zero-pad channels
  var skip_val:f32=0.0;
  if(oc<params.channel_pad){
    if(params.stride==2u){
      var max_val:f32=-1e38;
      for(var py:u32=0u;py<2u;py=py+1u){
        for(var px:u32=0u;px<2u;px=px+1u){
          let skip_y=out_y*2u+py; let skip_x=out_x*2u+px;
          if(skip_y<params.in_height && skip_x<params.in_width){
            let skip_idx=batch*params.channel_pad*params.in_height*params.in_width+oc*params.in_height*params.in_width+skip_y*params.in_width+skip_x;
            max_val=max(max_val,skip_input[skip_idx]);
          }
        }
      }
      skip_val=max_val;
    } else {
      let skip_idx=batch*params.channel_pad*params.height*params.width+oc*params.height*params.width+out_y*params.width+out_x;
      skip_val=skip_input[skip_idx];
    }
  }
  let v=sum+skip_val;
  // ReLU
  let result=max(0.0,v);
  let out_idx=batch*params.out_channels*params.height*params.width+oc*params.height*params.width+out_y*params.width+out_x;
  output[out_idx]=result;
}
`),st=be(`
struct Conv1x1Params { batch:u32, in_channels:u32, out_channels:u32, height:u32, width:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:Conv1x1Params;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let oc_batch=gid.z;
  let oc=oc_batch%params.out_channels; let batch=oc_batch/params.out_channels;
  if(out_x>=params.width||out_y>=params.height||batch>=params.batch){return;}
  var sum:f32=0.0;
  let in_base=batch*params.in_channels*params.height*params.width+out_y*params.width+out_x;
  let w_base=oc*params.in_channels;
  let spatial_stride=params.height*params.width;
  let ic4=params.in_channels/4u;
  for(var i:u32=0u;i<ic4;i=i+1u){
    let ic=i*4u;
    let inp=vec4<f32>(
      input[in_base+ic*spatial_stride],
      input[in_base+(ic+1u)*spatial_stride],
      input[in_base+(ic+2u)*spatial_stride],
      input[in_base+(ic+3u)*spatial_stride]
    );
    let w=vec4<f32>(
      weight[w_base+ic],
      weight[w_base+ic+1u],
      weight[w_base+ic+2u],
      weight[w_base+ic+3u]
    );
    sum+=dot(inp,w);
  }
  sum=sum+bias[oc];
  let out_idx=batch*params.out_channels*params.height*params.width+oc*params.height*params.width+out_y*params.width+out_x;
  output[out_idx]=sum;
}
`),it=be(`
struct CanvasParams { in_width:u32, in_height:u32, out_size:u32, }
@group(0)@binding(0) var input_tex:texture_2d<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> params:CanvasParams;
@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let x=gid.x; let y=gid.y;
  if(x>=params.in_width||y>=params.in_height){return;}
  let pixel=textureLoad(input_tex,vec2<u32>(x,y),0);
  let out_stride=params.out_size*params.out_size;
  output[0u*out_stride+y*params.out_size+x]=pixel.r;
  output[1u*out_stride+y*params.out_size+x]=pixel.g;
  output[2u*out_stride+y*params.out_size+x]=pixel.b;
}
`),ut=be(`
struct LBParams {
  src_w:u32, src_h:u32, dst_size:u32, _pad:u32,
  scale_x:f32, scale_y:f32, offset_x:f32, offset_y:f32,
}
@group(0)@binding(0) var input_tex:texture_2d<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> params:LBParams;
@group(0)@binding(3) var input_sampler:sampler;
@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let dx=gid.x; let dy=gid.y;
  if(dx>=params.dst_size||dy>=params.dst_size){return;}

  let out_stride=params.dst_size*params.dst_size;

  let src_x = (f32(dx) - params.offset_x + 0.5) * params.scale_x - 0.5;
  let src_y = (f32(dy) - params.offset_y + 0.5) * params.scale_y - 0.5;

  let in_region = src_x >= -0.5 && src_x < f32(params.src_w) - 0.5
               && src_y >= -0.5 && src_y < f32(params.src_h) - 0.5;

  if(!in_region){
    output[0u*out_stride+dy*params.dst_size+dx]=0.0;
    output[1u*out_stride+dy*params.dst_size+dx]=0.0;
    output[2u*out_stride+dy*params.dst_size+dx]=0.0;
    return;
  }

  let u = (src_x + 0.5) / f32(params.src_w);
  let v = (src_y + 0.5) / f32(params.src_h);
  let pixel = textureSampleLevel(input_tex, input_sampler, vec2<f32>(u, v), 0.0);

  output[0u*out_stride+dy*params.dst_size+dx]=pixel.r;
  output[1u*out_stride+dy*params.dst_size+dx]=pixel.g;
  output[2u*out_stride+dy*params.dst_size+dx]=pixel.b;
}
`),ct=be(`
@group(0)@binding(0) var<storage,read_write> buf:array<f32>;
@group(0)@binding(1) var<uniform> count:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x;
  if(idx>=count){return;}
  let v=buf[idx];
  buf[idx]=unpack2x16float(pack2x16float(vec2(v,0.0))).x;
}
`);async function kt(a){let[o,t]=await Promise.all([fetch(`${a}/face_detector.json`),fetch(`${a}/face_detector.bin`)]);if(!o.ok)throw new Error(`Failed to load face detector metadata: ${o.status}`);if(!t.ok)throw new Error(`Failed to load face detector weights: ${t.status}`);let y=await o.json(),d=await t.arrayBuffer(),u=new Map;for(let c of y){let h=new Float32Array(d,c.offset,c.size/4);u.set(c.key,{data:h,shape:c.shape})}return u}async function je(a,o){let t;if(o)t=o;else{if(!navigator.gpu)throw new Error("WebGPU not supported");let e=await navigator.gpu.requestAdapter();if(!e)throw new Error("No GPU adapter found");t=await e.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(e.limits.maxStorageBuffersPerShaderStage,8)}})}let y={r:"read-only-storage",s:"storage",u:"uniform"};function d(e){return t.createBindGroupLayout({entries:e.map((r,p)=>r==="t"?{binding:p,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:r==="sm"?{binding:p,visibility:GPUShaderStage.COMPUTE,sampler:{}}:{binding:p,visibility:GPUShaderStage.COMPUTE,buffer:{type:y[r]}})})}let u=t.createSampler({magFilter:"linear",minFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge"}),c=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,h=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,_=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,m=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function s(e,r){return t.createBuffer({size:Math.max(e,4),usage:r})}function l(e,r,p){t.queue.writeBuffer(e,r,p)}function f(e){let r=s(e.data.byteLength,c);return l(r,0,e.data),r}let P=Array.from(a.keys());function i(e){let r=a.get(e);if(!r)throw new Error(`Weight not found: ${e}`);return r}function g(...e){let r=P.find(p=>e.every(C=>p.includes(C)));if(!r)throw new Error(`Weight not found for: ${e.join(", ")}`);return i(r)}function T(e){let[,r,p,C]=e.shape,U=new Float32Array(C*9);for(let B=0;B<C;B++)for(let k=0;k<r;k++)for(let L=0;L<p;L++)U[B*9+k*3+L]=e.data[k*p*C+L*C+B];return U}function w(e){let[r,,,p]=e.shape,C=new Float32Array(r*p);for(let U=0;U<r;U++)for(let B=0;B<p;B++)C[U*p+B]=e.data[U*p+B];return C}let ne=t.createShaderModule({code:nt}),ae=t.createShaderModule({code:at}),Q=t.createShaderModule({code:ot}),Ce=t.createShaderModule({code:st}),Ue=t.createShaderModule({code:it}),me=t.createShaderModule({code:ut}),xe=t.createShaderModule({code:ct}),ue=d(["r","r","r","s","u"]),J=d(["r","r","r","s","u"]),Ee=d(["r","r","r","r","s","u"]),Te=d(["r","r","r","s","u"]),Fe=d(["t","s","u"]),Me=d(["t","s","u","sm"]),De=d(["s","u"]);function j(e,r){return t.createComputePipeline({layout:t.createPipelineLayout({bindGroupLayouts:[e]}),compute:{module:r,entryPoint:"main"}})}let we=j(ue,ne),Se=j(J,ae),Ie=j(Ee,Q),Le=j(Te,Ce),ke=j(Fe,Ue),Oe=j(Me,me),qe=j(De,xe),Ke=i("conv2d/Kernel"),n=i("conv2d/Bias"),x=f(Ke),b=f(n),D=[{dwKey:"depthwise_conv2d/Kernel",pwKey:"conv2d_1/Kernel",biasKey:"conv2d_1/Bias",inCh:24,outCh:24,stride:1,inH:64},{dwKey:"depthwise_conv2d_1/Kernel",pwKey:"conv2d_2/Kernel",biasKey:"conv2d_2/Bias",inCh:24,outCh:28,stride:1,inH:64},{dwKey:"depthwise_conv2d_2/Kernel",pwKey:"conv2d_3/Kernel",biasKey:"conv2d_3/Bias",inCh:28,outCh:32,stride:2,inH:64},{dwKey:"depthwise_conv2d_3/Kernel",pwKey:"conv2d_4/Kernel",biasKey:"conv2d_4/Bias",inCh:32,outCh:36,stride:1,inH:32},{dwKey:"depthwise_conv2d_4/Kernel",pwKey:"conv2d_5/Kernel",biasKey:"conv2d_5/Bias",inCh:36,outCh:42,stride:1,inH:32},{dwKey:"depthwise_conv2d_5/Kernel",pwKey:"conv2d_6/Kernel",biasKey:"conv2d_6/Bias",inCh:42,outCh:48,stride:2,inH:32},{dwKey:"depthwise_conv2d_6/Kernel",pwKey:"conv2d_7/Kernel",biasKey:"conv2d_7/Bias",inCh:48,outCh:56,stride:1,inH:16},{dwKey:"depthwise_conv2d_7/Kernel",pwKey:"conv2d_8/Kernel",biasKey:"conv2d_8/Bias",inCh:56,outCh:64,stride:1,inH:16},{dwKey:"depthwise_conv2d_8/Kernel",pwKey:"conv2d_9/Kernel",biasKey:"conv2d_9/Bias",inCh:64,outCh:72,stride:1,inH:16},{dwKey:"depthwise_conv2d_9/Kernel",pwKey:"conv2d_10/Kernel",biasKey:"conv2d_10/Bias",inCh:72,outCh:80,stride:1,inH:16},{dwKey:"depthwise_conv2d_10/Kernel",pwKey:"conv2d_11/Kernel",biasKey:"conv2d_11/Bias",inCh:80,outCh:88,stride:1,inH:16},{dwKey:"depthwise_conv2d_11/Kernel",pwKey:"conv2d_12/Kernel",biasKey:"conv2d_12/Bias",inCh:88,outCh:96,stride:2,inH:16}].map(e=>{let r=i(e.dwKey),p=i(e.pwKey),C=i(e.biasKey),U=T(r),B=s(U.byteLength,c);l(B,0,U);let k=new Float32Array(e.inCh),L=s(k.byteLength,c);l(L,0,k);let N=w(p),re=s(N.byteLength,c);l(re,0,N);let G=f(C);return{dwWeightBuf:B,dwBiasBuf:L,pwWeightBuf:re,pwBiasBuf:G,inCh:e.inCh,outCh:e.outCh,stride:e.stride,inH:e.inH}}),H=w(i("classificator_8/Kernel")),z=s(H.byteLength,c);l(z,0,H);let A=f(i("classificator_8/Bias")),S=w(i("regressor_8/Kernel")),I=s(S.byteLength,c);l(I,0,S);let K=f(i("regressor_8/Bias")),v=w(i("classificator_16/Kernel")),R=s(v.byteLength,c);l(R,0,v);let W=f(i("classificator_16/Bias")),M=w(i("regressor_16/Kernel")),ee=s(M.byteLength,c);l(ee,0,M);let V=f(i("regressor_16/Bias")),Y=Math.max(16384*3,4096*96,1024*96,256*96,6144)*4,q=s(16384*3*4,c),ce=s(Y,h),Pe=s(Y,h),ge=s(Y,h),te=s(256*2*4,_),de=s(256*32*4,_),fe=s(384*4,_),le=s(6144*4,_),oe=s(256*2*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),se=s(256*32*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),pe=s(384*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Be=s(6144*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),$e=t.createTexture({size:[128,128,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});function $(e,r){return Math.ceil(e/r)}function he(e){let r=s(e.byteLength,m);return l(r,0,e),r}let pt=he(new Uint32Array([1,3,24,128,128,64,64])),mt=D.map(e=>{let r=e.stride===2?e.inH/2:e.inH,p=r,C=e.stride===2?0:1,U=e.inCh;return{dw:he(new Uint32Array([1,e.inCh,e.inH,e.inH,r,p,e.stride,C])),pw:he(new Uint32Array([1,e.inCh,e.outCh,r,p,U,e.stride,e.inH,e.inH])),outH:r,outW:p}}),ft=he(new Uint32Array([1,88,2,16,16])),lt=he(new Uint32Array([1,88,32,16,16])),ht=he(new Uint32Array([1,96,6,8,8])),_t=he(new Uint32Array([1,96,96,8,8])),gt=he(new Uint32Array([128,128,128])),bt=t.createBindGroup({layout:Fe,entries:[{binding:0,resource:$e.createView()},{binding:1,resource:{buffer:q}},{binding:2,resource:{buffer:gt}}]}),ve=null,Ze=0,Qe=0,Je=s(32,m);function yt(e,r){return ve&&Ze===e&&Qe===r||(ve&&ve.destroy(),ve=t.createTexture({size:[e,r,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),Ze=e,Qe=r),ve}let xt=t.createBindGroup({layout:ue,entries:[{binding:0,resource:{buffer:q}},{binding:1,resource:{buffer:x}},{binding:2,resource:{buffer:b}},{binding:3,resource:{buffer:ce}},{binding:4,resource:{buffer:pt}}]});function Lt(e,r,p){}function wt(e,r,p,C,U,B){let k=B.outH,L=t.createBindGroup({layout:J,entries:[{binding:0,resource:{buffer:p}},{binding:1,resource:{buffer:r.dwWeightBuf}},{binding:2,resource:{buffer:r.dwBiasBuf}},{binding:3,resource:{buffer:ge}},{binding:4,resource:{buffer:B.dw}}]}),N=e.beginComputePass();N.setPipeline(Se),N.setBindGroup(0,L),N.dispatchWorkgroups($(k,8),$(B.outH,8),r.inCh),N.end(),r.inCh*B.outH*k;let re=t.createBindGroup({layout:Ee,entries:[{binding:0,resource:{buffer:ge}},{binding:1,resource:{buffer:U}},{binding:2,resource:{buffer:r.pwWeightBuf}},{binding:3,resource:{buffer:r.pwBiasBuf}},{binding:4,resource:{buffer:C}},{binding:5,resource:{buffer:B.pw}}]}),G=e.beginComputePass();G.setPipeline(Ie),G.setBindGroup(0,re),G.dispatchWorkgroups($(k,8),$(B.outH,8),r.outCh),G.end(),r.outCh*B.outH*k}function Ae(e,r,p,C,U,B,k,L,N){let re=t.createBindGroup({layout:Te,entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:p}},{binding:2,resource:{buffer:C}},{binding:3,resource:{buffer:U}},{binding:4,resource:{buffer:B}}]}),G=e.beginComputePass();G.setPipeline(Le),G.setBindGroup(0,re),G.dispatchWorkgroups($(N,8),$(L,8),k),G.end()}async function et(e){16384*3;{let E=e.beginComputePass();E.setPipeline(we),E.setBindGroup(0,xt),E.dispatchWorkgroups($(64,8),$(64,8),24),E.end()}4096*24;let r=ce,p=Pe,C=r;for(let E=0;E<D.length;E++){let O=D[E];wt(e,O,r,p,r,mt[E]);let X=r;r=p,p=X,E===10&&(C=r)}Ae(e,C,z,A,te,ft,2,16,16),256*2,Ae(e,C,I,K,de,lt,32,16,16),256*32,Ae(e,r,R,W,fe,ht,6,8,8),Ae(e,r,ee,V,le,_t,96,8,8),t.queue.submit([e.finish()]);let U=t.createCommandEncoder();U.copyBufferToBuffer(te,0,oe,0,256*2*4),U.copyBufferToBuffer(de,0,se,0,256*32*4),U.copyBufferToBuffer(fe,0,pe,0,384*4),U.copyBufferToBuffer(le,0,Be,0,6144*4),t.queue.submit([U.finish()]),await Promise.all([oe.mapAsync(GPUMapMode.READ),se.mapAsync(GPUMapMode.READ),pe.mapAsync(GPUMapMode.READ),Be.mapAsync(GPUMapMode.READ)]);let B=new Float32Array(oe.getMappedRange()).slice(),k=new Float32Array(se.getMappedRange()).slice(),L=new Float32Array(pe.getMappedRange()).slice(),N=new Float32Array(Be.getMappedRange()).slice();oe.unmap(),se.unmap(),pe.unmap(),Be.unmap();let re=896,G=new Float32Array(re),_e=new Float32Array(re*16),ie=0;for(let E=0;E<16;E++)for(let O=0;O<16;O++)for(let X=0;X<2;X++){G[ie]=B[X*256+E*16+O];for(let Z=0;Z<16;Z++){let Re=X*16+Z;_e[ie*16+Z]=k[Re*256+E*16+O]}ie++}for(let E=0;E<8;E++)for(let O=0;O<8;O++)for(let X=0;X<6;X++){G[ie]=L[X*64+E*8+O];for(let Z=0;Z<16;Z++){let Re=X*16+Z;_e[ie*16+Z]=N[Re*64+E*8+O]}ie++}return{scores:G,regressors:_e}}async function Pt(e){t.queue.copyExternalImageToTexture({source:e},{texture:$e},[128,128]);let r=t.createCommandEncoder();{let p=r.beginComputePass();p.setPipeline(ke),p.setBindGroup(0,bt),p.dispatchWorkgroups($(128,16),$(128,16),1),p.end()}return et(r)}async function Bt(e,r,p){let C=Math.min(128/r,128/p),U=Math.round(r*C),B=Math.round(p*C),k=Math.floor((128-U)/2),L=Math.floor((128-B)/2),N=k/128,re=L/128,G=yt(r,p),_e;e instanceof HTMLVideoElement?_e=await createImageBitmap(e,{colorSpaceConversion:"none"}):e instanceof HTMLImageElement?_e=await createImageBitmap(e,{colorSpaceConversion:"none"}):_e=e,t.queue.copyExternalImageToTexture({source:_e},{texture:G},[r,p]);let ie=new ArrayBuffer(32),E=new Uint32Array(ie),O=new Float32Array(ie);E[0]=r,E[1]=p,E[2]=128,E[3]=0,O[4]=r/U,O[5]=p/B,O[6]=k,O[7]=L,t.queue.writeBuffer(Je,0,ie);let X=t.createBindGroup({layout:Me,entries:[{binding:0,resource:G.createView()},{binding:1,resource:{buffer:q}},{binding:2,resource:{buffer:Je}},{binding:3,resource:u}]}),Z=t.createCommandEncoder();{let Ge=Z.beginComputePass();Ge.setPipeline(Oe),Ge.setBindGroup(0,X),Ge.dispatchWorkgroups($(128,16),$(128,16),1),Ge.end()}return{output:await et(Z),lbPadX:N,lbPadY:re}}return{device:t,run:Pt,runWithResize:Bt}}function At(a){return a.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var Rt=At(`
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
`);function Ve(a){let o=a.createShaderModule({code:Rt}),t=a.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:4,visibility:GPUShaderStage.COMPUTE,sampler:{}}]}),y=a.createComputePipeline({layout:a.createPipelineLayout({bindGroupLayouts:[t]}),compute:{module:o,entryPoint:"main"}}),d=a.createSampler({magFilter:"linear",minFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge"}),u=a.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),c=a.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),h=new Float32Array(8);function _(m,s,l,f,P,i,g){a.queue.writeBuffer(u,0,new Uint32Array([P,i,g,0])),h.set(f),a.queue.writeBuffer(c,0,h);let T=a.createBindGroup({layout:t,entries:[{binding:0,resource:s.createView()},{binding:1,resource:{buffer:l}},{binding:2,resource:{buffer:u}},{binding:3,resource:{buffer:c}},{binding:4,resource:d}]}),w=m.beginComputePass();w.setPipeline(y),w.setBindGroup(0,T),w.dispatchWorkgroups(Math.ceil(g/16),Math.ceil(g/16),1),w.end()}return{crop:_}}var Gt="https://cdn.jsdelivr.net/npm/@svenflow/micro-facemesh@0.1.0/weights",ye=256,dt=478,Dt=33,St=263;async function It(a={}){let{weightsUrl:o,scoreThreshold:t=.5,faceScoreThreshold:y=.5,maxFaces:d=1}=a;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-facemesh requires WebGPU. Check browser support at https://webgpureport.org");let u=(o??Gt).replace(/\/$/,"")+"/",[c,h,_,m]=await Promise.all([fetch(`${u}face_landmarks_weights_f16.json`),fetch(`${u}face_landmarks_weights_f16.bin`),fetch(`${u}face_detector_weights.json`),fetch(`${u}face_detector_weights.bin`)]);if(!c.ok)throw new Error(`Failed to fetch landmark weights: ${c.status}`);if(!h.ok)throw new Error(`Failed to fetch landmark weights: ${h.status}`);if(!_.ok)throw new Error(`Failed to fetch face detector weights: ${_.status}`);if(!m.ok)throw new Error(`Failed to fetch face detector weights: ${m.status}`);let[s,l,f,P]=await Promise.all([c.json(),h.arrayBuffer(),_.json(),m.arrayBuffer()]),i=He(s,l),g=new Map;for(let n=0;n<f.keys.length;n++){let x=f.keys[n],b=f.shapes[n],F=f.offsets[n],D=b.reduce((z,A)=>z*A,1),H=new Float32Array(P,F,D);g.set(x,{data:H,shape:b})}let T=await je(g),w=Xe(T,{scoreThreshold:y,maxFaces:d}),{compileFaceLandmarkModel:ne}=await import("./face_landmark_model-A2HWSD7C.js"),ae=await ne(i,T.device),Q=[];function Ce(n,x,b){let F=n[Dt],D=n[St],H=(D.x-F.x)*x,z=(D.y-F.y)*b,A=Math.atan2(z,H),S=Math.cos(A),I=Math.sin(A),K=1/0,v=-1/0,R=1/0,W=-1/0;for(let de=0;de<dt;de++){let fe=n[de],le=fe.x*x,oe=fe.y*b,se=S*le+I*oe,pe=-I*le+S*oe;K=Math.min(K,se),v=Math.max(v,se),R=Math.min(R,pe),W=Math.max(W,pe)}let M=(K+v)/2,ee=(R+W)/2,V=v-K,Y=W-R,q=S*M-I*ee,ce=I*M+S*ee,te=Math.max(V,Y)*1.5;return{centerXpx:q,centerYpx:ce,sizePx:te,rotation:A}}function Ue(n,x,b){let[F,D,H,z]=n.box,A=n.keypoints[0],S=n.keypoints[1],I=(A[0]+S[0])/2,K=(A[1]+S[1])/2,v=n.keypoints[2],R=v[0]-I,W=v[1]-K,M=Math.atan2(W,R),V=Math.PI/2-M,Y=V-2*Math.PI*Math.floor((V+Math.PI)/(2*Math.PI)),Pe=Math.max(H*x,z*b)*1.5;return{centerXpx:F*x,centerYpx:D*b,sizePx:Pe,rotation:Y}}let me=ae.device,xe=null,ue=null,J=null,Ee=0,Te=0;function Fe(){return xe||(xe=Ve(me)),xe}function Me(){return ue||(ue=me.createBuffer({size:3*ye*ye*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC})),ue}function De(n,x){return(!J||Ee!==n||Te!==x)&&(J&&J.destroy(),J=me.createTexture({size:[n,x],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),Ee=n,Te=x),J}let j=0,we=0;function Se(n){let x=1/(1-2*j),b=1/(1-2*we);return{score:n.score,box:[(n.box[0]-j)*x,(n.box[1]-we)*b,n.box[2]*x,n.box[3]*b],keypoints:n.keypoints.map(([F,D])=>[(F-j)*x,(D-we)*b])}}function Ie(n){return n instanceof HTMLCanvasElement||n instanceof OffscreenCanvas?[n.width,n.height]:typeof ImageBitmap<"u"&&n instanceof ImageBitmap?[n.width,n.height]:n instanceof ImageData?[n.width,n.height]:n instanceof HTMLVideoElement?[n.videoWidth,n.videoHeight]:n instanceof HTMLImageElement?[n.naturalWidth,n.naturalHeight]:[ye,ye]}function Le(n){return 1/(1+Math.exp(-n))}async function ke(n,x,b,F,D,H,z=!1){let A=Math.cos(n.rotation),S=Math.sin(n.rotation),I=n.sizePx/ye,K=ye/2,v=A*I/b,R=-S*I/b,W=n.centerXpx/b-K*(v+R),M=S*I/F,ee=A*I/F,V=n.centerYpx/F-K*(M+ee),Y=me.createCommandEncoder();D.crop(Y,x,H,[v,R,W,M,ee,V],b,F,ye),me.queue.submit([Y.finish()]);let q=await ae.runFromGPUBuffer(H),ce=Le(q.facePresence[0]),Pe=z?Math.min(t,.1):t;if(ce<Pe)return null;let ge=[];for(let te=0;te<dt;te++){let de=q.landmarks[te*3],fe=q.landmarks[te*3+1],le=q.landmarks[te*3+2],oe=(de-.5)*n.sizePx,se=(fe-.5)*n.sizePx,pe=A*oe-S*se+n.centerXpx,Be=S*oe+A*se+n.centerYpx;ge.push({x:pe/b,y:Be/F,z:le})}return{landmarks:ge,score:ce}}async function Oe(n){let x,b,F;if(n instanceof HTMLVideoElement)x=n.videoWidth,b=n.videoHeight,F=n;else if(n instanceof HTMLImageElement)x=n.naturalWidth,b=n.naturalHeight,F=n;else if(n instanceof ImageData){let v=await createImageBitmap(n,{colorSpaceConversion:"none"});[x,b]=[v.width,v.height],F=v}else[x,b]=Ie(n),F=n;let D=Fe(),H=Me(),z=De(x,b);if(me.queue.copyExternalImageToTexture({source:F},{texture:z},[x,b]),Q.length>0){let v=[];for(let R of Q){let W=Ce(R.landmarks,x,b),M=await ke(W,z,x,b,D,H,!0);if(M){let V=[133,362,1,13,234,454].map(Y=>M.landmarks[Y]);v.push({score:M.score,landmarks:M.landmarks,keypoints:We(V)})}}if(v.length>0)return Q=v.map(R=>({landmarks:R.landmarks})),v;Q=[]}let{detections:A,lbPadX:S,lbPadY:I}=await w.detectRawWithResize(F,x,b);if(j=S,we=I,A.length===0)return Q=[],[];let K=[];for(let v of A){let R=Se(v),W=Ue(R,x,b),M=await ke(W,z,x,b,D,H);if(M){let V=[133,362,1,13,234,454].map(Y=>M.landmarks[Y]);K.push({score:M.score,landmarks:M.landmarks,keypoints:We(V)})}}return Q=K.map(v=>({landmarks:v.landmarks})),K}function qe(){J&&J.destroy(),ue&&ue.destroy(),J=null,ue=null,xe=null,ae.device.destroy(),T.device.destroy()}function Ke(){Q=[]}return{detect:Oe,dispose:qe,reset:Ke}}export{ze as FACE_KEYPOINT_NAMES,Ct as FACE_LANDMARK_INDICES,je as compileFaceDetectorModel,vt as compileFaceLandmarkModel,Ft as computeCropTransform,Ve as createCropPipeline,Xe as createFaceDetector,It as createFacemesh,rt as detectionToROI,kt as loadFaceDetectorWeights,He as loadFaceLandmarkWeights,Mt as projectLandmarksToOriginal};
