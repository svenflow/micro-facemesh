import{a as Ve,b as kt}from"./chunk-FNDNR72C.js";var qe=["rightEye","leftEye","noseTip","mouthCenter","rightEarTragion","leftEarTragion"];function st(a){let o={};for(let t=0;t<qe.length;t++)o[qe[t]]=a[t];return o}var Mt={silhouetteTop:10,silhouetteBottom:152,leftEyeInner:133,leftEyeOuter:33,rightEyeInner:362,rightEyeOuter:263,leftEyebrowUpper:66,rightEyebrowUpper:296,noseTip:1,noseBottom:2,noseBridgeTop:6,upperLipTop:13,lowerLipBottom:14,mouthLeft:61,mouthRight:291,chin:152,leftEarTragion:234,rightEarTragion:454,leftIrisCenter:468,leftIrisRight:469,leftIrisTop:470,leftIrisLeft:471,leftIrisBottom:472,rightIrisCenter:473,rightIrisRight:474,rightIrisTop:475,rightIrisLeft:476,rightIrisBottom:477};function Rt(){let a=[];for(let o=0;o<16;o++)for(let t=0;t<16;t++){let w=(t+.5)/16,m=(o+.5)/16;for(let c=0;c<2;c++)a.push({x:w,y:m})}for(let o=0;o<8;o++)for(let t=0;t<8;t++){let w=(t+.5)/8,m=(o+.5)/8;for(let c=0;c<6;c++)a.push({x:w,y:m})}return a}var it=Rt();function At(a){return 1/(1+Math.exp(-a))}function $e(a,o){let t=[],{scores:w,regressors:m}=a,c=128;for(let d=0;d<it.length;d++){let g=At(w[d]);if(g<o)continue;let y=it[d],l=d*16,u=y.x+m[l+0]/c,_=y.y+m[l+1]/c,h=m[l+2]/c,v=m[l+3]/c,s=[];for(let x=0;x<6;x++){let F=y.x+m[l+4+x*2]/c,P=y.y+m[l+4+x*2+1]/c;s.push([F,P])}t.push({score:g,box:[u,_,h,v],keypoints:s})}return t}function Ze(a,o){if(a.length===0)return[];let t=[...a].sort((c,d)=>d.score-c.score),w=[],m=new Set;for(let c=0;c<t.length;c++){if(m.has(c))continue;let d=[c];for(let s=c+1;s<t.length;s++)m.has(s)||St(t[c],t[s])>o&&(d.push(s),m.add(s));let g=0,y=0,l=0,u=0,_=0,h=[];for(let s=0;s<6;s++)h.push([0,0]);for(let s of d){let x=t[s],F=x.score;g+=F,y+=x.box[0]*F,l+=x.box[1]*F,u+=x.box[2]*F,_+=x.box[3]*F;for(let P=0;P<6;P++)h[P][0]+=x.keypoints[P][0]*F,h[P][1]+=x.keypoints[P][1]*F}let v=1/g;w.push({score:t[c].score,box:[y*v,l*v,u*v,_*v],keypoints:h.map(([s,x])=>[s*v,x*v])})}return w}function St(a,o){let t=a.box[0]-a.box[2]/2,w=a.box[1]-a.box[3]/2,m=a.box[0]+a.box[2]/2,c=a.box[1]+a.box[3]/2,d=o.box[0]-o.box[2]/2,g=o.box[1]-o.box[3]/2,y=o.box[0]+o.box[2]/2,l=o.box[1]+o.box[3]/2,u=Math.max(t,d),_=Math.max(w,g),h=Math.min(m,y),v=Math.min(c,l),s=Math.max(0,h-u),x=Math.max(0,v-_),F=s*x,P=(m-t)*(c-w),se=(y-d)*(l-g),X=P+se-F;return X>0?F/X:0}function ct(a){let[o,t,w,m]=a.box,c=a.keypoints[0],d=a.keypoints[1],g=(c[0]+d[0])/2,y=(c[1]+d[1])/2,l=a.keypoints[2],u=l[0]-g,_=l[1]-y,h=Math.atan2(_,u),s=Math.PI/2-h,x=Math.max(w,m),P=x*1.5,se=0*x,X=Math.cos(s),j=Math.sin(s),xe=se*j,Fe=se*X;return{centerX:o+xe,centerY:t+Fe,width:P,height:P,rotation:s}}function Qe(a,o={}){let{scoreThreshold:t=.5,nmsThreshold:w=.3,maxFaces:m=1}=o;async function c(l){let u=await a.run(l),_=$e(u,t);return Ze(_,w).slice(0,m).map(ct)}async function d(l){let u=await a.run(l),_=$e(u,t);return Ze(_,w).slice(0,m)}async function g(l,u,_){let{output:h,lbPadX:v,lbPadY:s}=await a.runWithResize(l,u,_),x=$e(h,t);return{detections:Ze(x,w).slice(0,m),lbPadX:v,lbPadY:s}}async function y(l,u,_){let{output:h,lbPadX:v,lbPadY:s}=await a.runWithResize(l,u,_);return{scores:h.scores,regressors:h.regressors,lbPadX:v,lbPadY:s}}return{detect:c,detectRaw:d,detectRawWithResize:g,detectRawSSD:y,model:a}}function Gt(a,o=256){let t=Math.cos(a.rotation),w=Math.sin(a.rotation),m=a.width/o,c=a.height/o,d=m*t,g=m*w,y=-c*w,l=c*t,u=a.centerX-(d*o/2+g*o/2),_=a.centerY-(y*o/2+l*o/2),h=d*l-g*y,v=l/h,s=-g/h,x=-y/h,F=d/h,P=-(v*u+s*_),se=-(x*u+F*_);return{forward:[d,g,u,y,l,_],inverse:[v,s,P,x,F,se]}}function Lt(a,o,t,w){let m=Math.cos(o.rotation),c=Math.sin(o.rotation),d=Math.min(t,w),g=o.width*d,y=g/t,l=g/w;return a.map(u=>{let _=u.x-.5,h=u.y-.5,v=m*_+c*h,s=-c*_+m*h;return{x:v*y+o.centerX,y:s*l+o.centerY,z:u.z}})}function ge(a){return a.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var ut=ge(`
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
`),dt=ge(`
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
`),pt=ge(`
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
`),mt=ge(`
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
`),ft=ge(`
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
  output[0u*out_stride+y*params.out_size+x]=pixel.r*2.0-1.0;
  output[1u*out_stride+y*params.out_size+x]=pixel.g*2.0-1.0;
  output[2u*out_stride+y*params.out_size+x]=pixel.b*2.0-1.0;
}
`),lt=ge(`
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

  output[0u*out_stride+dy*params.dst_size+dx]=pixel.r*2.0-1.0;
  output[1u*out_stride+dy*params.dst_size+dx]=pixel.g*2.0-1.0;
  output[2u*out_stride+dy*params.dst_size+dx]=pixel.b*2.0-1.0;
}
`),ht=ge(`
@group(0)@binding(0) var<storage,read_write> buf:array<f32>;
@group(0)@binding(1) var<uniform> count:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x;
  if(idx>=count){return;}
  let v=buf[idx];
  buf[idx]=unpack2x16float(pack2x16float(vec2(v,0.0))).x;
}
`);async function Dt(a){let[o,t]=await Promise.all([fetch(`${a}/face_detector.json`),fetch(`${a}/face_detector.bin`)]);if(!o.ok)throw new Error(`Failed to load face detector metadata: ${o.status}`);if(!t.ok)throw new Error(`Failed to load face detector weights: ${t.status}`);let w=await o.json(),m=await t.arrayBuffer(),c=new Map;for(let d of w){let g=new Float32Array(m,d.offset,d.size/4);c.set(d.key,{data:g,shape:d.shape})}return c}async function Je(a,o){let t;if(o)t=o;else{if(!navigator.gpu)throw new Error("WebGPU not supported");let e=await navigator.gpu.requestAdapter();if(!e)throw new Error("No GPU adapter found");t=await e.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(e.limits.maxStorageBuffersPerShaderStage,8)}})}let w={r:"read-only-storage",s:"storage",u:"uniform"};function m(e){return t.createBindGroupLayout({entries:e.map((r,f)=>r==="t"?{binding:f,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:r==="sm"?{binding:f,visibility:GPUShaderStage.COMPUTE,sampler:{}}:{binding:f,visibility:GPUShaderStage.COMPUTE,buffer:{type:w[r]}})})}let c=t.createSampler({magFilter:"linear",minFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge"}),d=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,g=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,y=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,l=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function u(e,r){return t.createBuffer({size:Math.max(e,4),usage:r})}function _(e,r,f){t.queue.writeBuffer(e,r,f)}function h(e){let r=u(e.data.byteLength,d);return _(r,0,e.data),r}let v=Array.from(a.keys());function s(e){let r=a.get(e);if(!r)throw new Error(`Weight not found: ${e}`);return r}function x(...e){let r=v.find(f=>e.every(E=>f.includes(E)));if(!r)throw new Error(`Weight not found for: ${e.join(", ")}`);return s(r)}function F(e){let[,r,f,E]=e.shape,U=new Float32Array(E*9);for(let C=0;C<E;C++)for(let O=0;O<r;O++)for(let Y=0;Y<f;Y++)U[C*9+O*3+Y]=e.data[O*f*E+Y*E+C];return U}function P(e){let[r,,,f]=e.shape,E=new Float32Array(r*f);for(let U=0;U<r;U++)for(let C=0;C<f;C++)E[U*f+C]=e.data[U*f+C];return E}let se=t.createShaderModule({code:ut}),X=t.createShaderModule({code:dt}),j=t.createShaderModule({code:pt}),xe=t.createShaderModule({code:mt}),Fe=t.createShaderModule({code:ft}),V=t.createShaderModule({code:lt}),we=t.createShaderModule({code:ht}),de=m(["r","r","r","s","u"]),ne=m(["r","r","r","s","u"]),Ue=m(["r","r","r","r","s","u"]),ke=m(["r","r","r","s","u"]),Me=m(["t","s","u"]),Re=m(["t","s","u","sm"]),Le=m(["s","u"]);function Q(e,r){return t.createComputePipeline({layout:t.createPipelineLayout({bindGroupLayouts:[e]}),compute:{module:r,entryPoint:"main"}})}let Pe=Q(de,se),Ye=Q(ne,X),Ne=Q(Ue,j),tt=Q(ke,xe),Ae=Q(Me,Fe),Se=Q(Re,V),De=Q(Le,we),Xe=s("conv2d/Kernel"),re=s("conv2d/Bias"),ve=h(Xe),Be=h(re),be=[{dwKey:"depthwise_conv2d/Kernel",pwKey:"conv2d_1/Kernel",biasKey:"conv2d_1/Bias",inCh:24,outCh:24,stride:1,inH:64},{dwKey:"depthwise_conv2d_1/Kernel",pwKey:"conv2d_2/Kernel",biasKey:"conv2d_2/Bias",inCh:24,outCh:28,stride:1,inH:64},{dwKey:"depthwise_conv2d_2/Kernel",pwKey:"conv2d_3/Kernel",biasKey:"conv2d_3/Bias",inCh:28,outCh:32,stride:2,inH:64},{dwKey:"depthwise_conv2d_3/Kernel",pwKey:"conv2d_4/Kernel",biasKey:"conv2d_4/Bias",inCh:32,outCh:36,stride:1,inH:32},{dwKey:"depthwise_conv2d_4/Kernel",pwKey:"conv2d_5/Kernel",biasKey:"conv2d_5/Bias",inCh:36,outCh:42,stride:1,inH:32},{dwKey:"depthwise_conv2d_5/Kernel",pwKey:"conv2d_6/Kernel",biasKey:"conv2d_6/Bias",inCh:42,outCh:48,stride:2,inH:32},{dwKey:"depthwise_conv2d_6/Kernel",pwKey:"conv2d_7/Kernel",biasKey:"conv2d_7/Bias",inCh:48,outCh:56,stride:1,inH:16},{dwKey:"depthwise_conv2d_7/Kernel",pwKey:"conv2d_8/Kernel",biasKey:"conv2d_8/Bias",inCh:56,outCh:64,stride:1,inH:16},{dwKey:"depthwise_conv2d_8/Kernel",pwKey:"conv2d_9/Kernel",biasKey:"conv2d_9/Bias",inCh:64,outCh:72,stride:1,inH:16},{dwKey:"depthwise_conv2d_9/Kernel",pwKey:"conv2d_10/Kernel",biasKey:"conv2d_10/Bias",inCh:72,outCh:80,stride:1,inH:16},{dwKey:"depthwise_conv2d_10/Kernel",pwKey:"conv2d_11/Kernel",biasKey:"conv2d_11/Bias",inCh:80,outCh:88,stride:1,inH:16},{dwKey:"depthwise_conv2d_11/Kernel",pwKey:"conv2d_12/Kernel",biasKey:"conv2d_12/Bias",inCh:88,outCh:96,stride:2,inH:16},{dwKey:"depthwise_conv2d_12/Kernel",pwKey:"conv2d_13/Kernel",biasKey:"conv2d_13/Bias",inCh:96,outCh:96,stride:1,inH:8},{dwKey:"depthwise_conv2d_13/Kernel",pwKey:"conv2d_14/Kernel",biasKey:"conv2d_14/Bias",inCh:96,outCh:96,stride:1,inH:8},{dwKey:"depthwise_conv2d_14/Kernel",pwKey:"conv2d_15/Kernel",biasKey:"conv2d_15/Bias",inCh:96,outCh:96,stride:1,inH:8},{dwKey:"depthwise_conv2d_15/Kernel",pwKey:"conv2d_16/Kernel",biasKey:"conv2d_16/Bias",inCh:96,outCh:96,stride:1,inH:8}].map(e=>{let r=s(e.dwKey),f=s(e.pwKey),E=s(e.biasKey),U=F(r),C=u(U.byteLength,d);_(C,0,U);let O=new Float32Array(e.inCh),Y=u(O.byteLength,d);_(Y,0,O);let $=P(f),oe=u($.byteLength,d);_(oe,0,$);let z=h(E);return{dwWeightBuf:C,dwBiasBuf:Y,pwWeightBuf:oe,pwBiasBuf:z,inCh:e.inCh,outCh:e.outCh,stride:e.stride,inH:e.inH}}),Ie=P(s("classificator_8/Kernel")),Ke=u(Ie.byteLength,d);_(Ke,0,Ie);let je=h(s("classificator_8/Bias")),n=P(s("regressor_8/Kernel")),i=u(n.byteLength,d);_(i,0,n);let p=h(s("regressor_8/Bias")),b=P(s("classificator_16/Kernel")),k=u(b.byteLength,d);_(k,0,b);let H=h(s("classificator_16/Bias")),I=P(s("regressor_16/Kernel")),B=u(I.byteLength,d);_(B,0,I);let G=h(s("regressor_16/Bias")),R=Math.max(16384*3,4096*96,1024*96,256*96,6144)*4,L=u(16384*3*4,d),A=u(R,g),M=u(R,g),S=u(R,g),D=u(256*2*4,y),K=u(256*32*4,y),W=u(384*4,y),ae=u(6144*4,y),q=256*2*4,J=256*32*4,fe=384*4,Ce=6144*4,Ee=0,ie=q,pe=ie+J,me=pe+fe,Oe=me+Ce,ce=u(Oe,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Ge=t.createTexture({size:[128,128,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});function ee(e,r){return Math.ceil(e/r)}function le(e){let r=u(e.byteLength,l);return _(r,0,e),r}let bt=le(new Uint32Array([1,3,24,128,128,64,64])),gt=be.map(e=>{let r=e.stride===2?e.inH/2:e.inH,f=r,E=e.stride===2?0:1,U=e.inCh;return{dw:le(new Uint32Array([1,e.inCh,e.inH,e.inH,r,f,e.stride,E])),pw:le(new Uint32Array([1,e.inCh,e.outCh,r,f,U,e.stride,e.inH,e.inH])),outH:r,outW:f}}),yt=le(new Uint32Array([1,88,2,16,16])),xt=le(new Uint32Array([1,88,32,16,16])),wt=le(new Uint32Array([1,96,6,8,8])),Pt=le(new Uint32Array([1,96,96,8,8])),vt=le(new Uint32Array([128,128,128])),Bt=t.createBindGroup({layout:Me,entries:[{binding:0,resource:Ge.createView()},{binding:1,resource:{buffer:L}},{binding:2,resource:{buffer:vt}}]}),Te=null,nt=0,rt=0,at=u(32,l);function Ct(e,r){return Te&&nt===e&&rt===r||(Te&&Te.destroy(),Te=t.createTexture({size:[e,r,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),nt=e,rt=r),Te}let Et=t.createBindGroup({layout:de,entries:[{binding:0,resource:{buffer:L}},{binding:1,resource:{buffer:ve}},{binding:2,resource:{buffer:Be}},{binding:3,resource:{buffer:A}},{binding:4,resource:{buffer:bt}}]});function Yt(e,r,f){}function Tt(e,r,f,E,U,C){let O=C.outH,Y=t.createBindGroup({layout:ne,entries:[{binding:0,resource:{buffer:f}},{binding:1,resource:{buffer:r.dwWeightBuf}},{binding:2,resource:{buffer:r.dwBiasBuf}},{binding:3,resource:{buffer:S}},{binding:4,resource:{buffer:C.dw}}]}),$=e.beginComputePass();$.setPipeline(Ye),$.setBindGroup(0,Y),$.dispatchWorkgroups(ee(O,8),ee(C.outH,8),r.inCh),$.end(),r.inCh*C.outH*O;let oe=t.createBindGroup({layout:Ue,entries:[{binding:0,resource:{buffer:S}},{binding:1,resource:{buffer:U}},{binding:2,resource:{buffer:r.pwWeightBuf}},{binding:3,resource:{buffer:r.pwBiasBuf}},{binding:4,resource:{buffer:E}},{binding:5,resource:{buffer:C.pw}}]}),z=e.beginComputePass();z.setPipeline(Ne),z.setBindGroup(0,oe),z.dispatchWorkgroups(ee(O,8),ee(C.outH,8),r.outCh),z.end(),r.outCh*C.outH*O}function He(e,r,f,E,U,C,O,Y,$){let oe=t.createBindGroup({layout:ke,entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:f}},{binding:2,resource:{buffer:E}},{binding:3,resource:{buffer:U}},{binding:4,resource:{buffer:C}}]}),z=e.beginComputePass();z.setPipeline(tt),z.setBindGroup(0,oe),z.dispatchWorkgroups(ee($,8),ee(Y,8),O),z.end()}async function ot(e){16384*3;{let T=e.beginComputePass();T.setPipeline(Pe),T.setBindGroup(0,Et),T.dispatchWorkgroups(ee(64,8),ee(64,8),24),T.end()}4096*24;let r=A,f=M,E=r;for(let T=0;T<be.length;T++){let N=be[T];Tt(e,N,r,f,r,gt[T]);let Z=r;r=f,f=Z,T===10&&(E=r)}He(e,E,Ke,je,D,yt,2,16,16),256*2,He(e,E,i,p,K,xt,32,16,16),256*32,He(e,r,k,H,W,wt,6,8,8),He(e,r,B,G,ae,Pt,96,8,8),e.copyBufferToBuffer(D,0,ce,Ee,q),e.copyBufferToBuffer(K,0,ce,ie,J),e.copyBufferToBuffer(W,0,ce,pe,fe),e.copyBufferToBuffer(ae,0,ce,me,Ce),t.queue.submit([e.finish()]),await ce.mapAsync(GPUMapMode.READ);let U=ce.getMappedRange(),C=new Float32Array(U,Ee,q/4).slice(),O=new Float32Array(U,ie,J/4).slice(),Y=new Float32Array(U,pe,fe/4).slice(),$=new Float32Array(U,me,Ce/4).slice();ce.unmap();let oe=896,z=new Float32Array(oe),he=new Float32Array(oe*16),ue=0;for(let T=0;T<16;T++)for(let N=0;N<16;N++)for(let Z=0;Z<2;Z++){z[ue]=C[Z*256+T*16+N];for(let te=0;te<16;te++){let ze=Z*16+te;he[ue*16+te]=O[ze*256+T*16+N]}ue++}for(let T=0;T<8;T++)for(let N=0;N<8;N++)for(let Z=0;Z<6;Z++){z[ue]=Y[Z*64+T*8+N];for(let te=0;te<16;te++){let ze=Z*16+te;he[ue*16+te]=$[ze*64+T*8+N]}ue++}return{scores:z,regressors:he}}async function Ft(e){t.queue.copyExternalImageToTexture({source:e},{texture:Ge},[128,128]);let r=t.createCommandEncoder();{let f=r.beginComputePass();f.setPipeline(Ae),f.setBindGroup(0,Bt),f.dispatchWorkgroups(ee(128,16),ee(128,16),1),f.end()}return ot(r)}async function Ut(e,r,f){let E=Math.min(128/r,128/f),U=Math.round(r*E),C=Math.round(f*E),O=Math.floor((128-U)/2),Y=Math.floor((128-C)/2),$=O/128,oe=Y/128,z=Ct(r,f),he;e instanceof HTMLVideoElement?he=await createImageBitmap(e,{colorSpaceConversion:"none"}):e instanceof HTMLImageElement?he=await createImageBitmap(e,{colorSpaceConversion:"none"}):he=e,t.queue.copyExternalImageToTexture({source:he},{texture:z},[r,f]);let ue=new ArrayBuffer(32),T=new Uint32Array(ue),N=new Float32Array(ue);T[0]=r,T[1]=f,T[2]=128,T[3]=0,N[4]=r/U,N[5]=f/C,N[6]=O,N[7]=Y,t.queue.writeBuffer(at,0,ue);let Z=t.createBindGroup({layout:Re,entries:[{binding:0,resource:z.createView()},{binding:1,resource:{buffer:L}},{binding:2,resource:{buffer:at}},{binding:3,resource:c}]}),te=t.createCommandEncoder();{let We=te.beginComputePass();We.setPipeline(Se),We.setBindGroup(0,Z),We.dispatchWorkgroups(ee(128,16),ee(128,16),1),We.end()}return{output:await ot(te),lbPadX:$,lbPadY:oe}}return{device:t,run:Ft,runWithResize:Ut}}function It(a){return a.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var Kt=It(`
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
`);function et(a){let o=a.createShaderModule({code:Kt}),t=a.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:4,visibility:GPUShaderStage.COMPUTE,sampler:{}}]}),w=a.createComputePipeline({layout:a.createPipelineLayout({bindGroupLayouts:[t]}),compute:{module:o,entryPoint:"main"}}),m=a.createSampler({magFilter:"linear",minFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge"}),c=a.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),d=a.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),g=new Float32Array(8);function y(l,u,_,h,v,s,x){a.queue.writeBuffer(c,0,new Uint32Array([v,s,x,0])),g.set(h),a.queue.writeBuffer(d,0,g);let F=a.createBindGroup({layout:t,entries:[{binding:0,resource:u.createView()},{binding:1,resource:{buffer:_}},{binding:2,resource:{buffer:c}},{binding:3,resource:{buffer:d}},{binding:4,resource:m}]}),P=l.beginComputePass();P.setPipeline(w),P.setBindGroup(0,F),P.dispatchWorkgroups(Math.ceil(x/16),Math.ceil(x/16),1),P.end()}return{crop:y}}var Ot="https://cdn.jsdelivr.net/npm/@svenflow/micro-facemesh@0.1.0/weights",ye=256,_t=478,Ht=33,zt=263;async function Wt(a={}){let{weightsUrl:o,scoreThreshold:t=.5,faceScoreThreshold:w=.5,maxFaces:m=1}=a;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-facemesh requires WebGPU. Check browser support at https://webgpureport.org");let c=(o??Ot).replace(/\/$/,"")+"/",[d,g,y,l]=await Promise.all([fetch(`${c}face_landmarks_weights_f16.json`),fetch(`${c}face_landmarks_weights_f16.bin`),fetch(`${c}face_detector_weights.json`),fetch(`${c}face_detector_weights.bin`)]);if(!d.ok)throw new Error(`Failed to fetch landmark weights: ${d.status}`);if(!g.ok)throw new Error(`Failed to fetch landmark weights: ${g.status}`);if(!y.ok)throw new Error(`Failed to fetch face detector weights: ${y.status}`);if(!l.ok)throw new Error(`Failed to fetch face detector weights: ${l.status}`);let[u,_,h,v]=await Promise.all([d.json(),g.arrayBuffer(),y.json(),l.arrayBuffer()]),s=Ve(u,_),x=new Map;for(let n=0;n<h.keys.length;n++){let i=h.keys[n],p=h.shapes[n],b=h.offsets[n],k=p.reduce((I,B)=>I*B,1),H=new Float32Array(v,b,k);x.set(i,{data:H,shape:p})}let F=await Je(x),P=Qe(F,{scoreThreshold:w,maxFaces:m}),{compileFaceLandmarkModel:se}=await import("./face_landmark_model-5FQNDOH3.js"),X=await se(s,F.device),j=[];function xe(n,i,p){let b=n[Ht],k=n[zt],H=(k.x-b.x)*i,I=(k.y-b.y)*p,B=Math.atan2(I,H),G=Math.cos(B),R=Math.sin(B),L=1/0,A=-1/0,M=1/0,S=-1/0;for(let ie=0;ie<_t;ie++){let pe=n[ie],me=pe.x*i,Oe=pe.y*p,ce=G*me+R*Oe,Ge=-R*me+G*Oe;L=Math.min(L,ce),A=Math.max(A,ce),M=Math.min(M,Ge),S=Math.max(S,Ge)}let D=(L+A)/2,K=(M+S)/2,W=A-L,ae=S-M,q=G*D-R*K,J=R*D+G*K,Ee=Math.max(W,ae)*1.5;return{centerXpx:q,centerYpx:J,sizePx:Ee,rotation:B}}function Fe(n,i,p){let[b,k,H,I]=n.box,B=n.keypoints[0],G=n.keypoints[1],R=(B[0]+G[0])/2,L=(B[1]+G[1])/2,A=n.keypoints[2],M=A[0]-R,S=A[1]-L,D=Math.atan2(S,M),W=Math.PI/2-D,ae=W-2*Math.PI*Math.floor((W+Math.PI)/(2*Math.PI)),fe=Math.max(H*i,I*p)*1.5;return{centerXpx:b*i,centerYpx:k*p,sizePx:fe,rotation:ae}}let V=X.device,we=null,de=null,ne=null,Ue=0,ke=0;function Me(){return we||(we=et(V)),we}function Re(){return de||(de=V.createBuffer({size:3*ye*ye*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC})),de}function Le(n,i){return(!ne||Ue!==n||ke!==i)&&(ne&&ne.destroy(),ne=V.createTexture({size:[n,i],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),Ue=n,ke=i),ne}let Q=0,Pe=0;function Ye(n){let i=1/(1-2*Q),p=1/(1-2*Pe);return{score:n.score,box:[(n.box[0]-Q)*i,(n.box[1]-Pe)*p,n.box[2]*i,n.box[3]*p],keypoints:n.keypoints.map(([b,k])=>[(b-Q)*i,(k-Pe)*p])}}function Ne(n){return n instanceof HTMLCanvasElement||n instanceof OffscreenCanvas?[n.width,n.height]:typeof ImageBitmap<"u"&&n instanceof ImageBitmap?[n.width,n.height]:n instanceof ImageData?[n.width,n.height]:n instanceof HTMLVideoElement?[n.videoWidth,n.videoHeight]:n instanceof HTMLImageElement?[n.naturalWidth,n.naturalHeight]:[ye,ye]}function tt(n){return 1/(1+Math.exp(-n))}function Ae(n,i,p,b,k,H,I){let B=Math.cos(n.rotation),G=Math.sin(n.rotation),R=n.sizePx/ye,L=ye/2,A=B*R/p,M=-G*R/p,S=n.centerXpx/p-L*(A+M),D=G*R/b,K=B*R/b,W=n.centerYpx/b-L*(D+K);k.crop(I,i,H,[A,M,S,D,K,W],p,b,ye),X.encodeFromGPUBuffer(H,I)}function Se(n,i,p,b){let k=Math.cos(i.rotation),H=Math.sin(i.rotation),I=[];for(let B=0;B<_t;B++){let G=n.landmarks[B*3],R=n.landmarks[B*3+1],L=n.landmarks[B*3+2],A=(G-.5)*i.sizePx,M=(R-.5)*i.sizePx,S=k*A-H*M+i.centerXpx,D=H*A+k*M+i.centerYpx;I.push({x:S/p,y:D/b,z:L})}return I}function De(n,i){let b=[133,362,1,13,234,454].map(k=>n[k]);return{score:i,landmarks:n,keypoints:st(b)}}async function Xe(n,i,p,b,k,H,I=!1){let B=V.createCommandEncoder();Ae(n,i,p,b,k,H,B),V.queue.submit([B.finish()]);let G=await X.readbackLandmarks(),R=G.facePresence[0],L=I?Math.min(t,.1):t;return R<L?null:{landmarks:Se(G,n,p,b),score:R}}let re=null,ve=null,Be=null,_e=null;function be(){re=null,ve=null,Be=null,_e=null}async function Ie(n){let i,p,b;if(n instanceof HTMLVideoElement?(i=n.videoWidth,p=n.videoHeight,b=createImageBitmap(n,{colorSpaceConversion:"none"})):n instanceof HTMLImageElement?(i=n.naturalWidth,p=n.naturalHeight,b=createImageBitmap(n,{colorSpaceConversion:"none"})):n instanceof ImageData?(b=createImageBitmap(n,{colorSpaceConversion:"none"}),i=n.width,p=n.height):([i,p]=Ne(n),b=n),j.length>0&&re){let M=await re;re=null;let S=ve,[D,K]=Be,W=M.facePresence[0],ae=Math.min(t,.1);if(W>=ae){let q=Se(M,S,D,K),J=De(q,W),fe=b instanceof Promise?await b:b,Ce=Me(),Ee=Re(),ie=Le(i,p);V.queue.copyExternalImageToTexture({source:fe},{texture:ie},[i,p]);let pe=xe(q,i,p);X.flipReadbackBuffer();let me=V.createCommandEncoder();return Ae(pe,ie,i,p,Ce,Ee,me),V.queue.submit([me.finish()]),re=X.beginReadbackLandmarks(),ve=pe,Be=[i,p],_e=[J],j=[{landmarks:q}],[J]}be(),j=[]}let k=b instanceof Promise?await b:b,H=Me(),I=Re(),B=Le(i,p);if(V.queue.copyExternalImageToTexture({source:k},{texture:B},[i,p]),j.length>0&&!re){let M=j[0],S=xe(M.landmarks,i,p);X.flipReadbackBuffer();let D=V.createCommandEncoder();if(Ae(S,B,i,p,H,I,D),V.queue.submit([D.finish()]),re=X.beginReadbackLandmarks(),ve=S,Be=[i,p],_e)return _e;let K=await re;re=null;let W=K.facePresence[0],ae=Math.min(t,.1);if(W>=ae){let q=Se(K,S,i,p),J=De(q,W);return j=[{landmarks:q}],_e=[J],[J]}be(),j=[]}be();let{detections:G,lbPadX:R,lbPadY:L}=await P.detectRawWithResize(k,i,p);if(Q=R,Pe=L,G.length===0)return j=[],[];let A=[];for(let M of G){let S=Ye(M),D=Fe(S,i,p),K=await Xe(D,B,i,p,H,I);K&&A.push(De(K.landmarks,K.score))}return j=A.map(M=>({landmarks:M.landmarks})),_e=A,A}function Ke(){ne&&ne.destroy(),de&&de.destroy(),ne=null,de=null,we=null,X.device.destroy(),F.device.destroy()}function je(){j=[]}return{detect:Ie,dispose:Ke,reset:je}}export{qe as FACE_KEYPOINT_NAMES,Mt as FACE_LANDMARK_INDICES,Je as compileFaceDetectorModel,kt as compileFaceLandmarkModel,Gt as computeCropTransform,et as createCropPipeline,Qe as createFaceDetector,Wt as createFacemesh,ct as detectionToROI,Dt as loadFaceDetectorWeights,Ve as loadFaceLandmarkWeights,Lt as projectLandmarksToOriginal};
