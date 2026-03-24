import{a as He,b as Ct}from"./chunk-WOYQSGEU.js";var Ke=["rightEye","leftEye","noseTip","mouthCenter","rightEarTragion","leftEarTragion"];function We(a){let o={};for(let t=0;t<Ke.length;t++)o[Ke[t]]=a[t];return o}var Bt={silhouetteTop:10,silhouetteBottom:152,leftEyeInner:133,leftEyeOuter:33,rightEyeInner:362,rightEyeOuter:263,leftEyebrowUpper:66,rightEyebrowUpper:296,noseTip:1,noseBottom:2,noseBridgeTop:6,upperLipTop:13,lowerLipBottom:14,mouthLeft:61,mouthRight:291,chin:152,leftEarTragion:234,rightEarTragion:454,leftIrisCenter:468,leftIrisRight:469,leftIrisTop:470,leftIrisLeft:471,leftIrisBottom:472,rightIrisCenter:473,rightIrisRight:474,rightIrisTop:475,rightIrisLeft:476,rightIrisBottom:477};function Ut(){let a=[];for(let o=0;o<16;o++)for(let t=0;t<16;t++){let b=(t+.5)/16,p=(o+.5)/16;for(let i=0;i<2;i++)a.push({x:b,y:p})}for(let o=0;o<8;o++)for(let t=0;t<8;t++){let b=(t+.5)/8,p=(o+.5)/8;for(let i=0;i<6;i++)a.push({x:b,y:p})}return a}var tt=Ut();function Et(a){return 1/(1+Math.exp(-a))}function Ye(a,o){let t=[],{scores:b,regressors:p}=a,i=128;for(let u=0;u<tt.length;u++){let _=Et(b[u]);if(_<o)continue;let g=tt[u],m=u*16,s=g.x+p[m+0]/i,f=g.y+p[m+1]/i,h=p[m+2]/i,P=p[m+3]/i,l=[];for(let c=0;c<6;c++){let T=g.x+p[m+4+c*2]/i,w=g.y+p[m+4+c*2+1]/i;l.push([T,w])}t.push({score:_,box:[s,f,h,P],keypoints:l})}return t}function Ne(a,o){if(a.length===0)return[];let t=[...a].sort((i,u)=>u.score-i.score),b=[],p=new Set;for(let i=0;i<t.length;i++){if(p.has(i))continue;let u=[i];for(let l=i+1;l<t.length;l++)p.has(l)||Tt(t[i],t[l])>o&&(u.push(l),p.add(l));let _=0,g=0,m=0,s=0,f=0,h=[];for(let l=0;l<6;l++)h.push([0,0]);for(let l of u){let c=t[l],T=c.score;_+=T,g+=c.box[0]*T,m+=c.box[1]*T,s+=c.box[2]*T,f+=c.box[3]*T;for(let w=0;w<6;w++)h[w][0]+=c.keypoints[w][0]*T,h[w][1]+=c.keypoints[w][1]*T}let P=1/_;b.push({score:t[i].score,box:[g*P,m*P,s*P,f*P],keypoints:h.map(([l,c])=>[l*P,c*P])})}return b}function Tt(a,o){let t=a.box[0]-a.box[2]/2,b=a.box[1]-a.box[3]/2,p=a.box[0]+a.box[2]/2,i=a.box[1]+a.box[3]/2,u=o.box[0]-o.box[2]/2,_=o.box[1]-o.box[3]/2,g=o.box[0]+o.box[2]/2,m=o.box[1]+o.box[3]/2,s=Math.max(t,u),f=Math.max(b,_),h=Math.min(p,g),P=Math.min(i,m),l=Math.max(0,h-s),c=Math.max(0,P-f),T=l*c,w=(p-t)*(i-b),re=(g-u)*(m-_),ae=w+re-T;return ae>0?T/ae:0}function nt(a){let[o,t,b,p]=a.box,i=a.keypoints[0],u=a.keypoints[1],_=(i[0]+u[0])/2,g=(i[1]+u[1])/2,m=a.keypoints[2],s=m[0]-_,f=m[1]-g,h=Math.atan2(f,s),l=Math.PI/2-h,c=Math.max(b,p),w=c*1.5,re=0*c,ae=Math.cos(l),Q=Math.sin(l),Be=re*Q,Ue=re*ae;return{centerX:o+Be,centerY:t+Ue,width:w,height:w,rotation:l}}function Xe(a,o={}){let{scoreThreshold:t=.5,nmsThreshold:b=.3,maxFaces:p=1}=o;async function i(m){let s=await a.run(m),f=Ye(s,t);return Ne(f,b).slice(0,p).map(nt)}async function u(m){let s=await a.run(m),f=Ye(s,t);return Ne(f,b).slice(0,p)}async function _(m,s,f){let{output:h,lbPadX:P,lbPadY:l}=await a.runWithResize(m,s,f),c=Ye(h,t);return{detections:Ne(c,b).slice(0,p),lbPadX:P,lbPadY:l}}async function g(m,s,f){let{output:h,lbPadX:P,lbPadY:l}=await a.runWithResize(m,s,f);return{scores:h.scores,regressors:h.regressors,lbPadX:P,lbPadY:l}}return{detect:i,detectRaw:u,detectRawWithResize:_,detectRawSSD:g,model:a}}function Ft(a,o=256){let t=Math.cos(a.rotation),b=Math.sin(a.rotation),p=a.width/o,i=a.height/o,u=p*t,_=p*b,g=-i*b,m=i*t,s=a.centerX-(u*o/2+_*o/2),f=a.centerY-(g*o/2+m*o/2),h=u*m-_*g,P=m/h,l=-_/h,c=-g/h,T=u/h,w=-(P*s+l*f),re=-(c*s+T*f);return{forward:[u,_,s,g,m,f],inverse:[P,l,w,c,T,re]}}function Mt(a,o,t,b){let p=Math.cos(o.rotation),i=Math.sin(o.rotation),u=Math.min(t,b),_=o.width*u,g=_/t,m=_/b;return a.map(s=>{let f=s.x-.5,h=s.y-.5,P=p*f+i*h,l=-i*f+p*h;return{x:P*g+o.centerX,y:l*m+o.centerY,z:s.z}})}function be(a){return a.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var rt=be(`
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
`);async function kt(a){let[o,t]=await Promise.all([fetch(`${a}/face_detector.json`),fetch(`${a}/face_detector.bin`)]);if(!o.ok)throw new Error(`Failed to load face detector metadata: ${o.status}`);if(!t.ok)throw new Error(`Failed to load face detector weights: ${t.status}`);let b=await o.json(),p=await t.arrayBuffer(),i=new Map;for(let u of b){let _=new Float32Array(p,u.offset,u.size/4);i.set(u.key,{data:_,shape:u.shape})}return i}async function je(a,o){let t;if(o)t=o;else{if(!navigator.gpu)throw new Error("WebGPU not supported");let e=await navigator.gpu.requestAdapter();if(!e)throw new Error("No GPU adapter found");t=await e.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(e.limits.maxStorageBuffersPerShaderStage,8)}})}let b={r:"read-only-storage",s:"storage",u:"uniform"};function p(e){return t.createBindGroupLayout({entries:e.map((n,d)=>n==="t"?{binding:d,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:n==="sm"?{binding:d,visibility:GPUShaderStage.COMPUTE,sampler:{}}:{binding:d,visibility:GPUShaderStage.COMPUTE,buffer:{type:b[n]}})})}let i=t.createSampler({magFilter:"linear",minFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge"}),u=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,_=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,g=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,m=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function s(e,n){return t.createBuffer({size:Math.max(e,4),usage:n})}function f(e,n,d){t.queue.writeBuffer(e,n,d)}function h(e){let n=s(e.data.byteLength,u);return f(n,0,e.data),n}let P=Array.from(a.keys());function l(e){let n=a.get(e);if(!n)throw new Error(`Weight not found: ${e}`);return n}function c(...e){let n=P.find(d=>e.every(B=>d.includes(B)));if(!n)throw new Error(`Weight not found for: ${e.join(", ")}`);return l(n)}function T(e){let[,n,d,B]=e.shape,U=new Float32Array(B*9);for(let v=0;v<B;v++)for(let k=0;k<n;k++)for(let I=0;I<d;I++)U[v*9+k*3+I]=e.data[k*d*B+I*B+v];return U}function w(e){let[n,,,d]=e.shape,B=new Float32Array(n*d);for(let U=0;U<n;U++)for(let v=0;v<d;v++)B[U*d+v]=e.data[U*d+v];return B}let re=t.createShaderModule({code:rt}),ae=t.createShaderModule({code:at}),Q=t.createShaderModule({code:ot}),Be=t.createShaderModule({code:st}),Ue=t.createShaderModule({code:it}),me=t.createShaderModule({code:ut}),xe=t.createShaderModule({code:ct}),ue=p(["r","r","r","s","u"]),J=p(["r","r","r","s","u"]),Ee=p(["r","r","r","r","s","u"]),Te=p(["r","r","r","s","u"]),Fe=p(["t","s","u"]),Me=p(["t","s","u","sm"]),Ge=p(["s","u"]);function N(e,n){return t.createComputePipeline({layout:t.createPipelineLayout({bindGroupLayouts:[e]}),compute:{module:n,entryPoint:"main"}})}let we=N(ue,re),Se=N(J,ae),Ie=N(Ee,Q),Le=N(Te,Be),ke=N(Fe,Ue),Oe=N(Me,me),qe=N(Ge,xe),ze=c("conv2d/Conv2D"),r=c("batch_normalization/","conv2d/Conv2D"),y=h(ze),x=h(r),O=[{dwKey:"depthwise_conv2d/",pwKey:"conv2d_1/",bnKey:"batch_normalization_1/",inCh:24,outCh:24,stride:1,inH:64},{dwKey:"depthwise_conv2d_1/",pwKey:"conv2d_2/",bnKey:"batch_normalization_2/",inCh:24,outCh:28,stride:1,inH:64},{dwKey:"depthwise_conv2d_2/",pwKey:"conv2d_3/",bnKey:"batch_normalization_3/",inCh:28,outCh:32,stride:2,inH:64},{dwKey:"depthwise_conv2d_3/",pwKey:"conv2d_4/",bnKey:"batch_normalization_4/",inCh:32,outCh:36,stride:1,inH:32},{dwKey:"depthwise_conv2d_4/",pwKey:"conv2d_5/",bnKey:"batch_normalization_5/",inCh:36,outCh:42,stride:1,inH:32},{dwKey:"depthwise_conv2d_5/",pwKey:"conv2d_6/",bnKey:"batch_normalization_6/",inCh:42,outCh:48,stride:2,inH:32},{dwKey:"depthwise_conv2d_6/",pwKey:"conv2d_7/",bnKey:"batch_normalization_7/",inCh:48,outCh:56,stride:1,inH:16},{dwKey:"depthwise_conv2d_7/",pwKey:"conv2d_8/",bnKey:"batch_normalization_8/",inCh:56,outCh:64,stride:1,inH:16},{dwKey:"depthwise_conv2d_8/",pwKey:"conv2d_9/",bnKey:"batch_normalization_9/",inCh:64,outCh:72,stride:1,inH:16},{dwKey:"depthwise_conv2d_9/",pwKey:"conv2d_10/",bnKey:"batch_normalization_10/",inCh:72,outCh:80,stride:1,inH:16},{dwKey:"depthwise_conv2d_10/",pwKey:"conv2d_11/",bnKey:"batch_normalization_11/",inCh:80,outCh:88,stride:1,inH:16},{dwKey:"depthwise_conv2d_11/",pwKey:"conv2d_12/",bnKey:"batch_normalization_12/",inCh:88,outCh:96,stride:2,inH:16}].map(e=>{let n=c(e.dwKey),d=c(e.pwKey),B=c(e.bnKey),U=T(n),v=s(U.byteLength,u);f(v,0,U);let k=new Float32Array(e.inCh),I=s(k.byteLength,u);f(I,0,k);let W=w(d),ne=s(W.byteLength,u);f(ne,0,W);let D=h(B);return{dwWeightBuf:v,dwBiasBuf:I,pwWeightBuf:ne,pwBiasBuf:D,inCh:e.inCh,outCh:e.outCh,stride:e.stride,inH:e.inH}}),X=w(c("classificator_8","Conv2D")),j=s(X.byteLength,u);f(j,0,X);let G=h(c("classificator_8","BiasAdd")),R=w(c("regressor_8","Conv2D")),S=s(R.byteLength,u);f(S,0,R);let z=h(c("regressor_8","BiasAdd")),C=w(c("classificator_16","Conv2D")),A=s(C.byteLength,u);f(A,0,C);let H=h(c("classificator_16","BiasAdd")),M=w(c("regressor_16","Conv2D")),ee=s(M.byteLength,u);f(ee,0,M);let V=h(c("regressor_16","BiasAdd")),K=Math.max(16384*3,4096*96,1024*96,256*96,6144)*4,q=s(16384*3*4,u),ce=s(K,_),Pe=s(K,_),ge=s(K,_),te=s(256*2*4,g),pe=s(256*32*4,g),fe=s(384*4,g),le=s(6144*4,g),oe=s(256*2*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),se=s(256*32*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),de=s(384*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),ve=s(6144*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),$e=t.createTexture({size:[128,128,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});function $(e,n){return Math.ceil(e/n)}function he(e){let n=s(e.byteLength,m);return f(n,0,e),n}let dt=he(new Uint32Array([1,3,24,128,128,64,64])),mt=O.map(e=>{let n=e.stride===2?e.inH/2:e.inH,d=n,B=e.stride===2?0:1,U=e.inCh;return{dw:he(new Uint32Array([1,e.inCh,e.inH,e.inH,n,d,e.stride,B])),pw:he(new Uint32Array([1,e.inCh,e.outCh,n,d,U,e.stride,e.inH,e.inH])),outH:n,outW:d}}),ft=he(new Uint32Array([1,88,2,16,16])),lt=he(new Uint32Array([1,88,32,16,16])),ht=he(new Uint32Array([1,96,6,8,8])),_t=he(new Uint32Array([1,96,96,8,8])),gt=he(new Uint32Array([128,128,128])),bt=t.createBindGroup({layout:Fe,entries:[{binding:0,resource:$e.createView()},{binding:1,resource:{buffer:q}},{binding:2,resource:{buffer:gt}}]}),Ce=null,Ze=0,Qe=0,Je=s(32,m);function yt(e,n){return Ce&&Ze===e&&Qe===n||(Ce&&Ce.destroy(),Ce=t.createTexture({size:[e,n,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),Ze=e,Qe=n),Ce}let xt=t.createBindGroup({layout:ue,entries:[{binding:0,resource:{buffer:q}},{binding:1,resource:{buffer:y}},{binding:2,resource:{buffer:x}},{binding:3,resource:{buffer:ce}},{binding:4,resource:{buffer:dt}}]});function Lt(e,n,d){}function wt(e,n,d,B,U,v){let k=v.outH,I=t.createBindGroup({layout:J,entries:[{binding:0,resource:{buffer:d}},{binding:1,resource:{buffer:n.dwWeightBuf}},{binding:2,resource:{buffer:n.dwBiasBuf}},{binding:3,resource:{buffer:ge}},{binding:4,resource:{buffer:v.dw}}]}),W=e.beginComputePass();W.setPipeline(Se),W.setBindGroup(0,I),W.dispatchWorkgroups($(k,8),$(v.outH,8),n.inCh),W.end(),n.inCh*v.outH*k;let ne=t.createBindGroup({layout:Ee,entries:[{binding:0,resource:{buffer:ge}},{binding:1,resource:{buffer:U}},{binding:2,resource:{buffer:n.pwWeightBuf}},{binding:3,resource:{buffer:n.pwBiasBuf}},{binding:4,resource:{buffer:B}},{binding:5,resource:{buffer:v.pw}}]}),D=e.beginComputePass();D.setPipeline(Ie),D.setBindGroup(0,ne),D.dispatchWorkgroups($(k,8),$(v.outH,8),n.outCh),D.end(),n.outCh*v.outH*k}function Ae(e,n,d,B,U,v,k,I,W){let ne=t.createBindGroup({layout:Te,entries:[{binding:0,resource:{buffer:n}},{binding:1,resource:{buffer:d}},{binding:2,resource:{buffer:B}},{binding:3,resource:{buffer:U}},{binding:4,resource:{buffer:v}}]}),D=e.beginComputePass();D.setPipeline(Le),D.setBindGroup(0,ne),D.dispatchWorkgroups($(W,8),$(I,8),k),D.end()}async function et(e){16384*3;{let E=e.beginComputePass();E.setPipeline(we),E.setBindGroup(0,xt),E.dispatchWorkgroups($(64,8),$(64,8),24),E.end()}4096*24;let n=ce,d=Pe,B=n;for(let E=0;E<O.length;E++){let L=O[E];wt(e,L,n,d,n,mt[E]);let Y=n;n=d,d=Y,E===10&&(B=n)}Ae(e,B,j,G,te,ft,2,16,16),256*2,Ae(e,B,S,z,pe,lt,32,16,16),256*32,Ae(e,n,A,H,fe,ht,6,8,8),Ae(e,n,ee,V,le,_t,96,8,8),t.queue.submit([e.finish()]);let U=t.createCommandEncoder();U.copyBufferToBuffer(te,0,oe,0,256*2*4),U.copyBufferToBuffer(pe,0,se,0,256*32*4),U.copyBufferToBuffer(fe,0,de,0,384*4),U.copyBufferToBuffer(le,0,ve,0,6144*4),t.queue.submit([U.finish()]),await Promise.all([oe.mapAsync(GPUMapMode.READ),se.mapAsync(GPUMapMode.READ),de.mapAsync(GPUMapMode.READ),ve.mapAsync(GPUMapMode.READ)]);let v=new Float32Array(oe.getMappedRange()).slice(),k=new Float32Array(se.getMappedRange()).slice(),I=new Float32Array(de.getMappedRange()).slice(),W=new Float32Array(ve.getMappedRange()).slice();oe.unmap(),se.unmap(),de.unmap(),ve.unmap();let ne=896,D=new Float32Array(ne),_e=new Float32Array(ne*16),ie=0;for(let E=0;E<16;E++)for(let L=0;L<16;L++)for(let Y=0;Y<2;Y++){D[ie]=v[Y*256+E*16+L];for(let Z=0;Z<16;Z++){let De=Y*16+Z;_e[ie*16+Z]=k[De*256+E*16+L]}ie++}for(let E=0;E<8;E++)for(let L=0;L<8;L++)for(let Y=0;Y<6;Y++){D[ie]=I[Y*64+E*8+L];for(let Z=0;Z<16;Z++){let De=Y*16+Z;_e[ie*16+Z]=W[De*64+E*8+L]}ie++}return{scores:D,regressors:_e}}async function Pt(e){t.queue.copyExternalImageToTexture({source:e},{texture:$e},[128,128]);let n=t.createCommandEncoder();{let d=n.beginComputePass();d.setPipeline(ke),d.setBindGroup(0,bt),d.dispatchWorkgroups($(128,16),$(128,16),1),d.end()}return et(n)}async function vt(e,n,d){let B=Math.min(128/n,128/d),U=Math.round(n*B),v=Math.round(d*B),k=Math.floor((128-U)/2),I=Math.floor((128-v)/2),W=k/128,ne=I/128,D=yt(n,d),_e;e instanceof HTMLVideoElement?_e=await createImageBitmap(e,{colorSpaceConversion:"none"}):e instanceof HTMLImageElement?_e=await createImageBitmap(e,{colorSpaceConversion:"none"}):_e=e,t.queue.copyExternalImageToTexture({source:_e},{texture:D},[n,d]);let ie=new ArrayBuffer(32),E=new Uint32Array(ie),L=new Float32Array(ie);E[0]=n,E[1]=d,E[2]=128,E[3]=0,L[4]=n/U,L[5]=d/v,L[6]=k,L[7]=I,t.queue.writeBuffer(Je,0,ie);let Y=t.createBindGroup({layout:Me,entries:[{binding:0,resource:D.createView()},{binding:1,resource:{buffer:q}},{binding:2,resource:{buffer:Je}},{binding:3,resource:i}]}),Z=t.createCommandEncoder();{let Re=Z.beginComputePass();Re.setPipeline(Oe),Re.setBindGroup(0,Y),Re.dispatchWorkgroups($(128,16),$(128,16),1),Re.end()}return{output:await et(Z),lbPadX:W,lbPadY:ne}}return{device:t,run:Pt,runWithResize:vt}}function At(a){return a.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var Dt=At(`
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
`);function Ve(a){let o=a.createShaderModule({code:Dt}),t=a.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:4,visibility:GPUShaderStage.COMPUTE,sampler:{}}]}),b=a.createComputePipeline({layout:a.createPipelineLayout({bindGroupLayouts:[t]}),compute:{module:o,entryPoint:"main"}}),p=a.createSampler({magFilter:"linear",minFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge"}),i=a.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),u=a.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),_=new Float32Array(8);function g(m,s,f,h,P,l,c){a.queue.writeBuffer(i,0,new Uint32Array([P,l,c,0])),_.set(h),a.queue.writeBuffer(u,0,_);let T=a.createBindGroup({layout:t,entries:[{binding:0,resource:s.createView()},{binding:1,resource:{buffer:f}},{binding:2,resource:{buffer:i}},{binding:3,resource:{buffer:u}},{binding:4,resource:p}]}),w=m.beginComputePass();w.setPipeline(b),w.setBindGroup(0,T),w.dispatchWorkgroups(Math.ceil(c/16),Math.ceil(c/16),1),w.end()}return{crop:g}}var Rt="https://cdn.jsdelivr.net/npm/@svenflow/micro-facemesh@0.1.0/weights",ye=256,pt=478,Gt=33,St=263;async function It(a={}){let{weightsUrl:o,scoreThreshold:t=.5,faceScoreThreshold:b=.5,maxFaces:p=1}=a;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-facemesh requires WebGPU. Check browser support at https://webgpureport.org");let i=(o??Rt).replace(/\/$/,"")+"/",[u,_,g,m]=await Promise.all([fetch(`${i}face_landmarks_weights_f16.json`),fetch(`${i}face_landmarks_weights_f16.bin`),fetch(`${i}face_detector_weights.json`),fetch(`${i}face_detector_weights.bin`)]);if(!u.ok)throw new Error(`Failed to fetch landmark weights: ${u.status}`);if(!_.ok)throw new Error(`Failed to fetch landmark weights: ${_.status}`);if(!g.ok)throw new Error(`Failed to fetch face detector weights: ${g.status}`);if(!m.ok)throw new Error(`Failed to fetch face detector weights: ${m.status}`);let[s,f,h,P]=await Promise.all([u.json(),_.arrayBuffer(),g.json(),m.arrayBuffer()]),l=He(s,f),c=new Map;for(let r of h){let y=new Float32Array(P,r.offset,r.size/4);c.set(r.key,{data:y,shape:r.shape})}let T=await je(c),w=Xe(T,{scoreThreshold:b,maxFaces:p}),{compileFaceLandmarkModel:re}=await import("./face_landmark_model-A2HWSD7C.js"),ae=await re(l,T.device),Q=[];function Be(r,y,x){let F=r[Gt],O=r[St],X=(O.x-F.x)*y,j=(O.y-F.y)*x,G=Math.atan2(j,X),R=Math.cos(G),S=Math.sin(G),z=1/0,C=-1/0,A=1/0,H=-1/0;for(let pe=0;pe<pt;pe++){let fe=r[pe],le=fe.x*y,oe=fe.y*x,se=R*le+S*oe,de=-S*le+R*oe;z=Math.min(z,se),C=Math.max(C,se),A=Math.min(A,de),H=Math.max(H,de)}let M=(z+C)/2,ee=(A+H)/2,V=C-z,K=H-A,q=R*M-S*ee,ce=S*M+R*ee,te=Math.max(V,K)*1.5;return{centerXpx:q,centerYpx:ce,sizePx:te,rotation:G}}function Ue(r,y,x){let[F,O,X,j]=r.box,G=r.keypoints[0],R=r.keypoints[1],S=(G[0]+R[0])/2,z=(G[1]+R[1])/2,C=r.keypoints[2],A=C[0]-S,H=C[1]-z,M=Math.atan2(H,A),V=Math.PI/2-M,K=V-2*Math.PI*Math.floor((V+Math.PI)/(2*Math.PI)),Pe=Math.max(X*y,j*x)*1.5;return{centerXpx:F*y,centerYpx:O*x,sizePx:Pe,rotation:K}}let me=ae.device,xe=null,ue=null,J=null,Ee=0,Te=0;function Fe(){return xe||(xe=Ve(me)),xe}function Me(){return ue||(ue=me.createBuffer({size:3*ye*ye*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC})),ue}function Ge(r,y){return(!J||Ee!==r||Te!==y)&&(J&&J.destroy(),J=me.createTexture({size:[r,y],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),Ee=r,Te=y),J}let N=0,we=0;function Se(r){let y=1/(1-2*N),x=1/(1-2*we);return{score:r.score,box:[(r.box[0]-N)*y,(r.box[1]-we)*x,r.box[2]*y,r.box[3]*x],keypoints:r.keypoints.map(([F,O])=>[(F-N)*y,(O-we)*x])}}function Ie(r){return r instanceof HTMLCanvasElement||r instanceof OffscreenCanvas?[r.width,r.height]:typeof ImageBitmap<"u"&&r instanceof ImageBitmap?[r.width,r.height]:r instanceof ImageData?[r.width,r.height]:r instanceof HTMLVideoElement?[r.videoWidth,r.videoHeight]:r instanceof HTMLImageElement?[r.naturalWidth,r.naturalHeight]:[ye,ye]}function Le(r){return 1/(1+Math.exp(-r))}async function ke(r,y,x,F,O,X,j=!1){let G=Math.cos(r.rotation),R=Math.sin(r.rotation),S=r.sizePx/ye,z=ye/2,C=G*S/x,A=-R*S/x,H=r.centerXpx/x-z*(C+A),M=R*S/F,ee=G*S/F,V=r.centerYpx/F-z*(M+ee),K=me.createCommandEncoder();O.crop(K,y,X,[C,A,H,M,ee,V],x,F,ye),me.queue.submit([K.finish()]);let q=await ae.runFromGPUBuffer(X),ce=Le(q.facePresence[0]),Pe=j?Math.min(t,.1):t;if(ce<Pe)return null;let ge=[];for(let te=0;te<pt;te++){let pe=q.landmarks[te*3],fe=q.landmarks[te*3+1],le=q.landmarks[te*3+2],oe=(pe-.5)*r.sizePx,se=(fe-.5)*r.sizePx,de=G*oe-R*se+r.centerXpx,ve=R*oe+G*se+r.centerYpx;ge.push({x:de/x,y:ve/F,z:le})}return{landmarks:ge,score:ce}}async function Oe(r){let y,x,F;if(r instanceof HTMLVideoElement)y=r.videoWidth,x=r.videoHeight,F=r;else if(r instanceof HTMLImageElement)y=r.naturalWidth,x=r.naturalHeight,F=r;else if(r instanceof ImageData){let C=await createImageBitmap(r,{colorSpaceConversion:"none"});[y,x]=[C.width,C.height],F=C}else[y,x]=Ie(r),F=r;let O=Fe(),X=Me(),j=Ge(y,x);if(me.queue.copyExternalImageToTexture({source:F},{texture:j},[y,x]),Q.length>0){let C=[];for(let A of Q){let H=Be(A.landmarks,y,x),M=await ke(H,j,y,x,O,X,!0);if(M){let V=[133,362,1,13,234,454].map(K=>M.landmarks[K]);C.push({score:M.score,landmarks:M.landmarks,keypoints:We(V)})}}if(C.length>0)return Q=C.map(A=>({landmarks:A.landmarks})),C;Q=[]}let{detections:G,lbPadX:R,lbPadY:S}=await w.detectRawWithResize(F,y,x);if(N=R,we=S,G.length===0)return Q=[],[];let z=[];for(let C of G){let A=Se(C),H=Ue(A,y,x),M=await ke(H,j,y,x,O,X);if(M){let V=[133,362,1,13,234,454].map(K=>M.landmarks[K]);z.push({score:M.score,landmarks:M.landmarks,keypoints:We(V)})}}return Q=z.map(C=>({landmarks:C.landmarks})),z}function qe(){J&&J.destroy(),ue&&ue.destroy(),J=null,ue=null,xe=null,ae.device.destroy(),T.device.destroy()}function ze(){Q=[]}return{detect:Oe,dispose:qe,reset:ze}}export{Ke as FACE_KEYPOINT_NAMES,Bt as FACE_LANDMARK_INDICES,je as compileFaceDetectorModel,Ct as compileFaceLandmarkModel,Ft as computeCropTransform,Ve as createCropPipeline,Xe as createFaceDetector,It as createFacemesh,nt as detectionToROI,kt as loadFaceDetectorWeights,He as loadFaceLandmarkWeights,Mt as projectLandmarksToOriginal};
