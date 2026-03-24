function P(n){return n.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var vr=P(`
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
`),hr=P(`
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
`),Pr=P(`
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
`),kr=P(`
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
`),Ur=P(`
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
`),Br=P(`
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
`),Lr=P(`
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
`),Ar=P(`
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
`),Gr=P(`
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
`),Mr=P(`
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> count:u32;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x;
  if(idx>=count){return;}
  output[idx]=1.0/(1.0+exp(-input[idx]));
}
`),Cr=P(`
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
`);async function pa(n){let[u,e]=await Promise.all([fetch(`${n}/face_landmarks_weights.json`),fetch(`${n}/face_landmarks_weights.bin`)]);if(!u.ok)throw new Error(`Failed to load landmark metadata: ${u.status}`);if(!e.ok)throw new Error(`Failed to load landmark weights: ${e.status}`);let i=await u.json(),_=await e.arrayBuffer();return na(i,_)}function na(n,u){let e=new Map,i=n.dtype??"float32";for(let _=0;_<n.keys.length;_++){let g=n.keys[_],k=n.shapes[_],l=n.offsets[_],y=k.reduce((d,U)=>d*U,1),M;if(i==="float32")M=new Float32Array(u,l,y);else{let d=new DataView(u);M=new Float32Array(y);for(let U=0;U<y;U++)M[U]=ia(d.getUint16(l+U*2,!0))}e.set(g,{data:M,shape:k})}return e}function ia(n){let u=n>>15&1,e=n>>10&31,i=n&1023;return e===0?i===0?u?-0:0:(u?-1:1)*Math.pow(2,-14)*(i/1024):e===31?i===0?u?-1/0:1/0:NaN:(u?-1:1)*Math.pow(2,e-15)*(1+i/1024)}function oa(n,u,...e){let i=u.find(_=>e.every(g=>_.includes(g)));if(!i)throw new Error(`Weight not found for: ${e.join(", ")}`);return n.get(i)}function ea(n){let[,u,e,i]=n.shape,_=u*e,g=new Float32Array(i*_);for(let k=0;k<i;k++)for(let l=0;l<u;l++)for(let y=0;y<e;y++)g[k*_+l*e+y]=n.data[l*e*i+y*i+k];return g}async function ca(n,u){let e=Array.from(u.keys());function i(...r){return oa(u,e,...r)}let _={r:"read-only-storage",s:"storage",u:"uniform"};function g(r){return n.createBindGroupLayout({entries:r.map((t,a)=>({binding:a,visibility:GPUShaderStage.COMPUTE,buffer:{type:_[t]}}))})}let k=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,l=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,y=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,M=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function d(r,t){return n.createBuffer({size:Math.max(r,4),usage:t})}function U(r,t){n.queue.writeBuffer(r,0,t)}function w(r){let t=d(r.byteLength,k);return U(t,r),t}function v(r,t){return n.createComputePipeline({layout:n.createPipelineLayout({bindGroupLayouts:[r]}),compute:{module:n.createShaderModule({code:t}),entryPoint:"main"}})}function s(r,t){return n.createBindGroup({layout:r,entries:t.map((a,f)=>({binding:f,resource:{buffer:a}}))})}let V=g(["r","r","r","r","s","u"]),h=g(["r","r","r","s","u"]),Z=g(["r","r","s","u"]),rr=g(["r","s","u"]),ar=g(["r","s","u"]),O=g(["r","r","s","u"]),tr=g(["r","s","u"]),Sr=v(V,vr),Dr=v(V,hr),nr=v(h,Pr),ir=v(h,kr),or=v(Z,Ur),Er=v(h,Br),Rr=v(rr,Lr),Or=v(ar,Ar),er=v(h,Gr),Tr=v(tr,Mr),$=v(O,Cr),T=16384*128*4,q=d(T,l),ur=d(T,l),Hr=d(T,l),Nr=d(T,l),zr=d(T,l),sr=d(1434*4,y),pr=d(4,y),cr=d(4,y),H=d(1435*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),dr=132,fr=126,_r=82,J=60;function gr(){let r=i(`batch_normalization_${dr}`);return dr++,w(r.data)}function mr(){let r=i(`p_re_lu_${fr}`);return fr++,w(r.data)}function j(){let r=i(`conv2d_${_r}`,"Conv2D");return _r++,{buf:w(r.data),shape:r.shape}}function lr(){let r;try{r=i(`depthwise_conv2d_${J}/depthwise`)}catch{r=i(`depthwise_conv2d_${J}/depthwise1`)}return J++,w(ea(r))}function I(){return{bn:gr(),prelu:mr()}}let yr=[];function p(r,t,a){yr.push({pipe:r,bg:t,wg:a})}let o=r=>Math.ceil(r/8),Fr=r=>Math.ceil(r/64),xr=r=>Math.ceil(r/256);function c(r){let t=d(r.length*4,M);return U(t,new Uint32Array(r)),t}let Xr=w(i("conv2d_81","Conv2D").data),Wr=gr(),Vr=mr(),$r=c([3,16,256,256,128,128]);p(Sr,s(V,[q,Xr,Wr,Vr,ur,$r]),[o(128),o(128),16]);let m=ur,B=16,C=128,[x,b,N,br]=[Hr,Nr,zr,q];function L(r,t,a,f){let z=j(),A=I(),F=c([r,t,a,a]);p(Dr,s(V,[m,z.buf,A.bn,A.prelu,x,F]),[o(a),o(a),t]);let E=lr(),G=w(new Float32Array(t)),R=c([t,a,a,a,a,1,1]);p(nr,s(h,[x,E,G,b,R]),[o(a),o(a),t]);let X=j(),W=I(),K=c([t,r,a,a]);if(p(ir,s(h,[b,X.buf,W.bn,x,K]),[o(a),o(a),r]),f){let Y=r*a*a,Q=c([Y]);p(or,s(Z,[x,m,b,Q]),[xr(Y),1,1]);let aa=c([r,a,a]);p($,s(O,[b,W.prelu,N,aa]),[o(a),o(a),r]);let ta=m;m=N,N=ta}else{let Y=c([r,a,a]);p($,s(O,[x,W.prelu,b,Y]),[o(a),o(a),r]);let Q=m;m=b,b=Q}B=r}function S(r,t,a){let f=lr(),z=w(new Float32Array(B)),A=c([B,a,a,a,a,1,1]);p(nr,s(h,[m,f,z,x,A]),[o(a),o(a),B]);let F=j(),E=I(),G=c([B,r,a,a]);p(ir,s(h,[x,F.buf,E.bn,b,G]),[o(a),o(a),r]);let R=c([r,a,a]);p($,s(O,[b,E.prelu,x,R]),[o(a),o(a),r]);let X=m;m=x,x=X,B=r}function D(r,t){let a=C/2,f=B,z=j(),A=I(),F=c([f,r,C,C,a,a]);p(Er,s(h,[m,z.buf,A.bn,x,F]),[o(a),o(a),r]);let E=c([f,C,C,a,a]);p(Rr,s(rr,[m,b,E]),[o(a),o(a),f]);let G;if(t!==null&&t>f){let K=c([f,t,a,a]);p(Or,s(ar,[b,N,K]),[o(a),o(a),t]),G=N}else G=b;let R=r*a*a,X=c([R]);p(or,s(Z,[x,G,br,X]),[xr(R),1,1]);let W=c([r,a,a]);p($,s(O,[br,A.prelu,m,W]),[o(a),o(a),r]),B=r,C=a}for(let r=0;r<4;r++)L(16,8,128,!0);D(16,32),S(32,16,64);for(let r=0;r<4;r++)L(32,16,64,!0);D(32,64),S(64,32,32);for(let r=0;r<4;r++)L(64,32,32,!0);D(64,128),S(128,64,16);for(let r=0;r<4;r++)L(128,64,16,!0);D(64,null),S(128,64,8);for(let r=0;r<4;r++)L(128,64,8,!0);D(64,null),S(128,64,4);for(let r=0;r<4;r++)L(128,64,4,!0);D(64,null),S(128,64,2);for(let r=0;r<4;r++)L(128,64,2,!0);let jr=w(i("conv2d_150","Conv2D").data),Ir=w(i("conv2d_150","BiasAdd").data),Yr=c([128,1434]);p(er,s(h,[m,jr,Ir,sr,Yr]),[Fr(1434),1,1]);let Zr=w(i("conv2d_152","Conv2D").data),qr=w(i("conv2d_152","BiasAdd").data),Jr=c([128,1]);p(er,s(h,[m,Zr,qr,pr,Jr]),[1,1,1]);let Kr=c([1]);p(Tr,s(tr,[pr,cr,Kr]),[1,1,1]);let wr=3*256*256*4;function Qr(r,t){t.copyBufferToBuffer(r,0,q,0,wr);let a=t.beginComputePass();for(let f of yr)a.setPipeline(f.pipe),a.setBindGroup(0,f.bg),a.dispatchWorkgroups(...f.wg);a.end(),t.copyBufferToBuffer(sr,0,H,0,1434*4),t.copyBufferToBuffer(cr,0,H,1434*4,4)}async function ra(){await H.mapAsync(GPUMapMode.READ);let r=new Float32Array(H.getMappedRange().slice(0));return H.unmap(),{landmarks:r.subarray(0,1434),score:r[1434]}}return{device:n,run:Qr,readback:ra,inputBufferSize:wr}}export{ca as compileLandmarkModel,pa as loadLandmarkWeights,na as loadLandmarkWeightsFromBuffer};
