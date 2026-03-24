function b(i){return i.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var wr=b(`
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
          let w_idx=oc*9u*params.in_channels+ky*3u*params.in_channels+kx*params.in_channels+ic;
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
`),hr=b(`
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
`),vr=b(`
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
`),Pr=b(`
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
`),kr=b(`
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
`),Ur=b(`
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
          let w_idx=oc*4u*params.in_channels+ky*2u*params.in_channels+kx*params.in_channels+ic;
          sum+=input[in_idx]*weight[w_idx];
        }
      }
    }
  }
  sum+=bias[oc];
  output[oc*params.out_h*params.out_w+out_y*params.out_w+out_x]=sum;
}
`),Br=b(`
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
`),Lr=b(`
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
`),Ar=b(`
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
        let w_idx=oc*4u*params.in_channels+ky*2u*params.in_channels+kx*params.in_channels+ic;
        sum+=input[in_idx]*weight[w_idx];
      }
    }
  }
  sum+=bias[oc];
  output[oc]=sum;
}
`),Dr=b(`
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> count:u32;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x;
  if(idx>=count){return;}
  output[idx]=1.0/(1.0+exp(-input[idx]));
}
`),Gr=b(`
struct Params { channels:u32, height:u32, width:u32, }
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read> alpha:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:Params;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let x=gid.x; let y=gid.y; let c=gid.z;
  if(x>=params.width||y>=params.height||c>=params.channels){return;}
  let idx=c*params.height*params.width+y*params.width+x;
  var val=a[idx]+b[idx];
  if(val<0.0){ val=val*alpha[c]; }
  output[idx]=val;
}
`),Mr=b(`
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
`);async function fa(i){let[s,u]=await Promise.all([fetch(`${i}/face_landmarks_weights.json`),fetch(`${i}/face_landmarks_weights.bin`)]);if(!s.ok)throw new Error(`Failed to load landmark metadata: ${s.status}`);if(!u.ok)throw new Error(`Failed to load landmark weights: ${u.status}`);let e=await s.json(),g=await u.arrayBuffer();return ea(e,g)}function ea(i,s){let u=new Map,e=i.dtype??"float32";for(let g=0;g<i.keys.length;g++){let _=i.keys[g],v=i.shapes[g],l=i.offsets[g],y=v.reduce((f,P)=>f*P,1),A;if(e==="float32")A=new Float32Array(s,l,y);else{let f=new DataView(s);A=new Float32Array(y);for(let P=0;P<y;P++)A[P]=oa(f.getUint16(l+P*2,!0))}u.set(_,{data:A,shape:v})}return u}function oa(i){let s=i>>15&1,u=i>>10&31,e=i&1023;return u===0?e===0?s?-0:0:(s?-1:1)*Math.pow(2,-14)*(e/1024):u===31?e===0?s?-1/0:1/0:NaN:(s?-1:1)*Math.pow(2,u-15)*(1+e/1024)}function ua(i,s,...u){let e=s.find(g=>u.every(_=>g.includes(_)));if(!e)throw new Error(`Weight not found for: ${u.join(", ")}`);return i.get(e)}function sa(i){let[,s,u,e]=i.shape,g=s*u,_=new Float32Array(e*g);for(let v=0;v<e;v++)for(let l=0;l<s;l++)for(let y=0;y<u;y++)_[v*g+l*u+y]=i.data[l*u*e+y*e+v];return _}async function ga(i,s){let u=Array.from(s.keys());function e(...r){return ua(s,u,...r)}let g={r:"read-only-storage",s:"storage",u:"uniform"};function _(r){return i.createBindGroupLayout({entries:r.map((n,t)=>({binding:t,visibility:GPUShaderStage.COMPUTE,buffer:{type:g[n]}}))})}let v=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,l=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,y=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,A=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function f(r,n){return i.createBuffer({size:Math.max(r,4),usage:n})}function P(r,n){i.queue.writeBuffer(r,0,n)}function w(r){let n=f(r.byteLength,v);return P(n,r),n}function x(r,n){return i.createComputePipeline({layout:i.createPipelineLayout({bindGroupLayouts:[r]}),compute:{module:i.createShaderModule({code:n}),entryPoint:"main"}})}function p(r,n){return i.createBindGroup({layout:r,entries:n.map((t,a)=>({binding:a,resource:{buffer:t}}))})}let H=_(["r","r","r","r","s","u"]),h=_(["r","r","r","s","u"]),Sr=_(["r","r","s","u"]),K=_(["r","s","u"]),Q=_(["r","s","u"]),W=_(["r","r","s","u"]),V=_(["r","r","r","s","u"]),rr=_(["r","s","u"]),Er=x(H,wr),Rr=x(H,hr),ar=x(h,vr),tr=x(h,Pr),pa=x(Sr,kr),Cr=x(h,Ur),Or=x(K,Br),Tr=x(Q,Lr),nr=x(h,Ar),Hr=x(rr,Dr),ir=x(W,Mr),er=x(V,Gr),M=16384*128*4,$=f(M,l),or=f(M,l),zr=f(M,l),Nr=f(M,l),Fr=f(M,l),ur=f(1434*4,y),sr=f(4,y),pr=f(4,y),S=f(1435*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),cr=132,dr=126,_r=82,j=60;function fr(){let r=e(`batch_normalization_${cr}`);return cr++,w(r.data)}function gr(){let r=e(`p_re_lu_${dr}`);return dr++,w(r.data)}function z(){let r=e(`conv2d_${_r}`,"Conv2D");return _r++,{buf:w(r.data),shape:r.shape}}function lr(){let r;try{r=e(`depthwise_conv2d_${j}/depthwise`)}catch{r=e(`depthwise_conv2d_${j}/depthwise1`)}return j++,w(sa(r))}function N(){return{bn:fr(),prelu:gr()}}let mr=[];function c(r,n,t){mr.push({pipe:r,bg:n,wg:t})}let o=r=>Math.ceil(r/8),Xr=r=>Math.ceil(r/64),ca=r=>Math.ceil(r/256);function d(r){let n=f(r.length*4,A);return P(n,new Uint32Array(r)),n}let Wr=w(e("conv2d_81","Conv2D").data),Vr=fr(),$r=gr(),jr=d([3,16,256,256,128,128]);c(Er,p(H,[$,Wr,Vr,$r,or,jr]),[o(128),o(128),16]);let m=or,I=16,D=128,[k,U,E,yr]=[zr,Nr,Fr,$];function B(r,n,t,a){let L=z(),F=N(),X=d([r,n,t,t]);c(Rr,p(H,[m,L.buf,F.bn,F.prelu,k,X]),[o(t),o(t),n]);let Y=lr(),Z=w(new Float32Array(n)),R=d([n,t,t,t,t,1,1]);c(ar,p(h,[k,Y,Z,U,R]),[o(t),o(t),n]);let q=z(),C=N(),J=d([n,r,t,t]);if(c(tr,p(h,[U,q.buf,C.bn,k,J]),[o(t),o(t),r]),a){let O=d([r,t,t]);c(er,p(V,[k,m,C.prelu,E,O]),[o(t),o(t),r]);let T=m;m=E,E=T}else{let O=d([r,t,t]);c(ir,p(W,[k,C.prelu,U,O]),[o(t),o(t),r]);let T=m;m=U,U=T}I=r}function G(r,n,t){let a=D/2,L=I,F=z(),X=N(),Y=d([L,r,D,D,a,a]);c(Cr,p(h,[m,F.buf,X.bn,k,Y]),[o(a),o(a),r]);let Z=d([L,D,D,a,a]);c(Or,p(K,[m,U,Z]),[o(a),o(a),L]);let R;if(t!==null&&t>L){let ia=d([L,t,a,a]);c(Tr,p(Q,[U,E,ia]),[o(a),o(a),t]),R=E}else R=U;let q=d([r,a,a]);c(ir,p(W,[k,X.prelu,yr,q]),[o(a),o(a),r]);let C=lr(),J=w(new Float32Array(r)),O=d([r,a,a,a,a,1,1]);c(ar,p(h,[yr,C,J,m,O]),[o(a),o(a),r]);let T=z(),br=N(),ta=d([r,n,a,a]);c(tr,p(h,[m,T.buf,br.bn,k,ta]),[o(a),o(a),n]);let na=d([n,a,a]);c(er,p(V,[k,R,br.prelu,m,na]),[o(a),o(a),n]),I=n,D=a}for(let r=0;r<4;r++)B(16,8,128,!0);G(16,32,32);for(let r=0;r<4;r++)B(32,16,64,!0);G(32,64,64);for(let r=0;r<4;r++)B(64,32,32,!0);G(64,128,128);for(let r=0;r<4;r++)B(128,64,16,!0);G(64,128,null);for(let r=0;r<4;r++)B(128,64,8,!0);G(64,128,null);for(let r=0;r<4;r++)B(128,64,4,!0);G(64,128,null);for(let r=0;r<4;r++)B(128,64,2,!0);let Ir=w(e("conv2d_150","Conv2D").data),Yr=w(e("conv2d_150","BiasAdd").data),Zr=d([128,1434]);c(nr,p(h,[m,Ir,Yr,ur,Zr]),[Xr(1434),1,1]);let qr=w(e("conv2d_152","Conv2D").data),Jr=w(e("conv2d_152","BiasAdd").data),Kr=d([128,1]);c(nr,p(h,[m,qr,Jr,sr,Kr]),[1,1,1]);let Qr=d([1]);c(Hr,p(rr,[sr,pr,Qr]),[1,1,1]);let xr=3*256*256*4;function ra(r,n){n.copyBufferToBuffer(r,0,$,0,xr);let t=n.beginComputePass();for(let a of mr)t.setPipeline(a.pipe),t.setBindGroup(0,a.bg),t.dispatchWorkgroups(...a.wg);t.end(),n.copyBufferToBuffer(ur,0,S,0,1434*4),n.copyBufferToBuffer(pr,0,S,1434*4,4)}async function aa(){await S.mapAsync(GPUMapMode.READ);let r=new Float32Array(S.getMappedRange().slice(0));return S.unmap(),{landmarks:r.subarray(0,1434),score:r[1434]}}return{device:i,run:ra,readback:aa,inputBufferSize:xr}}export{ga as compileLandmarkModel,fa as loadLandmarkWeights,ea as loadLandmarkWeightsFromBuffer};
