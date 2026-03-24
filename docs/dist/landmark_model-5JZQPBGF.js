function x(i){return i.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var ha=x(`
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
`),va=x(`
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
`),Pa=x(`
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
`),ka=x(`
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
`),Ua=x(`
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
`),Ba=x(`
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
`),La=x(`
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
`),Aa=x(`
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
`),Da=x(`
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
`),Ga=x(`
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> count:u32;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x;
  if(idx>=count){return;}
  output[idx]=1.0/(1.0+exp(-input[idx]));
}
`),Ma=x(`
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
`),Ra=x(`
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
`);async function mr(i){let[s,u]=await Promise.all([fetch(`${i}/face_landmarks_weights.json`),fetch(`${i}/face_landmarks_weights.bin`)]);if(!s.ok)throw new Error(`Failed to load landmark metadata: ${s.status}`);if(!u.ok)throw new Error(`Failed to load landmark weights: ${u.status}`);let e=await s.json(),g=await u.arrayBuffer();return sr(e,g)}function sr(i,s){let u=new Map,e=i.dtype??"float32";for(let g=0;g<i.keys.length;g++){let _=i.keys[g],v=i.shapes[g],l=i.offsets[g],y=v.reduce((p,P)=>p*P,1),A;if(e==="float32")A=new Float32Array(s,l,y);else{let p=new DataView(s);A=new Float32Array(y);for(let P=0;P<y;P++)A[P]=pr(p.getUint16(l+P*2,!0))}u.set(_,{data:A,shape:v})}return u}function pr(i){let s=i>>15&1,u=i>>10&31,e=i&1023;return u===0?e===0?s?-0:0:(s?-1:1)*Math.pow(2,-14)*(e/1024):u===31?e===0?s?-1/0:1/0:NaN:(s?-1:1)*Math.pow(2,u-15)*(1+e/1024)}function cr(i,s,...u){let e=s.find(g=>u.every(_=>g.includes(_)));if(!e)throw new Error(`Weight not found for: ${u.join(", ")}`);return i.get(e)}function dr(i){let[,s,u,e]=i.shape,g=s*u,_=new Float32Array(e*g);for(let v=0;v<e;v++)for(let l=0;l<s;l++)for(let y=0;y<u;y++)_[v*g+l*u+y]=i.data[l*u*e+y*e+v];return _}async function yr(i,s){let u=Array.from(s.keys());function e(...a){return cr(s,u,...a)}let g={r:"read-only-storage",s:"storage",u:"uniform"};function _(a){return i.createBindGroupLayout({entries:a.map((r,n)=>({binding:n,visibility:GPUShaderStage.COMPUTE,buffer:{type:g[r]}}))})}let v=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,l=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,y=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,A=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function p(a,r){return i.createBuffer({size:Math.max(a,4),usage:r})}function P(a,r){i.queue.writeBuffer(a,0,r)}function w(a){let r=p(a.byteLength,v);return P(r,a),r}function b(a,r){return i.createComputePipeline({layout:i.createPipelineLayout({bindGroupLayouts:[a]}),compute:{module:i.createShaderModule({code:r}),entryPoint:"main"}})}function c(a,r){return i.createBindGroup({layout:a,entries:r.map((n,t)=>({binding:t,resource:{buffer:n}}))})}let H=_(["r","r","r","r","s","u"]),h=_(["r","r","r","s","u"]),Sa=_(["r","r","s","u"]),Q=_(["r","s","u"]),aa=_(["r","s","u"]),V=_(["r","r","s","u"]),I=_(["r","r","r","s","u"]),ra=_(["r","s","u"]),Ea=b(H,ha),Ca=b(H,va),ta=b(h,Pa),na=b(h,ka),fr=b(Sa,Ua),Oa=b(h,Ba),Ta=b(Q,La),Ha=b(aa,Aa),ia=b(h,Da),za=b(ra,Ga),ea=b(V,Ra),oa=b(I,Ma),R=16384*128*4,$=p(R,l),ua=p(R,l),Na=p(R,l),Fa=p(R,l),Xa=p(R,l),sa=p(1434*4,y),pa=p(4,y),ca=p(4,y),z=[p(1435*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),p(1435*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST)],D=0,da=132,fa=126,_a=82,j=60;function ga(){let a=e(`batch_normalization_${da}`);return da++,w(a.data)}function la(){let a=e(`p_re_lu_${fa}`);return fa++,w(a.data)}function N(){let a=e(`conv2d_${_a}`,"Conv2D");return _a++,{buf:w(a.data),shape:a.shape}}function ma(){let a;try{a=e(`depthwise_conv2d_${j}/depthwise`)}catch{a=e(`depthwise_conv2d_${j}/depthwise1`)}return j++,w(dr(a))}function F(){return{bn:ga(),prelu:la()}}let ya=[];function d(a,r,n){ya.push({pipe:a,bg:r,wg:n})}let o=a=>Math.ceil(a/8),Wa=a=>Math.ceil(a/64),_r=a=>Math.ceil(a/256);function f(a){let r=p(a.length*4,A);return P(r,new Uint32Array(a)),r}let Va=w(e("conv2d_81","Conv2D").data),Ia=ga(),$a=la(),ja=f([3,16,256,256,128,128]);d(Ea,c(H,[$,Va,Ia,$a,ua,ja]),[o(128),o(128),16]);let m=ua,Y=16,G=128,[k,U,S,ba]=[Na,Fa,Xa,$];function B(a,r,n,t){let L=N(),X=F(),W=f([a,r,n,n]);d(Ca,c(H,[m,L.buf,X.bn,X.prelu,k,W]),[o(n),o(n),r]);let Z=ma(),q=w(new Float32Array(r)),E=f([r,n,n,n,n,1,1]);d(ta,c(h,[k,Z,q,U,E]),[o(n),o(n),r]);let J=N(),C=F(),K=f([r,a,n,n]);if(d(na,c(h,[U,J.buf,C.bn,k,K]),[o(n),o(n),a]),t){let O=f([a,n,n]);d(oa,c(I,[k,m,C.prelu,S,O]),[o(n),o(n),a]);let T=m;m=S,S=T}else{let O=f([a,n,n]);d(ea,c(V,[k,C.prelu,U,O]),[o(n),o(n),a]);let T=m;m=U,U=T}Y=a}function M(a,r,n){let t=G/2,L=Y,X=N(),W=F(),Z=f([L,a,G,G,t,t]);d(Oa,c(h,[m,X.buf,W.bn,k,Z]),[o(t),o(t),a]);let q=f([L,G,G,t,t]);d(Ta,c(Q,[m,U,q]),[o(t),o(t),L]);let E;if(n!==null&&n>L){let ur=f([L,n,t,t]);d(Ha,c(aa,[U,S,ur]),[o(t),o(t),n]),E=S}else E=U;let J=f([a,t,t]);d(ea,c(V,[k,W.prelu,ba,J]),[o(t),o(t),a]);let C=ma(),K=w(new Float32Array(a)),O=f([a,t,t,t,t,1,1]);d(ta,c(h,[ba,C,K,m,O]),[o(t),o(t),a]);let T=N(),wa=F(),er=f([a,r,t,t]);d(na,c(h,[m,T.buf,wa.bn,k,er]),[o(t),o(t),r]);let or=f([r,t,t]);d(oa,c(I,[k,E,wa.prelu,m,or]),[o(t),o(t),r]),Y=r,G=t}for(let a=0;a<4;a++)B(16,8,128,!0);M(16,32,32);for(let a=0;a<4;a++)B(32,16,64,!0);M(32,64,64);for(let a=0;a<4;a++)B(64,32,32,!0);M(64,128,128);for(let a=0;a<4;a++)B(128,64,16,!0);M(64,128,null);for(let a=0;a<4;a++)B(128,64,8,!0);M(64,128,null);for(let a=0;a<4;a++)B(128,64,4,!0);M(64,128,null);for(let a=0;a<4;a++)B(128,64,2,!0);let Ya=w(e("conv2d_150","Conv2D").data),Za=w(e("conv2d_150","BiasAdd").data),qa=f([128,1434]);d(ia,c(h,[m,Ya,Za,sa,qa]),[Wa(1434),1,1]);let Ja=w(e("conv2d_152","Conv2D").data),Ka=w(e("conv2d_152","BiasAdd").data),Qa=f([128,1]);d(ia,c(h,[m,Ja,Ka,pa,Qa]),[1,1,1]);let ar=f([1]);d(za,c(ra,[pa,ca,ar]),[1,1,1]);let xa=3*256*256*4;function rr(a,r){r.copyBufferToBuffer(a,0,$,0,xa);let n=r.beginComputePass();for(let t of ya)n.setPipeline(t.pipe),n.setBindGroup(0,t.bg),n.dispatchWorkgroups(...t.wg);n.end(),r.copyBufferToBuffer(sa,0,z[D],0,1434*4),r.copyBufferToBuffer(ca,0,z[D],1434*4,4)}async function tr(){let a=z[D];await a.mapAsync(GPUMapMode.READ);let r=new Float32Array(a.getMappedRange().slice(0));return a.unmap(),{landmarks:r.subarray(0,1434),score:r[1434]}}function nr(){let a=z[D];return a.mapAsync(GPUMapMode.READ).then(()=>{let r=new Float32Array(a.getMappedRange().slice(0));return a.unmap(),{landmarks:r.subarray(0,1434),score:r[1434]}})}function ir(){D=1-D}return{device:i,run:rr,readback:tr,beginReadback:nr,flipReadbackBuffer:ir,inputBufferSize:xa}}export{yr as compileLandmarkModel,mr as loadLandmarkWeights,sr as loadLandmarkWeightsFromBuffer};
