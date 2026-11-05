// Independent dense-prefix algebra of pinned Pathway BDH-GPU, inference mode.
// MIT Pathway notice retained with source in evals/fixtures/bdh-reference-v1.
const f=Math.fround;
export function normalize(v){const mean=v.reduce((s,x)=>s+x,0)/v.length,variance=v.reduce((s,x)=>s+(x-mean)**2,0)/v.length;return v.map(x=>f((x-mean)/Math.sqrt(variance+1e-5)));}
function project(v,m,width){const out=Array(width).fill(0);for(let i=0;i<v.length;i++)for(let j=0;j<width;j++)out[j]=f(out[j]+f(v[i]*f(m[i*width+j])));return out;}
export function densePrefix(spec,inputs){const {dimensions:d,neurons:n,layers,position}=spec.genes;let x=inputs.map(normalize);
  for(let level=0;level<layers;level++){const values=x,positive=x.map(v=>project(v,spec.encoder,n).map(a=>Math.max(0,a))),keys=positive.map((v,t)=>v.map((a,j)=>{if(position!=='rotary')return a;const angle=(t/65536**(Math.floor(j/2)*2/n))%(Math.PI*2),rot=j%2?v[j-1]:-v[j+1];return f(f(a*f(Math.cos(angle)))+f(rot*f(Math.sin(angle))));}));
    x=values.map((value,t)=>{const y=Array(d).fill(0);for(let past=0;past<t;past++){let dot=0;for(let j=0;j<n;j++)dot=f(dot+f(keys[t][j]*keys[past][j]));for(let j=0;j<d;j++)y[j]=f(y[j]+f(dot*values[past][j]));}const gate=project(normalize(y),spec.encoder_v,n).map((a,j)=>f(Math.max(0,a)*positive[t][j])),decoded=normalize(project(gate,spec.decoder,d));return normalize(value.map((a,j)=>f(a+decoded[j])));});
  }return x;
}
export function features(spec,v){const k=project(normalize(v),spec.encoder,spec.genes.neurons).map(a=>Math.max(0,a));let sq=0;for(const a of k)sq=f(sq+f(a*a));const norm=f(Math.sqrt(sq));if(norm<=1e-12)throw Error('unsupported zero features');return k.map(a=>f(a/norm));}
export class AssociationOracle{
  constructor(spec){this.spec=spec;this.values=Array(spec.genes.dimensions*spec.genes.neurons).fill(0);}
  query(key){const k=features(this.spec,key),d=this.spec.genes.dimensions,out=Array(d).fill(0);for(let i=0;i<k.length;i++)for(let j=0;j<d;j++)out[j]=f(out[j]+f(k[i]*this.values[i*d+j]));return out;}
  learn(key,value){const {rule,decay,learning_rate:rate,dimensions:d}=this.spec.genes;if(rule!=='frozen'&&decay!==1)this.values=this.values.map(a=>f(a*decay));const out=this.query(key),k=features(this.spec,key);if(rule!=='frozen')for(let i=0;i<k.length;i++)for(let j=0;j<d;j++){const error=rule==='delta'?f(value[j]-out[j]):value[j];this.values[i*d+j]=f(this.values[i*d+j]+f(f(rate*k[i])*error));}return out;}
}
export function argmax(v){let best=0;for(let i=1;i<v.length;i++)if(v[i]>v[best])best=i;return best;}
export function compare(a,b){if(a.length!==b.length)throw Error('reference width');let max=0;for(let i=0;i<a.length;i++){const error=Math.abs(a[i]-b[i]);if(!Number.isFinite(error)||error>1e-4+1e-4*Math.abs(b[i]))throw Error(`reference mismatch ${i}: ${a[i]} / ${b[i]}`);max=Math.max(max,error);}return max;}
