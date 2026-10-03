import { ByteLRU } from './lru.js';
const abortError = () => new DOMException('Demand superseded','AbortError');
export class Scheduler {
  constructor({fetch:fetcher=fetch,concurrency=8,cacheBytes=32*1024*1024,onResult=()=>{},onError=()=>{},onStats=()=>{},clock={now:()=>Date.now(),setTimeout,clearTimeout},backoff=150}={}) {
    if(!Number.isInteger(concurrency)||concurrency<1)throw new RangeError('Concurrency must be a positive integer');
    this.heap=[];this.onStats=onStats;this.fetch=fetcher; this.concurrency=concurrency; this.onResult=onResult; this.onError=onError; this.clock=clock; this.backoff=backoff;
    this.cache=new ByteLRU(cacheBytes); this.absent=new Set(); this.pending=new Map(); this.demand=new Map(); this.active=0; this.destroyed=false; this.seq=0;
    this.metrics={requests:0,cancelled:0,bytes:0,retries:0};
  }
  setDemand(requests) {
    const next=new Map();
    for(const r of requests) { const old=next.get(r.key); if(!old || compare(r,old)<0) next.set(r.key,r); }
    this.demand=next;
    for(const [key,job] of this.pending) if(!next.has(key)) { this.pending.delete(key); job.controller.abort(abortError()); if(job.timer!=null) this.clock.clearTimeout(job.timer); if(job.running) this.metrics.cancelled++; }
    for(const [key,r] of next) {
      const job=this.pending.get(key);
      if(job) { job.request=r; continue; }
      const cached=this.cache.get(key);
      if(cached || this.absent.has(key)) { this.onResult(r,cached??null); continue; }
      this.pending.set(key,{request:r,controller:new AbortController(),attempt:0,seq:this.seq++,running:false,timer:null});
    }
    this.heap=[];for(const job of this.pending.values())if(!job.running&&job.timer==null)this.push(job);
    this.pump();this.onStats(this.stats());
  }
  pump() {
    if(this.destroyed) return;
    while(this.active<this.concurrency) {
      const best=this.pop();
      if(!best)break;
      if(this.pending.get(best.request.key)!==best || best.running || best.timer!=null)continue;
      best.running=true; this.active++; void this.run(best);
    }
  }
  push(job) {
    const heap=this.heap;let i=heap.length;heap.push(job);
    while(i>0){const p=(i-1)>>>1;if(compareJob(heap[p],job)<=0)break;heap[i]=heap[p];i=p;}heap[i]=job;
  }
  pop() {
    const heap=this.heap;if(!heap.length)return null;const first=heap[0],last=heap.pop();
    if(heap.length){let i=0;while(i*2+1<heap.length){let c=i*2+1;if(c+1<heap.length&&compareJob(heap[c+1],heap[c])<0)c++;if(compareJob(last,heap[c])<=0)break;heap[i]=heap[c];i=c;}heap[i]=last;}
    return first;
  }
  async run(job) {
    const {key,url}=job.request, signal=job.controller.signal;
    try {
      this.metrics.requests++;
      const response=await this.fetch(url,{signal});
      if(signal.aborted) throw abortError();
      if(response.status!==204 && !response.ok) { const e=new Error(`Tile HTTP ${response.status}`); e.retryable=response.status>=500 || response.status===408 || response.status===429; throw e; }
      const bytes=response.status===204 ? null : new Uint8Array(await response.arrayBuffer());
      if(signal.aborted) throw abortError();
      if(bytes) { this.metrics.bytes+=bytes.byteLength; this.cache.set(key,bytes); } else this.absent.add(key);
      if(this.pending.get(key)===job) { this.pending.delete(key); this.onResult({...job.request,contentType:response.headers?.get('content-type')},bytes); }
    } catch(error) {
      if(!signal.aborted && this.pending.get(key)===job) {
        if(error.retryable!==false && job.attempt<2) {
          this.metrics.retries++; const delay=this.backoff*2**job.attempt++;
          job.timer=this.clock.setTimeout(()=>{job.timer=null;if(this.pending.get(key)===job)this.push(job);this.pump();},delay);
        } else { this.pending.delete(key); this.onError({key,error}); }
      }
    } finally { job.running=false; this.active--; this.pump();this.onStats(this.stats()); }
  }
  stats() { return {inflight:this.active,queued:Array.from(this.pending.values()).filter(j=>!j.running).length,bytesCached:this.cache.bytes,...this.metrics}; }
  destroy() { this.destroyed=true; this.setDemand([]); this.cache.clear(); this.absent.clear(); }
}
const compare = (a,b) => (a.priority??1)-(b.priority??1) || (a.distance??0)-(b.distance??0);

const compareJob=(a,b)=>compare(a.request,b.request)||a.seq-b.seq;
