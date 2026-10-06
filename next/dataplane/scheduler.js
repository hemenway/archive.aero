import { ByteLRU } from './lru.js';
const abortError = () => new DOMException('Demand superseded','AbortError');
// The clock wraps the globals: browsers throw "Illegal invocation" when window.setTimeout is called as a method of another object.
export class Scheduler {
  constructor({fetch:fetcher=fetch,concurrency=8,cacheBytes=32*1024*1024,linger,onResult=()=>{},onError=()=>{},onStats=()=>{},clock={now:()=>Date.now(),setTimeout:(fn,ms)=>setTimeout(fn,ms),clearTimeout:t=>clearTimeout(t)},backoff=150,timeout=20000}={}) {
    if(!Number.isInteger(concurrency)||concurrency<1)throw new RangeError('Concurrency must be a positive integer');
    this.heap=[];this.onStats=onStats;this.fetch=fetcher; this.concurrency=concurrency; this.onResult=onResult; this.onError=onError; this.clock=clock; this.backoff=backoff; this.timeout=timeout;
    this.cache=new ByteLRU(cacheBytes); this.absent=new Set(); this.pending=new Map(); this.demand=new Map(); this.early=new Map(); this.active=0; this.destroyed=false; this.seq=0;
    // Fetches already on the wire when their key leaves the demand are not aborted: an aborted response never
    // reaches the browser's HTTP cache, so a scrub that came back to the date paid the round trip again. Up to
    // `linger` of them finish into the byte cache beside the demanded fetches (they do not take its slots).
    this.linger=linger??Math.max(1,Math.floor(concurrency/2)); this.lingering=new Map();
    this.metrics={requests:0,cancelled:0,bytes:0,retries:0,cacheHits:0,lingered:0,readopted:0};
  }
  setDemand(requests) {
    const next=new Map();
    for(const r of requests) { const old=next.get(r.key); if(!old || compare(r,old)<0) next.set(r.key,r); }
    this.demand=next;
    for(const [key,job] of this.pending) if(!next.has(key)) {
      this.pending.delete(key);
      // Queued jobs and backoff timers have cost nothing yet and are dropped; a running fetch lingers.
      if(job.running && this.linger>0) { this.lingering.set(key,job); this.metrics.lingered++; }
      else { job.controller.abort(abortError()); if(job.timer!=null) this.clock.clearTimeout(job.timer); if(job.running) this.metrics.cancelled++; }
    }
    for(const [key,r] of next) {
      const job=this.pending.get(key);
      if(job) { job.request=r; continue; }
      // A date scrubbed back to re-adopts its lingering fetches instead of starting them again.
      const lingering=this.lingering.get(key);
      if(lingering) { this.lingering.delete(key); lingering.request=r; this.pending.set(key,lingering); this.metrics.readopted++; continue; }
      const cached=this.cache.get(key);
      if(cached || this.absent.has(key)) { if(cached) this.metrics.cacheHits++; this.onResult(r,cached??null); continue; }
      this.pending.set(key,{request:r,controller:new AbortController(),attempt:0,seq:this.seq++,running:false,timer:null});
    }
    // The oldest lingering fetches make way once there are more than the allowance.
    while(this.lingering.size>this.linger) { const [key,job]=this.lingering.entries().next().value; this.lingering.delete(key); job.controller.abort(abortError()); this.metrics.cancelled++; }
    this.heap=[];for(const job of this.pending.values())if(!job.running&&job.timer==null)this.push(job);
    this.pump();this.onStats(this.stats());
  }
  // A response the page already requested (the boot script's early fetches) is
  // consumed in place of this scheduler's own fetch the first time the key runs.
  adopt(key,promise) { promise.catch(()=>{}); this.early.set(key,promise); }
  pump() {
    if(this.destroyed) return;
    while(this.active-this.lingering.size<this.concurrency) {
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
    // Each attempt owns a fresh controller: a timed-out one must not poison the retry.
    const {key,url}=job.request, controller=job.controller=new AbortController(), signal=controller.signal;
    const timer=this.timeout>0 ? this.clock.setTimeout(()=>controller.abort(new DOMException('Tile timeout','TimeoutError')),this.timeout) : null;
    try {
      this.metrics.requests++;
      const early=this.early.get(key); this.early.delete(key);
      const response=await (early??this.fetch(url,{signal}));
      if(signal.aborted) throw signal.reason??abortError();
      if(response.status!==204 && !response.ok) {
        const e=new Error(`Tile HTTP ${response.status}`); e.retryable=response.status>=500 || response.status===408 || response.status===429;
        const after=Number(response.headers?.get('retry-after')); if(after>0) e.retryAfter=Math.min(after,5)*1000; throw e;
      }
      const bytes=response.status===204 ? null : new Uint8Array(await response.arrayBuffer());
      if(signal.aborted) throw signal.reason??abortError();
      if(bytes) { this.metrics.bytes+=bytes.byteLength; this.cache.set(key,bytes); } else this.absent.add(key);
      if(this.pending.get(key)===job) { this.pending.delete(key); this.onResult({...job.request,contentType:response.headers?.get('content-type')},bytes); }
    } catch(error) {
      // Superseded demand already dropped the job; a timeout did not, and retries. A lingering fetch that fails
      // is simply forgotten: nothing is waiting for it.
      if(this.pending.get(key)===job && error?.name!=='AbortError') {
        if(error.retryable!==false && job.attempt<2) {
          this.metrics.retries++; const delay=error.retryAfter??this.backoff*2**job.attempt; job.attempt++;
          job.timer=this.clock.setTimeout(()=>{job.timer=null;if(this.pending.get(key)===job)this.push(job);this.pump();},delay);
        } else { this.pending.delete(key); this.onError({key,error}); }
      }
    } finally { if(timer!=null)this.clock.clearTimeout(timer); if(this.lingering.get(key)===job)this.lingering.delete(key); job.running=false; this.active--; this.pump();this.onStats(this.stats()); }
  }
  // inflight counts the fetches the current demand waits for; lingering ones are reported apart.
  stats() { return {inflight:this.active-this.lingering.size,lingering:this.lingering.size,queued:Array.from(this.pending.values()).filter(j=>!j.running).length,bytesCached:this.cache.bytes,...this.metrics}; }
  destroy() { this.destroyed=true; this.linger=0; this.setDemand([]); this.cache.clear(); this.absent.clear(); this.early.clear(); }
}
const compare = (a,b) => (a.priority??1)-(b.priority??1) || (a.distance??0)-(b.distance??0);

const compareJob=(a,b)=>compare(a.request,b.request)||a.seq-b.seq;
