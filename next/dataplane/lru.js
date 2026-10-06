export class ByteLRU {
  constructor(limit=32*1024*1024) { this.limit=limit; this.bytes=0; this.entries=new Map(); }
  get(key) { const v=this.entries.get(key); if(v) { this.entries.delete(key); this.entries.set(key,v); } return v; }
  set(key,value) { this.delete(key); if(value.byteLength > this.limit) return; this.entries.set(key,value); this.bytes+=value.byteLength; while(this.bytes>this.limit) this.delete(this.entries.keys().next().value); }
  delete(key) { const v=this.entries.get(key); if(v) this.bytes-=v.byteLength; this.entries.delete(key); }
  clear() { this.entries.clear(); this.bytes=0; }
}
