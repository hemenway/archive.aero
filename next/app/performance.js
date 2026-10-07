// A bounded local record plus User Timing entries for DevTools / observers.
// Metrics stay on this device; consumers can subscribe to archive-performance.
export function recordPerformance(metric,start,detail={}) {
  const end=performance.now(),sample={metric,ms:Math.round((end-start)*10)/10,at:end,...detail};
  const samples=window.archivePerformance??(window.archivePerformance=[]);
  samples.push(sample);if(samples.length>60)samples.shift();
  performance.clearMeasures(metric);
  performance.measure(metric,{start,end,detail});
  window.dispatchEvent(new CustomEvent('archive-performance',{detail:sample}));
  return sample;
}
