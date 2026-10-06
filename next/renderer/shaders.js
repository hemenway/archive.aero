// GLSL ES 3.00; positions are normalized Web Mercator, sizes are CSS pixels.
export const camera = `uniform vec2 center; uniform vec2 viewport; uniform float world;
vec2 screen(vec2 p){return (p-center)*world+viewport*.5;}
vec4 clip(vec2 p){return vec4(p/viewport*vec2(2.,-2.)+vec2(-1.,1.),0.,1.);}`;
export const tileVertex = `#version 300 es
precision highp float; precision highp int;
${camera}
uniform vec3 tile; uniform vec4 uvRect; out vec2 uv;
void main(){vec2 q=vec2(float(gl_VertexID&1),float((gl_VertexID>>1)&1));
uv=uvRect.xy+q*uvRect.zw;gl_Position=clip(screen(tile.xy+q*tile.z));}`;
export const tileFragment = `#version 300 es
precision highp float; precision highp int; precision highp sampler2DArray;
uniform sampler2DArray atlas; uniform float layer; uniform float opacity;
in vec2 uv; out vec4 color;
void main(){color=texture(atlas,vec3(uv,layer));color.a*=opacity;}`;
export const ringVertex = `#version 300 es
precision highp float; precision highp int; layout(location=0) in vec2 position;
${camera}
uniform float wrap; void main(){gl_Position=clip(screen(position+vec2(wrap,0.)));}`;
export const ringFragment = `#version 300 es
precision highp float; precision highp int; out vec4 color; void main(){color=vec4(1.);}`;
export const fieldVertex = `#version 300 es
precision highp float; precision highp int;
layout(location=0) in vec2 position; layout(location=1) in vec3 datesStatus;
${camera}
uniform float wrap; uniform float radius; uniform float year; uniform uint statusMask;
out vec2 local; flat out int status; flat out float faint;
void main(){status=int(datesStatus.z);float start=datesStatus.x,end=datesStatus.y;
bool on=(statusMask & (1u<<uint(status)))!=0u && (year==0. || ((start==0.||year>=start) && (status==0||end==0.||year<=end)));
vec2 q=vec2(float(gl_VertexID&1),float((gl_VertexID>>1)&1))*2.-1.;
local=q*(radius+1.);faint=start==0.&&end==0.? .45:1.;
gl_Position=on?clip(screen(position+vec2(wrap,0.))+local):vec4(2.,2.,0.,1.);}`;
export const fieldFragment = `#version 300 es
precision highp float; precision highp int; in vec2 local; flat in int status; flat in float faint;
uniform float radius; out vec4 color;
void main(){float d=length(local);if(d>radius+1.)discard;
vec3 fill=status==0?vec3(54.,163.,93.)/255.:status==1?vec3(194.,59.,42.)/255.:vec3(139.,130.,116.)/255.;
color=vec4(d>=radius-1.?vec3(1.):fill,d>=radius-1.?max(.55,faint):faint);}`;
export const lineVertex = `#version 300 es
precision highp float; precision highp int;
layout(location=0) in vec2 position; layout(location=1) in vec2 previous;
layout(location=2) in vec2 next; layout(location=3) in vec2 sideDistance;
layout(location=4) in vec4 datesMasks; layout(location=5) in vec2 styleFloor;
${camera}
uniform float wrap; uniform float zoom; uniform float day; uniform uint classMask; uniform uint regionMask; uniform int pass;
out float distancePx; out float across; flat out int style; flat out float band;
void main(){style=int(styleFloor.x);bool floor=style==6;
bool on=(classMask & uint(datesMasks.z))!=0u && (regionMask & uint(datesMasks.w))!=0u &&
(day>=datesMasks.x&&day<datesMasks.y) && (!floor||zoom>=7.);
float base=style==0?2.8:style==1?2.6:style==2?2.4:style==3?1.8:style==4?1.6:2.;
float width=base*(zoom>=10.?1.:zoom>=8.?.8:.65)+(pass==0?2.2:0.);
band=zoom>=10.?11.:zoom>=9.?9.:zoom>=8.?7.:5.;
vec2 a=position-previous,b=next-position;if(length(a)<1e-10)a=b;if(length(b)<1e-10)b=a;
a=normalize(a);b=normalize(b);vec2 n1=vec2(a.y,-a.x),n2=vec2(b.y,-b.x);
vec2 sum=n1+n2;vec2 n=length(sum)<.001?n2:normalize(sum);
float m=min(2.5,1./max(.4,abs(dot(n,n2))));
float offset=floor?(sideDistance.x+1.)*.5*band:sideDistance.x*width*.5;
across=(sideDistance.x+1.)*.5;distancePx=sideDistance.y*world;
gl_Position=on?clip(screen(position+vec2(wrap,0.))+n*offset*m):vec4(2.,2.,0.,1.);}`;
export const lineFragment = `#version 300 es
precision highp float; precision highp int; in float distancePx; in float across; flat in int style; flat in float band;
uniform int pass; uniform float floorColor; out vec4 color;
void main(){float dash=style==3?7.:style==4?5.:0.;if(dash>0.&&mod(distancePx,dash+4.)>=dash)discard;
vec3 blue=vec3(79.,140.,245.)/255.,magenta=vec3(238.,82.,178.)/255.;
if(style==6){if(pass==1)discard;float alpha=(across<=1./3.?3.:across<=2./3.?2.:1.)/6.;color=vec4(floorColor==700.?magenta:blue,alpha);}
else color=pass==0?vec4(8.,12.,18.,153.)/255.:vec4(style==2||style==4?magenta:blue,1.);}`;
export const pinVertex = `#version 300 es
precision highp float; precision highp int; ${camera}
uniform vec2 position; out vec2 local;
void main(){local=vec2(float(gl_VertexID&1),float((gl_VertexID>>1)&1))*2.-1.;gl_Position=clip(screen(position)+local*10.);}`;
export const pinFragment = `#version 300 es
precision highp float; precision highp int; in vec2 local; out vec4 color;
void main(){float d=length(local);if(d>1.)discard;color=vec4(d>.75?vec3(1.):vec3(.12,.56,1.),1.);}`;
