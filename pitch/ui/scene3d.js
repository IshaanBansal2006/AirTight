// ---------- site scene (three.js r128, loaded on first visit to the Map tab) ----------
// One unit is one metre. Data (x, y) maps to world (x - cx, height, -(y - cz)); three.js y is up.
const viewer = $('#viewer'), labelLayer = $('#mapLabels');
// var, not let: the shell may call ensureMap/resizeViewer/requestRender (deep link to #map) before this part has run
var renderer, scene, camera, threeOk, mapLoading;
const cx = (site.bounds[0]+site.bounds[2])/2, cz = (site.bounds[1]+site.bounds[3])/2;
const siteW = site.bounds[2]-site.bounds[0], siteD = site.bounds[3]-site.bounds[1];
const DRONE_ALT = 12, PLAY_RATE = 6;
const qs = new URLSearchParams(location.search);
const episodes = D.episodes;
const st = {ep: qs.get('ep') === 'miss' ? 'miss' : 'catch', t: Math.max(0, parseFloat(qs.get('t')) || 0), playing: false};
const toggles = {sec: true, atk: true, cover: true}; let roundFilter = 'all';

// ---- colour tokens (light fallback, dark fallback) ----
const TOKENS = {
  drone: ['--fleet-drone', '#3b4a9a', '#8f9be0'], go2: ['--fleet-go2', '#2e6f6b', '#5fb3ad'], guard: ['--fleet-guard', '#9a6a1f', '#d9b36a'],
  camera: ['--fleet-camera', '#6b5b95', '#b3a2dc'], intruder: ['--intruder', '#c2410c', '#ff8a5c'], benign: ['--benign', '#a9a59c', '#6d7090'],
  alarm: ['--alarm', '#d62839', '#ff6b7a'], ground: ['--ground', '#fff9f2', '#171a2e'], grid: ['--grid', '#e4d6c3', '#404768'],
  fence: ['--fence', '#5d627f', '#9a9db3'], asset: ['--asset', '#2b3157', '#e8e2ff'], ink: ['--ink', '#1c2033', '#f3efe8'],
  ink2: ['--ink-2', '#5d627f', '#b6b9cc'], surface: ['--surface', '#ffffff', '#252a47'], hair: ['--hair', '#f0e4d4', '#343a5c'],
  round: ['--intruder', '#c2410c', '#ff8a5c'], decoy: ['--warn', '#bbb092', '#bbb092'],
  ramp0: ['--ramp-0', '#6fa8d6', '#4f6fb5'], ramp1: ['--ramp-1', '#5a7fd0', '#7a6fd8'], ramp2: ['--ramp-2', '#6f5fc8', '#b565c8'],
  ramp3: ['--ramp-3', '#a83fa8', '#e0609a'], ramp4: ['--ramp-4', '#d6336c', '#ff8f70'],
};
const isDark = () => { const th = document.documentElement.dataset.theme; return th === 'dark' || (th !== 'light' && matchMedia('(prefers-color-scheme: dark)').matches); };
const normCtx = document.createElement('canvas').getContext('2d');
function tokenCss(key){ const [name, light, dark] = TOKENS[key]; const fb = isDark() ? dark : light; const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim(); if (!v) return fb; normCtx.fillStyle = '#000'; normCtx.fillStyle = v; const n = normCtx.fillStyle; return (n === '#000000' && !/^(#0{3,6}|black|rgb\(0)/i.test(v)) ? fb : (n.startsWith('#') ? n : fb); }

// ---- the one table that builds both the glyphs and the legend ----
const SENSOR_OF = {drone: 'drone_camera', go2: 'go2_camera', guard: 'human_eye', camera: 'fixed_camera'};
const GLYPHS = [
  {key: 'drone', label: `Drone at ${DRONE_ALT} m, drop line, view disc on the ground`, svg: c => `<path d="M4 4l10 10M14 4L4 14" stroke="${c}" stroke-width="2"/><circle cx="4" cy="4" r="2.4" fill="${c}"/><circle cx="14" cy="4" r="2.4" fill="${c}"/><circle cx="4" cy="14" r="2.4" fill="${c}"/><circle cx="14" cy="14" r="2.4" fill="${c}"/>`},
  {key: 'go2', label: 'Go2 robot dog and its forward view', svg: c => `<rect x="2" y="6" width="11" height="6" rx="1.5" fill="${c}"/><rect x="12" y="4" width="4.5" height="4.5" rx="1" fill="${c}"/>`},
  {key: 'guard', label: 'Guard and field of view', svg: c => `<circle cx="9" cy="4.5" r="3" fill="${c}"/><path d="M4.5 16l1.5-7h6l1.5 7z" fill="${c}"/>`},
  {key: 'camera', label: 'Fixed camera and its wedge', svg: c => `<path d="M9 9L17 4v10z" fill="${c}" opacity=".3"/><rect x="3" y="6.5" width="7" height="5" fill="${c}"/><rect x="5.5" y="11" width="1.6" height="6" fill="${c}"/>`},
  {key: 'intruder', label: 'Intruder', svg: c => `<path d="M9 1l6 8-6 8-6-8z" fill="${c}"/>`},
  {key: 'ramp', label: 'Intruder path, coloured by time (early to late)', svg: () => `<defs><linearGradient id="m3ramp"><stop offset="0" stop-color="${tokenCss('ramp0')}"/><stop offset=".25" stop-color="${tokenCss('ramp1')}"/><stop offset=".5" stop-color="${tokenCss('ramp2')}"/><stop offset=".75" stop-color="${tokenCss('ramp3')}"/><stop offset="1" stop-color="${tokenCss('ramp4')}"/></linearGradient></defs><rect x="1" y="7" width="16" height="4" rx="2" fill="url(#m3ramp)"/>`},
  {key: 'ink', label: 'Critical detection point: last moment an alarm is in time', svg: c => `<circle cx="9" cy="9" r="5.5" fill="none" stroke="${c}" stroke-width="2"/><circle cx="9" cy="9" r="1.8" fill="${c}"/>`},
  {key: 'alarm', label: 'Where the intruder was when the alarm fired', svg: c => `<circle cx="9" cy="5.5" r="4" fill="${c}"/><rect x="8.2" y="8" width="1.6" height="9" fill="${c}"/>`},
  {key: 'asset', label: 'Asset (nominal footprint), and the ring inside which an alarm is too late', svg: c => `<circle cx="9" cy="9" r="7.5" fill="none" stroke="${c}" stroke-width="1.4" stroke-dasharray="3 2"/><rect x="5.5" y="5.5" width="7" height="7" fill="${c}"/>`},
  {key: 'gateV', tok: 'fence', label: 'Vehicle gate, 8 m opening with posts', svg: c => `<path d="M0 13h3M15 13h3" stroke="${c}" stroke-width="2"/><rect x="2.5" y="5" width="1.8" height="9" fill="${c}"/><rect x="13.7" y="5" width="1.8" height="9" fill="${c}"/>`},
  {key: 'gateS', tok: 'fence', label: 'Service gate, 4 m opening with posts', svg: c => `<path d="M0 13h5M13 13h5" stroke="${c}" stroke-width="2"/><rect x="4.5" y="5" width="1.8" height="9" fill="${c}"/><rect x="11.7" y="5" width="1.8" height="9" fill="${c}"/>`},
  {key: 'gateP', tok: 'fence', label: 'Pedestrian gate, 2 m opening with posts', svg: c => `<path d="M0 13h7M11 13h7" stroke="${c}" stroke-width="2"/><rect x="6" y="5" width="1.8" height="9" fill="${c}"/><rect x="10.2" y="5" width="1.8" height="9" fill="${c}"/>`},
  {key: 'gap', tok: 'fence', label: 'Fence gap: a break with ragged ends, no posts', svg: c => `<path d="M0 13h5l2-3 1 4M18 13h-5l-2 3-1-4" stroke="${c}" stroke-width="1.8" fill="none"/>`},
  {key: 'lane', tok: 'benign', label: 'Vehicle lane from the site file, 6 m, dashed centre line', svg: c => `<rect x="0" y="4" width="18" height="10" fill="${c}" opacity=".25"/><path d="M1 9h16" stroke="${c}" stroke-width="1.6" stroke-dasharray="4 3"/>`},
  {key: 'foot', tok: 'benign', label: 'Footpath from the site file, 1.5 m, dotted edges', svg: c => `<rect x="0" y="6.5" width="18" height="5" fill="${c}" opacity=".2"/><path d="M1 6.5h16M1 11.5h16" stroke="${c}" stroke-width="1.4" stroke-dasharray="1.5 2.5"/>`},
  {key: 'fence', label: 'Fence; drone docks are square pads with a corner mark, the Go2 dock is round', svg: c => `<path d="M1 14h16" stroke="${c}" stroke-width="2"/><rect x="5" y="4" width="1.8" height="10" fill="${c}"/><rect x="11" y="4" width="1.8" height="10" fill="${c}"/>`},
  {key: 'grid', label: 'Ground grid: a line every 10 m, heavier every 50 m', svg: c => `<path d="M1 5h16M1 13h16M5 1v16M13 1v16" stroke="${c}" stroke-width="1.4"/>`},
  {key: 'benign', label: 'Benign traffic', svg: c => `<path d="M1 9h16" stroke="${c}" stroke-width="1.6" stroke-dasharray="3 2"/><rect x="6.5" y="6" width="5" height="5" fill="${c}"/>`},
  {key: 'round', label: 'Red team paths, the worst tactic of each round', svg: c => `<path d="M1 14L8 8l9-4" stroke="${c}" stroke-width="2" fill="none"/>`},
];
function renderLegend(){ $('#mapLegend').innerHTML = GLYPHS.map(g => `<span><svg viewBox="0 0 18 18">${g.svg(g.key === 'ramp' ? '' : tokenCss(g.tok || g.key))}</svg>${esc(g.label)}</span>`).join('') + `<span class="m3-note">Only what is in the site file is drawn. The simulation has no walls or occlusion, so none are shown.</span>`; }

// ---- materials: made once, recoloured on theme change ----
const mats = [];
function mat(key, opts){ const o = opts || {}; const M = o.basic ? THREE.MeshBasicMaterial : THREE.MeshLambertMaterial; const m = new M({color: tokenCss(key), transparent: o.opacity != null, opacity: o.opacity == null ? 1 : o.opacity, depthWrite: o.opacity == null, side: o.basic ? THREE.DoubleSide : THREE.FrontSide}); if (o.opacity != null) { m.polygonOffset = true; m.polygonOffsetFactor = -1; m.polygonOffsetUnits = -1; } mats.push({m, key}); return m; }
const matCache = {};
const solid = key => matCache[key] || (matCache[key] = mat(key));
const dim = key => matCache[key + ':dim'] || (matCache[key + ':dim'] = mat(key, {opacity: 0.4}));

// ---- ribbons: flat quads whose width follows the zoom, so lines keep a constant pixel weight ----
const ribbons = [];
function makeRibbon(pts, y, material, px, opts){
  const o = opts || {}, closed = !!o.closed, P = [];
  for (const p of pts) { const X = p[0]-cx, Z = -(p[1]-cz); if (!P.length || Math.hypot(X-P[P.length-1][0], Z-P[P.length-1][1]) > 0.05) P.push([X, Z, p[2] == null ? 0 : p[2]]); }
  if (P.length < 2) P.push([P[0][0]+0.01, P[0][1], P[0][2]]);
  const n = P.length, segs = closed ? n : n-1, pos = new Float32Array(n*6), nrm = new Float32Array(n*2), idx = [];
  for (let i=0;i<n;i++){ const a = P[closed ? (i-1+n)%n : Math.max(0,i-1)], b = P[i], c = P[closed ? (i+1)%n : Math.min(n-1,i+1)]; let d1x=b[0]-a[0], d1z=b[1]-a[1], d2x=c[0]-b[0], d2z=c[1]-b[1]; const l1=Math.hypot(d1x,d1z)||1, l2=Math.hypot(d2x,d2z)||1; d1x/=l1; d1z/=l1; d2x/=l2; d2z/=l2; if (!closed && i===0) { d1x=d2x; d1z=d2z; } if (!closed && i===n-1) { d2x=d1x; d2z=d1z; } let tx=d1x+d2x, tz=d1z+d2z; const tl=Math.hypot(tx,tz); if (tl < 1e-6) { tx=d2x; tz=d2z; } else { tx/=tl; tz/=tl; } const nx=-tz, nz=tx, k = 1/Math.max(0.5, Math.abs(nx*-d2z + nz*d2x)); nrm[i*2]=nx*k; nrm[i*2+1]=nz*k; }
  for (let i=0;i<segs;i++){ const a=i*2, b=((i+1)%n)*2; idx.push(a, a+1, b, b, a+1, b+1); }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setIndex(idx);
  const mesh = new THREE.Mesh(g, material); mesh.frustumCulled = false; mesh.renderOrder = o.order || 2;
  const rb = {mesh, P, n, y, px, pos, nrm, times: P.map(p => p[2]), w: -1};
  if (o.colors) { g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n*6), 3)); }
  ribbons.push(rb); return rb;
}
function setRibbonWidth(rb, w){ if (Math.abs(w - rb.w) < rb.w*0.02) return; rb.w = w; const {P, n, pos, nrm, y} = rb, h = w/2; for (let i=0;i<n;i++){ const ox = nrm[i*2]*h, oz = nrm[i*2+1]*h; pos[i*6]=P[i][0]+ox; pos[i*6+1]=y; pos[i*6+2]=P[i][1]+oz; pos[i*6+3]=P[i][0]-ox; pos[i*6+4]=y; pos[i*6+5]=P[i][1]-oz; } rb.mesh.geometry.attributes.position.needsUpdate = true; }
function dashMesh(pts, offsets, on, off, width, y, material){ const v = [], h = width/2; for (const o of offsets) for (let i=0;i<pts.length-1;i++) { const a = pts[i], b = pts[i+1], len = Math.hypot(b[0]-a[0], b[1]-a[1]), ux = (b[0]-a[0])/len, uy = (b[1]-a[1])/len, nx = -uy, ny = ux; for (let u = off/2; u < len; u += on+off) { const u1 = Math.min(len, u+on), c = [[u, o-h], [u, o+h], [u1, o-h], [u1, o-h], [u, o+h], [u1, o+h]]; for (const [uu, oo] of c) v.push(a[0]+ux*uu+nx*oo-cx, y, -(a[1]+uy*uu+ny*oo-cz)); } } const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(v, 3)); const m = new THREE.Mesh(g, material); m.renderOrder = 2; return m; }
const circlePts = (x, y, r, k) => Array.from({length: k}, (_, i) => [x + r*Math.cos(i/k*2*Math.PI), y + r*Math.sin(i/k*2*Math.PI)]);

// ---- glyph builders (nominal metres; the group is scaled with zoom so marks stay readable) ----
function box(w, h, d, m, x, y, z){ const b = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m); b.position.set(x||0, y||0, z||0); return b; }
function cyl(r1, r2, h, m, x, y, z, seg){ const c = new THREE.Mesh(new THREE.CylinderGeometry(r1, r2, h, seg||16), m); c.position.set(x||0, y||0, z||0); return c; }
const BUILD = {
  drone(m){ const g = new THREE.Group(); const a = box(3.4, 0.22, 0.4, m); a.rotation.y = Math.PI/4; const b = box(3.4, 0.22, 0.4, m); b.rotation.y = -Math.PI/4; g.add(a, b, box(1, 0.5, 1, m)); for (const [sx, sz] of [[1,1],[1,-1],[-1,1],[-1,-1]]) g.add(cyl(0.8, 0.8, 0.14, m, sx*1.2, 0.18, sz*1.2, 18)); return g; },
  go2(m){ const g = new THREE.Group(); g.add(box(2.4, 0.8, 1, m, 0, 1.1, 0), box(0.8, 0.7, 0.8, m, 1.45, 1.55, 0)); for (const [sx, sz] of [[1,1],[1,-1],[-1,1],[-1,-1]]) g.add(box(0.28, 0.8, 0.28, m, sx*0.95, 0.4, sz*0.36)); return g; },
  guard(m){ const g = new THREE.Group(); const head = new THREE.Mesh(new THREE.SphereGeometry(0.55, 16, 12), m); head.position.y = 2.6; g.add(cyl(0.45, 0.8, 2, m, 0, 1, 0), head); return g; },
  camera(m){ const g = new THREE.Group(); g.add(cyl(0.16, 0.16, 4, m, 0, 2, 0, 8), box(1.5, 0.8, 0.8, m, 0.35, 4.3, 0)); return g; },
  intruder(m){ const g = new THREE.Group(); const o = new THREE.Mesh(new THREE.OctahedronGeometry(1.1), m); o.scale.set(1, 1.5, 1); o.position.y = 1.8; g.add(o); return g; },
  benign(m){ const g = new THREE.Group(); g.add(box(1.3, 1, 1.3, m, 0, 0.5, 0)); return g; },
  pin(m){ const g = new THREE.Group(); const s = new THREE.Mesh(new THREE.SphereGeometry(0.75, 16, 12), m); s.position.y = 3.4; g.add(cyl(0.1, 0.1, 3, m, 0, 1.5, 0, 8), s); return g; },
};
function wedge(range, fovDeg, material){ const f = THREE.MathUtils.degToRad(Math.min(360, fovDeg)); const g = new THREE.CircleGeometry(range, 48, -f/2, f); g.rotateX(-Math.PI/2); const m = new THREE.Mesh(g, material); m.renderOrder = 1; return m; }

// ---- scene state ----
const zoomScaled = [];           // groups whose scale follows the zoom
const actors = {};               // per episode: [{id, kind, track, glyph, cover, drop, heading, label}]
const epNodes = {};              // per episode: THREE.Group holding everything that belongs to it
const epInfo = {};               // per episode: speed, tReach, ring radius, cdp and alarm positions
const labels = []; const roundObjs = []; const roundLabels = [];
let gridObjs = [], groundMesh, gridMinorMat, gridMajorMat, fenceWallMat, hemi;
const tmpV = {v: null};
function addLabel(text, cls, prio, radiusPx, getPos, isVisible){ const el = document.createElement('div'); el.className = 'm3-label' + (cls ? ' ' + cls : ''); el.textContent = text; labelLayer.appendChild(el); const L = {el, prio, r: radiusPx, getPos, isVisible: isVisible || (() => true), w: 0, h: 0, x: -1, y: -1, shown: true, ax: 0, ay: 0, on: false}; labels.push(L); return L; }
function episodeInfo(ep){ const tr = ep.tracks.intruder; let len = 0, tReach = tr[tr.length-1][0]; for (let i=1;i<tr.length;i++) len += Math.hypot(tr[i][1]-tr[i-1][1], tr[i][2]-tr[i-1][2]); const end = tr[tr.length-1]; for (let i=0;i<tr.length;i++) if (Math.hypot(tr[i][1]-end[1], tr[i][2]-end[2]) < 0.3) { tReach = tr[i][0]; break; } if (ep.t_cdp != null && ep.t_cdp + site.response_time_s < tReach) tReach = ep.t_cdp + site.response_time_s; const speed = len / (tReach || 1); return {speed, tReach, ring: speed * site.response_time_s, tMax: Math.max(ep.t_end || 0, tr[tr.length-1][0])}; }

function ensureMap(){ if (threeOk || mapLoading) return; mapLoading = true; const boot = () => { try { initScene(); applyMapFilter(); } catch(e) { console.error(e); viewer.insertAdjacentHTML('afterbegin', `<p class="empty">3D view unavailable: ${esc(e.message||e)}</p>`); } }; if (window.THREE) { boot(); return; } const sc = document.createElement('script'); sc.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js'; sc.onload = boot; sc.onerror = () => { viewer.insertAdjacentHTML('afterbegin', '<p class="empty">3D view could not load.</p>'); }; document.head.appendChild(sc); }

function initScene(){
  renderer = new THREE.WebGLRenderer({antialias: true, powerPreference: 'low-power'}); renderer.setPixelRatio(Math.min(window.devicePixelRatio||1, 2));
  viewer.insertBefore(renderer.domElement, viewer.firstChild);
  scene = new THREE.Scene(); scene.background = new THREE.Color(tokenCss('ground'));
  camera = new THREE.OrthographicCamera(-100, 100, 60, -60, 1, 3000); tmpV.v = new THREE.Vector3();
  hemi = new THREE.HemisphereLight(0xffffff, 0xbfc4d6, 0.85); scene.add(hemi); const sun = new THREE.DirectionalLight(0xffffff, 0.35); sun.position.set(-60, 200, 90); scene.add(sun);

  // ground, grid (10 m, heavier every 50 m), fence
  groundMesh = new THREE.Mesh(new THREE.PlaneGeometry(siteW, siteD), mat('ground', {basic: true})); groundMesh.rotation.x = -Math.PI/2; groundMesh.position.y = -0.05; groundMesh.renderOrder = 0; scene.add(groundMesh);
  const minor = [], major = []; const [x0, y0, x1, y1] = site.bounds;
  for (let x = Math.ceil(x0/10)*10; x <= x1; x += 10) (x % 50 === 0 ? major : minor).push(x-cx, 0, -(y0-cz), x-cx, 0, -(y1-cz));
  for (let y = Math.ceil(y0/10)*10; y <= y1; y += 10) (y % 50 === 0 ? major : minor).push(x0-cx, 0, -(y-cz), x1-cx, 0, -(y-cz));
  gridMinorMat = new THREE.LineBasicMaterial({color: tokenCss('grid'), transparent: true, opacity: 0.7, depthWrite: false}); gridMajorMat = new THREE.LineBasicMaterial({color: tokenCss('grid'), depthWrite: false});
  for (const [arr, m, yy] of [[minor, gridMinorMat, 0.01], [major, gridMajorMat, 0.02]]) { const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(arr, 3)); const ls = new THREE.LineSegments(g, m); ls.position.y = yy; ls.renderOrder = 1; scene.add(ls); gridObjs.push(ls); }
  const majorBold = makeRibbon([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], 0.02, mat('grid', {basic: true, opacity: 0.9}), 1.5, {closed: true, order: 1}); scene.add(majorBold.mesh);
  for (let x = Math.ceil(x0/50)*50; x <= x1; x += 50) scene.add(makeRibbon([[x, y0], [x, y1]], 0.02, majorBold.mesh.material, 1.6, {order: 1}).mesh);
  for (let y = Math.ceil(y0/50)*50; y <= y1; y += 50) scene.add(makeRibbon([[x0, y], [x1, y]], 0.02, majorBold.mesh.material, 1.6, {order: 1}).mesh);
  const per = site.perimeter.map(p => [p.x, p.y]);
  const inside = new THREE.Shape(per.map(p => new THREE.Vector2(p[0]-cx, p[1]-cz))); const insideG = new THREE.ShapeGeometry(inside); insideG.rotateX(-Math.PI/2); const insideM = new THREE.Mesh(insideG, mat('surface', {basic: true})); insideM.position.y = -0.02; insideM.renderOrder = 0; scene.add(insideM);
  // entries by kind: the opening is cut out of the fence line, to scale
  const ENTRY = {vehicle_gate: {w: 8, posts: true, words: 'vehicles'}, service_gate: {w: 4, posts: true, words: 'service'}, pedestrian_gate: {w: 2, posts: true, words: 'pedestrians'}, fence_gap: {w: 3, posts: false, words: 'break in the fence'}};
  const cuts = per.map(() => []); const fenceLine = mat('fence', {basic: true, opacity: 1}); fenceWallMat = mat('fence', {basic: true, opacity: 0.22});
  const pretty = id => id.replaceAll('_', ' ');
  site.entry_points.forEach(e => { let best = 0, bd = 1e9, bu = 0; per.forEach((a, i) => { const b = per[(i+1)%per.length], l2 = (b[0]-a[0])**2 + (b[1]-a[1])**2, u = Math.max(0, Math.min(1, ((e.position.x-a[0])*(b[0]-a[0]) + (e.position.y-a[1])*(b[1]-a[1]))/l2)), d = Math.hypot(e.position.x-a[0]-u*(b[0]-a[0]), e.position.y-a[1]-u*(b[1]-a[1])); if (d < bd) { bd = d; best = i; bu = u*Math.sqrt(l2); } }); const k = ENTRY[e.kind] || ENTRY.service_gate; cuts[best].push([bu - k.w/2, bu + k.w/2, k]); addLabel(`${pretty(e.id)}, ${k.words}`, 'quiet', 5, 12, v => v.set(e.position.x-cx, 2, -(e.position.y-cz))); });
  per.forEach((a, i) => { const b = per[(i+1)%per.length], len = Math.hypot(b[0]-a[0], b[1]-a[1]), ux = (b[0]-a[0])/len, uy = (b[1]-a[1])/len, ang = Math.atan2(uy, ux); const at = u => [a[0]+ux*u, a[1]+uy*u]; let u = 0; const cs = cuts[i].sort((p, q) => p[0]-q[0]);
    const piece = (u0, u1) => { if (u1 - u0 < 0.1) return; const p0 = at(u0), p1 = at(u1); scene.add(makeRibbon([p0, p1], 0.05, fenceLine, 2.2, {order: 3}).mesh); const wall = new THREE.Mesh(new THREE.PlaneGeometry(u1-u0, 2), fenceWallMat); wall.position.set((p0[0]+p1[0])/2-cx, 1, -((p0[1]+p1[1])/2-cz)); wall.rotation.y = ang; wall.renderOrder = 6; scene.add(wall); };
    for (const [c0, c1, k] of cs) { piece(u, Math.max(0, c0)); for (const [uu, dir] of [[Math.max(0, c0), 1], [Math.min(len, c1), -1]]) { const p = at(uu); if (k.posts) { const g = new THREE.Group(); g.add(cyl(0.22, 0.22, 2.6, solid('fence'), 0, 1.3, 0, 10)); g.position.set(p[0]-cx, 0, -(p[1]-cz)); scene.add(g); zoomScaled.push(g); } else { const q = [p[0] + ux*dir*0.9 - uy*0.8*dir, p[1] + uy*dir*0.9 + ux*0.8*dir], r = [q[0] + ux*dir*0.5 + uy*1.1*dir, q[1] + uy*dir*0.5 - ux*1.1*dir]; scene.add(makeRibbon([p, q, r], 0.05, fenceLine, 1.6, {order: 3}).mesh); } } u = Math.min(len, c1); }
    piece(u, len); });
  site.docks.forEach(dk => { const kind = dk.id.startsWith('drone') ? 'drone' : dk.id.startsWith('go2') ? 'go2' : 'guard', x = dk.position.x, y = dk.position.y, line = mat(kind, {basic: true, opacity: 1}); if (kind === 'drone') { scene.add(box(7, 0.25, 7, solid('hair'), x-cx, 0.125, -(y-cz))); scene.add(makeRibbon([[x-3.5, y-3.5], [x+3.5, y-3.5], [x+3.5, y+3.5], [x-3.5, y+3.5]], 0.3, line, 1.8, {closed: true, order: 5}).mesh); const mk = new THREE.Mesh(new THREE.PlaneGeometry(1.8, 1.8), line); mk.rotation.x = -Math.PI/2; mk.position.set(x-cx-2.6, 0.31, -(y-cz)-2.6); mk.renderOrder = 5; scene.add(mk); } else { scene.add(cyl(2.6, 2.6, 0.25, solid('hair'), x-cx, 0.125, -(y-cz), 32)); scene.add(makeRibbon(circlePts(x, y, 2.6, 32), 0.3, line, 1.8, {closed: true, order: 5}).mesh); } addLabel(pretty(dk.id), 'quiet', 7, 12, v => v.set(x-cx, 0.5, -(y-cz))); });
  const assetG = new THREE.Group(); assetG.add(box(12, 3, 9, solid('asset'), 0, 1.5, 0)); assetG.position.set(site.asset.x-cx, 0, -(site.asset.y-cz)); scene.add(assetG); { const ax = site.asset.x, ay = site.asset.y; scene.add(makeRibbon([[ax-5.4, ay-3.9], [ax+5.4, ay-3.9], [ax+5.4, ay+3.9], [ax-5.4, ay+3.9]], 3.04, mat('surface', {basic: true, opacity: 0.9}), 1.2, {closed: true, order: 7}).mesh); }
  addLabel('asset', 'strong', 1, 14, v => v.set(site.asset.x-cx, 3, -(site.asset.y-cz)));
  site.fixed_sensors.forEach((s, i) => { const c = D.curves[s.sensor_type] || {fov_deg: 60, range: 30}, a = THREE.MathUtils.degToRad(s.heading_deg); const w = wedge(c.range, c.fov_deg, mat('camera', {basic: true, opacity: 0.14})); w.position.set(s.position.x-cx, 0.06 + i*0.004, -(s.position.y-cz)); w.rotation.y = a; w.userData.cover = true; scene.add(w); coverObjs.push(w); const g = BUILD.camera(solid('camera')); g.position.set(s.position.x-cx, 0, -(s.position.y-cz)); g.rotation.y = a; scene.add(g); zoomScaled.push(g); addLabel(pretty(s.id.replace(/^cam_/, 'camera ')), 'quiet', 8, 12, v => v.set(s.position.x-cx, 4, -(s.position.y-cz))); });
  // benign routes: quiet
  const laneFill = mat('benign', {basic: true, opacity: 0.16}), pathFill = mat('benign', {basic: true, opacity: 0.12}), laneMark = mat('benign', {basic: true, opacity: 0.75});
  (site.benign_routes || []).forEach((r, ri) => { const pts = []; for (const p of r.waypoints) if (!pts.some(q => Math.hypot(q[0]-p.x, q[1]-p.y) < 0.5)) pts.push([p.x, p.y]); if (pts.length < 2) return; const veh = r.cls === 'vehicle'; const strip = makeRibbon(pts, 0.035 + ri*0.003, veh ? laneFill : pathFill, 1, {order: 1}); strip.metres = veh ? 6 : 1.5; scene.add(strip.mesh); scene.add(dashMesh(pts, veh ? [0] : [-0.75, 0.75], veh ? 3 : 0.5, veh ? 3 : 1.2, veh ? 0.45 : 0.35, 0.05 + ri*0.003, laneMark)); let bi = 0, bl = 0; for (let i=0;i<pts.length-1;i++) { const l = Math.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1]); if (l > bl) { bl = l; bi = i; } } const lx = (pts[bi][0]+pts[bi+1][0])/2, ly = (pts[bi][1]+pts[bi+1][1])/2; addLabel(pretty(r.id) + (veh ? ' lane' : ' path'), 'quiet', 8, 3, v => v.set(lx-cx, 0.1, -(ly-cz))); });

  // per episode: ring, intruder path, markers, actors
  for (const key of Object.keys(episodes)) buildEpisode(key);

  // red-team rounds: quiet by default
  const roundMat = mat('round', {basic: true, opacity: 0.42}), decoyMat = mat('decoy', {basic: true, opacity: 0.9});
  D.rounds.forEach(r => { if (!r.tactic || !r.tactic.path.length) return; const pts = [...r.tactic.path, [site.asset.x, site.asset.y]]; const rb = makeRibbon(pts, 0.1, roundMat, 2, {order: 3}); rb.mesh.userData.round = r.k; roundObjs.push(rb.mesh); scene.add(rb.mesh); const mid = pts[Math.min(1, pts.length-1)], p0 = pts[0], lx = p0[0] + (mid[0]-p0[0])*0.3, ly = p0[1] + (mid[1]-p0[1])*0.3; const L = addLabel(`Round ${r.k}, ${Math.round((r.tactic.miss||0)*100)}% missed`, 'quiet', 9, 4, v => v.set(lx-cx, 0.2, -(ly-cz)), () => toggles.atk && roundFilter === r.k); roundLabels.push(L); if (r.tactic.decoy) { const ring = makeRibbon(circlePts(r.tactic.decoy[0], r.tactic.decoy[1], 3, 24), 0.1, decoyMat, 2, {closed: true, order: 3}); ring.mesh.userData.round = r.k; roundObjs.push(ring.mesh); scene.add(ring.mesh); } });

  labels.sort((a, b) => a.prio - b.prio);
  paintRamp(); threeOk = true; buildHud(); renderLegend(); selectEpisode(st.ep, true); resizeViewer();
  const cam = qs.get('cam'); setPreset(['top', 'angled', 'follow'].includes(cam) ? cam : 'angled', true);
  new ResizeObserver(resizeViewer).observe(viewer);
  new MutationObserver(refreshColours).observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme', 'class', 'style']});
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', refreshColours);
  requestRender();
}
const coverObjs = [];

function buildEpisode(key){
  const ep = episodes[key], info = epInfo[key] = episodeInfo(ep), root = epNodes[key] = new THREE.Group(); scene.add(root); const list = actors[key] = [];
  const ax = site.asset.x, ay = site.asset.y;
  const fill = new THREE.Mesh(new THREE.CircleGeometry(info.ring, 64), mat('asset', {basic: true, opacity: 0.06})); fill.rotation.x = -Math.PI/2; fill.position.set(ax-cx, 0.03, -(ay-cz)); fill.renderOrder = 1; root.add(fill);
  root.add(makeRibbon(circlePts(ax, ay, info.ring, 72), 0.12, mat('asset', {basic: true, opacity: 0.85}), 1.8, {closed: true, order: 3}).mesh);
  addLabel('too late inside this ring', 'quiet', 6, 2, v => v.set(ax-cx + info.ring*0.72, 0.2, -(ay-cz) + info.ring*0.72), () => st.ep === key);
  const tr = ep.tracks.intruder;
  root.add(makeRibbon(tr.map(p => [p[1], p[2]]), 0.14, mat('intruder', {basic: true, opacity: 0.22}), 3, {order: 4}).mesh);
  const pathMat = new THREE.MeshBasicMaterial({vertexColors: true, side: THREE.DoubleSide, depthWrite: false, transparent: true, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2});
  const path = makeRibbon(tr.map(p => [p[1], p[2], p[0]]), 0.16, pathMat, 5, {order: 5, colors: true}); path.info = info; root.add(path.mesh); info.path = path; rampRibbons.push(path);
  const mark = (t, colorKey, text, prio) => { const p = positionAt(tr, t); const g = new THREE.Group(); g.position.set(p[1]-cx, 0, -(p[2]-cz)); if (colorKey === 'alarm') g.add(BUILD.pin(solid('alarm'))); else { const post = BUILD.pin(solid('ink')); g.add(post); } root.add(g); zoomScaled.push(g); root.add(makeRibbon(circlePts(p[1], p[2], 1, 20), 0.18, mat(colorKey, {basic: true, opacity: 1}), 2.2, {closed: true, order: 6}).mesh); ribbons[ribbons.length-1].pxRadius = 7; addLabel(text, colorKey === 'alarm' ? 'strong' : '', prio, 12, v => v.set(p[1]-cx, 3.5, -(p[2]-cz)), () => st.ep === key); };
  if (ep.t_cdp != null) mark(ep.t_cdp, 'ink', `critical detection point, ${ep.t_cdp.toFixed(1)} s`, 2);
  if (ep.t_alarm != null) mark(ep.t_alarm, 'alarm', `alarm at ${ep.t_alarm.toFixed(1)} s`, 2);
  Object.keys(ep.tracks).sort().forEach((id, i) => {
    let kind = kindOf(id); if (kind === 'decoy') kind = 'benign'; const track = ep.tracks[id];
    const glyph = BUILD[kind](solid(kind)); root.add(glyph); zoomScaled.push(glyph);
    const a = {id, kind, track, glyph, cover: null, drop: null, heading: 0, docked: false, t0: track[0][0], t1: track[track.length-1][0]};
    const curve = D.curves[SENSOR_OF[kind]];
    if (curve) { a.cover = wedge(curve.range, curve.fov_deg, mat(kind, {basic: true, opacity: curve.fov_deg >= 360 ? 0.1 : 0.13})); a.cover.position.y = 0.07 + i*0.004; root.add(a.cover); a.edge = makeRibbon(curve.fov_deg >= 360 ? circlePts(cx, cz, curve.range, 64) : [[cx, cz], ...Array.from({length: 25}, (_, j) => { const an = THREE.MathUtils.degToRad(-curve.fov_deg/2 + curve.fov_deg*j/24); return [cx + curve.range*Math.cos(an), cz + curve.range*Math.sin(an)]; })], 0.09 + i*0.004, mat(kind, {basic: true, opacity: 0.5}), 1.2, {closed: true, order: 2}); root.add(a.edge.mesh); }
    if (a.cover) { const trail = makeRibbon(track.map(p => [p[1], p[2]]), 0.1 + i*0.002, mat(kind, {basic: true, opacity: 0.4}), 1.6, {order: 3}); root.add(trail.mesh); (trailObjs[key] = trailObjs[key] || []).push(trail.mesh); }
    if (kind === 'drone') { a.drop = cyl(0.08, 0.08, 1, dim('drone'), 0, 0, 0, 6); root.add(a.drop); }
    a.label = addLabel(kind === 'benign' ? (id.startsWith('decoy') ? 'decoy' : '') : id.replace('_', ' '), kind === 'benign' ? 'quiet' : '', kind === 'intruder' ? 0 : 3, 13, v => v.set(glyph.position.x, glyph.position.y + 2.5*glyph.scale.y, glyph.position.z), () => st.ep === key && glyph.visible && !!a.label.el.textContent);
    list.push(a);
  });
}
const rampRibbons = [];
function paintRamp(){ const stops = [0,1,2,3,4].map(i => new THREE.Color(tokenCss('ramp'+i))), c = new THREE.Color(); for (const rb of rampRibbons) { const col = rb.mesh.geometry.attributes.color; for (let i=0;i<rb.n;i++){ const f = Math.max(0, Math.min(0.9999, rb.times[i]/rb.info.tReach))*4, k = Math.floor(f); c.copy(stops[k]).lerp(stops[k+1], f-k); col.setXYZ(i*2, c.r, c.g, c.b); col.setXYZ(i*2+1, c.r, c.g, c.b); } col.needsUpdate = true; } }
function refreshColours(){ if (!threeOk) return; for (const {m, key} of mats) m.color.set(tokenCss(key)); scene.background.set(tokenCss('ground')); gridMinorMat.color.set(tokenCss('grid')); gridMajorMat.color.set(tokenCss('grid')); hemi.intensity = isDark() ? 0.95 : 0.85; paintRamp(); renderLegend(); requestRender(); }

// ---- time ----
function selectEpisode(key, silent){ if (!episodes[key]) return; st.ep = key; for (const k of Object.keys(epNodes)) epNodes[k].visible = k === key; const info = epInfo[key]; st.t = Math.min(st.t, info.tMax); const sc = $('#mapScrub'); sc.max = info.tMax.toFixed(1); $('#mapEp').querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.ep === key))); renderCaption(); sceneDirty = true; if (!silent) announce(); requestRender(); }
function renderCaption(){ const ep = episodes[st.ep], info = epInfo[st.ep]; const outcome = ep.timely ? `caught in time, alarm at ${ep.t_alarm.toFixed(1)} s via ${esc((ep.via||'').replace('_',' '))}, ${(ep.t_cdp-ep.t_alarm).toFixed(1)} s before the critical point` : (ep.t_alarm != null ? `too late, alarm at ${ep.t_alarm.toFixed(1)} s, after the critical point at ${ep.t_cdp.toFixed(1)} s` : `missed, no alarm before the critical point at ${ep.t_cdp.toFixed(1)} s`); $('#mapCaption').innerHTML = `Shown: <b>${esc(prettyFleet(ep.fleet))}</b>, tactic <b>${esc(ep.family.replaceAll('_',' '))}</b> through the ${esc(ep.entry.replaceAll('_',' '))} (${esc(ep.tactic)}), seed <b>${ep.seed}</b>. The intruder moves at ${info.speed.toFixed(1)} m/s, so the ring is ${info.ring.toFixed(0)} m. Outcome: <b>${outcome}</b>.`; }
let sceneDirty = true, lastAnnounce = 0;
function announce(){ window.dispatchEvent(new CustomEvent('airtight:time', {detail: {t: st.t, episode: st.ep, source: 'map'}})); }
function setTime(t, key, fromOutside){ if (key && key !== st.ep && episodes[key]) { if (threeOk) selectEpisode(key, true); else st.ep = key; } const max = threeOk ? epInfo[st.ep].tMax : 1e9; st.t = Math.max(0, Math.min(max, +t || 0)); sceneDirty = true; if (!fromOutside) announce(); requestRender(); }
window.sceneSetTime = (t, episodeKey) => setTime(t, episodeKey, true);
window.addEventListener('airtight:time', e => { const d = e.detail || {}; if (d.source === 'map') return; setTime(d.t, d.episode, true); });
$('#mapScrub').addEventListener('input', e => { setPlaying(false); setTime(parseFloat(e.target.value)); });
$('#mapPlay').addEventListener('click', () => { if (!st.playing && threeOk && st.t >= epInfo[st.ep].tMax - 0.05) st.t = 0; setPlaying(!st.playing); });
$('#mapEp').addEventListener('click', e => { const b = e.target.closest('button[data-ep]'); if (b) { st.ep = b.dataset.ep; if (threeOk) selectEpisode(b.dataset.ep); } });
function setPlaying(on){ st.playing = on; const b = $('#mapPlay'); b.textContent = on ? '❚❚' : '▶'; b.setAttribute('aria-label', on ? 'Pause' : 'Play'); if (on) requestRender(); }

const DOCKS = site.docks.map(d => [d.position.x, d.position.y]);
function updateScene(){
  const t = st.t, info = epInfo[st.ep], ep = episodes[st.ep];
  for (const a of actors[st.ep]) {
    const present = t >= a.t0 - 0.01 && t <= a.t1 + 0.6; a.glyph.visible = present; if (a.cover) { a.cover.visible = false; a.edge.mesh.visible = false; } if (a.drop) a.drop.visible = false; if (!present) continue;
    const p = positionAt(a.track, t), q = positionAt(a.track, Math.max(a.t0, t-5)), n = positionAt(a.track, Math.min(a.t1, t+0.5)), b = positionAt(a.track, Math.max(a.t0, t-0.5));
    const X = p[1]-cx, Z = -(p[2]-cz); const dx = n[1]-b[1], dy = n[2]-b[2]; if (Math.hypot(dx, dy) > 0.05) a.heading = Math.atan2(dy, dx);
    let docked = false; if (a.cover && Math.hypot(p[1]-q[1], p[2]-q[2]) < 0.2 && (t - a.t0 >= 5 || Math.hypot(p[1]-positionAt(a.track, t+5)[1], p[2]-positionAt(a.track, t+5)[2]) < 0.2)) for (const d of DOCKS) if (Math.hypot(d[0]-p[1], d[1]-p[2]) <= 3) docked = true;
    { const tm = a.cover && window.__agentMode ? window.__agentMode(ep, a.id, t) : null; if (tm != null) docked = tm === 2; }
    if (docked !== a.docked || !a.inited) { a.docked = docked; a.inited = true; const m = docked ? dim(a.kind) : solid(a.kind); for (const c of a.glyph.children) c.material = m; }
    a.flat = docked; a.glyph.position.set(X, a.kind === 'drone' && !docked ? DRONE_ALT : (docked ? 0.3 : 0), Z); a.glyph.rotation.y = a.kind === 'drone' ? 0 : a.heading;
    if (a.cover && !docked) { const on = toggles.cover; a.cover.visible = on; a.edge.mesh.visible = on; a.cover.position.x = X; a.cover.position.z = Z; a.cover.rotation.y = a.heading; a.edge.mesh.position.set(X, 0, Z); a.edge.mesh.rotation.y = a.heading; }
    if (a.drop && !docked) { a.drop.visible = true; a.drop.position.set(X, DRONE_ALT/2, Z); a.drop.scale.set(glyphScale*1.6, DRONE_ALT, glyphScale*1.6); }
  }
  const path = info.path; let k = 0; while (k < path.n-1 && path.times[k+1] <= t + 1e-6) k++; path.mesh.geometry.setDrawRange(0, k*6);
  $('#mapScrub').value = t.toFixed(1);
  const left = ep.t_cdp - t; $('#mapClock').textContent = `${t.toFixed(1)} s of ${info.tMax.toFixed(0)} s, ${left >= 0 ? left.toFixed(0) + ' s to the critical point' : 'past the critical point'}`;
  sceneDirty = false;
}

// ---- camera: damped orthographic orbit about a ground target ----
const CAM_R = 900, PHI_MIN = 0.002, PHI_MAX = 1.25;
const cur = {tx: 0, tz: 0, th: Math.PI/2, ph: 0.8, h: 80}, goal = {...cur};
let aspect = 1.6, viewH = 600, follow = false, fitted = true, preset = 'angled', dragging = false, appliedH = -1, appliedAspect = -1;
function fitHeight(th, ph){ return fitView(th, ph).h; }
const fitOut = {h: 80, tx: 0, tz: 0};
function fitView(th, ph){ let u0 = 1e9, u1 = -1e9, r0 = 1e9, r1 = -1e9; for (const sx of [-1, 1]) for (const sz of [-1, 1]) for (const Y of [0, DRONE_ALT + 3]) { const X = sx*siteW/2, Z = sz*siteD/2, r = X*Math.sin(th) - Z*Math.cos(th), u = -X*Math.cos(ph)*Math.cos(th) + Y*Math.sin(ph) - Z*Math.cos(ph)*Math.sin(th); r0 = Math.min(r0, r); r1 = Math.max(r1, r); u0 = Math.min(u0, u); u1 = Math.max(u1, u); } const padTop = 2*46/viewH; fitOut.h = Math.max((u1-u0)/2 / (0.9 - padTop/2), (r1-r0)/2/aspect/0.9); const uc = (u0+u1)/2 + fitOut.h*padTop/2, rc = (r0+r1)/2, cp = Math.max(0.2, Math.cos(ph)); fitOut.tx = Math.sin(th)*rc - Math.cos(th)*uc/cp; fitOut.tz = -Math.cos(th)*rc - Math.sin(th)*uc/cp; return fitOut; }
function fitHeightOld(th, ph){ let mu = 0, mr = 0; for (const sx of [-1, 1]) for (const sz of [-1, 1]) for (const Y of [0, DRONE_ALT + 3]) { const X = sx*siteW/2, Z = sz*siteD/2; mr = Math.max(mr, Math.abs(X*Math.sin(th) - Z*Math.cos(th))); mu = Math.max(mu, Math.abs(-X*Math.cos(ph)*Math.cos(th) + Y*Math.sin(ph) - Z*Math.cos(ph)*Math.sin(th))); } return Math.max(mu, mr/aspect) / 0.9; }
const hMax = () => fitHeight(goal.th, goal.ph) * 1.6, H_MIN = 10;
function setPreset(name, snap){ preset = name; follow = name === 'follow'; fitted = !follow; if (name === 'top') { goal.th = Math.PI/2; goal.ph = PHI_MIN; } else if (name === 'angled') { goal.th = Math.PI/2 + 0.12; goal.ph = 0.66; } else if (name === 'follow') { goal.ph = 0.95; goal.h = 42; } if (name !== 'follow') applyFit(); else trackIntruder(); const d = goal.th - cur.th; if (Math.abs(d) > Math.PI) cur.th += Math.sign(d)*2*Math.PI; if (snap) Object.assign(cur, goal); syncHud(); requestRender(); }
function applyFit(){ const f = fitView(goal.th, goal.ph); goal.h = f.h; goal.tx = f.tx; goal.tz = f.tz; }
function fitSite(){ follow = false; fitted = true; applyFit(); preset = ''; syncHud(); requestRender(); }
function trackIntruder(){ const p = positionAt(episodes[st.ep].tracks.intruder, st.t); goal.tx = p[1]-cx; goal.tz = -(p[2]-cz); }
function stepCamera(dt){ if (follow) trackIntruder(); const k = 1 - Math.exp(-dt*11); let moving = false; for (const f of ['tx', 'tz', 'th', 'ph']) { const d = goal[f] - cur[f]; if (Math.abs(d) > (f === 'tx' || f === 'tz' ? 0.02 : 0.0004)) { cur[f] += d*k; moving = true; } else cur[f] = goal[f]; } const dh = Math.log(goal.h/cur.h); if (Math.abs(dh) > 0.001) { cur.h *= Math.exp(dh*k); moving = true; } else cur.h = goal.h;
  const sp = Math.sin(cur.ph), cp = Math.cos(cur.ph); camera.position.set(cur.tx + CAM_R*sp*Math.cos(cur.th), CAM_R*cp, cur.tz + CAM_R*sp*Math.sin(cur.th)); camera.lookAt(cur.tx, 0, cur.tz);
  if (cur.h !== appliedH || aspect !== appliedAspect) { appliedH = cur.h; appliedAspect = aspect; camera.top = cur.h; camera.bottom = -cur.h; camera.left = -cur.h*aspect; camera.right = cur.h*aspect; camera.updateProjectionMatrix(); applyZoom(); }
  return moving; }
function applyZoom(){ const mpp = 2*cur.h/viewH; for (const rb of ribbons) { if (rb.pxRadius) { const s = rb.pxRadius*mpp; rb.mesh.scale.set(1, 1, 1); if (!rb.c0) rb.c0 = [rb.P.reduce((a, p) => a+p[0], 0)/rb.n, rb.P.reduce((a, p) => a+p[1], 0)/rb.n]; rb.mesh.position.set(rb.c0[0]*(1-s), 0, rb.c0[1]*(1-s)); rb.mesh.scale.set(s, 1, s); setRibbonWidth(rb, rb.px*mpp/s); } else setRibbonWidth(rb, rb.metres || rb.px*mpp); } const gs = Math.max(0.8, Math.min(4.2, mpp*7.5)); for (const g of zoomScaled) g.scale.set(gs, gs, gs); glyphScale = gs; sceneDirty = true; updateScaleBar(mpp); }
let glyphScale = 1;
function updateScaleBar(mpp){ const target = 110*mpp; let nice = 1; for (const v of [1, 2, 5, 10, 20, 50, 100, 200]) if (v <= target) nice = v; const el = $('#mapScale'); el.firstElementChild.textContent = `${nice} m`; el.lastElementChild.style.width = (nice/mpp).toFixed(1) + 'px'; }
// ground vector for a screen-space shift of (a right, b up) metres in the view plane
function shiftTarget(a, b){ const th = goal.th, cp = Math.max(0.2, Math.cos(goal.ph)); goal.tx += Math.sin(th)*a - Math.cos(th)*b/cp; goal.tz += -Math.cos(th)*a - Math.sin(th)*b/cp; goal.tx = Math.max(-siteW/2-20, Math.min(siteW/2+20, goal.tx)); goal.tz = Math.max(-siteD/2-20, Math.min(siteD/2+20, goal.tz)); }
function zoomAt(factor, clientX, clientY){ const r = viewer.getBoundingClientRect(), nx = ((clientX-r.left)/r.width)*2-1, ny = -(((clientY-r.top)/r.height)*2-1); const h0 = goal.h, h1 = Math.max(H_MIN, Math.min(hMax(), h0*factor)); goal.h = h1; if (!follow) shiftTarget(nx*aspect*(h0-h1), ny*(h0-h1)); fitted = false; preset = follow ? 'follow' : ''; syncHud(); requestRender(); }
function userPan(dxPx, dyPx){ const mpp = 2*goal.h/viewH; follow = false; fitted = false; preset = ''; shiftTarget(-dxPx*mpp, dyPx*mpp); syncHud(); }
function userOrbit(dxPx, dyPx){ fitted = false; goal.th += dxPx*0.006; goal.ph = Math.max(PHI_MIN, Math.min(PHI_MAX, goal.ph - dyPx*0.006)); if (!follow) preset = ''; goal.h = Math.min(goal.h, hMax()); syncHud(); }
function resizeViewer(){ if (!renderer) return; const w = viewer.clientWidth, h = viewer.clientHeight; if (w < 10 || h < 10) return; renderer.setSize(w, h, false); aspect = w/h; viewH = h; $('#mapHint').hidden = w < 560; if (fitted) { applyFit(); cur.h = goal.h; cur.tx = goal.tx; cur.tz = goal.tz; } for (const L of labels) L.w = 0; appliedH = -1; requestRender(); }
window.addEventListener('resize', resizeViewer);

// pointers: one = orbit (right button, middle or shift = pan), two = pinch zoom and pan
const ptr = [{id: -1, x: 0, y: 0}, {id: -1, x: 0, y: 0}]; let panMode = false, pinchD = 0, pinchX = 0, pinchY = 0;
const nPtr = () => (ptr[0].id >= 0 ? 1 : 0) + (ptr[1].id >= 0 ? 1 : 0);
function pinchReset(){ pinchD = Math.hypot(ptr[0].x-ptr[1].x, ptr[0].y-ptr[1].y); pinchX = (ptr[0].x+ptr[1].x)/2; pinchY = (ptr[0].y+ptr[1].y)/2; }
viewer.addEventListener('pointerdown', e => { if (e.target.closest('.hud')) return; const s = ptr[0].id < 0 ? ptr[0] : ptr[1].id < 0 ? ptr[1] : null; if (!s) return; s.id = e.pointerId; s.x = e.clientX; s.y = e.clientY; panMode = e.button === 2 || e.button === 1 || e.shiftKey; try { viewer.setPointerCapture(e.pointerId); } catch(_) {} dragging = true; viewer.classList.add('dragging'); if (nPtr() === 2) pinchReset(); e.preventDefault(); });
viewer.addEventListener('pointermove', e => { const s = ptr[0].id === e.pointerId ? ptr[0] : ptr[1].id === e.pointerId ? ptr[1] : null; if (!s) return; const dx = e.clientX-s.x, dy = e.clientY-s.y; s.x = e.clientX; s.y = e.clientY; if (nPtr() === 2) { const d0 = pinchD, x0 = pinchX, y0 = pinchY; pinchReset(); if (d0 > 0 && pinchD > 0) zoomAt(d0/pinchD, pinchX, pinchY); userPan(pinchX-x0, pinchY-y0); } else if (panMode) userPan(dx, dy); else userOrbit(dx, dy); requestRender(); });
const ptrUp = e => { for (const s of ptr) if (s.id === e.pointerId) s.id = -1; if (nPtr() === 0) { dragging = false; viewer.classList.remove('dragging'); } };
viewer.addEventListener('pointerup', ptrUp); viewer.addEventListener('pointercancel', ptrUp);
viewer.addEventListener('contextmenu', e => e.preventDefault());
viewer.addEventListener('dblclick', e => { if (!e.target.closest('.hud')) fitSite(); });
viewer.addEventListener('wheel', e => { e.preventDefault(); if (!threeOk) return; zoomAt(Math.exp(Math.max(-60, Math.min(60, e.deltaY)) * (e.ctrlKey ? 0.01 : 0.0022)), e.clientX, e.clientY); }, {passive: false});

function buildHud(){ $('#hud').innerHTML = `<button data-cam="top" aria-pressed="false">Top</button><button data-cam="angled" aria-pressed="false">Angled</button><button data-cam="follow" aria-pressed="false">Follow intruder</button><button data-cam="fit">Fit site</button>`; $('#hud').addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; if (b.dataset.cam === 'fit') fitSite(); else setPreset(b.dataset.cam); }); }
function syncHud(){ $('#hud').querySelectorAll('button[data-cam]').forEach(b => { if (b.dataset.cam !== 'fit') b.setAttribute('aria-pressed', String(b.dataset.cam === preset)); }); }

// ---- labels: fixed-size HTML, greedy placement against marks and each other ----
const rects = new Float32Array(4*160); let nRects = 0;
const hits = (x, y, w, h) => { for (let i=0;i<nRects;i++){ const o = i*4; if (x < rects[o+2] && x+w > rects[o] && y < rects[o+3] && y+h > rects[o+1]) return true; } return false; };
const pushRect = (x, y, w, h) => { if (nRects >= 160) return; const o = nRects*4; rects[o] = x; rects[o+1] = y; rects[o+2] = x+w; rects[o+3] = y+h; nRects++; };
function updateLabels(){ const W = viewer.clientWidth, H = viewer.clientHeight, v = tmpV.v; nRects = 0; pushRect(0, 0, Math.min(W, 330), 46); pushRect(0, H-40, 150, 40);
  for (const L of labels) { L.on = L.isVisible(); if (!L.on) continue; if (!L.w) { L.el.style.display = ''; L.w = L.el.offsetWidth; L.h = L.el.offsetHeight; } L.getPos(v); v.project(camera); L.ax = (v.x+1)/2*W; L.ay = (1-v.y)/2*H; if (L.r > 3) pushRect(L.ax-L.r, L.ay-L.r, L.r*2, L.r*2 + 4); }
  for (const L of labels) { let ok = false, x = 0, y = 0; if (L.on && L.ax > -20 && L.ax < W+20 && L.ay > -20 && L.ay < H+20) { for (let c=0;c<6 && !ok;c++){ if (c === 0) { x = L.ax - L.w/2; y = L.ay - L.r - L.h - 3; } else if (c === 1) { x = L.ax + L.r + 4; y = L.ay - L.h/2; } else if (c === 2) { x = L.ax - L.r - 4 - L.w; y = L.ay - L.h/2; } else if (c === 3) { x = L.ax - L.w/2; y = L.ay + L.r + 6; } else if (c === 4) { x = L.ax + L.r + 2; y = L.ay - L.r - L.h - 8; } else { x = L.ax - L.r - 2 - L.w; y = L.ay + L.r + 8; } ok = x >= 2 && y >= 2 && x+L.w <= W-2 && y+L.h <= H-2 && !hits(x, y, L.w, L.h); } }
    if (ok) { pushRect(x-2, y-1, L.w+4, L.h+2); x = Math.round(x); y = Math.round(y); if (x !== L.x || y !== L.y) { L.x = x; L.y = y; L.el.style.transform = `translate(${x}px,${y}px)`; } if (!L.shown) { L.shown = true; L.el.style.visibility = 'visible'; } } else if (L.shown) { L.shown = false; L.el.style.visibility = 'hidden'; } } }

// ---- render on demand; continuous only while playing, orbiting or settling ----
let rafId = 0, lastNow = 0; const stats = window.__sceneStats = {frames: 0, ms: 0, last: 0};
stats.bench = n => { if (!threeOk) return null; const keep = st.t, t0 = performance.now(); for (let i=0;i<n;i++) { st.t = (i*0.5) % epInfo[st.ep].tMax; stepCamera(0.016); updateScene(); updateLabels(); renderer.render(scene, camera); } renderer.getContext().finish(); const ms = (performance.now()-t0)/n; st.t = keep; sceneDirty = true; requestRender(); return ms; };
function requestRender(){ if (!threeOk || rafId) return; rafId = requestAnimationFrame(tick); }
function tick(now){ rafId = 0; if (!viewer.closest('.page').classList.contains('active')) { lastNow = 0; return; } const t0 = performance.now(); const dt = lastNow ? Math.min(0.05, (now-lastNow)/1000) : 0.016; lastNow = now;
  if (st.playing) { st.t += dt*PLAY_RATE; const max = epInfo[st.ep].tMax; if (st.t >= max) { st.t = max; setPlaying(false); } sceneDirty = true; if (now - lastAnnounce > 100 || !st.playing) { lastAnnounce = now; announce(); } }
  const moving = stepCamera(dt); if (sceneDirty) updateScene(); updateLabels(); renderer.render(scene, camera);
  stats.last = performance.now()-t0; stats.frames++; stats.ms += stats.last;
  if (st.playing || moving || dragging) rafId = requestAnimationFrame(tick); else lastNow = 0; }

// ---- layers, rounds, the list of holes ----
function applyMapFilter(){ if (!threeOk) return; const on = o => roundFilter === 'all' || o.userData.round === roundFilter; roundObjs.forEach(o => { o.visible = toggles.atk && on(o); }); coverObjs.forEach(o => { o.visible = toggles.cover; }); for (const k of Object.keys(trailObjs)) trailObjs[k].forEach(o => { o.visible = toggles.sec; }); sceneDirty = true; requestRender(); }
const trailObjs = {};
const rounds = D.rounds;
$('#mapLead').textContent = rounds.length ? 'Red team paths are the routes that still landed after each round of fixes.' : '';
$('#roundSeg').innerHTML = `<button data-round="all" aria-pressed="true">All rounds</button>` + rounds.map(r => `<button data-round="${r.k}" aria-pressed="false">Round ${r.k}</button>`).join('') + `<button class="m3-layer" data-layer="sec" aria-pressed="true"><i style="background:var(--fleet-drone,#3b4a9a)"></i>Fleet trails</button><button class="m3-layer" data-layer="cover" aria-pressed="true"><i style="background:var(--fleet-camera,#6b5b95)"></i>Fleet sensors</button><button class="m3-layer" data-layer="atk" aria-pressed="true"><i style="background:var(--intruder,#c2410c)"></i>Red team paths</button>`;
$('#roundSeg').addEventListener('click', e => { const b = e.target.closest('button'); if(!b) return; if (b.dataset.round != null) { roundFilter = b.dataset.round === 'all' ? 'all' : parseInt(b.dataset.round); $('#roundSeg').querySelectorAll('button[data-round]').forEach(x => x.setAttribute('aria-pressed', x===b ? 'true' : 'false')); } else { toggles[b.dataset.layer] = !toggles[b.dataset.layer]; b.setAttribute('aria-pressed', String(toggles[b.dataset.layer])); } applyMapFilter(); });
$('#holeList').innerHTML = rounds.length ? rounds.map(r => { const t = r.tactic || {}; const miss = t.miss == null ? (1 - r.worst_pd) : t.miss; return `<div class="row ${miss >= 0.9 ? 'sev-high' : miss >= 0.5 ? 'sev-med' : 'sev-low'}" data-round="${r.k}"><div class="dot"></div><div><div class="t">Round ${r.k}: ${esc((r.worst_family||'').replaceAll('_',' '))} through the ${esc((t.entry||'?').replaceAll('_',' '))}</div><div class="s">${esc(prettyFleet(r.fleet))} at $${r.cost.toFixed(0)}/h${t.phase!=null ? `, enters at phase ${fmt(t.phase)} of the charge cycle` : ''}${r.worst_pd_schedule_blind != null ? `, ${pct(r.worst_pd_schedule_blind)} caught with the schedule hidden` : ''} → ${r.move ? 'fix: ' + r.move.replaceAll('_',' ') : 'budget reached'}</div></div><div class="n">${Math.round(miss*100)}%<span>still missed</span></div></div>`; }).join('') : D.threats.map(t => `<div class="row ${t.miss >= 0.9 ? 'sev-high' : t.miss >= 0.5 ? 'sev-med' : 'sev-low'}"><div class="dot"></div><div><div class="t">${esc(t.family.replaceAll('_',' '))} through the ${esc(t.entry.replaceAll('_',' '))}</div><div class="s">phase ${fmt(t.phase)} of the charge cycle at ${fmt(t.speed,1)} m/s, found by ${esc(t.origin)}</div></div><div class="n">${Math.round(t.miss*100)}%<span>missed</span></div></div>`).join('');
$('#holeList').addEventListener('click', e => { const row = e.target.closest('.row[data-round]'); if(!row) return; const r = rounds.find(x => x.k === parseInt(row.dataset.round)); if(!r || !r.tactic) return; follow = false; fitted = false; preset = ''; goal.tx = r.tactic.path[0][0] - cx; goal.tz = -(r.tactic.path[0][1] - cz); goal.h = 38; roundFilter = r.k; toggles.atk = true; $('#roundSeg').querySelectorAll('button[data-round]').forEach(x => x.setAttribute('aria-pressed', String(x.dataset.round == r.k))); const ab = $('#roundSeg').querySelector('button[data-layer="atk"]'); if (ab) ab.setAttribute('aria-pressed', 'true'); if (threeOk) syncHud(); applyMapFilter(); showTab('tab-map', true); });
if (location.hash === '#map') ensureMap();

