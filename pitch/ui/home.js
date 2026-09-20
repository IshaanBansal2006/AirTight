// ---------- home ----------
const HM_BUILD = 'Run airtight-sweep, then pitch/build_app.py.';
const hmPct = (v) => Math.round(v*100) + '%';
const hmHasCi = (c) => c && Array.isArray(c.ci) && c.ci.length === 2 && c.ci.every(v => typeof v === 'number' && !Number.isNaN(v));
const hmRange = (c) => hmHasCi(c) ? `${Math.round(c.ci[0]*100)} to ${Math.round(c.ci[1]*100)}%` : '';
const hmPd = (c) => c.pd == null ? 'not in the report' : hmPct(c.pd) + (hmHasCi(c) ? ` (${hmRange(c)})` : '');
const hmUsd = (v) => '$' + Number(v).toFixed(0);
const hmShort = (n) => prettyFleet(n).replace(/, (staggered|synchronized)$/, '').replace(/^0 drones, /, '');
const hmStag = (n) => /stagger/.test(n.split('+')[0]);
const hmEmpty = (what) => `<div class="hm-empty">${esc(what)} ${HM_BUILD}</div>`;
const hmPanel = (sel, what, fn) => { try { fn(); } catch (e) { const el = $(sel); if (el) el.innerHTML = hmEmpty(what); } };
const hmOk = Array.isArray(configs) && configs.length > 0 && fixed && fixed.pd != null;

hmPanel('#heroS', 'No report found.', () => {
  if (!hmOk) throw new Error('no report');
  const pds = configs.map(c => c.pd), lo = Math.min(...pds), hi = Math.max(...pds);
  const loC = configs.find(c => c.pd === lo), hiC = configs.find(c => c.pd === hi);
  $('#heroK').textContent = `Current configuration: ${prettyFleet(fixed.name)}, ${hmUsd(fixed.cost)} per hour`;
  $('#heroBig').innerHTML = `${Math.round(fixed.pd*100)}<small>%</small>` + (hmHasCi(fixed) ? `<span class="ci">(${hmRange(fixed)}) of intrusions</span>` : `<span class="ci">of intrusions; no interval in the report</span>`);
  $('#worstBig').innerHTML = fixed.worst_pd == null ? `<span class="ci">Not in the report. Run airtight-redteam, then airtight-sweep.</span>` : fixed.blind_pd != null ? `${Math.round(fixed.blind_pd*100)}<small>%</small><span class="ci">with the charge schedule private; ${hmPct(fixed.worst_pd)} if the adversary has the schedule too. Baseline's worst attack: ${base.worst_pd == null ? 'not in the report' : hmPct(base.worst_pd)}.</span>` : `${Math.round(fixed.worst_pd*100)}<small>%</small><span class="ci">Baseline ${base.worst_pd == null ? 'not in the report' : hmPct(base.worst_pd)}.</span>`;
  $('#heroS').textContent = base.name === fixed.name ? `This site ranges from ${hmPct(lo)} to ${hmPct(hi)} across ${configs.length} configurations.` : `The baseline, ${prettyFleet(base.name)} at ${hmUsd(base.cost)} per hour, caught ${hmPd(base)}. Across ${configs.length} configurations this site ranges from ${hmPct(lo)} to ${hmPct(hi)}.`;
  const span = (hi - lo) || 1;
  $('#range').innerHTML = `<div class="fill" style="width:${((fixed.pd-lo)/span*100).toFixed(1)}%"></div><div class="mark base" style="left:${((base.pd-lo)/span*100).toFixed(1)}%" title="baseline"></div><div class="mark" style="left:${((fixed.pd-lo)/span*100).toFixed(1)}%" title="current configuration"></div>`;
  $('#rangeLo').textContent = `Lowest ${hmPct(lo)}, ${hmShort(loC.name)}, ${hmUsd(loC.cost)} per hour`;
  $('#rangeHi').textContent = `Highest ${hmPct(hi)}, ${hmUsd(hiC.cost)} per hour`;
});

hmPanel('#condList', 'No conditions found in the report.', () => {
  const rows = [];
  rows.push(['False alarms per hour', report.far == null ? 'Not in the report.' : `${Number(report.far).toFixed(report.far % 1 ? 1 : 0)} per hour`]);
  rows.push(['Adversary knowledge', hmOk && fixed.worst_pd != null ? (fixed.blind_pd != null ? 'Worst attack found knows the site; the headline keeps the charge schedule private' : 'Worst attack found knows the site and the charge schedule') : 'Not in the payload.']);
  const tail = String(report.conditions || '').split(/,\s*/).slice(2);
  rows.push(['Sensor calibration', tail.length ? tail[0].replace(/^\w/, c => c.toUpperCase()) : 'Not in the payload.']);
  rows.push(['Seeds', report.n_seeds == null ? 'Not in the report.' : `${report.n_seeds}, shared by every configuration`]);
  $('#condList').innerHTML = rows.map(r => `<div><dt>${esc(r[0])}</dt><dd>${esc(r[1])}</dd></div>`).join('');
  $('#condLine').textContent = tail.length > 1 ? 'Model: ' + tail.slice(1).join(', ') + '.' : '';
});

const ppd = (c) => (c.pd*100)/c.cost;
hmPanel('#savings', 'No cost figures found in the report.', () => {
  if (!hmOk) throw new Error('no report');
  $('#costBig').innerHTML = `${hmUsd(fixed.cost)}<small> per hour</small>`;
  $('#ppd').innerHTML = `${ppd(fixed).toFixed(2)}<small> points per $ per hour</small>`;
  const droneCost = D.fleet && D.fleet.cost_by_type && D.fleet.cost_by_type.drone;
  const sameCostStagger = configs.find(c => c.cost === base.cost && c.name !== base.name && c.name.endsWith('stagger'));
  const savings = [];
  if (sameCostStagger) savings.push({t:`+${Math.round((sameCostStagger.pd-base.pd)*100)} percentage points at the same cost`, s:`Staggering the charge schedule lifted ${prettyFleet(base.name)} from ${hmPd(base)} to ${hmPd(sameCostStagger)} caught in time, both at ${hmUsd(base.cost)} per hour.`});
  if (base.blind_pd != null && base.worst_pd != null) savings.push({t:`Worst attack found: ${hmPct(base.worst_pd)} to ${hmPct(base.blind_pd)} at no cost`, s:`Keeping the baseline's charge schedule private.${droneCost ? ` For scale, one more drone is ${hmUsd(droneCost)} per hour, $${Math.round(droneCost*8760/1000)}k a year at 8,760 hours.` : ''}`});
  if (base.name !== fixed.name) savings.push({t:`${(ppd(fixed)/ppd(base)).toFixed(1)} times more caught per dollar`, s:`${ppd(base).toFixed(2)} percentage points per $ per hour at the baseline, ${ppd(fixed).toFixed(2)} now, on the same ${report.n_seeds} seeds.`});
  $('#savings').innerHTML = savings.map(s => `<div class="saving"><div><div class="t">${esc(s.t)}</div><div class="s">${esc(s.s)}</div></div></div>`).join('');
});

hmPanel('#tiles', 'No operator load figures found in the report.', () => {
  if (!hmOk) throw new Error('no report');
  const perShift = fixed.decisions*8;
  const nRounds = (D.rounds && D.rounds.length) || 0, nThreats = (D.threats && D.threats.length) || 0;
  const tiles = [
    {v:fmt(fixed.decisions), unit:'per hour', k:'Decisions asked of you', d:`about ${perShift < 1.5 ? 'one' : Math.round(perShift)} per 8-hour shift`},
    {v:fmt(report.far, report.far % 1 ? 1 : 0), unit:'per hour', k:'False alarms per hour', d:'the operating point for every number here'},
    {v:fmt(fixed.gap,0), unit:'s per hour', k:'Nobody on patrol', d: fixed.gap === base.gap ? 'same as the baseline' : `baseline ${fmt(base.gap,0)} s per hour`},
    {v:String(nRounds || nThreats), unit:nRounds ? 'rounds' : 'families', k:nRounds ? 'Attack, fix, attack again' : 'Attack families searched', d:`${report.n_seeds} shared seeds per configuration`},
  ];
  $('#tiles').innerHTML = tiles.map(t => `<div class="stat"><div class="v">${t.v}<small> ${t.unit}</small></div><div class="k">${t.k}</div><div class="d">${t.d}</div></div>`).join('');
});

// cost against caught in time
function hmChart(){
  const svg = $('#frontier'), card = $('#frontierCard'); if (!svg || !card) return;
  if (!hmOk) { card.innerHTML = hmEmpty('No report found, so there is nothing to plot.'); return; }
  const W = Math.max(300, Math.round(svg.getBoundingClientRect().width || card.clientWidth - 16)); if (hmChart.w === W) return; hmChart.w = W;
  const wide = W >= 560, fs = wide ? 12 : 11, cw = fs * 0.54;
  const H = Math.round(Math.min(520, Math.max(360, W*0.6))), L = 46, R = 14, T = 14, B = 46;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('width', W); svg.setAttribute('height', H);
  const pts = configs.filter(c => c.pd != null && c.cost != null);
  const cmin = Math.min(...pts.map(c => c.cost)), cmax = Math.max(...pts.map(c => c.cost));
  const xstep = (cmax - cmin) / (wide ? 8 : 5) > 10 ? 20 : (cmax - cmin) / (wide ? 8 : 5) > 5 ? 10 : 5;
  const xmin = Math.floor((cmin - 2)/xstep)*xstep, xmax = Math.ceil((cmax + 2)/xstep)*xstep;
  const top = Math.max(...pts.map(c => hmHasCi(c) ? c.ci[1] : c.pd)), ymax = Math.min(1, Math.ceil((top + 0.02)/0.2)*0.2);
  const X = (c) => L + (c - xmin)/(xmax - xmin) * (W-L-R), Y = (p) => T + (1 - p/ymax) * (H-T-B);
  let s = `<style>#frontier text{font-size:${fs}px}</style>`;
  for (let p = 0; p <= ymax + 1e-9; p += 0.2) s += `<line class="${p ? 'grid' : 'axis'}" x1="${L}" x2="${W-R}" y1="${Y(p)}" y2="${Y(p)}"/><text x="${L-8}" y="${Y(p)+4}" text-anchor="end">${Math.round(p*100)}</text>`;
  for (let x = xmin; x <= xmax; x += xstep) s += `<line class="grid" x1="${X(x)}" x2="${X(x)}" y1="${T}" y2="${H-B}"/><text x="${X(x)}" y="${H-B+16}" text-anchor="middle">${x}</text>`;
  s += `<text class="ttl" x="${(L+W-R)/2}" y="${H-8}" text-anchor="middle">Cost, $ per hour</text>`;
  s += `<text class="ttl" transform="translate(12 ${(T+H-B)/2}) rotate(-90)" text-anchor="middle">Caught in time, %</text>`;
  // frontier: best caught-in-time rate at each cost or less, drawn as steps
  const sorted = [...pts].sort((a,b) => a.cost-b.cost || b.pd-a.pd); const front = []; let best = -1; sorted.forEach(c => { if (c.pd > best) { best = c.pd; front.push(c); } });
  let d = ''; front.forEach((c,i) => { d += i ? `H${X(c.cost).toFixed(1)}V${Y(c.pd).toFixed(1)}` : `M${X(c.cost).toFixed(1)},${Y(c.pd).toFixed(1)}`; }); d += `H${W-R}`;
  s += `<path class="front" d="${d}"/>`;
  // obstacles for label placement: every marker with its whisker, then the arrow
  const boxes = pts.map(c => { const y0 = hmHasCi(c) ? Y(c.ci[1]) : Y(c.pd), y1 = hmHasCi(c) ? Y(c.ci[0]) : Y(c.pd); const r = (c.name === fixed.name || c.name === base.name) ? 11 : 7; return [X(c.cost)-r, Math.min(y0, Y(c.pd)-r), 2*r, Math.max(y1, Y(c.pd)+r) - Math.min(y0, Y(c.pd)-r)]; });
  { let px = null, py = null; const seg = (x1,y1,x2,y2) => { const n = Math.max(1, Math.ceil(Math.hypot(x2-x1,y2-y1)/8)); for (let k = 0; k <= n; k++) boxes.push([x1+(x2-x1)*k/n-3, y1+(y2-y1)*k/n-3, 6, 6]); }; front.forEach(c => { const x = X(c.cost), y = Y(c.pd); if (px != null) { seg(px, py, x, py); seg(x, py, x, y); } px = x; py = y; }); if (px != null) seg(px, py, W-R, py); }
  const hit = (a, b) => Math.max(0, Math.min(a[0]+a[2], b[0]+b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[1]+a[3], b[1]+b[3]) - Math.max(a[1], b[1]));
  let ctx = null; try { ctx = document.createElement('canvas').getContext('2d'); } catch (e) { ctx = null; }
  const fam = getComputedStyle(document.documentElement).getPropertyValue('--font').trim() || 'sans-serif';
  const measure = (text, strong) => { if (ctx) { ctx.font = `${strong ? 600 : 400} ${fs}px ${fam}`; const m = ctx.measureText(text).width; if (m > 0) return m; } return text.length * cw * (strong ? 1.06 : 1); };
  const place = (px, py, text, strong, tryOnly) => {
    const w = measure(text, strong) + 8, h = fs + 5; let bestC = null;
    const cands = []; [10, 22, 38, 58].forEach((r, ri) => { [[1,0],[-1,0],[1,-1],[1,1],[-1,-1],[-1,1],[0,-1],[0,1]].forEach(([ux,uy]) => { if (ri === 0 && ux === 0) { cands.push([0, uy*16, ux]); return; } cands.push([ux*r, uy*(ri ? r*0.8 : 12), ux]); }); });
    for (const [dx, dy, ux] of cands) {
      const tx = px + dx, ty = py + dy, bx = ux > 0 ? tx : ux < 0 ? tx - w : tx - w/2, box = [bx, ty - h/2, w, h];
      const out = box[0] < L+2 || box[0]+w > W-2 || box[1] < T-4 || box[1]+h > H-B-2;
      const n = boxes.reduce((a, b) => a + hit(box, b), 0) + (out ? 1e4 : 0);
      if (!bestC || n < bestC.n) bestC = {n, tx, ty, ux, box, far: Math.hypot(dx, dy) > 18};
      if (n === 0) break;
    }
    if (tryOnly && bestC.n > 0) return null;
    boxes.push(bestC.box);
    const anchor = bestC.ux > 0 ? 'start' : bestC.ux < 0 ? 'end' : 'middle';
    return (bestC.far ? `<line class="lead" x1="${px.toFixed(1)}" y1="${py.toFixed(1)}" x2="${(bestC.ux > 0 ? bestC.box[0] : bestC.ux < 0 ? bestC.box[0]+w : bestC.tx).toFixed(1)}" y2="${bestC.ty.toFixed(1)}"/>` : '') + `<text class="lbl halo${strong ? ' strong' : ''}" x="${bestC.tx.toFixed(1)}" y="${(bestC.ty + fs*0.35).toFixed(1)}" text-anchor="${anchor}">${esc(text)}</text>`;
  };
  let labels = '';
  // the fix: baseline to current configuration
  if (base.name !== fixed.name && base.pd != null) {
    const x1 = X(base.cost), y1 = Y(base.pd), x2 = X(fixed.cost), y2 = Y(fixed.pd), len = Math.hypot(x2-x1, y2-y1) || 1, ux = (x2-x1)/len, uy = (y2-y1)/len;
    const ax = x1 + ux*9, ay = y1 + uy*9, bx = x2 - ux*11, by = y2 - uy*11;
    s += `<path class="arrow" d="M${ax.toFixed(1)},${ay.toFixed(1)}L${(bx-ux*6).toFixed(1)},${(by-uy*6).toFixed(1)}"/><path class="arrowhead" d="M${bx.toFixed(1)},${by.toFixed(1)}L${(bx-ux*9-uy*4).toFixed(1)},${(by-uy*9+ux*4).toFixed(1)}L${(bx-ux*9+uy*4).toFixed(1)},${(by-uy*9-ux*4).toFixed(1)}Z"/>`;
    for (let k = 0; k <= 8; k++) boxes.push([ax + (bx-ax)*k/8 - 4, ay + (by-ay)*k/8 - 4, 8, 8]);
    const dp = Math.round((fixed.pd - base.pd)*100), dc = fixed.cost - base.cost;
    const txt = `${dp >= 0 ? '+' : '-'}${Math.abs(dp)} points for ${dc >= 0 ? '+' : '-'}${hmUsd(Math.abs(dc))} per hour`;
    let dl = null; for (const t of [0.5, 0.35, 0.65, 0.25, 0.75, 0.15]) { dl = place(ax+(bx-ax)*t, ay+(by-ay)*t, txt, true, true); if (dl) break; }
    labels += (dl || place((ax+bx)/2, (ay+by)/2, txt, true)).replace('class="lbl halo strong"', 'class="delta halo"').replace('class="lead"', 'class="lead" style="display:none"');
  }
  // labels: current and baseline first, then one label per fleet make-up at each cost when its points sit together
  const order = [...pts].sort((a,b) => (b.name===fixed.name)-(a.name===fixed.name) || (b.name===base.name)-(a.name===base.name) || b.pd-a.pd);
  const done = new Set();
  order.forEach(c => {
    if (done.has(c.name)) return; done.add(c.name);
    const isSpecial = c.name === fixed.name || c.name === base.name;
    const mates = isSpecial ? [] : pts.filter(o => !done.has(o.name) && o.cost === c.cost && hmShort(o.name) === hmShort(c.name) && o.name !== base.name && o.name !== fixed.name);
    mates.forEach(o => done.add(o.name));
    const py = Y(c.pd);
    const special = c.name === fixed.name ? 'Current' : c.name === base.name ? 'Baseline' : '';
        const text = special ? (wide ? `${special}: ${hmShort(c.name)}, ${hmPct(c.pd)}` : `${special}, ${hmPct(c.pd)}`) : hmShort(c.name);
    labels += place(X(c.cost), py, text, !!special);
  });
  // points, drawn last so they sit above lines; staggered is filled, synchronized is hollow, the baseline is a diamond
  let marks = '';
  [...pts].sort((a,b) => hmStag(b.name) - hmStag(a.name)).forEach(c => {
    const x = X(c.cost), y = Y(c.pd), isB = c.name === base.name, isF = c.name === fixed.name, r = isB || isF ? 5 : 4;
    const role = isF ? 'current configuration' : isB ? 'baseline' : 'configuration';
    const aria = `${prettyFleet(c.name)}, ${role}. Caught in time ${hmPd(c)}. Worst attack found ${c.worst_pd == null ? 'not in the report' : hmPct(c.worst_pd)}. Cost ${hmUsd(c.cost)} per hour.`;
    marks += `<g class="pt${isB ? ' basept' : ''}" tabindex="0" role="img" aria-label="${esc(aria)}" data-name="${esc(c.name)}" data-x="${x.toFixed(1)}" data-y="${y.toFixed(1)}">`;
    if (hmHasCi(c)) marks += `<path class="whisk" d="M${x},${Y(c.ci[0]).toFixed(1)}V${Y(c.ci[1]).toFixed(1)}M${x-4},${Y(c.ci[0]).toFixed(1)}h8M${x-4},${Y(c.ci[1]).toFixed(1)}h8"/>`;
    marks += isB ? `<path class="mk ${hmStag(c.name) ? 'stag' : 'sync'}" d="M${x},${y-r-1}l${r+1},${r+1}l${-r-1},${r+1}l${-r-1},${-r-1}Z"/>` : `<circle class="mk ${hmStag(c.name) ? 'stag' : 'sync'}" cx="${x}" cy="${y}" r="${r}"/>`;
    if (isF) marks += `<circle class="ring" cx="${x}" cy="${y}" r="${r+4}"/>`;
    marks += `<circle class="focus" cx="${x}" cy="${y}" r="${r+7}"/><circle class="hit" cx="${x}" cy="${y}" r="14"/></g>`;
  });
  svg.innerHTML = s + labels + marks;
  const tip = $('#frontierTip');
  const show = (g) => {
    const c = pts.find(o => o.name === g.dataset.name); if (!c) return;
    const role = c.name === fixed.name ? ' (current)' : c.name === base.name ? ' (baseline)' : '';
    tip.innerHTML = `<b>${esc(prettyFleet(c.name))}${role}</b>Caught in time <span>${hmPd(c)}</span>${hmHasCi(c) ? '' : ', no interval in the report'}<br>Worst attack found <span>${c.worst_pd == null ? 'not in the report' : hmPct(c.worst_pd)}</span>${c.blind_pd == null ? '' : `, schedule hidden <span>${hmPct(c.blind_pd)}</span>`}<br>Cost <span>${hmUsd(c.cost)} per hour</span>`;
    tip.hidden = false;
    const ox = svg.offsetLeft, oy = svg.offsetTop, px = +g.dataset.x, py = +g.dataset.y, tw = tip.offsetWidth, th = tip.offsetHeight;
    tip.style.left = Math.max(4, Math.min(card.clientWidth - tw - 4, ox + px - tw/2)) + 'px';
    tip.style.top = (py - th - 16 < 0 ? oy + py + 16 : oy + py - th - 16) + 'px';
  };
  svg.querySelectorAll('.pt').forEach(g => {
    g.addEventListener('mouseenter', () => show(g)); g.addEventListener('focus', () => show(g)); g.addEventListener('click', () => show(g));
    g.addEventListener('mouseleave', () => { if (document.activeElement !== g) tip.hidden = true; }); g.addEventListener('blur', () => { tip.hidden = true; });
    g.addEventListener('keydown', e => { if (e.key === 'Escape') tip.hidden = true; });
  });
  const ico = (inner) => `<svg viewBox="0 0 14 12" aria-hidden="true">${inner}</svg>`;
  $('#frontierLegend').innerHTML = [
    [ico('<circle cx="7" cy="6" r="4" style="fill:var(--fleet-drone);stroke:var(--fleet-drone);stroke-width:1.5"/>'), 'staggered charging'],
    [ico('<circle cx="7" cy="6" r="4" style="fill:var(--surface);stroke:var(--fleet-drone);stroke-width:1.5"/>'), 'synchronized charging'],
    [ico('<path d="M7,0l6,6l-6,6l-6,-6Z" style="fill:var(--surface);stroke:var(--ink);stroke-width:1.5"/>'), 'baseline'],
    [ico('<circle cx="7" cy="6" r="2.5" style="fill:var(--fleet-drone)"/><circle cx="7" cy="6" r="5.5" style="fill:none;stroke:var(--fleet-drone);stroke-width:1.25"/>'), 'current configuration'],
    [ico('<path d="M7,0V12M3,0h8M3,12h8" style="fill:none;stroke:var(--fleet-drone);stroke-width:1.25"/>'), 'interval from the report'],
  ].map(([i, t]) => `<span>${i}${t}</span>`).join('');
  $('#frontierCap').textContent = `The stepped line is the highest caught-in-time rate you can buy at each cost or less; a configuration below it costs as much as a better one. All ${pts.length} configurations face the same ${report.n_seeds} seeds at ${Number(report.far)} false alarm${Number(report.far) === 1 ? '' : 's'} per hour.`;
}
try { hmChart(); } catch (e) { const el = $('#frontierCard'); if (el) el.innerHTML = hmEmpty('The chart could not be drawn from this report.'); }
if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { hmChart.w = 0; try { hmChart(); } catch (e) {} });
if (window.ResizeObserver && $('#frontierCard')) new ResizeObserver(() => { try { hmChart(); } catch (e) {} }).observe($('#frontierCard'));

const fleetAgents = (D.fleet && D.fleet.agents) || agentsOf(catchEp).map(id => ({id, type: kindOf(id)}));
function batteryAt(agent, t){ if(!agent.endurance_s) return null; const cycle = agent.endurance_s + agent.charge_time_s; const off = (D.fleet && D.fleet.stagger && D.fleet.stagger[agent.id]) || 0; const x = ((t + off) % cycle + cycle) % cycle; return x < agent.endurance_s ? {flying:true, frac: 1 - x/agent.endurance_s} : {flying:false, frac: (x - agent.endurance_s)/agent.charge_time_s}; }
$('#fleetStrip').innerHTML = fleetAgents.length ? fleetAgents.map(a => { const b = batteryAt(a, catchEp.t_alarm || 0); const cost = (D.fleet && D.fleet.cost_by_type && D.fleet.cost_by_type[a.type]); const kind = a.type === 'go2' ? 'Ground robot' : String(a.type).replace(/^\w/, c => c.toUpperCase()); return `<div class="agentchip"><div class="t">${esc(a.id.replaceAll('_',' ').replace(/^\w/, c => c.toUpperCase()))}</div><div class="s">${esc(kind)}${cost != null ? `, ${hmUsd(cost)} per hour` : ''}</div>${b ? `<div class="b"><span>${b.flying ? 'Battery' : 'Charging'}</span><div class="bar"><i style="width:${Math.round(b.frac*100)}%${b.flying?'':';background:var(--ramp-2)'}"></i></div><span>${Math.round(b.frac*100)}%</span></div>` : `<div class="b"><span>${a.type==='guard' ? 'On foot, radio' : 'Patrolling'}</span></div>`}</div>`; }).join('') : hmEmpty('No fleet found in the payload.');
