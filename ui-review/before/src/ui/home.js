// ---------- home ----------
const pds = configs.map(c => c.pd), lo = Math.min(...pds), hi = Math.max(...pds);
const cheapest = configs.reduce((a,c) => c.cost < a.cost ? c : a, configs[0]);
$('#heroBig').innerHTML = `${Math.round(fixed.pd*100)}<small>of 100</small>`;
$('#heroS').textContent = `${prettyFleet(fixed.name)} at $${fixed.cost.toFixed(0)} an hour. This site scores between ${Math.round(lo*100)} and ${Math.round(hi*100)} across ${configs.length} fleet mixes.`;
const span = (hi - lo) || 1;
$('#range').innerHTML = `<div class="fill" style="width:${((fixed.pd-lo)/span*100).toFixed(1)}%"></div><div class="mark base" style="left:${((base.pd-lo)/span*100).toFixed(1)}%" title="baseline"></div><div class="mark" style="left:${((fixed.pd-lo)/span*100).toFixed(1)}%" title="deployed"></div>`;
$('#rangeLo').textContent = `${Math.round(lo*100)} · cheapest mix $${cheapest.cost.toFixed(0)}/h`;
$('#rangeHi').textContent = `best mix ${Math.round(hi*100)}`;
$('#heroChips').innerHTML = [
  `<span class="chip">worst tactic <b>${pct(fixed.worst_pd)}</b></span>`,
  fixed.blind_pd == null ? '' : `<span class="chip">schedule hidden <b>${pct(fixed.blind_pd)}</b></span>`,
  `<span class="chip">was <b>${Math.round(base.pd*100)}</b> before the fix</span>`,
].join('');
const ppd = (c) => (c.pd*100)/c.cost;
$('#costBig').innerHTML = `$${fixed.cost.toFixed(0)}<small>/h</small>`;
$('#ppd').innerHTML = `${ppd(fixed).toFixed(2)}<small> pts/$</small>`;
const droneCost = (D.fleet && D.fleet.cost_by_type && D.fleet.cost_by_type.drone) || 7;
const sameCostStagger = configs.find(c => c.cost === base.cost && c.name !== base.name && c.name.endsWith('stagger'));
const savings = [];
if (sameCostStagger) savings.push({t:`+${Math.round((sameCostStagger.pd-base.pd)*100)} points for $0`, s:`Staggering the charge schedule lifted ${prettyFleet(base.name)} from ${Math.round(base.pd*100)} to ${Math.round(sameCostStagger.pd*100)} at the same $${base.cost.toFixed(0)}/h.`});
if (base.blind_pd != null) savings.push({t:`Worst case ${pct(base.worst_pd)} to ${pct(base.blind_pd)} for $0`, s:`Keeping the charge schedule private is worth about one more drone: $${droneCost.toFixed(0)}/h, $${Math.round(droneCost*8760/1000)}k a year at 24/7.`});
savings.push({t:`${(ppd(fixed)/ppd(base)).toFixed(1)}× more detection per dollar`, s:`${ppd(base).toFixed(2)} points per dollar before, ${ppd(fixed).toFixed(2)} now, on the same ${report.n_seeds} intrusion seeds.`});
$('#savings').innerHTML = savings.map(s => `<div class="saving"><div class="ico">$</div><div><div class="t">${esc(s.t)}</div><div class="s">${esc(s.s)}</div></div></div>`).join('');
const dir = (a,b,higherBetter) => a===b ? 'flat' : ((higherBetter ? a>b : a<b) ? 'up' : 'down');
const perShift = (fixed.decisions*8);
const tiles = [
  {v:fmt(fixed.decisions), unit:'/h', k:'decisions asked of you', d:`about ${perShift < 1 ? 'one' : Math.round(perShift)} per 8-hour shift`, cls:'flat'},
  {v:report.far, unit:'/h', k:'false alarms at the operating point', d:'the threshold is set on quiet nights', cls:'flat'},
  {v:fmt(fixed.gap,0), unit:' s/h', k:'nobody on patrol', d:`was ${fmt(base.gap,0)} s/h`, cls:dir(fixed.gap, base.gap, false)},
  {v:String(D.rounds.length || D.threats.length), unit:'', k:D.rounds.length ? 'attack, fix, re-attack rounds' : 'tactic families searched', d:`${report.n_seeds} shared seeds per fleet`, cls:'flat'},
];
$('#tiles').innerHTML = tiles.map(t => `<div class="stat"><div class="v">${t.v}<small>${t.unit||''}</small></div><div class="k">${t.k}</div><div class="d ${t.cls}">${t.d}</div></div>`).join('');
$('#condLine').textContent = report.conditions;
// frontier chart
(function(){
  const W=360, H=230, L=44, R=16, T=16, B=40; const xs = configs.map(c=>c.cost); const xmin = Math.min(...xs)-4, xmax = Math.max(...xs)+4;
  const X = (c) => L + (c - xmin)/(xmax-xmin) * (W-L-R), Y = (p) => T + (1-p) * (H-T-B);
  let s = '';
  for (let p=0; p<=1; p+=0.25) s += `<line x1="${L}" x2="${W-R}" y1="${Y(p)}" y2="${Y(p)}" stroke="var(--hair)" /><text x="${L-6}" y="${Y(p)+4}" text-anchor="end">${Math.round(p*100)}</text>`;
  const ticks = [...new Set(xs)].sort((a,b)=>a-b);
  ticks.forEach(x => s += `<text x="${X(x)}" y="${H-B+16}" text-anchor="middle">$${x.toFixed(0)}</text>`);
  s += `<text x="${(L+W-R)/2}" y="${H-6}" text-anchor="middle">fleet cost per hour</text>`;
  // frontier: best detection at each cost or less
  const sorted = [...configs].sort((a,b)=>a.cost-b.cost || b.pd-a.pd); const front=[]; let best=-1; sorted.forEach(c => { if(c.pd > best){ best=c.pd; front.push(c);} });
  s += `<path d="${front.map((c,i)=>`${i?'L':'M'}${X(c.cost).toFixed(1)},${Y(c.pd).toFixed(1)}`).join(' ')}" fill="none" stroke="var(--warn)" stroke-width="2" stroke-linejoin="round" opacity=".7"/>`;
  configs.forEach(c => { const col = c.name===fixed.name ? 'var(--good)' : c.name===base.name ? 'var(--atk)' : 'var(--ink3)'; const r = (c.name===fixed.name||c.name===base.name) ? 7 : 5; s += `<line x1="${X(c.cost)}" x2="${X(c.cost)}" y1="${Y(c.ci[0])}" y2="${Y(c.ci[1])}" stroke="${col}" stroke-width="1.5" opacity=".6"/><circle cx="${X(c.cost)}" cy="${Y(c.pd)}" r="${r}" fill="${col}" stroke="var(--surface)" stroke-width="2"><title>${esc(prettyFleet(c.name))}: ${Math.round(c.pd*100)} at $${c.cost.toFixed(0)}/h</title></circle>`; });
  [[fixed,'deployed'],[base,'baseline']].forEach(([c,l]) => { const above = c.pd < 0.5; s += `<text class="lbl" x="${X(c.cost)}" y="${Y(c.pd) + (above ? -12 : 20)}" text-anchor="middle">${l} ${Math.round(c.pd*100)}</text>`; });
  $('#frontier').innerHTML = s;
})();
const fleetAgents = (D.fleet && D.fleet.agents) || agentsOf(catchEp).map(id => ({id, type: kindOf(id)}));
function batteryAt(agent, t){ if(!agent.endurance_s) return null; const cycle = agent.endurance_s + agent.charge_time_s; const off = (D.fleet && D.fleet.stagger && D.fleet.stagger[agent.id]) || 0; const x = ((t + off) % cycle + cycle) % cycle; return x < agent.endurance_s ? {flying:true, frac: 1 - x/agent.endurance_s} : {flying:false, frac: (x - agent.endurance_s)/agent.charge_time_s}; }
$('#fleetStrip').innerHTML = fleetAgents.map(a => { const b = batteryAt(a, catchEp.t_alarm || 0); const cost = (D.fleet && D.fleet.cost_by_type[a.type]) || null; return `<div class="agentchip"><div class="t">${esc(a.id.replace('_',' '))}</div><div class="s">${a.type === 'go2' ? 'ground robot' : a.type}${cost!=null ? ` · $${cost.toFixed(0)}/h` : ''}</div>${b ? `<div class="b"><span>${b.flying ? 'battery' : 'charging'}</span><div class="bar"><i style="width:${Math.round(b.frac*100)}%${b.flying?'':';background:var(--warn)'}"></i></div><span>${Math.round(b.frac*100)}%</span></div>` : `<div class="b"><span>${a.type==='guard' ? 'on foot, radio' : 'patrolling'}</span></div>`}</div>`; }).join('');

