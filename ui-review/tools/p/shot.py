"""Screenshot one view of the operator console in headless Chrome.

usage: shot.py PAGE VIEW WIDTHxHEIGHT THEME OUT.png [--click CSS]... [--scroll PX] [--wait MS] [--gl swiftshader|metal|none]

PAGE   path to the built app.html
VIEW   home | map | swarm | perception | logs
THEME  light | dark

The page is loaded in an iframe of the exact viewport (headless Chrome refuses windows narrower
than 500 px), the theme is set BEFORE the tab is opened (the 3D scene reads the theme once, when
it is first built), the tab button is clicked, optional --click selectors are clicked in order,
then the shot is taken at device scale 2 and cropped to the viewport. The map camera is never
touched, so it is always HOME_VIEW from pitch/ui/scene3d.js. Console errors are printed.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
HERE = Path(__file__).resolve().parent
PROFILE = HERE.parent / ".chrome-profile"
GL = {
    "swiftshader": ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
    "metal": ["--use-gl=angle", "--use-angle=metal", "--ignore-gpu-blocklist"],
    "none": ["--disable-gpu"],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("page", type=Path)
    ap.add_argument("view", choices=["home", "map", "swarm", "perception", "logs"])
    ap.add_argument("viewport")
    ap.add_argument("theme", choices=["light", "dark"])
    ap.add_argument("out", type=Path)
    ap.add_argument("--click", action="append", default=[])
    ap.add_argument("--scroll", type=int, default=0)
    ap.add_argument("--wait", type=int, default=0)
    ap.add_argument("--query", default="", help="query string for the page, e.g. mode=perceived&ep=catch&t=30")
    ap.add_argument("--gl", choices=list(GL), default="swiftshader")
    a = ap.parse_args()
    w, h = (int(v) for v in a.viewport.lower().split("x"))
    wait = a.wait or (4000 if a.view == "map" else 1500)
    page = a.page.resolve()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    wrap = page.parent / f".wrap_{a.out.stem}.html"
    clicks = json.dumps(a.click)
    wrap.write_text(f"""<!doctype html><meta charset=utf-8><style>html,body{{margin:0;background:#888}}iframe{{border:0;width:{w}px;height:{h}px;display:block}}</style>
<iframe id=f src="{page.name}{("?" + a.query) if a.query else ""}"></iframe><pre id=out></pre>
<script>
const f=document.getElementById('f'); f.addEventListener('load',()=>{{ const d=f.contentDocument, win=f.contentWindow; const errs=[];
 win.addEventListener('error',e=>errs.push(String(e.message)));
 d.documentElement.dataset.theme='{a.theme}'; win.dispatchEvent(new Event('resize'));
 const b=d.querySelector('nav.tabbar button[data-tab="tab-{a.view}"]'); if(b) b.click(); else errs.push('no tab button');
 setTimeout(()=>{{ for(const sel of {clicks}){{ const e=d.querySelector(sel); if(e) e.click(); else errs.push('no element '+sel); }}
  if({a.scroll}) d.getElementById('main').scrollTop={a.scroll}; }}, {wait // 2});
 setTimeout(()=>{{ const over=[...d.querySelectorAll('section.page.active *')].filter(e=>{{const r=e.getBoundingClientRect(); return r.width>0 && (r.right>win.innerWidth+0.5 || r.left<-0.5);}}).map(e=>e.id||e.className||e.tagName).slice(0,8);
  const cv=d.querySelector('#viewer canvas');
  document.getElementById('out').textContent=JSON.stringify({{innerWidth:win.innerWidth, scrollWidth:d.documentElement.scrollWidth, mainScrollHeight:d.getElementById('main').scrollHeight, three: !!win.THREE, mapCanvas: cv?[cv.width,cv.height]:null, viewerText:(d.querySelector('#viewer .empty')||{{}}).textContent||null, overflowing:over, perceived: win.__perceived ? win.__perceived.stats() : null, errs}}); }}, {wait}); }});
</script>""")
    profile = PROFILE / a.out.stem
    base = [CHROME, "--headless=new", *GL[a.gl], "--hide-scrollbars", "--no-first-run", "--allow-file-access-from-files",
            "--force-device-scale-factor=2", f"--user-data-dir={profile}", f"--window-size={max(w, 500)},{h}",
            f"--virtual-time-budget={wait + 2500}", "--enable-logging=stderr", "--v=0"]

    def run(extra: list[str]) -> tuple[str, str]:
        # Headless Chrome often does not exit by itself once WebGL is up: stop waiting when the work is done.
        p = subprocess.Popen(base + extra, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        shot = any(x.startswith("--screenshot") for x in extra)
        deadline = time.time() + (45 if shot else 14)
        while time.time() < deadline and p.poll() is None:
            time.sleep(0.5)
            if shot and a.out.exists() and a.out.stat().st_size > 0 and time.time() - a.out.stat().st_mtime > 1.5:
                break
        subprocess.run(["pkill", "-f", str(profile)], check=False)
        try:
            out, err = p.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate()
        return out, err

    url = wrap.as_uri()
    a.out.unlink(missing_ok=True)
    _, err = run([f"--screenshot={a.out}", url])
    out, err2 = run(["--dump-dom", url])
    wrap.unlink(missing_ok=True)
    m = re.search(r'<pre id="out">(.*?)</pre>', out, re.S)
    print(a.out.name, html.unescape(m.group(1)) if m else "NO MEASUREMENT")
    seen = set()
    for line in (err + err2).splitlines():
        if ("CONSOLE" in line or "Uncaught" in line) and line[-120:] not in seen:
            seen.add(line[-120:])
            print("  console:", line[:300])
    if not a.out.exists():
        print("  NO SCREENSHOT WRITTEN")
        return 1
    Image.open(a.out).crop((0, 0, w * 2, h * 2)).save(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
