import json, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "pitch"))
from airtight.contracts import FleetConfig, SensorCurves, Site
import perception_map as pm
SCEN = REPO / "scenarios" / "logistics_yard"
site = Site.model_validate_json((SCEN / "site.json").read_text())
curves = SensorCurves.model_validate_json((SCEN / "sensor_curve.json").read_text())
for kind in ("miss", "catch"):
    log = next((REPO / "data" / "clip_logs" / kind).glob("*.jsonl"))
    fleet = FleetConfig.model_validate_json((SCEN / "fleets" / (log.name.split("__")[0] + ".json")).read_text())
    t = time.time()
    v = pm.build_view(log, fleet, site, curves)
    blob = json.dumps(v, separators=(",", ":"))
    print(kind, "seconds", round(time.time() - t, 1), "bytes", len(blob), "frames", v["n"], "real_frames", v["pm"]["real_frames"])
    for k, x in v.items():
        print("  ", k, len(json.dumps(x)))
    for k, x in v["pm"].items():
        print("   pm", k, len(json.dumps(x)))
    for src, rec in v["pm"]["src"].items():
        print("   ", src, "ghosts max", max(rec["ghosts"]), "occ changes", len(rec["occ"]), "vis", rec["vis"][:6])
    print("   classes", v["classes"], "pmax", v["pmax"], "wmax", v["wmax"], "never0", v["never0"])
