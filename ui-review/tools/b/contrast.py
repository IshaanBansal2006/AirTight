"""Contrast (WCAG 2.x) and colour-blind separation check for the AirTight UI tokens."""
from __future__ import annotations
import numpy as np

LIGHT = dict(ground="#f3f4f0", surface="#ffffff", surface2="#e8ebe5", hair="#cfd4cc", grid="#dde1da", fence="#3c4650",
             ink="#14202b", ink2="#44515d", ink3="#5d6873", drone="#1f66b8", go2="#0a7273", guard="#2a2a78", camera="#7ba3c7",
             intruder="#b84f08", alarm="#c8102e", benign="#8b8f94", focus="#0b57d0", asset="#14202b")
DARK = dict(ground="#12181e", surface="#1a222a", surface2="#232d37", hair="#36434f", grid="#26313b", fence="#aab4bd",
            ink="#eef1ee", ink2="#b7c0c8", ink3="#8e9aa5", drone="#4a90e2", go2="#35c4b5", guard="#c3bcff", camera="#7d98b3",
            intruder="#f08a3c", alarm="#ff5a6a", benign="#6f767d", focus="#8ab4ff", asset="#eef1ee")

def rgb(h): return np.array([int(h[i:i+2], 16) for i in (1, 3, 5)]) / 255
def lin(c): return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
def lum(h): return float(lin(rgb(h)) @ [0.2126, 0.7152, 0.0722])
def cr(a, b):
    la, lb = sorted((lum(a), lum(b)), reverse=True); return (la + 0.05) / (lb + 0.05)
# Machado et al. 2009, severity 1.0
CVD = {"protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
       "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
       "tritan": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]}
def sim(h, kind):
    if kind == "normal": return lin(rgb(h))
    return np.clip(np.array(CVD[kind]) @ lin(rgb(h)), 0, 1)
def lab(l):
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ l / np.array([0.9505, 1, 1.089])
    f = np.where(m > 0.008856, np.cbrt(m), 7.787 * m + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])
def de(a, b, kind): return float(np.linalg.norm(lab(sim(a, kind)) - lab(sim(b, kind))))

for name, P in (("LIGHT", LIGHT), ("DARK", DARK)):
    print(name)
    for fg in ("ink", "ink2", "ink3", "drone", "go2", "guard", "intruder", "alarm", "focus"):
        print(f"  {fg:9s} on surface {cr(P[fg], P['surface']):5.2f}  ground {cr(P[fg], P['ground']):5.2f}  surface2 {cr(P[fg], P['surface2']):5.2f}")
    for g in ("camera", "benign", "fence", "hair"):
        print(f"  {g:9s} non-text on surface {cr(P[g], P['surface']):5.2f} ground {cr(P[g], P['ground']):5.2f}")
    sem = ["drone", "go2", "guard", "camera", "intruder", "alarm", "benign"]
    for kind in ("normal", "protan", "deutan", "tritan"):
        pairs = sorted((de(P[a], P[b], kind), a, b) for i, a in enumerate(sem) for b in sem[i + 1:])
        print(f"  {kind:7s} min dE76: " + ", ".join(f"{a}/{b} {d:.0f}" for d, a, b in pairs[:3]))
