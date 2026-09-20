---
marp: true
theme: default
paginate: true
style: |
  section { background: #fcfcfb; color: #0b0b0b; font-family: ui-sans-serif, system-ui, sans-serif; padding: 48px 64px; }
  h1 { color: #0b0b0b; font-size: 1.7em; margin-bottom: 0.2em; }
  h2 { color: #52514e; font-weight: 500; font-size: 1.1em; }
  a, strong { color: #2a78d6; }
  small { color: #898781; font-size: 0.6em; }
  code { background: #f0efe9; color: #0b0b0b; }
  section::after { color: #898781; }
  img { display: block; margin: 0 auto; }
---

# Airtight

## How secure is this site, and what should you buy?

A security score and a vulnerability map for building owners, insurers and security firms, before any robot is purchased, and a re-score after.

---

# Site twin and mixed fleet

![height:480px](charts/vulnerability_map.png)

Drones, a ground robot and guards bid in one auction. Batteries and docks make coverage continuity real.

---

# The red team

Four tactic families: charging window, decoy, blind spot, comms cut.

An LLM proposes; search attacks. Every tactic passes one validator.

- **charging_window**: entry `rear_fence_gap`, phase 1.00, 1.8 m/s, origin random
- **decoy**: entry `main_gate`, phase 0.98, 1.7 m/s, origin random
- **blind_spot**: entry `rear_fence_gap`, phase 0.33, 2.0 m/s, origin search
- **comms_cut**: entry `service_gate`, phase 0.52, 2.0 m/s, origin search

---

# What the adversary found

Charging-window attack: enter `rear_fence_gap` at phase 1.00 of the charge cycle at 1.8 m/s.

*Replay clip A: the miss.*

Clip A, seed 63663: d2_go2_guard_sync never raised a timely alarm against `charging_window-439068210` (deadline 34 s).

---

# The score

![height:470px](charts/cost_vs_detection.png)

---

# The fix and the re-attack

![height:400px](charts/before_after.png)

d2_go2_guard_sync to d3_go2_guard_stagger: detection 0.14 to 0.63; against the re-attacking worst tactic 0.04 to 0.49.

*Replay clip B: the catch.*

Clip B, same seed and tactic: d3_go2_guard_stagger alarmed at 24 s, before the 34 s deadline.

---

# Human attention is a cost

Human decisions per hour: 0.10 to 0.21. Coverage gap: 0 to 0 s/h.

![height:300px](charts/token_cost.png)

Adversary cost per scored configuration: 100x cheaper than an LLM planning every episode (mean of 2 real calls).

---

# Attack, fix, re-attack

Each round the red team searches the current fleet, every affordable fix is scored against what it found, the best worst case wins, and the red team attacks again.

| Round | Fleet | $/h | Worst case after re-attack | Fix chosen |
|---|---|---|---|---|
| 0 | 2 drones, Go2, guard, synchronized | 55 | 0.05 | stagger |
| 1 | 2 drones, Go2, guard, staggered | 55 | 0.00 | add drone |
| 2 | 3 drones, Go2, guard, staggered | 62 | 0.05 | add drone |
| 3 | 4 drones, Go2, guard, staggered | 69 | 0.00 | stop |

The re-attack number stays near zero: with full knowledge of the patrol and no reaction from the fleet, the adversary finds a new hole after every fix. The score reports the typical intruder and the worst case side by side, and the loop is how a site finds the next hole before an intruder does.

---

# What we sell, and what is next

- The score, the vulnerability map, and a re-score after purchase
- Next: learned adversary, calibrated sensors on more platforms, fleet memory under link loss

<small>Operating point 1 false alarm/h · 200 seeds · adversary: open-loop adversary with full knowledge of the patrol policy and charge schedule; search plus LLM proposals · sensor: hand-written stub curve with a 360-degree drone disc · reduced-order per-look Bernoulli model, truth association, engine v0; false alarms from 20 quiet nights of one charge cycle each</small>
