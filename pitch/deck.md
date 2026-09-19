---
marp: true
theme: default
paginate: true
---

# Airtight

## How secure is this site, and what should you buy?

A security score and a vulnerability map for building owners, insurers and security firms, before any robot is purchased, and a re-score after.

**EXAMPLE DATA, NOT A RESULT**

---

# Site twin and mixed fleet

![height:480px](charts/vulnerability_map.png)

Drones, a ground robot and guards bid in one auction. Batteries and docks make coverage continuity real.

**EXAMPLE DATA, NOT A RESULT**

---

# The red team

Four tactic families: charging window, decoy, blind spot, comms cut.

An LLM proposes; search attacks. Every tactic passes one validator.

- **charging_window**: entry `loading_dock`, phase 0.27, 0.9 m/s, origin random
- **decoy**: entry `east_fence`, phase 0.56, 2.2 m/s, origin random
- **blind_spot**: entry `loading_dock`, phase 0.40, 1.8 m/s, origin random
- **comms_cut**: entry `east_fence`, phase 0.86, 1.3 m/s, origin random

**EXAMPLE DATA, NOT A RESULT**

---

# What the adversary found

Charging-window attack: enter `loading_dock` at phase 0.27 of the charge cycle at 0.9 m/s.

*Replay clip A: the miss.*

**EXAMPLE DATA, NOT A RESULT**

---

# The score

![height:440px](charts/cost_vs_detection.png)

<small>Operating point 1 false alarm/h · 2 seeds · adversary: full knowledge of patrol policy and charge schedule · sensor: hand-written stub curve · reduced-order Bernoulli-per-look model, truth association</small>

**EXAMPLE DATA, NOT A RESULT**

---

# The fix and the re-attack

![height:400px](charts/before_after.png)

2drone_go2_sync to 3drone_go2_stagger: detection 0.70 to 0.90; against the re-attacking worst tactic 0.40 to 0.78.

*Replay clip B: the catch.*

**EXAMPLE DATA, NOT A RESULT**

---

# Human attention is a cost

Human decisions per hour: 2.50 to 2.80. Coverage gap: 410 to 60 s/h.

![height:300px](charts/token_cost.png)

Adversary cost per scored configuration: 100x cheaper than an LLM planning every episode (mean of 1 real calls).

**EXAMPLE DATA, NOT A RESULT**

---

# What we sell, and what is next

- The score, the vulnerability map, and a re-score after purchase
- Next: learned adversary, calibrated sensors on more platforms, fleet memory under link loss

<small>Operating point 1 false alarm/h · 2 seeds · adversary: full knowledge of patrol policy and charge schedule · sensor: hand-written stub curve · reduced-order Bernoulli-per-look model, truth association</small>

**EXAMPLE DATA, NOT A RESULT**
