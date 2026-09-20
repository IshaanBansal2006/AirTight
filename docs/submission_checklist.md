# Submission checklist (HackMIT, 2026-09-19 to 20)

What the public site says: every field of the project submission form must be completed, and a
partial submission (title plus a code link) is not accepted. The full rules, track list, judging
criteria and video limits are on the participant dashboard, which needs a login. **First task for
the lead at the next break: open the dashboard, paste the exact submission fields and the video
length limit into this file, and set the demo script to that length.**

Fields to have ready regardless of the form (all produced by scripts or already written):

- Project name and one-line description: from decision 001's framing.
- Longer description: the opening paragraph of `docs/plan.md` plus `docs/related_work.md`'s claim paragraph.
- Built with: dimOS, Python 3.12, MuJoCo, pydantic, numpy, matplotlib, the OpenAI API.
- Repository link: the repo goes public only at submission time, by an explicit instruction (parent rule §7).
- Video: `pitch/demo_script.md`, cut to the limit; clips from lane A; charts from `pitch/charts/`.
- Slides: `pitch/deck.md` rendered with Marp, watermark gone only once `report.json` is real.
- Tracks: decide which tracks fit once the list is known; the security-posture score fits a robotics or
  physical-AI track, the token-accounting chart fits an efficient-AI or agents track.
- Team members and roles: A, B, C as in the plan.

Public repo contents at submission: everything in `main`. The HELD documents (plan, lanes, decisions,
related work, this file) become public at that moment, which is the dated claim the disclosure rule
requires. Nothing else needs to change.
