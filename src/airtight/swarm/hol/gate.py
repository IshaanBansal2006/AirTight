"""L4 — human-on-the-loop approval gate.

`ApprovalGate` is pure logic (unit-testable, no ROS): proposals enter PENDING,
a human decision moves them to APPROVED/DENIED, and expiry auto-DENIES —
fail-safe: silence never authorizes anything. Every transition is appended to
an audit trail (in memory + optional JSONL file: who, what, when, proposed vs
decided). `main()` wraps it as a ROS node + terminal console: proposals arrive
on /engagement_proposals, the operator types y/n, decisions publish on
/engagement_decisions.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from airtight.swarm.schemas import EngagementProposal

log = logging.getLogger(__name__)


@dataclass
class _Pending:
    proposal: EngagementProposal
    submitted_at: float


@dataclass
class ApprovalGate:
    audit_path: Path | None = None
    pending: dict[str, _Pending] = field(default_factory=dict)
    audit: list[dict] = field(default_factory=list)

    def _record(
        self, event: str, proposal: EngagementProposal, now: float, operator: str | None = None
    ) -> None:
        entry = {
            "event": event,
            "proposal_id": proposal.proposal_id,
            "action": proposal.action,
            "target_track_id": proposal.target_track_id,
            "rationale": proposal.rationale,
            "operator": operator,
            "time": now,
        }
        self.audit.append(entry)
        if self.audit_path is not None:  # append-only JSONL — the L4 audit log
            with self.audit_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")

    def submit(self, proposal: EngagementProposal, now: float) -> None:
        self.pending[proposal.proposal_id] = _Pending(proposal, now)
        self._record("proposed", proposal, now)

    def decide(self, proposal_id: str, approve: bool, operator: str, now: float) -> bool:
        """Returns the decision; unknown/expired ids raise (never silently pass)."""
        p = self.pending.pop(proposal_id, None)
        if p is None:
            raise KeyError(f"no pending proposal '{proposal_id}' — pending: {sorted(self.pending)}")
        self._record("approved" if approve else "denied", p.proposal, now, operator)
        return approve

    def tick(self, now: float) -> list[str]:
        """Auto-deny everything past its deadline. Fail-safe default: a human
        who says nothing has NOT approved."""
        expired = [
            pid for pid, p in self.pending.items() if now - p.submitted_at >= p.proposal.deadline_s
        ]
        for pid in expired:
            p = self.pending.pop(pid)
            self._record("auto_denied_timeout", p.proposal, now)
            log.warning("proposal %s auto-denied (timeout %.1fs)", pid, p.proposal.deadline_s)
        return expired


def main() -> None:  # pragma: no cover — interactive ROS wrapper
    import rclpy
    from std_msgs.msg import String

    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", default="l4_audit.jsonl")
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("l4_gate")
    gate = ApprovalGate(audit_path=Path(args.audit))
    pub = node.create_publisher(String, "/engagement_decisions", 10)

    def on_proposal(msg: String) -> None:
        p = EngagementProposal.model_validate_json(msg.data)
        gate.submit(p, time.time())
        print(
            f"\n[L4] PROPOSAL {p.proposal_id}: {p.action} on track "
            f"{p.target_track_id} — {p.rationale}\n"
            f"     approve? y/n (auto-deny in {p.deadline_s:.0f}s): ",
            end="",
            flush=True,
        )

    node.create_subscription(String, "/engagement_proposals", on_proposal, 10)
    print(
        f"[L4] gate up — audit -> {args.audit}. Waiting for engagement "
        f"proposals (this stays quiet until one arrives)...",
        flush=True,
    )

    def stdin_loop() -> None:
        while True:
            line = input().strip().lower()
            if not gate.pending:
                continue
            pid = sorted(gate.pending)[0]  # oldest pending
            approve = line.startswith("y")
            gate.decide(pid, approve, operator="console", now=time.time())
            pub.publish(String(data=json.dumps({"proposal_id": pid, "approved": approve})))
            print(f"[L4] {pid}: {'APPROVED' if approve else 'DENIED'}")

    threading.Thread(target=stdin_loop, daemon=True).start()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
            for pid in gate.tick(time.time()):
                pub.publish(String(data=json.dumps({"proposal_id": pid, "approved": False})))
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
