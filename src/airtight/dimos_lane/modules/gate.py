"""Approval gate wrapped as a dimOS module. Silence never authorizes."""

from __future__ import annotations

import time
import uuid

from airtight.swarm.hol.gate import ApprovalGate
from airtight.swarm.schemas import EngagementProposal
from dimos.agents.annotation import skill
from dimos.core.module import Module


class Gate:
    def __init__(self, audit_path: str | None = None) -> None:
        from pathlib import Path

        self._gate = ApprovalGate(audit_path=Path(audit_path) if audit_path else None)

    def propose(
        self,
        action: str,
        rationale: str,
        *,
        deadline_s: float = 30.0,
        target_track_id: int | None = None,
        now: float | None = None,
    ) -> str:
        proposal_id = f"p-{uuid.uuid4().hex[:8]}"
        self._gate.submit(
            EngagementProposal(
                proposal_id=proposal_id,
                action=action,
                target_track_id=target_track_id,
                rationale=rationale,
                deadline_s=deadline_s,
            ),
            now if now is not None else time.time(),
        )
        return proposal_id

    def decide(
        self, proposal_id: str, approve: bool, operator: str, *, now: float | None = None
    ) -> bool:
        return self._gate.decide(proposal_id, approve, operator, now if now is not None else time.time())

    def pending_ids(self, *, now: float | None = None) -> list[str]:
        self._gate.tick(now if now is not None else time.time())
        return sorted(self._gate.pending)

    def audit(self) -> list[dict[str, object]]:
        return list(self._gate.audit)


class GateModule(Module):
    def __init__(self, config_args: dict[str, object] | None = None) -> None:
        super().__init__(dict(config_args or {}))
        self.inner = Gate()

    @skill
    def propose(self, action: str, rationale: str, deadline_s: float = 30.0) -> str:
        """Submit an engagement proposal. Does not authorize it."""
        return self.inner.propose(action, rationale, deadline_s=deadline_s)

    @skill
    def decide(self, proposal_id: str, approve: bool, operator: str = "operator") -> str:
        """Approve or deny a pending proposal. Unknown ids raise."""
        decided = self.inner.decide(proposal_id, approve, operator)
        return f"{proposal_id}: {'approved' if decided else 'denied'}"

    @skill
    def pending(self) -> str:
        """List pending proposal ids. Expired ones are auto-denied first."""
        ids = self.inner.pending_ids()
        return "none" if not ids else ",".join(ids)
