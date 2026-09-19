"""Minimal behavior-tree executor.

The deliberately small node set from the 020 scope guard: Sequence, Fallback,
and Action leaves. Trees are TICKED — each tick propagates down and returns
SUCCESS / FAILURE / RUNNING. RUNNING is what makes BTs reactive: the tree
yields control every tick, so a higher layer can re-tick, preempt, or rebuild
it as the world changes (vs. a plan executed open-loop to completion).

Composition semantics (the whiteboard core):
  Sequence = "and then": ticks children in order; fails fast; RUNNING sticks to
             the current child (memory sequence).
  Fallback = "else try":  succeeds fast; tries the next child on failure —
             this is where recovery behaviors hang.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class Status(enum.Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


class Node:
    def tick(self, ctx: dict) -> Status:  # ctx: shared blackboard for one tree
        raise NotImplementedError

    def reset(self) -> None:
        """Clear RUNNING memory (called when a tree is restarted/preempted)."""


class Action(Node):
    """Leaf: wraps a callable(ctx) -> Status. All world interaction lives here."""

    def __init__(self, name: str, fn: Callable[[dict], Status]) -> None:
        self.name = name
        self.fn = fn

    def tick(self, ctx: dict) -> Status:
        return self.fn(ctx)


class Sequence(Node):
    def __init__(self, *children: Node) -> None:
        self.children = list(children)
        self._current = 0

    def tick(self, ctx: dict) -> Status:
        while self._current < len(self.children):
            status = self.children[self._current].tick(ctx)
            if status == Status.RUNNING:
                return Status.RUNNING
            if status == Status.FAILURE:
                self.reset()
                return Status.FAILURE
            self._current += 1  # SUCCESS -> advance
        self.reset()
        return Status.SUCCESS

    def reset(self) -> None:
        self._current = 0
        for c in self.children:
            c.reset()


class Fallback(Node):
    def __init__(self, *children: Node) -> None:
        self.children = list(children)
        self._current = 0

    def tick(self, ctx: dict) -> Status:
        while self._current < len(self.children):
            status = self.children[self._current].tick(ctx)
            if status == Status.RUNNING:
                return Status.RUNNING
            if status == Status.SUCCESS:
                self.reset()
                return Status.SUCCESS
            self._current += 1  # FAILURE -> try next
        self.reset()
        return Status.FAILURE

    def reset(self) -> None:
        self._current = 0
        for c in self.children:
            c.reset()
