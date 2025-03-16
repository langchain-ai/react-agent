"""State module for managing agent state."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class State:
    """Class representing the state of the agent interaction."""

    interrupt_response: str = "example"