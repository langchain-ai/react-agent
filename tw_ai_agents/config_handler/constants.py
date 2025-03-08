"""Constants used throughout the TW AI Agents application."""

import re
from typing import Final, Literal

# Database constants
DB_CHECKPOINT_PATH: Final[str] = "checkpoints.sqlite"
WHITESPACE_RE = re.compile(r"\s+")
SUBAGENT_TOOL_NAME_PREFIX = f"transfer_to_"
SUBAGENT_TOOL_NAME_SUFFIX = "_agent"
OutputMode = Literal["full_history", "last_message"]
TW_SUPERVISOR_NAME = "tw_supervisor"
