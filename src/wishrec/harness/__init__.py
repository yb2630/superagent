"""R1 ReAct harness for wish-driven code agents.

The harness turns a mechanically deleted benchmark specification (L3) into a single-task
workspace with four tools, a stakeholder and a 12-turn budget, and evaluates the submitted
solution against the full hidden suite.  Nine conditions cover B1/H0-H4/B2/B3 and DISCARD;
H5 is a plan item.  The generator is a local OpenAI-compatible vLLM endpoint.
"""

from .conditions import CONDITION_ORDER, CONDITIONS, Condition, get_condition
from .config import TURN_CAP, generator_config
from .models import Task, load_tasks

__all__ = [
    "CONDITIONS",
    "CONDITION_ORDER",
    "TURN_CAP",
    "Condition",
    "Task",
    "generator_config",
    "get_condition",
    "load_tasks",
]
