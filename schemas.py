from typing import Annotated, List, Literal, Optional, Sequence, TypedDict

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field, create_model


# ---------- Planner output ----------

class SubTask(BaseModel):
    id: str = Field(description="Short unique id, e.g. 't1', 't2'")
    agent: str = Field(description="Sub-agent that executes this task.")
    title: str = Field(description="Short human-readable action (e.g. 'Fetch unread messages')")
    instruction: str = Field(
        description="Self-contained instruction for the sub-agent, including every detail it needs."
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="Ids of tasks whose output this task needs. Empty if it can run immediately.",
    )


class Plan(BaseModel):
    needs_clarification: bool = Field(
        description="True if critical information is missing and cannot be obtained from the inbox."
    )
    question: Optional[str] = Field(
        default=None, description="Question for the user when needs_clarification is true."
    )
    tasks: List[SubTask] = Field(default_factory=list)


def plan_schema(agent_names: Sequence[str]) -> type[Plan]:
    """Plan schema whose SubTask.agent is restricted to the registered sub-agent names,
    so the LLM's structured output can only pick sub-agents that exist."""
    task_model = create_model(
        "SubTask",
        __base__=SubTask,
        agent=(Literal[tuple(agent_names)], Field(description="Sub-agent that executes this task.")),
    )
    return create_model("Plan", __base__=Plan, tasks=(List[task_model], Field(default_factory=list)))


# ---------- HITL ----------

class EmailDraft(BaseModel):
    recipient: str
    subject: str
    body: str


class ReviewDecision(TypedDict, total=False):
    """Value the human sends back through Command(resume=...)."""
    action: Literal["approve", "edit", "revise", "reject"]
    draft: dict      # edited draft, for action == "edit"
    feedback: str    # requested changes, for action == "revise"


# ---------- Graph state ----------

def merge_results(left: list, right: Optional[list]) -> list:
    """Append worker results; a None update resets the list for a new turn."""
    if right is None:
        return []
    return (left or []) + right


class OfficeState(MessagesState):
    signature: Optional[str]
    plan: List[dict]
    results: Annotated[List[dict], merge_results]


class WorkerState(TypedDict):
    task: dict
    request: str
    context: List[dict]
    signature: Optional[str]
