
from pydantic import BaseModel, Field
from typing import List, Literal, Any, Optional


class ArgumentPair(BaseModel):
    key: str = Field(description="Parameter name, e.g. 'recipient', 'max_results', or 'question'")
    value: str = Field(description="Parameter value as a string")

class Task(BaseModel):
    step_id: int
    title: str = Field(description="Short human-readable action (e.g., 'Fetch unread messages')")
    tool_name: Literal["check_inbox", "send_email", "ask_user", "final_response", "generate_email"]
    arguments: List[ArgumentPair] = Field(
        default_factory=list,
        description="List of key-value parameter pairs for the tool."
    )

class PlanResponse(BaseModel):
    needs_clarification: bool = Field(
        description="Set to true if critical parameters are missing and an ask_user step is required."
    )
    plan: List[Task]

class EmailStructure(BaseModel):
    recipient: str = Field(description="Target email address")
    subject: str = Field(description="Concise email subject line")
    body: str = Field(description="Complete, well-formatted body content")

class VerificationResult(BaseModel):
    is_valid: bool = Field(
        description="True if all parameters are real, supplied by the user, and safe to execute."
    )
    issue_type: Literal["none", "missing_information", "hallucinated_data", "unsafe_action"] = Field(
        description="Category of the flaw if is_valid is False."
    )
    question_for_user: Optional[str] = Field(
        default=None,
        description="If information is missing or fake (like a placeholder email), phrase the question to ask the user."
    )
    critique: str = Field(
        description="Detailed explanation of why the plan passed or failed verification."
    )