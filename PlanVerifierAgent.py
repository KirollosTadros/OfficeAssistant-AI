from google import genai
from google.genai import types
from typing import Literal, Optional
from pydantic import BaseModel, Field
from schemas import *

class PlanVerifierAgent:
    def __init__(self, client: genai.Client, model_name: str = "gemini-3.5-flash"):
        self.client = client
        self.model_name = model_name

    def verify_plan(self, user_request: str, proposed_plan: PlanResponse, known_signature: Optional[str] = None) -> VerificationResult:

        sig_rule = f"Known user signature: '{known_signature}'. NEVER fail verification for missing signature/name." if known_signature else "If signature name is missing, accept standard closing placeholders or ask user."

        verifier_instruction = f"""
        You are an adversarial quality-assurance critic in a plan-and-execute system.
        {sig_rule}
        Audit the proposed execution plan against the conversation history.

        REPLY WORKFLOW RULE:
        - If the user asks to reply to an email or check inbox and respond:
          It is completely VALID and EXPECTED for `recipient` or details to be populated dynamically AT RUNTIME from the `check_inbox` step.
          Do NOT fail verification for missing recipient if a `check_inbox` step precedes `generate_email`/`send_email`.

        GENERAL RULES:
        1. Reject only if the planner invented synthetic/fake emails (like '@example.com') out of nowhere.
        2. Reject if the user directly requested a brand new email to an unknown person without checking inbox first.
        """


        prompt = f"""
        User Request and history:
        {user_request}

        Proposed Plan to Audit:
        {proposed_plan.model_dump_json(indent=2)}
        """

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=verifier_instruction,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=VerificationResult,
            ),
        )
        return VerificationResult.model_validate_json(response.text)