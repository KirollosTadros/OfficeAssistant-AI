import os
import json
from typing import List, Literal, Any
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from EmailManager import EmailManager
from PlanVerifierAgent import PlanVerifierAgent
from schemas import *

class InteractiveOfficeAssistant:
    def __init__(self, model_name: str = "gemini-3.5-flash-lite"):
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.email_manager = EmailManager()
        self.verifier = PlanVerifierAgent(self.client, self.model_name)
        self.user_profile = {"signature_name": None}

        self.tool_map = {
            "check_inbox": self.email_manager.check_inbox,
            "send_email": self.email_manager.send_email,
            "generate_email": self.generate_email,
        }

    def generate_email(self, context: str, recipient: str):
        
        sig_text = f"Sign the email with: {self.user_profile['signature_name']}" if self.user_profile.get("signature_name") else "Use a professional closing and placeholder signature."
        prompt = f"""
        Compose a clean, professional email based on this request.
        signature: {sig_text}
        Recipient: {recipient}
        Context & Goal: {context}

        CRITICAL FORMATTING INSTRUCTIONS:
        1. Maintain distinct paragraphs separated by blank lines (double newline '\\n\\n').
        2. Follow this structure:
           - Salutation (e.g., 'Dear [Name],' or 'Hi [Name],') followed by a blank line
           - Opening sentence/context paragraph
           - Core details/body paragraph (or bullet points with linebreaks)
           - Call to action or concluding sentence followed by a blank line
           - Sign-off (e.g., 'Best regards,')
           - Sender Name / Title
        3. Do NOT compress the email into a single paragraph.
        """

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=EmailStructure,
            ),
        )
        validated_email = EmailStructure.model_validate_json(response.text)
    
        return validated_email.model_dump()

    def get_args(self, arg_pairs: List[ArgumentPair]) -> dict:
        """Converts args to dictionary."""
        result = {}
        for pair in arg_pairs:
            val = pair.value

            if val.lower() == "true":
                result[pair.key] = True
            elif val.lower() == "false":
                result[pair.key] = False
            elif val.isdigit():
                result[pair.key] = int(val)
            else:
                result[pair.key] = val
        return result

    def generate_plan(self, conversation_history: List[str]) -> PlanResponse:

        known_sig = f"User signature is already known: '{self.user_profile['signature_name']}'. Do NOT ask for signature or name." if self.user_profile.get("signature_name") else "Signature is unknown."

        system_instruction = f"""
        You are an office assistant task planner.
        User State: {known_sig}
        Break down user requests into discrete, sequential steps using these tools:
        - check_inbox(max_results: int, unread_only: bool, primary: bool)
        - send_email(recipient: str, subject: str, body: str)
        - generate_email(context: str, recipient)
        - ask_user(question: str)
        - final_response(summary: str)

        RULES:
        1. If critical parameters (like recipient or core email body) are missing and cannot 
           be obtained by reading the inbox first, set `needs_clarification: true` and make step 1 `ask_user`.
        2. Provide arguments strictly as a list of key-value pairs.
        3. Make the title of each step concise and descriptive.
        4. Make sure to breakdown email generation from email sending
        """

        prompt = "\n".join(conversation_history)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=PlanResponse,
            ),
        )
        return PlanResponse.model_validate_json(response.text)

    def resolve_args(self, task: Task, raw_args: dict, context: List[dict]) -> dict:
        prompt = f"""
        Action: {task.title}
        Target Tool: {task.tool_name}
        Initial planned args: {raw_args}
        Context gathered from previous steps:
        {json.dumps(context, default=str)}

        Determine the final parameters for this tool call.
        Return ONLY valid JSON with keys matching the tool's signature:
        - For 'send_email': {{"recipient": "...", "subject": "...", "body": "..."}}
        """
        resolver = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return json.loads(resolver.text)

    def run_interactive_session(self):
        conversation: List[str] = []
        print("Office Assistant ready. Type 'exit' to quit.\n" + "-" * 50)

        while True:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                break

            conversation.append(f"User: {user_input}")

            # 1. Plan Phase
            plan_obj = self.generate_plan(conversation)

            print("Proposed Plan:")
            for task in plan_obj.plan:
                print(f"  Step {task.step_id}: {task.title}")
            print("-" * 50)

            #Run Verification Agent
            print("Verifying plan safety and parameters...")
            verification = self.verifier.verify_plan(conversation, 
                                                    plan_obj, 
                                                    known_signature=self.user_profile.get("signature_name"))

            if not verification.is_valid:
                print(f"Verification Failed: {verification.critique}")
                clarification_question = verification.question_for_user or "Could you clarify the missing details?"
                print(f"\Agent: {clarification_question}")
                conversation.append(f"Agent: {clarification_question}")
                continue

            print("Plan verified. Proceeding to execution.")

            #clarification
            if plan_obj.needs_clarification:
                first_step = plan_obj.plan[0]
                args_dict = self.get_args(first_step.arguments)
                question = args_dict.get("question", "Could you please provide the missing details?")
                print(f"\Agent: {question}")
                conversation.append(f"Agent: {question}")
                continue

            execution_context = []
            for task in plan_obj.plan:
                if task.tool_name in ["final_response", "ask_user"]:
                    continue

                print(f"⚙️ Running Step {task.step_id}: {task.title}...")

                try:
                    func = self.tool_map[task.tool_name]
                    args = self.get_args(task.arguments)

                    # Dynamic payload resolution for dependent steps
                    if execution_context and task.tool_name == "send_email":
                        args = self.resolve_args(task, args, execution_context)

                    result = func(**args)
                    execution_context.append({
                        "step_id": task.step_id,
                        "tool": task.tool_name,
                        "output": result
                    })
                except Exception as e:
                    print(f"Execution failed at step {task.step_id}: {e}")
                    break

            synthesis_prompt = f"""
            User Query: {user_input}
            Execution Results: {json.dumps(execution_context, default=str)}
            
            Give a brief, natural response confirming what was accomplished.
            """
            summary = self.client.models.generate_content(
                model=self.model_name,
                contents=synthesis_prompt
            )
            print(f"\nAgent: {summary.text}")
            conversation.append(f"Agent: {summary.text}")


if __name__ == "__main__":
    assistant = InteractiveOfficeAssistant()
    assistant.run_interactive_session()