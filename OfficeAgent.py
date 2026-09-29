import json
import os
import uuid
from typing import Iterator, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Interrupt, Send

from schemas import OfficeState, Plan, ReviewDecision, WorkerState, plan_schema
from SubAgents import SUB_AGENTS, build_sub_agents, build_task_message, describe_sub_agents

load_dotenv()

PLANNER_PROMPT = """
You are the planner of an office assistant. Break the user's latest request into sub-tasks
executed by these sub-agents:

{sub_agents}

User State: {signature_state}

RULES:
1. Each task's `instruction` must be self-contained; sub-agents do not see the conversation.
2. If a task needs another task's output (e.g. replying to an inbox email), list that task in `depends_on`.
   Independent tasks must have no dependencies so they run in parallel.
3. One generate_agent task per email to write, and one explain_agent task per email to explain.
   To reply to an email the user asked you to explain, make the generate_agent task depend on the explain_agent task.
4. Set `needs_clarification` and ask a `question` ONLY if a recipient or the email's purpose is missing
   and cannot be found by reading the inbox. Never invent email addresses (e.g. '@example.com').
5. If the message needs no email action (greeting, question about the conversation), return no tasks.
"""

SYNTHESIZER_PROMPT = """
You are an office assistant. Using the results of the sub-agents, give the user a brief, natural
response confirming what was accomplished. Mention clearly any email that was cancelled or failed.
When the user asked to read or explain emails, present that content in full (well formatted);
do not shorten an explanation into a one-line confirmation.
If there are no results, simply answer the user's message.
"""


def last_user_message(state: OfficeState) -> str:
    return next((m.text for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")


class OfficeAssistant:
    def __init__(self, model_name: Optional[str] = None):
        self.model = ChatGoogleGenerativeAI(
            model=model_name or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.2,
        )
        self.planner = self.model.with_structured_output(plan_schema(list(SUB_AGENTS)))
        self.sub_agents = build_sub_agents(self.model)
        self.graph = self._build_graph()

    # ---------- Nodes ----------

    def plan(self, state: OfficeState) -> dict:
        signature = state.get("signature")
        signature_state = (
            f"User signature is known: '{signature}'. Do NOT ask for it."
            if signature
            else "Signature is unknown; emails may use a professional closing without a name."
        )
        plan: Plan = self.planner.invoke(
            [SystemMessage(PLANNER_PROMPT.format(sub_agents=describe_sub_agents(), signature_state=signature_state)), *state["messages"]]
        )

        # Drop dependencies on unknown task ids so the scheduler cannot deadlock on them.
        ids = {t.id for t in plan.tasks}
        tasks = [
            {**t.model_dump(), "depends_on": [d for d in t.depends_on if d in ids and d != t.id]}
            for t in plan.tasks
        ]

        update = {"plan": tasks, "results": None}
        if plan.needs_clarification:
            update["plan"] = []
            update["messages"] = [AIMessage(plan.question or "Could you clarify the missing details?")]
        return update

    def route_after_plan(self, state: OfficeState) -> str:
        # The planner only adds a message when it asks a clarifying question. That ends the turn,
        # and the user's answer re-enters the planner with the full conversation.
        asked_question = isinstance(state["messages"][-1], AIMessage)
        return END if asked_question else "scheduler"

    def scheduler(self, state: OfficeState) -> dict:
        return {}

    def assign_workers(self, state: OfficeState):
        """Fan out every task whose dependencies are done, one Send per sub-agent task."""
        results = state.get("results") or []
        done = {r["task_id"] for r in results}
        pending = [t for t in state["plan"] if t["id"] not in done]
        ready = [t for t in pending if all(d in done for d in t["depends_on"])]
        if not ready:
            return "synthesizer"

        request = last_user_message(state)
        return [
            Send(
                t["agent"],
                {
                    "task": t,
                    "request": request,
                    "context": [r for r in results if r["task_id"] in t["depends_on"]],
                    "signature": state.get("signature"),
                },
            )
            for t in ready
        ]

    def make_worker(self, agent_name: str):
        agent = self.sub_agents[agent_name]

        def run(state: WorkerState) -> dict:
            task = state["task"]
            message = build_task_message(task, state["request"], state["context"], state.get("signature"))
            try:
                output = agent.invoke({"messages": [HumanMessage(message)]})["messages"][-1].text
            except GraphBubbleUp:
                raise  # interrupt() for human review must propagate to the parent graph
            except Exception as e:
                output = f"FAILED: {e}"
            return {"results": [{"task_id": task["id"], "agent": agent_name, "title": task["title"], "output": output}]}

        return run

    def synthesize(self, state: OfficeState) -> dict:
        prompt = (
            f"User request: {last_user_message(state)}\n\n"
            f"Sub-agent results:\n{json.dumps(state.get('results') or [], indent=2, default=str)}"
        )
        response = self.model.invoke([SystemMessage(SYNTHESIZER_PROMPT), HumanMessage(prompt)])
        return {"messages": [AIMessage(response.text)]}

    # ---------- Graph ----------

    def _build_graph(self):
        builder = StateGraph(OfficeState)
        builder.add_node("planner", self.plan)
        builder.add_node("scheduler", self.scheduler)
        builder.add_node("synthesizer", self.synthesize)
        for name in self.sub_agents:
            builder.add_node(name, self.make_worker(name))
            builder.add_edge(name, "scheduler")

        builder.add_edge(START, "planner")
        builder.add_conditional_edges("planner", self.route_after_plan, ["scheduler", END])
        builder.add_conditional_edges("scheduler", self.assign_workers, [*self.sub_agents, "synthesizer"])
        builder.add_edge("synthesizer", END)

        # The checkpointer persists conversation memory and lets interrupt() pause/resume the run.
        return builder.compile(checkpointer=InMemorySaver())

    # ---------- Public API ----------

    @staticmethod
    def config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def _stream(self, inputs, thread_id: str) -> Iterator[dict]:
        for update in self.graph.stream(inputs, self.config(thread_id), stream_mode="updates"):
            # On resume, writes of tasks that already finished are replayed as cached; skip them.
            if not update.get("__metadata__", {}).get("cached"):
                yield update

    def chat(self, thread_id: str, user_input: str, signature: Optional[str] = None) -> Iterator[dict]:
        """Run a new user turn, yielding node updates as they happen."""
        yield from self._stream({"messages": [HumanMessage(user_input)], "signature": signature}, thread_id)

    def resume(self, thread_id: str, decisions: dict[str, ReviewDecision]) -> Iterator[dict]:
        """Resume paused email reviews; `decisions` maps interrupt id -> ReviewDecision."""
        yield from self._stream(Command(resume=decisions), thread_id)

    def pending_reviews(self, thread_id: str) -> List[Interrupt]:
        # Read interrupts from unfinished tasks only: a parallel sibling that was already
        # resumed and finished keeps its stale interrupt in the snapshot until the step completes.
        tasks = self.graph.get_state(self.config(thread_id)).tasks
        return [i for t in tasks if t.result is None for i in t.interrupts]

    def last_reply(self, thread_id: str) -> str:
        messages = self.graph.get_state(self.config(thread_id)).values.get("messages", [])
        return messages[-1].text if messages and isinstance(messages[-1], AIMessage) else ""


def ask_review(draft: dict) -> ReviewDecision:
    print("\n" + "=" * 50 + "\nEmail awaiting your approval")
    print(f"To: {draft['recipient']}\nSubject: {draft['subject']}\n\n{draft['body']}\n" + "=" * 50)
    choice = input("[a]pprove / [e]dit / [r]equest changes / [c]ancel: ").strip().lower()
    if choice.startswith("a"):
        return {"action": "approve"}
    if choice.startswith("e"):
        edits = {k: input(f"{k} (blank = keep): ").strip() for k in ("recipient", "subject", "body")}
        return {"action": "edit", "draft": {k: v for k, v in edits.items() if v}}
    if choice.startswith("r"):
        return {"action": "revise", "feedback": input("What should change? ").strip()}
    return {"action": "reject"}


def print_updates(updates: Iterator[dict]):
    for update in updates:
        for node, value in update.items():
            if node == "planner" and value.get("plan"):
                print("Plan:")
                for t in value["plan"]:
                    deps = f" (after {', '.join(t['depends_on'])})" if t["depends_on"] else ""
                    print(f"  {t['id']} [{t['agent']}] {t['title']}{deps}")
            elif node in SUB_AGENTS:
                for r in value["results"]:
                    print(f"Done {r['task_id']}: {r['title']}")


if __name__ == "__main__":
    assistant = OfficeAssistant()
    thread_id = str(uuid.uuid4())
    print("Office Assistant ready. Type 'exit' to quit.\n" + "-" * 50)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        print_updates(assistant.chat(thread_id, user_input))
        while pending := assistant.pending_reviews(thread_id):
            decisions = {i.id: ask_review(i.value["draft"]) for i in pending}
            print_updates(assistant.resume(thread_id, decisions))

        print(f"\nAgent: {assistant.last_reply(thread_id)}")
