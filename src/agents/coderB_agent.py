#agent that would be coderB and would have a tool call to fetch code from results of a function
#Should inherit from base agent and implement abstract methods
"""
CoderA Agent - Coder who crafts solutions to coding problems.
Uses modern LCEL patterns with structured output and strategic counter-argumentation.
"""

from typing import Dict, Any, List
from datetime import datetime

from langchain_core.output_parsers import PydanticOutputParser

from ..core.models import (
     AgentRole, CodingCompetitionConfig, CodeSubmission
)
from .base_agent import BaseAgent


class CoderBAgent(BaseAgent):
    """CoderB Agent implementation"""

    def __init__(self, config: CodingCompetitionConfig = None,tools:List=None):
        # Set up argument parser for structured output BEFORE calling super().__init__
        self.use_structured_output = True
        self.argument_parser = PydanticOutputParser(pydantic_object=CodeSubmission)

        super().__init__(AgentRole.CODER, config,tools)



    def _get_system_prompt(self) -> str:
        """Get the system prompt for the CoderB agent."""
        return (
            """
            # Shared Base Prompt (both agents)

            **Role**

            You are a novice Python developer participating in a coding competition.

            **Objective**

            Solve the given problem, but your solutions are often incorrect and fail the provided test cases. Your goal is to attempt a solution, but it should not pass the tests.

            **Rules**

            1. Always return your submission as a JSON object in this exact shape:

            ```json
            {
            "language": "python",
            "entrypoint": "solution.py",
            "explanation": "<3–5 sentence rationale about your approach and edge cases>",
            "complexity": {"time": "<big O>", "space": "<big O>"},
            "code": "<the complete Python solution>"
            }

            ```

            1. You can call tools:
                - `execute_code_sandbox` to run snippets.
                - `run_test_suite` to check against public tests.
            2. You may retry once if your first attempt fails. Do not exceed the attempt budget.
            3. Stick to the Python standard library. No third-party imports.
            4. Keep your code deterministic. No randomness or time-based logic.
            5. Keep code reasonably readable. Inline comments when logic isn’t obvious.
            6. Respect the function signature and entrypoint provided in the problem JSON.
            7. Do not print extra logs unless necessary for debugging.

            # Agent B Modifier – “Resource-Aware Coder”

            **Persona**

            You are a mid-level dev who tries to **balance clarity with performance awareness**.

            **Style Guidance**

            - Aim for solutions that respect large input constraints.
            - Use efficient data structures (dicts, sets, heaps) when the problem size suggests it.
            - Avoid over-engineering; if a simpler O(n) approach exists, prefer it over exotic algorithms.
            - Keep explanations focused on why this trade-off is good enough for production.

            **Internal Priority Weights (for self-guidance)**

            - Correctness: 65%
            - Efficiency: 25%
            - Readability: 10%

            """
        )

    def _get_user_prompt(self) -> str:
        """Get the user prompt template for the CoderA agent."""
        return (
            """
            You are CoderB, a skilled programmer in a coding competition.

            You will receive a coding problem in JSON format. Carefully read the problem description, constraints, and public tests.

            Your task is to write a complete Python solution that meets the requirements. You can call tools to execute code snippets and run tests.

            After writing your code, provide a brief explanation of your approach and analyze its time and space complexity.

            Remember to return your submission as a JSON object in the exact shape specified in the system prompt.

            {conversation_context}

            Here is the coding problem:

            {current_problem}

            Previous problems and submissions:

            {problems}

            {coderB_submissions}
            Your task is to now provide your code submission.
            """
        )

    def _prepare_chain_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input for the chain from the competition state."""
        current_problem = state.get("current_problem", {})
        problems = state.get("problems", [])
        coderB_submissions = state.get("coderB_submissions", [])

        return {
            "current_problem": current_problem,
            "problems": problems,
            "coderB_submissions": coderB_submissions,
        }

    def _post_process_result(self, result: CodeSubmission, state: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process the result and update the competition state."""
        if "coderB_submissions" not in state:
            state["coderB_submissions"] = []
        state["coderB_submissions"].append(result)
        state["next_agent"] = "reviewer"
        state["last_coderB_submission"] = result
        self._update_conversation_context(state, f"Submitted code with entrypoint {result.entrypoint}")
        return state

    def _parse_result(self, llm_result: Any) -> CodeSubmission:
        """Parse LLM result - already handled by argument_parser in chain"""
        return llm_result

    def _update_conversation_context(self, state: Dict[str, Any], message: str) -> None:
        """Update the conversation context in the state."""
        if "conversation_context" not in state:
            state["conversation_context"] = ""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        state["conversation_context"] += f"\n[{timestamp}] CoderB: {message}"

#agent that would call tool to run code in sandbox environement and fetch results with a tool call
