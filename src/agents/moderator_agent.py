"""
Moderator Agent - Controls the competition flow and fetches coding problems.
Uses modern LCEL patterns with structured output.
"""

from typing import Dict, Any, List
from datetime import datetime

from langchain_core.output_parsers import PydanticOutputParser

from ..core.models import (
    AgentRole, CodingCompetitionConfig, CodingProblem
)
from .base_agent import BaseAgent


class ModeratorAgent(BaseAgent):
    """Moderator Agent implementation - manages competition flow and problem selection"""

    def __init__(self, config: CodingCompetitionConfig = None, tools: List = None):
        # Set up argument parser for structured output BEFORE calling super().__init__
        self.argument_parser = PydanticOutputParser(pydantic_object=CodingProblem)

        super().__init__(AgentRole.ROUND_CONTROLLER, config, tools)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Moderator agent."""
        return (
            """
            # Role

            You are the Moderator of a coding competition between two AI coders: CoderA and CoderB.

            # Objective

            Your job is to:
            1. Select or generate coding problems appropriate for the competition
            2. Ensure problems are fair, well-defined, and testable
            3. Manage the flow of the competition rounds
            4. Provide clear problem statements with constraints and test cases

            # Problem Selection Guidelines

            - Choose problems that allow for different approaches and trade-offs
            - Ensure the problem has clear input/output specifications
            - Provide at least 2-3 public test cases
            - Set reasonable time and space complexity expectations
            - Consider problems that highlight different coding styles (clarity vs efficiency)

            # Output Format

            You must return a coding problem as a JSON object with this exact structure:

            ```json
            {
                "id": "<unique problem identifier>",
                "title": "<problem title>",
                "entrypoint": "<function name>",
                "description": "<detailed problem description including examples>",
                "constraints": {
                    "time_limit": "<time limit in seconds>",
                    "space_limit": "<memory limit>",
                    "input_constraints": "<input size/type constraints>"
                },
                "public_tests": [
                    {
                        "name": "<test name>",
                        "input": "<test input>",
                        "expected_output": "<expected output>"
                    }
                ],
                "budgets": {
                    "max_attempts": 2,
                    "time_per_attempt": 300
                }
            }
            ```

            # Competition Flow

            - Track the current round number
            - Ensure both coders get the same problem
            - Move the competition forward after each round
            - Maintain fairness and consistency
            """
        )

    def _get_user_prompt(self) -> str:
        """Get the user prompt template for the Moderator agent."""
        return (
            """
            You are the Moderator of a coding competition.

            Current competition state:
            - Round: {current_round} of {max_rounds}
            - Competition Status: {competition_status}

            {conversation_context}

            Previous problems used in this competition:
            {problems}

            Your task is to create the next coding problem for this round.

            The problem should:
            1. Be different from previous problems
            2. Be appropriate for mid-level developers
            3. Allow for different coding approaches
            4. Have clear, testable requirements

            IMPORTANT: You MUST return ONLY valid JSON in the exact format specified in the system prompt.
            Do not include any explanatory text, markdown, or code blocks - just the raw JSON object.
            Start your response with {{ and end with }}

            """
        )

    def _prepare_chain_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input for the chain from the competition state."""
        current_round = state.get("current_round", 1)
        max_rounds = state.get("max_rounds", 3)
        problems = state.get("problems", [])
        competition_status = state.get("competition_status", "preparing")

        return {
            "current_round": current_round,
            "max_rounds": max_rounds,
            "problems": problems,
            "competition_status": competition_status,
        }

    def _post_process_result(self, result: CodingProblem, state: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process the result and update the competition state."""
        # Initialize problems list if it doesn't exist
        if "problems" not in state:
            state["problems"] = []

        # Add the new problem to the problems list
        state["problems"].append(result)

        # Set as current problem
        state["current_problem"] = result

        # Update competition flow
        if "current_round" not in state:
            state["current_round"] = 1

        # Set next agent to coderA (first coder to submit)
        state["next_agent"] = "coderA"

        # Update status to active
        state["competition_status"] = "active"

        # Update conversation context
        self._update_conversation_context(
            state,
            f"Selected problem '{result.title}' for round {state.get('current_round', 1)}"
        )

        return state

    def _parse_result(self, llm_result: Any) -> CodingProblem:
        """Parse the LLM result into a CodingProblem object."""
        import re

        # Extract content if it's an AIMessage
        if hasattr(llm_result, 'content'):
            content = llm_result.content
        else:
            content = str(llm_result)

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)

        # Try to find JSON object in the content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)

        try:
            return self.argument_parser.parse(content)
        except Exception as e:
            print(f"\n⚠️  Failed to parse moderator output:")
            print(f"   Error: {str(e)}")
            print(f"   Content: {content[:500]}...")
            raise

    def _update_conversation_context(self, state: Dict[str, Any], message: str) -> None:
        """Update the conversation context in the state."""
        if "conversation_context" not in state:
            state["conversation_context"] = ""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        state["conversation_context"] += f"\n[{timestamp}] Moderator: {message}"
