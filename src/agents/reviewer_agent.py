"""
Reviewer Agent - Reviews and evaluates code submissions from both coders.
Uses modern LCEL patterns with structured output and tool calling for testing.
"""

from typing import Dict, Any, List
from datetime import datetime

from langchain_core.output_parsers import PydanticOutputParser

from ..core.models import (
    AgentRole, CodingCompetitionConfig, CodeReview
)
from .base_agent import BaseAgent


class ReviewerAgent(BaseAgent):
    """Reviewer Agent implementation - evaluates code submissions and determines round winners"""

    def __init__(self, config: CodingCompetitionConfig = None, tools: List = None):
        # Set up argument parser for structured output BEFORE calling super().__init__
        self.argument_parser = PydanticOutputParser(pydantic_object=CodeReview)
        # self.use_structured_output = True  # Flag to use .with_structured_output()

        super().__init__(AgentRole.REVIEWER, config, tools)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Reviewer agent."""
        return (
            """
            # Role

            You are the Reviewer in a coding competition between CoderA and CoderB.

            # Objective

            Your job is to:
            1. Run and test both codes in a sandbox environment using e2b.
            2. Evaluate each submission on three criteria:
               - Correctness (0-5 points): Does it solve the problem correctly?
               - Efficiency (0-3 points): Is it optimized for time/space complexity?
               - Code Quality (0-2 points): Is it readable, maintainable, and well-structured?
            3. Determine the winner of the round
            4. Provide constructive feedback to both coders

            # Evaluation Guidelines

            **Correctness (0-5 points)**
            - 5: Passes all tests, handles all edge cases
            - 4: Passes all public tests, likely handles most edge cases
            - 3: Passes most tests, has minor issues
            - 2: Passes some tests, has significant bugs
            - 1: Barely functional, fails most tests
            - 0: Does not work or crashes

            **Efficiency (0-3 points)**
            - 3: Optimal time/space complexity for the problem
            - 2: Good complexity, minor inefficiencies
            - 1: Works but inefficient approach
            - 0: Very inefficient or doesn't meet complexity requirements

            **Code Quality (0-2 points)**
            - 2: Clean, readable, well-documented, follows best practices
            - 1: Functional but could be clearer or better structured
            - 0: Messy, hard to read, poor structure

            # Tools Available

            - `execute_code_sandbox`: Run code with specific inputs to see output
            - `run_test_suite`: Run all test cases at once and get comprehensive results

            Use these tools to thoroughly test both submissions before scoring.

            # Output Format

            You must return a review as a JSON object with this exact structure:

            ```json
            {
                "coderA_correctness_score": <0-5>,
                "coderA_efficiency_score": <0-3>,
                "coderA_quality_score": <0-2>,
                "coderA_feedback": "<detailed feedback for CoderA>",

                "coderB_correctness_score": <0-5>,
                "coderB_efficiency_score": <0-3>,
                "coderB_quality_score": <0-2>,
                "coderB_feedback": "<detailed feedback for CoderB>",

                "round_winner": "<coderA|coderB|draw>",
                "summary": "<overall comparison and round summary>"
            }
            ```

            # Guidelines

            - Be fair and objective in your evaluation
            - Base scores on actual test results, not just code inspection
            - Provide specific, actionable feedback
            - Explain your reasoning in the summary
            - A draw is acceptable if scores are very close
            """
        )

    def _get_user_prompt(self) -> str:
        """Get the user prompt template for the Reviewer agent."""
        return (
            """
            You are the Reviewer evaluating Round {current_round} of the coding competition.

            {conversation_context}

            **Problem:**
            {current_problem}

            **CoderA's Submission:**
            {coderA_submission}

            **CoderB's Submission:**
            {coderB_submission}

            **Scoring Rules:**
            - Correctness: {correctness_weight} points max
            - Efficiency: {efficiency_weight} points max
            - Code Quality: {quality_weight} points max

            **Current Scores:**
            - CoderA: {coderA_score}
            - CoderB: {coderB_score}

            Your task:
            1. Use the `execute_code_sandbox` tool to execute the code submitted by both coders with specific inputs.
            2. Use the `run_test_suite` tool to run all provided test cases for
            3. Evaluate both submissions on correctness, efficiency, and code quality
            4. Determine the round winner based on total scores
            5. Provide detailed, constructive feedback

            Remember to be thorough, fair, and use the tools to validate your evaluation.
            """
        )

    def _prepare_chain_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input for the chain from the competition state."""
        current_problem = state.get("current_problem", {})
        current_round = state.get("current_round", 1)

        # Get latest submissions
        coderA_submissions = state.get("coderA_submissions", [])
        coderB_submissions = state.get("coderB_submissions", [])

        coderA_submission = coderA_submissions[-1] if coderA_submissions else None
        coderB_submission = coderB_submissions[-1] if coderB_submissions else None

        # Get current scores
        coderA_score = state.get("coderA_Score", 0)
        coderB_score = state.get("coderB_Score", 0)

        # Get scoring rules from config
        scoring_rules = self.config.scoring_rules

        return {
            "current_problem": current_problem,
            "current_round": current_round,
            "coderA_submission": coderA_submission,
            "coderB_submission": coderB_submission,
            "coderA_score": coderA_score,
            "coderB_score": coderB_score,
            "correctness_weight": scoring_rules.get("correctness", 5),
            "efficiency_weight": scoring_rules.get("efficiency", 3),
            "quality_weight": scoring_rules.get("code_quality", 2),
        }

    def _post_process_result(self, result: CodeReview, state: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process the result and update the competition state."""


        # Calculate total scores for this round
        coderA_round_score = (
            result.coderA_correctness_score +
            result.coderA_efficiency_score +
            result.coderA_quality_score
        )

        coderB_round_score = (
            result.coderB_correctness_score +
            result.coderB_efficiency_score +
            result.coderB_quality_score
        )

        # Update cumulative scores
        if "coderA_Score" not in state:
            state["coderA_Score"] = 0
        if "coderB_Score" not in state:
            state["coderB_Score"] = 0

        state["coderA_Score"] += coderA_round_score
        state["coderB_Score"] += coderB_round_score

        # Add review comments
        if "reviewer_comments" not in state:
            state["reviewer_comments"] = []

        review_comment = (
            f"Round {state.get('current_round', 1)} - "
            f"CoderA: {coderA_round_score} points ({result.coderA_feedback}), "
            f"CoderB: {coderB_round_score} points ({result.coderB_feedback})"
        )
        state["reviewer_comments"].append(review_comment)

        # Set round winner
        state["round_winner"] = result.round_winner

        # Update conversation context
        self._update_conversation_context(
            state,
            f"Round {state.get('current_round', 1)} winner: {result.round_winner}. "
            f"Summary: {result.summary}"
        )

        # Determine next steps
        current_round = state.get("current_round", 1)
        max_rounds = state.get("max_rounds", 3)

        if current_round >= max_rounds:
            # Competition is over, determine overall winner
            if state["coderA_Score"] > state["coderB_Score"]:
                state["overall_winner"] = "coderA"
            elif state["coderB_Score"] > state["coderA_Score"]:
                state["overall_winner"] = "coderB"
            else:
                state["overall_winner"] = "draw"

            state["competition_status"] = "completed"
            state["next_agent"] = None
        else:
            # Move to next round
            state["current_round"] = current_round + 1
            state["next_agent"] = "moderator"
            state["competition_status"] = "preparing"

        return state

    def _parse_result(self, llm_result: Any) -> CodeReview:
        """Return LLM result as-is since structured output is enforced."""
        return self.argument_parser.invoke(llm_result)

    def _update_conversation_context(self, state: Dict[str, Any], message: str) -> None:
        """Update the conversation context in the state."""
        if "conversation_context" not in state:
            state["conversation_context"] = ""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        state["conversation_context"] += f"\n[{timestamp}] Reviewer: {message}"
