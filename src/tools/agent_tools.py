"""
Agent tools for the coding competition system.
Provides executable functions that agents can call to interact with external systems.
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from .codeUtils import executeCode, testCode, generateCodeQuestion


class CodeExecutionInput(BaseModel):
    """Input schema for code execution."""
    code: str = Field(description="The Python code to execute")
    input_data: str = Field(description="Input data to pass to the code")


class CodeTestInput(BaseModel):
    """Input schema for running test cases."""
    code: str = Field(description="The code to test")
    test_cases: list = Field(description="List of test cases to run")


@tool(args_schema=CodeExecutionInput)
def execute_code_sandbox(code: str, input_data: str) -> str:
    """
    Execute code in a sandboxed environment with given input data.
    Use this tool to run and test code submissions safely.

    Returns execution output or error messages.
    """
    return executeCode(code, input_data)


@tool(args_schema=CodeTestInput)
def run_test_suite(code: str, test_cases: list) -> str:
    """
    Run a complete test suite against code submission.
    Use this to validate code against multiple test cases at once.

    Returns summary of all test results including pass/fail for each case.
    """
    return testCode(code, test_cases)


@tool
def fetch_code_problem(prompt: str) -> str:
    """
    Generate or fetch a coding problem for the competition.
    Use this when you need to create a coding challenge.

    Args:
        prompt: Description or requirements for the coding problem

    Returns a coding problem description.
    """
    return generateCodeQuestion(prompt)


# Export tools grouped by agent type
REVIEWER_TOOLS = [
    execute_code_sandbox,
    run_test_suite
]

MODERATOR_TOOLS = [
    fetch_code_problem
]

ALL_TOOLS = REVIEWER_TOOLS + MODERATOR_TOOLS
