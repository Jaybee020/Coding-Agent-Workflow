"""
Agent tools for the coding competition system.
Provides executable functions that agents can call to interact with external systems.
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from .codeUtils import executeCode, testCode, generateCodeQuestion
from e2b_code_interpreter import Sandbox
import textwrap
import ast


class CodeExecutionInput(BaseModel):
    """Input schema for code execution."""
    code: str = Field(description="The Python code to execute")
    input_data: str = Field(description="Input data to pass to the code")


class CodeTestInput(BaseModel):
    """Input schema for running test cases."""
    code: str = Field(description="The code to test")
    test_cases: list = Field(description="List of test cases to run")


@tool("execute_code_sandbox", description="Execute Python code in a sandboxed environment with optional input data", args_schema=CodeExecutionInput)
def execute_code_sandbox(code: str, input_data: str) -> str:
    """
    Execute code in a sandboxed environment with given input data.
    Use this tool to run and test code submissions safely.

    Returns execution output or error messages.
    """
    try:
        # Safely evaluate input_data to ensure correct type
        try:
            evaluated_input = ast.literal_eval(input_data)
        except (ValueError, SyntaxError):
            evaluated_input = input_data  # Fallback to string if evaluation fails

        # Create a fresh sandbox for each execution
        with Sandbox.create() as sandbox:
            # Dedent and normalize the user code
            dedented_code = textwrap.dedent(code)

            # Prepare code with input if provided
            if input_data:
                full_code = f"""
# Input data
input_data = {evaluated_input}

# User code
{textwrap.indent(dedented_code, '')}
"""
            else:
                full_code = textwrap.indent(dedented_code, '    ')

            # Debugging: Print the full code to verify formatting
            print("Full Code to Execute:\n", full_code)

            # Execute the code with timeout
            execution = sandbox.run_code(full_code)
            print("Execution", execution)

            # Check for errors
            if execution.error:
                return f"""❌ EXECUTION ERROR

                Error Type: Runtime Error
                Error Message: {execution.error.name}: {execution.error.value}
Traceback: {execution.error.traceback}

Code executed:
{full_code}
"""

            # Success - return formatted output
            output_text = execution.text if execution.text else "[No output]"
            print("Output Text", output_text)

            return f"""✅ EXECUTION SUCCESS

Output:
{output_text}

Logs:
{execution.logs.stdout if execution.logs.stdout else '[No stdout]'}

Execution completed successfully.
"""

    except Exception as e:
        return f"""❌ SANDBOX ERROR: {str(e)}"""


@tool("run_test_suite", description="Run a suite of test cases against a code submission", args_schema=CodeTestInput)
def run_test_suite(code: str, test_cases: list) -> str:
    """
    Run a complete test suite against code submission.
    Use this to validate code against multiple test cases at once.

    Returns summary of all test results including pass/fail for each case.
    """

    if not test_cases:
        return "⚠️  NO TESTS PROVIDED\n\nNo test cases were provided to run."

    try:
        # Create one sandbox for all tests
        with Sandbox.create() as sandbox:
            results = []
            passed_count = 0
            failed_count = 0

            for i, test in enumerate(test_cases, 1):
                test_name = test.get("name", f"Test {i}")
                test_input = test.get("input", "")
                expected_output = test.get("expected_output", "")

                # Prepare test code
                test_code = f"""
# Test Case: {test_name}
test_input = {repr(test_input)}
expected_output = {repr(expected_output)}

# User code
{code}

# Run and capture result
try:
    actual_output = str(result) if 'result' in locals() else '[No result variable]'
    test_passed = actual_output.strip() == str(expected_output).strip()
    print(f"ACTUAL: {{actual_output}}")
    print(f"EXPECTED: {{expected_output}}")
    print(f"PASSED: {{test_passed}}")
except Exception as e:
    print(f"ERROR: {{e}}")
    print(f"PASSED: False")
"""

                # Execute test
                execution = sandbox.run_code(test_code)

                # Parse results
                if execution.error:
                    failed_count += 1
                    results.append(f"""
Test {i}: {test_name}
  Status: ❌ FAILED (Runtime Error)
  Input: {test_input}
  Expected: {expected_output}
  Error: {execution.error.name}: {execution.error.value}
""")
                else:
                    # Check if test passed based on output
                    output = execution.logs.stdout if execution.logs.stdout else ""

                    if "PASSED: True" in output:
                        passed_count += 1
                        results.append(f"""
Test {i}: {test_name}
  Status: ✅ PASSED
  Input: {test_input}
  Expected: {expected_output}
  Output: {output.split('ACTUAL: ')[1].split('EXPECTED:')[0].strip() if 'ACTUAL: ' in output else '[Unknown]'}
""")
                    else:
                        failed_count += 1
                        results.append(f"""
Test {i}: {test_name}
  Status: ❌ FAILED
  Input: {test_input}
  Expected: {expected_output}
  Output: {output.split('ACTUAL: ')[1].split('EXPECTED:')[0].strip() if 'ACTUAL: ' in output else output}
""")

            # Build summary
            total_tests = len(test_cases)
            pass_rate = (passed_count / total_tests * 100) if total_tests > 0 else 0

            summary = f"""
{'='*60}
TEST SUITE RESULTS
{'='*60}

Total Tests: {total_tests}
Passed: {passed_count} ✅
Failed: {failed_count} ❌
Pass Rate: {pass_rate:.1f}%

{'='*60}
DETAILED RESULTS
{'='*60}
{''.join(results)}

{'='*60}
SUMMARY: {'ALL TESTS PASSED ✅' if failed_count == 0 else f'{failed_count} TEST(S) FAILED ❌'}
{'='*60}
"""
            return summary

    except Exception as e:
        return f"""❌ TEST SUITE ERROR

Error: Failed to run test suite
Details: {str(e)}

This may be due to:
- E2B API key not set or invalid
- Network connectivity issues
- Test cases format invalid
- Code execution timeout

Test cases provided: {len(test_cases)}
"""


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
