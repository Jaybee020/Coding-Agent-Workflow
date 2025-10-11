"""
Code execution utilities using E2B sandboxes and LLM for problem generation.
"""

import os
import json
from typing import List, Dict, Any
from e2b_code_interpreter import Sandbox
from langchain_openai import ChatOpenAI


def generateCodeQuestion(prompt: str) -> str:
    """
    Generate a coding problem using LLM based on the given prompt.

    Args:
        prompt: Description or requirements for the coding problem

    Returns:
        JSON string containing the coding problem
    """
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
        )

        system_prompt = """You are a coding problem generator. Create a well-defined coding problem
        that includes:
        - A clear problem statement
        - Input/output examples
        - Constraints
        - At least 2-3 test cases

        Return ONLY a JSON object in this exact format:
        {
            "id": "unique_problem_id",
            "title": "Problem Title",
            "entrypoint": "function_name",
            "description": "Detailed problem description with examples",
            "public_tests": [
                {
                    "name": "Test 1",
                    "input": "example input",
                    "expected_output": "example output"
                }
            ],
        }
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate a coding problem: {prompt}"}
        ]

        result = llm.invoke(messages)
        return result.content

    except Exception as e:
        error_result = {
            "error": f"Failed to generate problem: {str(e)}",
            "prompt": prompt
        }
        return json.dumps(error_result, indent=2)


def executeCode(code: str, input_data: str = "") -> str:
    """
    Execute Python code in a fresh E2B sandbox with optional input data.

    Args:
        code: Python code to execute
        input_data: Optional input data for the code

    Returns:
        Formatted string with execution results
    """
    try:
        # Create a fresh sandbox for each execution
        with Sandbox.create() as sandbox:
            # Prepare code with input if provided
            if input_data:
                full_code = f"""
            # Input data
            input_data = {repr(input_data)}

            # User code
            {code}
            """
            else:
                full_code = code

            # Execute the code with timeout
            execution = sandbox.run_code(full_code)
            print("Execution",execution)

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
            print("Output Text",output_text)

            return f"""✅ EXECUTION SUCCESS

Output:
{output_text}

Logs:
{execution.logs.stdout if execution.logs.stdout else '[No stdout]'}

Execution completed successfully.
"""

    except Exception as e:
        return f"""❌ SANDBOX ERROR

Error: Failed to create or execute in sandbox
Details: {str(e)}

This may be due to:
- E2B API key not set or invalid
- Network connectivity issues
- E2B service unavailable
- Code execution timeout

Please check your E2B_API_KEY in the .env file.
"""


def testCode(code: str, test_cases: List[Dict[str, Any]]) -> str:
    """
    Run a complete test suite against code submission in one E2B sandbox.
    All tests run in the same sandbox for efficiency, but with isolated execution.

    Args:
        code: The code to test
        test_cases: List of test cases, each containing:
            - name: Test case name
            - input: Input data
            - expected_output: Expected output

    Returns:
        Formatted string with all test results
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


# Helper function to check E2B setup
def check_e2b_setup() -> bool:
    """
    Check if E2B is properly configured.

    Returns:
        True if E2B_API_KEY is set, False otherwise
    """
    api_key = os.getenv("E2B_API_KEY")
    if not api_key:
        print("⚠️  Warning: E2B_API_KEY not found in environment variables")
        print("Code execution will not work without it.")
        print("Please add E2B_API_KEY to your .env file")
        return False
    return True
