"""
Main entry point for running coding competitions.
"""

import asyncio
import os
from dotenv import load_dotenv

from src.core.models import CodingCompetitionConfig
from src.workflow.competition_graph import CodingCompetitionGraph


async def run_competition_example():
    """
    Example function demonstrating how to run a coding competition.
    """
    # Load environment variables (for API keys)
    load_dotenv()

    # Ensure required API keys are set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key to run the competition.")
        return

    if not os.getenv("E2B_API_KEY"):
        print("⚠️  Error: E2B_API_KEY not found in environment variables")
        print("Please set your E2B API key for code execution.")
        print("\nGet your E2B API key at: https://e2b.dev/")
        return

    # Configure the competition
    config = CodingCompetitionConfig(
        max_rounds=3,
        model_name="gpt-4o-mini",
        temperature=0.7,
        max_tokens=2000,
        scoring_rules={
            "correctness": 5,
            "efficiency": 3,
            "code_quality": 2
        }
    )

    # Create the competition graph
    competition = CodingCompetitionGraph(config=config)

    # Run the competition
    print("\n🚀 Starting Coding Competition...\n")
    final_state = await competition.run_competition()

    # Get agent performance metrics
    print("\n📊 Agent Performance Metrics:")
    metrics = competition.get_agent_metrics()
    for agent_name, agent_metrics in metrics.items():
        print(f"\n{agent_name.upper()}:")
        print(f"  Total Calls: {agent_metrics['calls']}")
        print(f"  Success Rate: {agent_metrics['success_rate']:.2%}")
        print(f"  Avg Execution Time: {agent_metrics['average_execution_time']:.2f}s")
        print(f"  Tool Calls: {agent_metrics['tool_calls']}")

    return final_state


async def run_custom_competition(
    max_rounds: int = 3,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7
):
    """
    Run a custom competition with specified parameters.

    Args:
        max_rounds: Number of rounds to run
        model: LLM model to use
        temperature: Temperature for LLM responses
    """
    config = CodingCompetitionConfig(
        max_rounds=max_rounds,
        model_name=model,
        temperature=temperature
    )

    competition = CodingCompetitionGraph(config=config)
    return await competition.run_competition()


if __name__ == "__main__":
    # Run the example competition
    asyncio.run(run_competition_example())
