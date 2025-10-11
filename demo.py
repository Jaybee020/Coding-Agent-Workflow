"""
Quick demo script for the AI Coding Competition System.
Shows how to run a simple competition and inspect results.
"""

import asyncio
import os
from dotenv import load_dotenv

from src.core.models import CodingCompetitionConfig
from src.workflow.competition_graph import CodingCompetitionGraph


async def run_quick_demo():
    """
    Run a quick 1-round demo competition.
    Perfect for testing the system.
    """
    print("\n" + "="*60)
    print("AI CODING COMPETITION - QUICK DEMO")
    print("="*60)

    # Load environment variables
    load_dotenv()

    # Check for required API keys
    missing_keys = []
    if not os.getenv("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")
    if not os.getenv("E2B_API_KEY"):
        missing_keys.append("E2B_API_KEY")

    if missing_keys:
        print("\n❌ Error: Missing required API keys!")
        print(f"Missing: {', '.join(missing_keys)}")
        print("\nSteps to fix:")
        print("1. Copy .env.example to .env")
        print("2. Add your OpenAI API key (get it from: https://platform.openai.com/api-keys)")
        print("3. Add your E2B API key (get it from: https://e2b.dev/)")
        print("\nYour .env file should look like:")
        print("OPENAI_API_KEY=sk-proj-xxxxx")
        print("E2B_API_KEY=e2b_xxxxx")
        return

    # Configure for a quick 1-round demo
    config = CodingCompetitionConfig(
        max_rounds=1,  # Just 1 round for demo
        model_name="gpt-4o-mini",
        temperature=0.7,
        max_tokens=1500,
        scoring_rules={
            "correctness": 5,
            "efficiency": 3,
            "code_quality": 2
        }
    )

    print("\n📋 Demo Configuration:")
    print(f"  - Rounds: {config.max_rounds}")
    print(f"  - Model: {config.model_name}")
    print(f"  - Temperature: {config.temperature}")

    # Create competition
    competition = CodingCompetitionGraph(config=config)

    # Run the competition
    print("\n🚀 Starting competition...\n")
    final_state = await competition.run_competition()

    # Show detailed results
    print("\n" + "="*60)
    print("📊 DETAILED RESULTS")
    print("="*60)

    # Show the problem
    if final_state.get("current_problem"):
        problem = final_state["current_problem"]
        print(f"\n📝 Problem: {problem.title}")
        print(f"Description: {problem.description[:100]}...")

    # Show submissions
    print("\n👨‍💻 CoderA's Approach:")
    if final_state.get("coderA_submissions"):
        submission = final_state["coderA_submissions"][-1]
        print(f"  Explanation: {submission.explanation}")
        print(f"  Complexity: {submission.complexity}")
        print(f"  Code Preview: {submission.code[:150]}...")

    print("\n👩‍💻 CoderB's Approach:")
    if final_state.get("coderB_submissions"):
        submission = final_state["coderB_submissions"][-1]
        print(f"  Explanation: {submission.explanation}")
        print(f"  Complexity: {submission.complexity}")
        print(f"  Code Preview: {submission.code[:150]}...")

    # Agent performance
    print("\n📈 Agent Performance:")
    metrics = competition.get_agent_metrics()
    for agent_name, agent_metrics in metrics.items():
        if agent_metrics['calls'] > 0:
            print(f"\n  {agent_name.upper()}:")
            print(f"    Calls: {agent_metrics['calls']}")
            print(f"    Success Rate: {agent_metrics['success_rate']:.1%}")
            print(f"    Avg Time: {agent_metrics['average_execution_time']:.2f}s")
            if agent_metrics['tool_calls'] > 0:
                print(f"    Tool Calls: {agent_metrics['tool_calls']}")

    print("\n" + "="*60)
    print("✅ Demo completed successfully!")
    print("="*60)

    return final_state


async def run_multi_round_demo():
    """
    Run a full 3-round competition.
    """
    print("\n" + "="*60)
    print("AI CODING COMPETITION - FULL DEMO (3 Rounds)")
    print("="*60)

    load_dotenv()

    # Check for required API keys
    missing_keys = []
    if not os.getenv("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")
    if not os.getenv("E2B_API_KEY"):
        missing_keys.append("E2B_API_KEY")

    if missing_keys:
        print(f"\n❌ Error: Missing {', '.join(missing_keys)}!")
        print("Please set up your .env file with both API keys.")
        return

    config = CodingCompetitionConfig(
        max_rounds=3,
        model_name="gpt-4o-mini",
        temperature=0.7
    )

    competition = CodingCompetitionGraph(config=config)
    final_state = await competition.run_competition()

    print("\n📊 Competition Summary:")
    print(f"  Total Rounds: {config.max_rounds}")
    print(f"  Final Scores - CoderA: {final_state.get('coderA_Score', 0)}")
    print(f"  Final Scores - CoderB: {final_state.get('coderB_Score', 0)}")
    print(f"  Overall Winner: {final_state.get('overall_winner', 'N/A').upper()}")

    return final_state


if __name__ == "__main__":
    # Choose which demo to run
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        print("\nRunning FULL 3-round competition...")
        asyncio.run(run_multi_round_demo())
    else:
        print("\nRunning QUICK 1-round demo...")
        print("(Use --full flag for a 3-round competition)")
        asyncio.run(run_quick_demo())
