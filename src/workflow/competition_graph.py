"""
LangGraph-based workflow for the coding competition.
Orchestrates the flow between Moderator, CoderA, CoderB, and Reviewer agents.
"""

from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from datetime import datetime

from ..core.models import CodingCompetitionConfig, CompetitionState, CompetitionMetrics
from ..agents.moderator_agent import ModeratorAgent
from ..agents.coderA_agent import CoderAAgent
from ..agents.coderB_agent import CoderBAgent
from ..agents.reviewer_agent import ReviewerAgent
from ..tools.agent_tools import REVIEWER_TOOLS


class CodingCompetitionGraph:
    """
    Main orchestration class for the coding competition workflow.
    Uses LangGraph to manage the state machine between agents.
    """

    def __init__(self, config: CodingCompetitionConfig = None):
        """
        Initialize the competition graph with configuration.

        Args:
            config: Competition configuration
        """
        self.config = config or CodingCompetitionConfig()

        # Initialize all agents with their respective tools
        # Moderator has no tools - it generates problems directly using JSON mode
        self.moderator = ModeratorAgent(config=self.config, tools=[])
        self.coderA = CoderAAgent(config=self.config, tools=[])
        self.coderB = CoderBAgent(config=self.config, tools=[])
        self.reviewer = ReviewerAgent(config=self.config, tools=REVIEWER_TOOLS)

        # Build the graph
        self.graph = self._build_graph()
        self.compiled_graph = self.graph.compile()

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state machine for the competition workflow.

        Returns:
            Compiled StateGraph
        """
        # Create the state graph
        workflow = StateGraph(CompetitionState)

        # Add nodes for each agent
        workflow.add_node("moderator", self._moderator_node)
        workflow.add_node("coderA", self._coderA_node)
        workflow.add_node("coderB", self._coderB_node)
        workflow.add_node("reviewer", self._reviewer_node)

        # Set the entry point
        workflow.set_entry_point("moderator")

        # Add conditional edges based on next_agent field
        workflow.add_conditional_edges(
            "moderator",
            self._route_next,
            {
                "coderA": "coderA",
                "coderB": "coderB",
                "reviewer": "reviewer",
                "moderator": "moderator",
                "end": END
            }
        )

        workflow.add_conditional_edges(
            "coderA",
            self._route_next,
            {
                "coderA": "coderA",
                "coderB": "coderB",
                "reviewer": "reviewer",
                "moderator": "moderator",
                "end": END
            }
        )

        workflow.add_conditional_edges(
            "coderB",
            self._route_next,
            {
                "coderA": "coderA",
                "coderB": "coderB",
                "reviewer": "reviewer",
                "moderator": "moderator",
                "end": END
            }
        )

        workflow.add_conditional_edges(
            "reviewer",
            self._route_next,
            {
                "coderA": "coderA",
                "coderB": "coderB",
                "reviewer": "reviewer",
                "moderator": "moderator",
                "end": END
            }
        )

        return workflow

    async def _moderator_node(self, state: CompetitionState) -> CompetitionState:
        """Execute the moderator agent."""
        print(f"\n{'='*60}")
        print(f"🎯 MODERATOR - Round {state.get('current_round', 1)}")
        print(f"{'='*60}")
        return await self.moderator(state)

    async def _coderA_node(self, state: CompetitionState) -> CompetitionState:
        """Execute CoderA agent."""
        print(f"\n{'='*60}")
        print(f"👨‍💻 CODER A - Submitting Solution")
        print(f"{'='*60}")
        return await self.coderA(state)

    async def _coderB_node(self, state: CompetitionState) -> CompetitionState:
        """Execute CoderB agent."""
        print(f"\n{'='*60}")
        print(f"👩‍💻 CODER B - Submitting Solution")
        print(f"{'='*60}")
        return await self.coderB(state)

    async def _reviewer_node(self, state: CompetitionState) -> CompetitionState:
        """Execute the reviewer agent."""
        print(f"\n{'='*60}")
        print(f"⚖️  REVIEWER - Evaluating Round {state.get('current_round', 1)}")
        print(f"{'='*60}")
        return await self.reviewer(state)

    def _route_next(self, state: CompetitionState) -> Literal["moderator", "coderA", "coderB", "reviewer", "end"]:
        """
        Routing function to determine the next agent based on state.

        Args:
            state: Current competition state

        Returns:
            Name of the next node to execute or END
        """
        next_agent = state.get("next_agent")
        competition_status = state.get("competition_status")

        # If there's an error, end the workflow
        if competition_status == "error":
            print(f"\n❌ Competition ended due to error. Check error_log in state.")
            if "error_log" in state:
                print(f"Latest error: {state['error_log'][-1]}")
            return "end"

        # If competition is completed, end the workflow
        if competition_status == "completed" or next_agent is None:
            return "end"

        # Route to the specified next agent
        if next_agent == "moderator":
            return "moderator"
        elif next_agent == "coderA":
            return "coderA"
        elif next_agent == "coderB":
            return "coderB"
        elif next_agent == "reviewer":
            return "reviewer"

        # Default to end if routing is unclear
        return "end"

    async def run_competition(self, initial_state: Dict[str, Any] = None) -> CompetitionState:
        """
        Run a complete coding competition.

        Args:
            initial_state: Optional initial state to start with

        Returns:
            Final competition state with results
        """
        # Initialize the state
        state: CompetitionState = initial_state or self._initialize_state()

        print("\n" + "="*60)
        print("🏆 CODING COMPETITION STARTING")
        print("="*60)
        print(f"Max Rounds: {state.get('max_rounds', 3)}")
        print(f"Model: {self.config.model_name}")
        print("="*60)

        # Run the graph
        final_state = await self.compiled_graph.ainvoke(state)

        # Print final results
        self._print_results(final_state)

        return final_state

    def _initialize_state(self) -> CompetitionState:
        """
        Initialize the competition state with default values.

        Returns:
            Initial competition state
        """
        return {
            "problems": [],
            "current_problem": None,
            "context": "",
            "current_round": 1,
            "max_rounds": self.config.max_rounds,
            "coderA_personality": "straightforward",
            "coderB_personality": "resourceaware",
            "coderA_submissions": [],
            "coderB_submissions": [],
            "coderA_Score": 0,
            "coderB_Score": 0,
            "reviewer_comments": [],
            "round_winner": None,
            "overall_winner": None,
            "metrics": CompetitionMetrics(),
            "error_log": [],
            "next_agent": "moderator",
            "competition_status": "preparing",
            "memory_thread_id": f"comp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "memory_checkpoint_id": None,
            "conversation_context": ""
        }

    def _print_results(self, state: CompetitionState):
        """
        Print the final competition results.

        Args:
            state: Final competition state
        """
        print("\n" + "="*60)
        print("🏆 COMPETITION COMPLETED")
        print("="*60)

        print(f"\n📊 FINAL SCORES:")
        print(f"  CoderA: {state.get('coderA_Score', 0)} points")
        print(f"  CoderB: {state.get('coderB_Score', 0)} points")

        winner = state.get('overall_winner', 'unknown')
        if winner == "coderA":
            print(f"\n🥇 WINNER: CoderA!")
        elif winner == "coderB":
            print(f"\n🥇 WINNER: CoderB!")
        else:
            print(f"\n🤝 RESULT: Draw!")

        print(f"\n📝 ROUND SUMMARIES:")
        for i, comment in enumerate(state.get('reviewer_comments', []), 1):
            print(f"  Round {i}: {comment}")

        metrics = state.get('metrics')
        if metrics:
            print(f"\n📈 METRICS:")
            print(f"  API Calls: {metrics.api_calls_made}")
            print(f"  Total Tokens: {metrics.total_tokens}")
            if metrics.total_duration:
                print(f"  Duration: {metrics.total_duration:.2f}s")

        print("="*60)

    def get_agent_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for all agents.

        Returns:
            Dictionary of agent metrics
        """
        return {
            "moderator": self.moderator.get_metrics(),
            "coderA": self.coderA.get_metrics(),
            "coderB": self.coderB.get_metrics(),
            "reviewer": self.reviewer.get_metrics()
        }
