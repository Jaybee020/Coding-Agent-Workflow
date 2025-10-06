"""
Core data models for the AI coding system.
Defines rich, validated state structures using Pydantic.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from typing_extensions import TypedDict
from datetime import datetime
from enum import Enum


class CodingPersonalities(str, Enum):
    """
    Enum representing different coding personalities.
    """
    DEFAULT = "default"
    STRAIGHT_FORWARD = "straightforward"
    RESOURCE_AWARE = "resourceaware"

class AgentRole(str, Enum):
    """
    Enum representing different agent roles.
    """
    CODER = "coder"
    REVIEWER = "reviewer"
    ROUND_CONTROLLER = "roundcontroller"
    AGGREGATOR = "aggregator"


class CodeTest(BaseModel):
    """
    Model representing a code test case.
    """
    name: str
    input: Any
    expected_output: Any

class CodingProblem(BaseModel):
    """
    Model representing a coding problem.
    """
    id:str
    title: str
    entrypoint: str
    description: str
    constraints:Dict[str, Any] = Field(default_factory=dict)
    public_tests: List[CodeTest] = Field(default_factory=list)
    budgets: Dict[str, int] = Field(default_factory=dict)

class CodeSubmission(BaseModel):
    """
    Model representing a code submission.
    """
    code: str
    language: str
    entrypoint: str
    explanation: Optional[str] = None
    complexity: Optional[Dict[str, str]] = None

class CompetitionMetrics(BaseModel):
    """Performance and usage metrics"""
    total_tokens: int = Field(default=0, ge=0)
    api_calls_made: int = Field(default=0, ge=0)
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_duration: Optional[float] = None

    def calculate_duration(self) -> float:
        """Calculate total duration in seconds"""
        if self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            self.total_duration = duration
            return duration
        return 0.0



class CodeExecutionResult(BaseModel):
    """
    Model representing the result of code execution.
    """
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None

class CodeTestResult(BaseModel):
    """
    Model representing the result of a code test.
    """
    test_input: Any
    expected_output: Any
    actual_output: Any
    passed: bool
    error: Optional[str] = None
    execution_time: Optional[float] = None


class CodeReview(BaseModel):
    """
    Model representing a code review result.
    """
    coderA_correctness_score: int = Field(ge=0, le=5, description="Correctness score for CoderA (0-5)")
    coderA_efficiency_score: int = Field(ge=0, le=3, description="Efficiency score for CoderA (0-3)")
    coderA_quality_score: int = Field(ge=0, le=2, description="Code quality score for CoderA (0-2)")
    coderA_feedback: str = Field(description="Detailed feedback for CoderA's submission")

    coderB_correctness_score: int = Field(ge=0, le=5, description="Correctness score for CoderB (0-5)")
    coderB_efficiency_score: int = Field(ge=0, le=3, description="Efficiency score for CoderB (0-3)")
    coderB_quality_score: int = Field(ge=0, le=2, description="Code quality score for CoderB (0-2)")
    coderB_feedback: str = Field(description="Detailed feedback for CoderB's submission")

    round_winner: Literal["coderA", "coderB", "draw"] = Field(description="Winner of this round")
    summary: str = Field(description="Overall summary of the round and comparison")


class CompetitionState(TypedDict , total=False):
    """
    TypedDict representing the state of a coding competition.
    """
    problems: List[CodingProblem]
    current_problem: Optional[CodingProblem]
    context: str
    current_round: int
    max_rounds: int

    coderA_personality: CodingPersonalities
    coderB_personality: CodingPersonalities
    coderA_submissions: List[CodeSubmission]
    coderB_submissions: List[CodeSubmission]
    coderA_Score: int
    coderB_Score: int

    reviewer_comments: List[str]
    round_winner: Optional[Literal["coderA", "coderB", "draw"]]
    overall_winner: Optional[Literal["coderA", "coderB", "draw"]]

     # AI System Metadata
    metrics: CompetitionMetrics
    error_log: List[str]

    # Workflow Control
    next_agent: Literal["moderator", "coderA", "coderB", "reviewer"]
    competition_status: Literal["active", "completed", "error", "preparing"]

    # Memory Management
    memory_thread_id: str
    memory_checkpoint_id: Optional[str]
    conversation_context: str  # Recent conversation context for AI


class CodingCompetitionConfig(BaseModel):
    """
    Model representing configuration for a coding competition.
    """
    max_rounds: int = 3
      # LLM Configuration
    model_name: str = "gpt-4-turbo-preview"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1500, ge=100)
    time_per_round: int = 600  # in seconds
    review_time: int = 300  # in seconds
    scoring_rules: Dict[str, int] = Field(default_factory=lambda: {
        "correctness": 5,
        "efficiency": 3,
        "code_quality": 2
    })
