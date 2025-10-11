"""
Core data models for the AI coding system.
Defines rich, validated state structures using Pydantic.
"""

from pydantic import BaseModel, Field, validator
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

class CodeComplexity(BaseModel):
    """
    Model representing code complexity analysis.
    """
    time: str = Field(..., description="Time complexity in Big O notation", type="string")
    space: str = Field(..., description="Space complexity in Big O notation", type="string")


class CodeTest(BaseModel):
    """
    Model representing a code test case.
    """
    name: str = Field(..., description="Name of the test case", min_length=1, type="string")
    input: str = Field(..., description="Input for the test case", type="string")
    expected_output: str = Field(..., description="Expected output for the test case", type="string")

    class Config:
        schema_extra = {
            "required": ["name", "input", "expected_output"]
        }

class CodingProblem(BaseModel):
    """
    Model representing a coding problem.
    """
    id: str = Field(..., description="Unique identifier for the coding problem", min_length=1)
    title: str = Field(..., description="Title of the coding problem", min_length=1)
    entrypoint: str = Field(..., description="Entrypoint function name", min_length=1)
    description: str = Field(..., description="Detailed description of the problem", min_length=1)
    public_tests: List[CodeTest] = Field(default_factory=list, description="List of public test cases", type="array")



class CodeSubmission(BaseModel):
    """
    Model representing a code submission.
    """
    code: str = Field(..., description="Code submitted by the user", type="string")
    language: str = Field(..., description="Programming language of the submission", type="string")
    entrypoint: str = Field(..., description="Entrypoint function name", type="string")
    explanation: Optional[str] = Field(None, description="Explanation of the code", type="string")
    complexity: Optional[CodeComplexity] = Field(None, description="Complexity analysis of the code", type="object")

class CompetitionMetrics(BaseModel):
    """Performance and usage metrics"""
    total_tokens: int = Field(default=0, ge=0, description="Total tokens used", type="integer")
    api_calls_made: int = Field(default=0, ge=0, description="Number of API calls made", type="integer")
    start_time: datetime = Field(default_factory=datetime.now, description="Start time of the competition", type="string")
    end_time: Optional[datetime] = Field(None, description="End time of the competition", type="string")
    total_duration: Optional[float] = Field(None, description="Total duration of the competition in seconds", type="number")





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
    model_name: str = "gpt-4o-mini"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1500, ge=100)
    time_per_round: int = 600  # in seconds
    review_time: int = 300  # in seconds
    scoring_rules: Dict[str, int] = Field(default_factory=lambda: {
        "correctness": 5,
        "efficiency": 3,
        "code_quality": 2
    })
