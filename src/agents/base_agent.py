"""
Base agent implementation using modern LangChain LCEL patterns.
Provides foundation for all agents with common functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
import time

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import ToolMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model

from ..core.models import  CodeExecutionResult, CodingPersonalities, AgentRole, CodingCompetitionConfig,CompetitionState



class BaseAgent(ABC):
    """Abstract base class for agents"""

    def __init__(self, role:AgentRole, config:CodingCompetitionConfig=None, tools:List=None):
        self.role = role
        self.config = config or CodingCompetitionConfig()
        self.tools = tools or []

        # Create tool name to function mapping
        self.tool_map = {tool.name: tool for tool in self.tools}

        # Only use JSON mode if there are no tools (JSON mode conflicts with tool calling)
        llm_kwargs = {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": 30,
            "max_retries": 3,

        }

        # # Add JSON mode only for agents without tools
        # if not self.tools:
        #     llm_kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

        self.llm = ChatOpenAI()

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(tools) if self.tools else self.llm

        print("Tools bound to LLM:", list(self.tool_map.keys()))
        print("LLM with tools:", self.llm_with_tools)

        # Apply structured output if the agent requests it
        # This must be done AFTER tool binding
        if hasattr(self, 'use_structured_output') and self.use_structured_output:
            # Get the Pydantic model from the parser
            if hasattr(self, 'argument_parser') and hasattr(self.argument_parser, 'pydantic_object'):
                pydantic_model = self.argument_parser.pydantic_object
                self.llm_with_tools = self.llm_with_tools.with_structured_output(pydantic_model)

         # Performance tracking
        self.metrics = {
            "calls": 0,
            "total_time": 0.0,
            "errors": 0,
            "success_rate": 1.0,
            "tool_calls": 0
        }

           # Initialize chain components
        self._initialize_chain()

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for the agent.
        """
        pass

    @abstractmethod
    def _get_user_prompt(self) -> str:
        """
        Get the human prompt for the agent.
        """
        pass

    @abstractmethod
    def _post_process_result(self,result:Any,state:CompetitionState):
        """
        Post-process the result from the LLM.
        """
        pass

    def _execute_tool(self, tool_call: Dict[str, Any]) -> ToolMessage:
        """
        Execute a single tool call and return a ToolMessage.

        Args:
            tool_call: Tool call from LLM containing name, args, and id

        Returns:
            ToolMessage with the tool execution result
        """
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        try:
            # Get the tool function from the map
            if tool_name not in self.tool_map:
                return ToolMessage(
                    content=f"Error: Tool '{tool_name}' not found",
                    tool_call_id=tool_call_id
                )

            tool_func = self.tool_map[tool_name]

            # Execute the tool
            result = tool_func.invoke(tool_args)

            print(f"Tool '{tool_name}' executed with args {tool_args}, result: {result}")

            # Track tool usage
            self.metrics["tool_calls"] += 1

            return ToolMessage(
                content=str(result),
                tool_call_id=tool_call_id
            )

        except Exception as e:
            return ToolMessage(
                content=f"Error executing {tool_name}: {str(e)}",
                tool_call_id=tool_call_id
            )

    def _handle_tool_calls(self, llm_result):
        """
        Execute any tool calls and re-invoke LLM with results.
        Implements the tool calling loop: LLM -> Tools -> LLM with results

        For structured output mode:
        - First call may return AIMessage with tool_calls
        - After tools execute, second call returns Pydantic object
        - If no tools needed, first call returns Pydantic object directly
        """

        if hasattr(llm_result, 'tool_calls') and llm_result.tool_calls:
            # Build message history for next LLM call
            messages = []

            # Add the AI message with tool calls
            messages.append(llm_result)
            print("llm result in handle tool calls",llm_result)

            # Execute each tool call and add results
            for tool_call in llm_result.tool_calls:
                tool_message = self._execute_tool(tool_call)
                messages.append(tool_message)

            # Re-invoke LLM with tool results to get final response
            # In structured output mode, this will return a Pydantic object
            return self.llm_with_tools.invoke(messages)

        # No tool calls - return as-is
        # In structured output mode, this is already a Pydantic object
        # In regular mode, this is an AIMessage that needs parsing
        return llm_result

    def _initialize_chain(self):
        """Initialize the LCEL chain with prompts and parsers"""

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self._get_system_prompt()),
            HumanMessagePromptTemplate.from_template(self._get_user_prompt())
        ])

        # Create the base chain
        # If using structured output, skip the parse step since it returns Pydantic objects directly
        if hasattr(self, 'use_structured_output') and self.use_structured_output:
            self.chain = (
                RunnablePassthrough.assign(
                    conversation_context=RunnableLambda(self._get_conversation_context)
                )
                | self.prompt
                | self.llm_with_tools
                | RunnableLambda(self._handle_tool_calls)
                | RunnableLambda(self._validate_result)
            )
        else:
            self.chain = (
                RunnablePassthrough.assign(
                    conversation_context=RunnableLambda(self._get_conversation_context)
                )
                | self.prompt
                | self.llm_with_tools
                | RunnableLambda(self._handle_tool_calls)
                | RunnableLambda(self._parse_result)
                | RunnableLambda(self._validate_result)
            )


    def _get_conversation_context(self, state: Dict[str, Any]) -> str:
        """Get conversation context from memory"""
        return state.get("conversation_context", "")

    @abstractmethod
    def _parse_result(self, llm_result: Any) -> Any:
        """Parse the LLM result into appropriate format"""
        pass

    def _validate_result(self, parsed_result: Any) -> Any:
        """Validate the parsed result"""
        return parsed_result

    async def __call__(self, state: CompetitionState) -> CompetitionState:
        """Execute the agent with performance tracking"""
        start_time = time.time()
        self.metrics["calls"] += 1

        try:
            # Prepare input for the chain
            chain_input = self._prepare_chain_input(state)

            # Execute the chain
            result = await self.chain.ainvoke(chain_input)

            # Post-process and update state
            updated_state = self._post_process_result(result, state)

            # Update performance metrics
            execution_time = time.time() - start_time
            self.metrics["total_time"] += execution_time

            # Add execution metadata
            if "metrics" in updated_state:
                updated_state["metrics"].api_calls_made += 1

            return updated_state

        except Exception as e:
            self.metrics["errors"] += 1
            self.metrics["success_rate"] = (
                (self.metrics["calls"] - self.metrics["errors"]) /
                max(1, self.metrics["calls"])
            )

            # Add error to state log with detailed info
            import traceback
            error_msg = f"{self.role.value} Error: {str(e)}"
            error_log = state.get("error_log", [])
            error_log.append(error_msg)

            # Print detailed error for debugging
            print(f"\n❌ Error in {self.role.value} agent:")
            print(f"   {str(e)}")
            print(f"\n   Full traceback:")
            traceback.print_exc()

            return {
                **state,
                "error_log": error_log,
                "competition_status": "error"
            }

    @abstractmethod
    def _prepare_chain_input(self, state:CompetitionState):
        """Prepare input dictionary for the LCEL chain"""
    pass

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this agent"""
        avg_time = self.metrics["total_time"] / max(1, self.metrics["calls"])
        return {
            **self.metrics,
            "average_execution_time": avg_time,
            "role": self.role.value
        }

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize agent with configuration"""
        pass

    async def cleanup(self) -> None:
        """Cleanup agent resources"""
        pass
