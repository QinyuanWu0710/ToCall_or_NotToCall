"""
Agent loop implementation with tool selection capabilities.
The agent decides whether to use tools and orchestrates the interaction.
"""

import json
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from vllm import LLM, SamplingParams
from config import AgentConfig

@dataclass
class ToolCall:
    """Represents a tool call decision."""
    needs_tool: bool
    tool_name: Optional[str]
    tool_input: Optional[str]
    reasoning: str


@dataclass
class AgentResponse:
    """Represents the final agent response."""
    response: str
    tool_calls: List[Dict[str, Any]]
    iterations: int
    tokens_generated: int
    tool_in_prompt_info: str
    final_prompt: str


class AgentLoop:
    """
    Agent loop that decides when to use tools and orchestrates the interaction.
    """
    
    def __init__(
        self,
        llm: LLM,
        sampling_params: SamplingParams,
        tool_manager: Any,  # ToolManager instance
        config: AgentConfig = None,
        tokenizer=None
    ):
        """
        Initialize the agent loop.
        
        Args:
            llm: vLLM model instance
            sampling_params: Sampling parameters for generation
            tool_manager: Tool manager for executing tools
            config: Agent configuration (uses default if None)
            tokenizer: Tokenizer for applying chat templates
        """
        self.llm = llm
        self.sampling_params = sampling_params
        self.tool_manager = tool_manager
        self.config = config or AgentConfig()
        self.tokenizer = tokenizer or llm.get_tokenizer()
        # Resolve model name for config dispatch (e.g. reasoning effort for gpt models)
        self.model_name = (
            getattr(llm, "model_name", None)
            or getattr(llm, "model", None)
            or getattr(getattr(llm, "llm_engine", None), "model_config", None) and
            llm.llm_engine.model_config.model
            or ""
)
    
    def _apply_chat_template(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """
        Apply chat template to messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            add_generation_prompt: Whether to add generation prompt
            
        Returns:
            Formatted prompt string
        """
        if self.tokenizer is None:
            # Fallback: just concatenate messages
            formatted = ""
            for msg in messages:
                formatted += f"{msg['role']}: {msg['content']}\n"
            return formatted
        
        try:
            # Use tokenizer's apply_chat_template method
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            )
            return formatted
        except Exception as e:
            print(f"Warning: Failed to apply chat template: {e}")
            # Fallback
            formatted = ""
            for msg in messages:
                formatted += f"{msg['role']}: {msg['content']}\n"
            return formatted
    
    def _create_tool_selection_prompt(self, query: str, override_description: str = None) -> str:
        """Create a prompt for tool selection with chat template applied."""
        tool_selection_content = self.config.format_tool_selection_prompt(
            query, override_description=override_description
        )
        messages = [{"role": "user", "content": tool_selection_content}]
        return self._apply_chat_template(messages, add_generation_prompt=True)

    def _create_force_search_prompt(self, query: str, override_description: str = None) -> str:
        """Create a prompt that *forces* web_search but lets the model craft tool_input."""
        content = self.config.format_tool_force_prompt(
            query, override_description=override_description
        )
        messages = [{"role": "user", "content": content}]
        return self._apply_chat_template(messages, add_generation_prompt=True)
    

    def _create_perceived_need_prompt(self, query: str) -> str:
        """Build the YES/NO actual-need probe prompt.

        The model is shown the testing question and asked whether it needs
        an external tool — no tool list is injected.
        """
        content = (
            f"{query}\n\n"
            "You can call for help from a tool.\n\n"
            "Respond with ONLY a JSON object in this exact schema:\n"
            '{\n'
            '  "needs_tool": true\n'
            '}\n'
            "or\n"
            '{\n'
            '  "needs_tool": false\n'
            '}\n'
        )
        messages = [{"role": "user", "content": content}]
        return self._apply_chat_template(messages, add_generation_prompt=True)

    def _create_perceived_need_prompt_v1(self, query: str) -> str:
        """Build the YES/NO actual-need probe prompt.

        The model is shown the testing question and asked whether it needs
        an external tool — no tool list is injected.
        """
        content = (
            f"{query}\n\n"
            "Do you need help to answer the question?\n\n"
            "Answer:"
        )
        messages = [{"role": "user", "content": content}]
        return self._apply_chat_template(messages, add_generation_prompt=True)
    
    def _create_perceived_need_prompt_v2(self, query: str) -> str:
        """Build the YES/NO actual-need probe prompt.

        The model is shown the testing question and asked whether it needs
        an external tool — no tool list is injected.
        """
        content = (
            f"{query}\n\n"
            "Do you know the answer of the tool?\n"
            "Answer:"
        )
        messages = [{"role": "user", "content": content}]
        return self._apply_chat_template(messages, add_generation_prompt=True)

    # def _parse_yes_no(self, response: str) -> str:
    #     """
    #     Extract a normalised 'yes' or 'no' from the model response.
    #     If 'analysis' and 'assistantfinal' are present, only search after 'assistantfinal'.
    #     Returns 'yes', 'no', or 'unknown' if neither word is found.
    #     """
    #     cleaned = response.strip().lower()

    #     import re as _re

    #     # If both markers exist, search only after 'assistantfinal'
    #     if "analysis" in cleaned and "assistantfinal" in cleaned:
    #         parts = cleaned.split("assistantfinal", 1)
    #         cleaned = parts[1]

    #     # Search for yes/no
    #     m = _re.search(r"\b(yes|no)\b", cleaned)
    #     if m:
    #         return m.group(1)

    #     return "unknown"

    def _parse_tool_decision(self, response: str) -> ToolCall:
        """
        Parse the model's tool selection decision.
        
        Args:
            response: Model's response text
            
        Returns:
            ToolCall object with the decision
        """
        try:
            if "assistantfinal" in response:
                response = response.split("assistantfinal", 1)[1]
            if "\n</think>\n" in response:
                response = response.split("\n</think>\n", 1)[1]
            # Try to find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                data = json.loads(json_str)
                
                return ToolCall(
                    needs_tool=data.get("needs_tool", False),
                    tool_name=data.get("tool_name"),
                    tool_input=data.get("tool_input", ""),
                    reasoning=data.get("reasoning", "")
                )
            else:
                # No JSON found, assume no tool needed
                print("No JSON file found")
                return ToolCall(
                    needs_tool=False,
                    tool_name=None,
                    tool_input=None,
                    reasoning="No valid JSON response"
                )
        
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse tool decision: {e}")
            print(f"Response was: {response}")
            return ToolCall(
                needs_tool=False,
                tool_name=None,
                tool_input=None,
                reasoning=f"JSON parse error: {str(e)}"
            )


    def _format_search_results(self, search_results: Dict) -> str:
        """Format search results for inclusion in the prompt."""
        try:
            if isinstance(search_results, str):
                data = json.loads(search_results)
            else:
                data = search_results
            
            if "error" in data:
                return f"Search error: {data['error']}"
            
            results = data.get("results", [])
            if not results:
                return "No search results found."
            
            formatted = "Search Results:\n\n"
            for idx, result in enumerate(results, 1):
                title = result.get("title", "No title")
                snippet = result.get("snippet", "No description")
                link = result.get("link", "")
                
                formatted += f"{idx}. {title}\n"
                formatted += f"   {snippet}\n"
                if link:
                    formatted += f"   Source: {link}\n"
                formatted += "\n"
            
            return formatted.strip()
        
        except Exception as e:
            return f"Error formatting search results: {str(e)}"

    
    async def run(
        self,
        query: str,
        enable_tool_selection: bool = True,
        force_search: bool = False,
        override_description: str = None,
        decision_only: bool = False,
        perceived_need: bool = False,
        perceived_need_v1: bool = False,
        perceived_need_v2: bool = False
    ) -> AgentResponse:
        """
        Run the agent loop for a single query.

        Args:
            query:                 The user's query.
            enable_tool_selection: Whether to enable tool selection.
            force_search:          Skip decision loop, always call web_search.
            override_description:  Runtime tool description (e.g. budget-aware).
            decision_only:         If True, stop after the tool-selection step —
                                   do NOT execute MCP / web search and do NOT
                                   generate a final answer.  Used for budget-aware
                                   variants where we only want to record whether
                                   the model would have called the tool.
            perceived_need:           If True, run the YES/NO actual-need probe
                                   instead of the normal tool-selection flow.
                                   Returns immediately with yes_no_decision set.
        """
        tool_calls = []
        iterations = 0
        total_tokens = 0

        tool_decision = None
        tool_in_prompt_info = ''

        # ── Perceived-need YES/NO probe (main-perceived-need variant) ──────────
        if perceived_need:
            iterations += 1
            probe_prompt = self._create_perceived_need_prompt(query)
            outputs = self.llm.generate([probe_prompt], self.sampling_params)
            raw = outputs[0].outputs[0].text
            total_tokens += len(outputs[0].outputs[0].token_ids)
            parsed = self._parse_tool_decision(raw)
            tool_calls.append({
                "type": "perceived_need_probe",
                "prompt": probe_prompt,
                "raw_response": raw,
                "yes_no_decision": parsed.needs_tool,
            })
            return AgentResponse(
                tool_in_prompt_info=probe_prompt,
                final_prompt="",
                response="",
                tool_calls=tool_calls,
                iterations=iterations,
                tokens_generated=total_tokens,
            )
        if perceived_need_v1:
            iterations += 1
            probe_prompt = self._create_perceived_need_prompt_v1(query)
            outputs = self.llm.generate([probe_prompt], self.sampling_params)
            raw = outputs[0].outputs[0].text
            total_tokens += len(outputs[0].outputs[0].token_ids)
            parsed = self._parse_tool_decision(raw)
            tool_calls.append({
                "type": "perceived_need_probe",
                "prompt": probe_prompt,
                "raw_response": raw,
                "yes_no_decision": parsed.needs_tool,
            })
            return AgentResponse(
                tool_in_prompt_info=probe_prompt,
                final_prompt="",
                response="",
                tool_calls=tool_calls,
                iterations=iterations,
                tokens_generated=total_tokens,
            )
        if perceived_need_v2:
            iterations += 1
            probe_prompt = self._create_perceived_need_prompt_v2(query)
            outputs = self.llm.generate([probe_prompt], self.sampling_params)
            raw = outputs[0].outputs[0].text
            total_tokens += len(outputs[0].outputs[0].token_ids)
            parsed = self._parse_tool_decision(raw)
            tool_calls.append({
                "type": "perceived_need_probe",
                "prompt": probe_prompt,
                "raw_response": raw,
                "yes_no_decision": parsed.needs_tool,
            })
            return AgentResponse(
                tool_in_prompt_info=probe_prompt,
                final_prompt="",
                response="",
                tool_calls=tool_calls,
                iterations=iterations,
                tokens_generated=total_tokens,
            )

        # FORCE SEARCH MODE
        if force_search and self.tool_manager.has_tools():
            iterations += 1
            force_prompt = self._create_force_search_prompt(query, override_description)
            outputs = self.llm.generate([force_prompt], self.sampling_params)
            decision_response = outputs[0].outputs[0].text

            parsed = self._parse_tool_decision(decision_response)

            tool_decision = ToolCall(
                needs_tool=True,
                tool_name="web_search",
                tool_input=(parsed.tool_input or "").strip() or query,
                reasoning=parsed.reasoning or "Force search enabled"
            )

        if not force_search and enable_tool_selection and self.tool_manager.has_tools():
            iterations += 1

            tool_selection_prompt = self._create_tool_selection_prompt(query, override_description)
            tool_in_prompt_info = tool_selection_prompt

            outputs = self.llm.generate([tool_selection_prompt], self.sampling_params)
            decision_response = outputs[0].outputs[0].text
            total_tokens += len(outputs[0].outputs[0].token_ids)

            tool_decision = self._parse_tool_decision(decision_response)

            tool_calls.append({
                "type": "tool_selection",
                "decision": {
                    "needs_tool": tool_decision.needs_tool,
                    "tool_name": tool_decision.tool_name,
                    "tool_input": tool_decision.tool_input,
                    "reasoning": tool_decision.reasoning
                },
                "raw_response": decision_response
            })
        
        # ── Decision-only mode: stop here, no MCP call, no final generation ──
        if decision_only:
            return AgentResponse(
                tool_in_prompt_info=tool_in_prompt_info,
                final_prompt="",
                response="",
                tool_calls=tool_calls,
                iterations=iterations,
                tokens_generated=total_tokens,
            )

        # Step 2: Execute tool if needed
        search_context = ""
        if tool_decision and tool_decision.needs_tool and tool_decision.tool_name:
            iterations += 1
            
            if tool_decision.tool_input != "":
                tool_input = tool_decision.tool_input
            else:
                tool_input = query
            # Execute the tool
            tool_result = await self.tool_manager.execute_tool(
                tool_decision.tool_name,
                query=tool_input
            )
            
            tool_calls.append({
                "type": "tool_execution",
                "tool_name": tool_decision.tool_name,
                "tool_input": tool_decision.tool_input,
                "result": tool_result
            })
            
            # Format search results for context
            if tool_decision.tool_name == "web_search":
                search_context = self._format_search_results(tool_result)
        
        # Step 3: Generate final response (with chat template applied)
        iterations += 1
        
        # Build final prompt with or without search context
        if search_context:
            final_content = f"""Based on the following search results, please answer the question.

                {search_context}

                Question: {query}"""
        else:
            final_content = query
        
        # Apply chat template to final prompt
        messages = [
            {"role": "user", "content": final_content}
        ]
        
        # Add system message if configured
        system_msg = self.config.get_system_message(model_name=self.model_name)
        if system_msg:
            messages.insert(0, {"role": "system", "content": system_msg})
        
        final_prompt = self._apply_chat_template(messages, add_generation_prompt=True)
        
        # Generate final response
        outputs = self.llm.generate([final_prompt], self.sampling_params)
        final_response = outputs[0].outputs[0].text
        total_tokens += len(outputs[0].outputs[0].token_ids)
        
        return AgentResponse(
            tool_in_prompt_info=tool_in_prompt_info,
            final_prompt=final_prompt,
            response=final_response,
            tool_calls=tool_calls,
            iterations=iterations,
            tokens_generated=total_tokens,
        )
    
    def run_sync(self, query: str, enable_tool_selection: bool = True) -> AgentResponse:
        """Synchronous wrapper for run()."""
        return asyncio.run(self.run(query, enable_tool_selection))