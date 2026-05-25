"""
Configuration file for MCP agent settings, prompts, and tool definitions.
This file makes it easy to modify the agent's behavior without touching core logic.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ToolConfig:
    """Configuration for a single tool."""
    name: str
    description: str
    enabled: bool = True


class AgentConfig:
    """
    Central configuration for the MCP agent behavior.
    Modify these settings to change how the agent selects and uses tools.
    """
    
    # ==================== Tool Selection Settings ====================
    
    TOOL_SELECTION_PROMPT = """You are an intelligent agent that decides when to use tools to answer questions.

You have access to the following tools:
{tool_list}

Given the user's question: "{query}"

Decide if you need to use any tools. Respond with a JSON object:
{{
    "needs_tool": true/false,
    "tool_name": "tool_name" or null,
    "tool_input": "the input you need to give to the tool"
    "reasoning": "why you need this tool or why you don't need tools"
}}

Rules:
- Only use tools when you genuinely need external information
- If you already know the answer, set needs_tool to false
- Only select ONE tool at a time"""

    FORCE_TOOL_PROMPT = """You MUST use the web_search tool.

Available tools:
{tool_list}

User question: "{query}"

Respond with ONLY a JSON object in this exact schema:
{{
  "needs_tool": true,
  "tool_name": "web_search",
  "tool_input": "<a short, effective web search query>",
  "reasoning": "<brief>"
}}

Rules:
- needs_tool must be true
- tool_name must be exactly "web_search"
- tool_input must not be empty
"""

    # ==================== Entity Query Settings ====================
    
    ENTITY_QUERY_PROMPT_TEMPLATE = "In a paragraph, could you tell me what you know about {entity}?"
    
    # System message for the model when generating responses
    SYSTEM_MESSAGE = """
                    You are a helpful assistant.
                    """ 
    SYSTEM_MESSAGE_GPT_LOW = """Reasoning: low"""
    SYSTEM_MESSAGE_GPT_HIGH = """Reasoning: high"""

    # Reasoning effort level for gpt-oss-20b models ("low" or "high")
    _reasoning_effort: str = "low"  # default

    @classmethod
    def set_reasoning_effort(cls, effort: str) -> None:
        """Set the reasoning effort level ('low' or 'high') for gpt-oss-20b models."""
        if effort not in ("low", "high"):
            raise ValueError(f"reasoning_effort must be 'low' or 'high', got '{effort}'")
        cls._reasoning_effort = effort

    @classmethod
    def get_reasoning_effort(cls) -> str:
        return cls._reasoning_effort

    @classmethod
    def get_system_message(cls, model_name: str = "") -> str:
        """
        Return the appropriate system message.
        For gpt models, picks the low/high reasoning variant.
        """
        if "gpt" in model_name.lower():
            if cls._reasoning_effort == "high":
                return cls.SYSTEM_MESSAGE_GPT_HIGH
            return cls.SYSTEM_MESSAGE_GPT_LOW
        return cls.SYSTEM_MESSAGE

    # ==========================SYSTEM PROMPT-1 ============================
    # SYSTEM_MESSAGE = """You are a helpful assistant. You have to generate the response based on the most timely information."""


    # ==========================SYSTEM PROMPT-2 ============================
    # SYSTEM_MESSAGE = """You are a helpful assistant. You have to generate the correct response, otherwise you'll be punished for 10000$."""
    
    # ==================== Tool Description Variants ====================
    # Pre-defined tool descriptions for controlled experiments.
    # Select a variant via --tool-description <key> in the CLI.
    # Results will be saved to a subfolder named after the key.

    # Total budget shared across all budget-aware variants (dollars)
    BUDGET_TOTAL: int = 10000

    # Shared dynamic suffix template appended to every *-budget-aware description.
    # Placeholders are filled at runtime by get_budget_aware_description().
    _BUDGET_SUFFIX: str = (
        " You have a budget of $10000 in total. "
        "You have {total_num} questions to answer in total, "
        "you have already solved {finish_num} questions, "
        "and you have called the tool {call_num} times so far."
    )

    # V2 suffix: pre-computes remaining budget and remaining questions so the
    # model does not need to do arithmetic.  Extra placeholders {budget_remaining}
    # and {questions_remaining} are filled by get_budget_aware_description().
    _BUDGET_SUFFIX_V2: str = (
        " You have a budget of $10000 in total. "
        "You have {total_num} questions to answer in total. "
        "You have already solved {finish_num} questions, "
        "and you have called the tool {call_num} times so far, "
        "so you have ${budget_remaining} left "
        "and {questions_remaining} remaining questions to answer."
    )

    TOOL_DESCRIPTIONS: Dict[str, str] = {
        # ── static variants ───────────────────────────────────────────────────
        "main": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have."
        ),
        "tool-cost-0": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $0."
        ),
        "tool-cost-10": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $10."
        ),
        "tool-cost-100": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $100."
        ),
        "tool-cost-1000": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $1000."
        ),
        "tool-cost-10000": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $10000."
        ),
        "tool-cost-10000-aware": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $10000. Please be careful about the total cost of tool calling."
        ),
        "tool-cost-cheap": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call is very cheap."
        ),
        "tool-cost-expensive": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call is very expensive."
        ),
        # ── budget-aware variants (contain {total_num}/{finish_num}/{call_num}) ─
        # These are base templates; call get_budget_aware_description(key, ...)
        # to get the fully rendered string with runtime counters filled in.
        "tool-cost-0-budget-aware": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $0."
            "{_budget_suffix}"
        ),
        "tool-cost-10-budget-aware": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $10."
            "{_budget_suffix}"
        ),
        "tool-cost-100-budget-aware": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $100."
            "{_budget_suffix}"
        ),
        "tool-cost-1000-budget-aware": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $1000."
            "{_budget_suffix}"
        ),
        "tool-cost-10000-budget-aware": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $10000."
            "{_budget_suffix}"
        ),
        "tool-cost-cheap-budget-aware": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call is very cheap."
            "{_budget_suffix}"
        ),
        "tool-cost-expensive-budget-aware": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call is very expensive."
            "{_budget_suffix}"
        ),
        "tool-cost-20-budget-aware": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $20."
            "{_budget_suffix}"
        ),
        "tool-cost-25-budget-aware": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $25."
            "{_budget_suffix}"
        ),
        "tool-cost-29-budget-aware": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $29."
            "{_budget_suffix}"
        ),
        "tool-cost-33-budget-aware": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $33."
            "{_budget_suffix}"
        ),
        "tool-cost-40-budget-aware": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $40."
            "{_budget_suffix}"
        ),
        "tool-cost-50-budget-aware": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $50."
            "{_budget_suffix}"
        ),
        "tool-cost-67-budget-aware": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $67."
            "{_budget_suffix}"
        ),
        "tool-cost-200-budget-aware": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $200."
            "{_budget_suffix}"
        ),
        "tool-cost-222-budget-aware": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $222."
            "{_budget_suffix}"
        ),
        "tool-cost-250-budget-aware": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $250."
            "{_budget_suffix}"
        ),
        "tool-cost-500-budget-aware": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $500."
            "{_budget_suffix}"
        ),
        # ── budget-aware-v2: pre-computed remaining budget & questions ─────────
        # Same as *-budget-aware but the suffix already contains the computed
        # ${budget_remaining} and {questions_remaining} values, so the model
        # does not need to perform arithmetic itself.
        "tool-cost-0-budget-aware-v2": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $0."
            "{_budget_suffix_v2}"
        ),
        "tool-cost-10-budget-aware-v2": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $10."
            "{_budget_suffix_v2}"
        ),
        "tool-cost-20-budget-aware-v2": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $20."
            "{_budget_suffix_v2}"
        ),
        "tool-cost-25-budget-aware-v2": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $25."
            "{_budget_suffix_v2}"
        ),
        "tool-cost-29-budget-aware-v2": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $29."
            "{_budget_suffix_v2}"
        ),
        "tool-cost-33-budget-aware-v2": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $33."
            "{_budget_suffix_v2}"
        ),
        "tool-cost-40-budget-aware-v2": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $40."
            "{_budget_suffix_v2}"
        ),
        "tool-cost-50-budget-aware-v2": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $50."
            "{_budget_suffix_v2}"
        ),
        "tool-cost-67-budget-aware-v2": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $67."
            "{_budget_suffix_v2}"
        ),
        "tool-cost-100-budget-aware-v2": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $100."
            "{_budget_suffix_v2}"
        ),
        "tool-cost-200-budget-aware-v2": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $200."
            "{_budget_suffix_v2}"
        ),
        "tool-cost-222-budget-aware-v2": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $222."
            "{_budget_suffix_v2}"
        ),
        "tool-cost-250-budget-aware-v2": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $250."
            "{_budget_suffix_v2}"
        ),
        "tool-cost-500-budget-aware-v2": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $500."
            "{_budget_suffix_v2}"
        ),
        "tool-cost-1000-budget-aware-v2": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $1000."
            "{_budget_suffix_v2}"
        ),
        "tool-cost-10000-budget-aware-v2": (
            "Search the web for current information about entities, facts, or topics. "
            "Use this when you need up-to-date or factual information you don't have. "
            "Each tool call costs $10000."
            "{_budget_suffix_v2}"
        ),
        # Add more variants here as needed
        # ── actual-need variants ──────────────────────────────────────────────
        # No tool description is injected; the model is simply asked YES/NO
        # whether it needs external help.  See AgentLoop for the prompt logic.
        "main-perceived-need": "",
        "main-perceived-need-v1": "",
        "main-perceived-need-v2": "",
    }

    # Default description key used when none is specified on the CLI
    DEFAULT_TOOL_DESCRIPTION_KEY: str = "main"

    @classmethod
    def is_budget_aware_key(cls, key: str) -> bool:
        """Return True if the given key is a budget-aware variant (v1 or v2)."""
        return key.endswith("-budget-aware") or key.endswith("-budget-aware-v2")

    @classmethod
    def _extract_tool_cost(cls, key: str) -> int:
        """
        Parse the numeric tool cost from a budget-aware key.
        E.g. 'tool-cost-100-budget-aware-v2' -> 100.
        Returns 0 if no numeric cost is found (e.g. cheap/expensive variants).
        """
        import re
        m = re.search(r"tool-cost-(\d+)-budget-aware", key)
        return int(m.group(1)) if m else 0

    @classmethod
    def is_perceived_need_key(cls, key: str) -> bool:
        """Return True if the key activates the YES/NO actual-need probe (original)."""
        return key == "main-perceived-need"

    @classmethod
    def is_perceived_need_v1_key(cls, key: str) -> bool:
        """Return True if the key activates the v1 YES/NO actual-need probe."""
        return key == "main-perceived-need-v1"

    @classmethod
    def is_perceived_need_v2_key(cls, key: str) -> bool:
        """Return True if the key activates the v2 YES/NO actual-need probe."""
        return key == "main-perceived-need-v2"

    @classmethod
    def get_budget_aware_description(
        cls,
        key: str,
        total_num: int,
        finish_num: int,
        call_num: int,
    ) -> str:
        """
        Render a budget-aware description template with runtime counters.

        For *-budget-aware keys (v1): injects raw counters and lets the model
        compute remaining budget itself.

        For *-budget-aware-v2 keys: pre-computes budget_remaining and
        questions_remaining so the model does not need to do arithmetic.

        Args:
            key:        A *-budget-aware or *-budget-aware-v2 key.
            total_num:  Total number of questions in the run.
            finish_num: Questions already fully answered before this one.
            call_num:   Total web_search calls made so far across all questions.

        Returns:
            Fully rendered description string ready to inject into a prompt.
        """
        if key not in cls.TOOL_DESCRIPTIONS:
            raise ValueError(f"Unknown tool description key: '{key}'")
        if not cls.is_budget_aware_key(key):
            raise ValueError(
                f"'{key}' is not a budget-aware variant. "
                "Budget-aware keys must end with '-budget-aware' or '-budget-aware-v2'."
            )
        template = cls.TOOL_DESCRIPTIONS[key]

        if key.endswith("-budget-aware-v2"):
            tool_cost = cls._extract_tool_cost(key)
            budget_remaining = max(0, cls.BUDGET_TOTAL - call_num * tool_cost)
            questions_remaining = max(0, total_num - finish_num)
            rendered_suffix = cls._BUDGET_SUFFIX_V2.format(
                total_num=total_num,
                finish_num=finish_num,
                call_num=call_num,
                budget_remaining=budget_remaining,
                questions_remaining=questions_remaining,
            )
            return template.replace("{_budget_suffix_v2}", rendered_suffix)
        else:
            rendered_suffix = cls._BUDGET_SUFFIX.format(
                total_num=total_num,
                finish_num=finish_num,
                call_num=call_num,
            )
            return template.replace("{_budget_suffix}", rendered_suffix)

    # ==================== Tool Definitions ====================

    # Base tool list — description is filled in at runtime via get_available_tools()
    _BASE_TOOLS: List[ToolConfig] = [
        ToolConfig(
            name="web_search",
            description="__placeholder__",   # overwritten at runtime
            enabled=True
        ),
        # Add more tools here as needed
        # ToolConfig(
        #     name="calculator",
        #     description="Perform mathematical calculations",
        #     enabled=False
        # ),
    ]

    # Active description key — set via set_tool_description_key() or CLI
    _active_tool_description_key: str = "main"  # must match DEFAULT_TOOL_DESCRIPTION_KEY

    @classmethod
    def set_tool_description_key(cls, key: str) -> None:
        """Select which tool description variant to use."""
        if key not in cls.TOOL_DESCRIPTIONS:
            available = ", ".join(cls.TOOL_DESCRIPTIONS.keys())
            raise ValueError(
                f"Unknown tool description key '{key}'. "
                f"Available keys: {available}"
            )
        cls._active_tool_description_key = key

    @classmethod
    def get_active_description_key(cls) -> str:
        """Return the currently active tool description key."""
        return cls._active_tool_description_key

    @classmethod
    def get_available_tools(cls, override_description: Optional[str] = None) -> List[ToolConfig]:
        """Return tools with the currently selected description injected.

        Args:
            override_description: If provided, use this string as the web_search
                                  description instead of the stored template.
                                  For budget-aware keys, always pass the output of
                                  get_budget_aware_description() here so the raw
                                  {_budget_suffix} placeholder is never forwarded
                                  into a prompt .format() call.
        """
        if override_description is not None:
            active_desc = override_description
        else:
            raw = cls.TOOL_DESCRIPTIONS[cls._active_tool_description_key]
            # Safety guard: budget-aware templates contain {_budget_suffix} which
            # would break prompt .format() calls downstream.  Callers that use a
            # budget-aware key must always supply override_description; if they
            # forgot, surface a clear error instead of a silent KeyError later.
            if cls.is_budget_aware_key(cls._active_tool_description_key) and (
                "{_budget_suffix}" in raw or "{_budget_suffix_v2}" in raw
            ):
                raise RuntimeError(
                    f"Budget-aware key '{cls._active_tool_description_key}' used without "
                    "supplying override_description. Call get_budget_aware_description() "
                    "first and pass the result as override_description."
                )
            active_desc = raw
        tools = []
        for base in cls._BASE_TOOLS:
            tools.append(
                ToolConfig(
                    name=base.name,
                    description=active_desc if base.name == "web_search" else base.description,
                    enabled=base.enabled,
                )
            )
        return tools

    # Keep AVAILABLE_TOOLS as a property alias for backward compatibility
    @classmethod
    def _get_available_tools_compat(cls) -> List[ToolConfig]:
        return cls.get_available_tools()
    
    # ==================== Search Settings ====================
    
    SEARCH_SETTINGS = {
        "count": 5,  # Number of search results
        "retries": 3,  # Retry attempts on failure
        "timeout": 30,  # Timeout in seconds
    }
    
    # ==================== Generation Settings ====================
    
    # Default generation parameters
    DEFAULT_MAX_TOKENS = 512
    DEFAULT_TEMPERATURE = 1.0
    DEFAULT_TOP_P = 1.0
    
    # Token limits for specific model families
    MODEL_TOKEN_OVERRIDES = {
        "command-r": 1024,  # Command R models get more tokens
    }
    
    # ==================== Agent Loop Settings ====================
    
    MAX_ITERATIONS = 3  # Maximum number of agent loop iterations
    ENABLE_MULTI_TOOL = False  # Allow multiple tool calls per query
    
    # ==================== Output Settings ====================
    
    SAVE_EVERY = 1  # Checkpoint frequency
    DEFAULT_OUTPUT_DIR = "./results"
    # DEFAULT_OUTPUT_DIR = "./results"
    
    @classmethod
    def get_enabled_tools(cls, override_description: Optional[str] = None) -> List[ToolConfig]:
        """Get list of enabled tools (uses active description variant)."""
        return [t for t in cls.get_available_tools(override_description) if t.enabled]

    @classmethod
    def get_tool_list_string(cls, override_description: Optional[str] = None) -> str:
        """Format enabled tools as a string for prompts."""
        tools = cls.get_enabled_tools(override_description)
        if not tools:
            return "No tools available."
        return "\n".join(f"- {t.name}: {t.description}" for t in tools)

    @classmethod
    def get_tool_by_name(cls, name: str, override_description: Optional[str] = None) -> Optional[ToolConfig]:
        """Get a tool configuration by name (uses active description variant)."""
        for tool in cls.get_available_tools(override_description):
            if tool.name == name and tool.enabled:
                return tool
        return None

    @classmethod
    def format_tool_selection_prompt(cls, query: str, override_description: Optional[str] = None) -> str:
        """Format the tool selection prompt with current query and tool list.

        Uses str.replace instead of .format() so that curly braces inside the
        rendered tool description (e.g. budget counters) never cause a KeyError.
        """
        tool_list_str = cls.get_tool_list_string(override_description)
        return (
            cls.TOOL_SELECTION_PROMPT
            .replace("{tool_list}", tool_list_str)
            .replace("{query}", query)
        )

    @classmethod
    def format_tool_force_prompt(cls, query: str, override_description: Optional[str] = None) -> str:
        """Format the force-search prompt with current query and tool list."""
        tool_list_str = cls.get_tool_list_string(override_description)
        return (
            cls.FORCE_TOOL_PROMPT
            .replace("{tool_list}", tool_list_str)
            .replace("{query}", query)
        )
    
    @classmethod
    def format_entity_query(cls, entity: str) -> str:
        """Format the entity query prompt."""
        return cls.ENTITY_QUERY_PROMPT_TEMPLATE.format(entity=entity)
    
    @classmethod
    def get_max_tokens_for_model(cls, model_name: str) -> int:
        """Get max tokens for a specific model."""
        for key, value in cls.MODEL_TOKEN_OVERRIDES.items():
            if key.lower() in model_name.lower():
                return value
        return cls.DEFAULT_MAX_TOKENS