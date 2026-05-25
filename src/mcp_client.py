"""
FastMCP client wrapper for connecting to MCP servers.
"""

import os
import asyncio
from typing import Optional, Dict, Any

try:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("Warning: MCP not installed. Install with: pip install mcp")


class FastMCPClient:
    """
    Client for connecting to FastMCP servers.
    """
    
    def __init__(
        self,
        server_script: str,
        provider: str = "brave",
        api_key: Optional[str] = None
    ):
        """
        Initialize FastMCP client.
        
        Args:
            server_script: Path to the MCP server script
            provider: Search provider ("google" or "brave")
            api_key: API key for the search provider
        """
        if not MCP_AVAILABLE:
            raise ImportError("MCP not available. Install with: pip install mcp")
        
        self.server_script = server_script
        self.provider = provider.lower()
        
        # Get API key
        if self.provider == "google":
            self.api_key = api_key or os.getenv("SERPAPI_API_KEY")
            if not self.api_key:
                raise ValueError("SERPAPI_API_KEY must be set")
            env_key = "SERPAPI_API_KEY"
        elif self.provider == "brave":
            self.api_key = api_key or os.getenv("BRAVE_API_KEY")
            if not self.api_key:
                raise ValueError("BRAVE_API_KEY must be set")
            env_key = "BRAVE_API_KEY"
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        self.env_key = env_key
        self.session = None
        self.stdio_context = None
    
    async def __aenter__(self):
        """Start the MCP server and establish connection."""
        # Set up server parameters
        server_params = StdioServerParameters(
            command="python",
            args=[self.server_script, self.provider],
            env={
                self.env_key: self.api_key,
                **os.environ  # Include existing env vars
            }
        )
        
        # Start stdio client
        self.stdio_context = stdio_client(server_params)
        read, write = await self.stdio_context.__aenter__()
        
        # Create session
        self.session = ClientSession(read, write)
        await self.session.__aenter__()
        
        # Initialize
        await self.session.initialize()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up MCP connection."""
        if self.session:
            try:
                await self.session.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                print(f"Warning: Error closing session: {e}")
        
        if self.stdio_context:
            try:
                await self.stdio_context.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                print(f"Warning: Error closing stdio context: {e}")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool result
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
        
        try:
            result = await self.session.call_tool(tool_name, arguments=arguments)
            # print(f'MCP server result: {result}')
            return result
        except Exception as e:
            # print(f"Error calling tool {tool_name}: {e}")
            raise
    
    async def list_tools(self) -> list:
        """List available tools from the server."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
        
        tools_list = await self.session.list_tools()
        return tools_list.tools if hasattr(tools_list, 'tools') else []


# Convenience function for quick tool calls
async def quick_search(
    query: str,
    provider: str = "brave",
    api_key: Optional[str] = None,
    server_script: str = "./mcp_server.py"
) -> str:
    """
    Quick search function for one-off searches.
    
    Args:
        query: Search query
        provider: Search provider
        api_key: API key
        server_script: Path to MCP server script
        
    Returns:
        Search results as JSON string
    """
    async with FastMCPClient(server_script, provider, api_key) as client:
        result = await client.call_tool("web_search", {"query": query, "count": 5})
        
        # Extract text from result
        if hasattr(result, 'content') and isinstance(result.content, list):
            for item in result.content:
                if hasattr(item, 'text'):
                    return item.text
        
        return str(result)