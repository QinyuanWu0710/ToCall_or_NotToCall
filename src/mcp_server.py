"""
FastMCP server implementation for web search tool.
This server provides web search capabilities via MCP protocol.
"""

import os
from typing import Optional
from fastmcp import FastMCP
from typing import Optional, Any, Dict, List


# Import search clients (Serper for Google, Brave native)
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("Warning: httpx not installed. Install with: pip install httpx")


class SearchProvider:
    """Base class for search providers."""
    
    async def search(self, query: str, count: int = 5) -> dict:
        """Perform a search and return results."""
        raise NotImplementedError
        
class SerperSearchProvider(SearchProvider):
    """Google search via SerpApi Search API.

    Docs: https://serpapi.com/search-api
    Endpoint: https://serpapi.com/search.json
    """

    def __init__(self, api_key: Optional[str] = None, *, engine: str = "google"):
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            raise ValueError("SERPAPI_API_KEY must be set")

        self.engine = engine
        self.url = "https://serpapi.com/search.json"

    async def search(
        self,
        query: str,
        count: int = 5,
        *,
        # Common SerpApi params (optional)
        google_domain: Optional[str] = None,
        gl: Optional[str] = None,   # country code (e.g., "us")
        hl: Optional[str] = None,   # language code (e.g., "en")
        location: Optional[str] = None,
        start: Optional[int] = None,     # pagination offset
        no_cache: Optional[bool] = None,
        async_search: Optional[bool] = None,  # SerpApi calls this param "async"
        **extra_params: Any,
    ) -> Dict[str, Any]:
        """Search using SerpApi.

        Returns SerpApi JSON on success, or {"error": "...", "results": []} on failure.
        """
        if not HTTPX_AVAILABLE:
            return {"error": "httpx not installed", "results": []}

        params: Dict[str, Any] = {
            "api_key": self.api_key,
            "engine": self.engine,
            "q": query,
            # SerpApi supports `num` for Google-family engines (how many results to return).
            # If the engine you use doesn't support it, SerpApi will ignore/handle it.
            "num": max(1, int(count)),
        }

        if google_domain:
            params["google_domain"] = google_domain
        if gl:
            params["gl"] = gl
        if hl:
            params["hl"] = hl
        if location:
            params["location"] = location
        if start is not None:
            params["start"] = int(start)
        if no_cache is not None:
            params["no_cache"] = "true" if no_cache else "false"
        if async_search is not None:
            # SerpApi parameter name is literally "async"
            params["async"] = "true" if async_search else "false"

        # Allow passing through any other SerpApi params (tbs, safe, tbm, etc.)
        params.update(extra_params)

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(self.url, params=params, timeout=30.0)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                return {"error": str(e), "results": []}

class BraveSearchProvider(SearchProvider):
    """Brave Search API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("BRAVE_API_KEY")
        if not self.api_key:
            raise ValueError("BRAVE_API_KEY must be set")
        
        self.url = "https://api.search.brave.com/res/v1/web/search"
    
    async def search(self, query: str, count: int = 5) -> dict:
        """Search using Brave API."""
        if not HTTPX_AVAILABLE:
            return {"error": "httpx not installed"}
        
        headers = {
            "X-Subscription-Token": self.api_key,
            "Accept": "application/json"
        }
        
        params = {
            "q": query,
            "count": count
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    self.url,
                    params=params,
                    headers=headers,
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                return {"error": str(e), "results": []}


def create_search_server(provider: str = "brave", api_key: Optional[str] = None) -> FastMCP:
    """
    Create a FastMCP server with web search capability.
    
    Args:
        provider: Search provider ("google" or "brave")
        api_key: API key for the provider
    
    Returns:
        FastMCP server instance
    """
    # Create FastMCP instance
    mcp = FastMCP("Web Search Server")
    
    # Initialize search provider
    if provider.lower() == "google":
        search_client = SerperSearchProvider(api_key)
    elif provider.lower() == "brave":
        search_client = BraveSearchProvider(api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    @mcp.tool()
    async def web_search(query: str, count: int = 5) -> str:
        """
        Search the web for information.
        
        Args:
            query: The search query string
            count: Number of results to return (default: 5)
        
        Returns:
            JSON string containing search results
        """
        import json
        
        results = await search_client.search(query, count)
        
        # Format results for the model
        if "error" in results:
            return json.dumps({
                "error": results["error"],
                "results": []
            })
        
        # Parse based on provider
        formatted_results = []
        
        if provider.lower() == "google":
            # Serper format
            organic_results = results.get("organic_results", [])
            for item in organic_results[:count]:
                formatted_results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", "")
                })
        
        elif provider.lower() == "brave":
            # Brave format
            web_results = results.get("web", {}).get("results", [])
            for item in web_results[:count]:
                formatted_results.append({
                    "title": item.get("title", ""),
                    "link": item.get("url", ""),
                    "snippet": item.get("description", "")
                })
        
        return json.dumps({
            "query": query,
            "results": formatted_results,
            "total_results": len(formatted_results)
        })
    
    return mcp


# For running the server standalone
if __name__ == "__main__":
    import sys
    
    # Get provider from command line or default to brave
    provider = sys.argv[1] if len(sys.argv) > 1 else "brave"
    
    # Create and run server
    mcp = create_search_server(provider=provider)
    mcp.run()