import os
import requests
from langchain.tools import tool
from tavily import TavilyClient

@tool
def get_weather(city: str) -> str:
    """Get current weather of an Indian city using wttr.in open framework."""
    try:
        url = f"https://wttr.in/{city}?format=3"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return f"Weather Update: {response.text.strip()}"
        return f"Error: Could not retrieve weather for {city} via wttr.in"
    except Exception as e:
        return f"Weather service failed: {str(e)}"


@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""
    tavily_api = os.getenv("TAVILY_API_KEY")
    if not tavily_api:
        return "Error: TAVILY_API_KEY is missing from environment secrets."
        
    tavily_client = TavilyClient(api_key=tavily_api)
    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )
    
    results = response.get("results", [])
    if not results:
        return f"No news updates found for {city}"
    
    news_list = []
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")
        news_list.append(f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}...")
    
    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)