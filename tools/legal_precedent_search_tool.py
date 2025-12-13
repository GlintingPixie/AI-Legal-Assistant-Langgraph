import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def legal_precedent_search_tool(query: str) -> list[dict]:
    """
    Search the web for Indian legal precedents related to IPC sections or case facts.
    """
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=5
    )

    results = []
    for item in response.get("results", []):
        results.append({
            "title": item.get("title"),
            "url": item.get("url"),
            "content": item.get("content")
        })

    return results
