from fastmcp import FastMCP
from langchain_community.tools import DuckDuckGoSearchRun
from serpapi import GoogleSearch
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv
import os


load_dotenv()

mcp=FastMCP("Search Tool")

#Wikipedia search tool
@mcp.tool()
def get_query_info(query:str)->str:
    """
    Fetch concise information from Wikipedia for a given student query.

    This function takes a query (usually a topic, concept, or question from a student),
    searches Wikipedia using the LangChain `WikipediaQueryRun` tool, and returns
    a summarized text result that can be used by EduRAG for explanation or reasoning.

    Parameters:
        query (str): The topic or question entered by the student.

    Returns:
        str: A concise, readable summary of the Wikipedia search result related to the query.
    """

    
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    result=wikipedia.run(query)
    return result

#SerpAPI search tool
@mcp.tool()
def fetch_web_context(query: str, num_results: int = 3):
    """
    Fetch top Google search results using SerpAPI GoogleSearch.

    Performs a real-time web search for the given query and returns a
    formatted text block containing titles, links, and snippets of the
    top results — useful for providing fresh context to EduRAG.

    Args:
        query (str): The topic or question to search.
        num_results (int): Number of results to fetch (default: 3).

    Returns:
        str: Combined search results with titles, URLs, and snippets.
    """

    params = {
        "engine": "google",
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "num": num_results
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    context = ""
    for i, res in enumerate(results.get("organic_results", []), 1):
        context += f"[Result {i}] {res.get('title')} - {res.get('link')}\n{res.get('snippet')}\n\n"
    return context

#DuckDuckGo search tool
@mcp.tool()
def get_web_serach(topic:str)->str:
    """Searches the web using DuckDuckGoSearchRun for a topic and returns results as text."""

    search_web=DuckDuckGoSearchRun()
    result=search_web.invoke(topic)
    return result


if __name__=="__main__":
    mcp.run(transport="streamable-http")