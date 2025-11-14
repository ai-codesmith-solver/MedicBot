from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
from utils import get_llm
from langgraph.prebuilt import create_react_agent
from langchain_core.output_parsers import StrOutputParser


gemini_llm=get_llm()

str_parse=StrOutputParser()




async def mian_agent(query:str):

    client=MultiServerMCPClient(
        {
            "Search Tool": {
            "url": "http://localhost:8000/mcp",  # Ensure server is running here
            "transport": "streamable_http",
        },
        }
    )

    tools=await client.get_tools()


    search_agent=create_react_agent(
        model=gemini_llm,
        tools=tools,
        prompt="""
        You are a tool-orchestration agent whose ONLY job is to fetch data from THREE tools 
        and return the raw results combined together EXACTLY as provided.

        You MUST ALWAYS call all three tools for every user query:
        1. WikipediaQueryRun  → 'get_query_info'
        2. DuckDuckGoSearchRun → 'get_web_serach'
        3. SerpAPI GoogleSearch → 'fetch_web_context'

        This is mandatory. Never skip any tool.

        --------------------------------------
        ### TOOL EXECUTION RULES (STRICT)
        - First → Call: get_query_info(query)
        - Second → Call: get_web_serach(query)
        - Third → Call: fetch_web_context(query)

        You must call the tools in this exact order.

        --------------------------------------
        ### OUTPUT RULES (STRICT)
        After all three tools have been executed:
        - Combine the three outputs.
        - Do NOT summarize.
        - Do NOT edit.
        - Do NOT shorten.
        - Do NOT explain.
        - Do NOT add your own text.
        - Do NOT add medical reasoning.
        - Do NOT modify tool results in any way.

        Return ONLY this final structure:

        [WIKIPEDIA RESULT]
        <raw result from get_query_info>

        [DUCKDUCKGO RESULT]
        <raw result from get_web_serach>

        [SERPAPI RESULT]
        <raw result from fetch_web_context>

        --------------------------------------
        ### IMPORTANT
        You must strictly follow:
        THINK → TOOL 1 → TOOL 2 → TOOL 3 → RETURN COMBINED RAW OUTPUT.

        Do NOT answer early.
        Do NOT refuse.
        Do NOT generate your own content.

        Begin.
        """
    )


    response = await search_agent.ainvoke(
        {"messages": [{"role": "user", "content": query}]}
        )
    bot_message = str_parse.parse(response['messages'][-1].content)
    
    return bot_message

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


def get_extra_context(query:str):
    result = loop.run_until_complete(mian_agent(query))
    return result

# query="dermatologists in Siliguri"
# result=get_extra_context(query)
# print(result)






    
