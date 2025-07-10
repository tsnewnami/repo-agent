import asyncio
import json
import logging
from dotenv import load_dotenv
from rich import print
from litellm import acompletion
from tools import search_issues, read_issue
from langchain_core.utils.function_calling import convert_to_openai_function

load_dotenv()


async def run_agent(repo: str,input: str) -> str:
    SYSTEM_PROMPT = """
        Use the tools provided to answer the user's question. Use the tools to answer the question.
    """
    MAX_TURNS = 1

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input},
    ]
    
    def search_repo(keywords: list[str]) -> list[dict]:
        """
            Search the repo for .
        """
        return search_issues(repo, keywords)
    
    tools = [search_repo, read_issue]
    tools_by_name = {tool.__name__: tool for tool in tools}
    tools = [
        {
            "type": "function",
            "function": convert_to_openai_function(search_repo)
        },
        # {
        #     "type": "function",
        #     "function": convert_to_openai_function(read_issue)
        # }
    ]

    turns = 0
    logging.info(f"Running agent with input: {input}")
    while turns < MAX_TURNS:
        logging.info(f"Turn {turns + 1}:")
        response = await acompletion(        
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
        )

        print(response)
        messages.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )
        
        for tool_call in response.choices[0].message.tool_calls:
            tool_name: str = tool_call.function.name # type: ignore
            if tool_name in tools_by_name:
                tool_args = json.loads(tool_call.function.arguments)
                tool_to_call = tools_by_name[tool_name]
                tool_result = tool_to_call(**tool_args)
                print(f"TOOL RESULT: {tool_result}")
                messages.append(
                    {"role": "tool", "content": tool_result}
                )
        turns += 1
        
        return messages[-1]["content"]
    

if __name__ == "__main__":
    asyncio.run(run_agent("pytorch/pytorch", "When was there tensor issues using cuda?"))