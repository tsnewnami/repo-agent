import asyncio
import json
import logging
import litellm
import art 

from textwrap import dedent
from pydantic import BaseModel
from dotenv import load_dotenv
from rich import print
from litellm import acompletion
from tools import search_issues, read_issue
from langchain_core.utils.function_calling import convert_to_openai_function
from litellm.caching.caching import LiteLLMCacheType, Cache
# from art.utils.litellm import convert_litellm_choice_to_openai

load_dotenv()

litellm.cache = Cache(type=LiteLLMCacheType.DISK)

class FinalAnswer(BaseModel):
    answer: str
    issues: list[str]
    
MAX_TURNS = 10

async def run_agent(repo: str,input: str) -> FinalAnswer | None:
    SYSTEM_PROMPT = dedent(f"""
        You are a github repo issue seacher. You are given a repo and a question.
        You will use the tools provided to search the repo for issues that match the question.
        You may operate for up to {MAX_TURNS}, so if your first search doesn't find the answer, you can use different keywords.

        The repo to search is {repo}.
    """)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input},
    ]
    
    def search_repo(keywords: list[str]) -> list[dict]:
        """
            Search the repo for issues that match the keywords.
            Return the issues in a list of dictionaries so the LLM can use them.
        """
        return search_issues(repo, keywords)

    def return_answer(answer: str, issues: list[str]) -> FinalAnswer:
        """
            Return the answer and the Issue #'s used to answer the question.
        """
        return FinalAnswer(answer=answer, issues=issues)
    
    tools = [search_repo, read_issue, return_answer]
    tools_by_name = {tool.__name__: tool for tool in tools}
    tools = [
        {
            "type": "function",
            "function": convert_to_openai_function(search_repo)
        },
        {
            "type": "function",
            "function": convert_to_openai_function(read_issue)
        },
        {
            "type": "function",
            "function": convert_to_openai_function(return_answer)
        }
    ]

    turns = 0
    logging.info(f"Running agent with input: {input}")
    while turns < MAX_TURNS:
        logging.info(f"Turn {turns + 1}:")
        response = await acompletion(        
            model="gpt-4.1",
            messages=messages,
            tools=tools,
            caching=True,
        )

        response_message = response.choices[0].message
        print(f"RESPONSE: {response_message}")
        if response.choices[0] is None:
            print("Response message is None")
        
        messages.append(
            response_message
        )
        
        # Terminate early. We always want tool calls.
        if response_message.tool_calls is None:
            print("Response message has no tool calls")
            return None
        
        # print(f"RESPONSE: {response}")
        for tool_call in response.choices[0].message.tool_calls:
            tool_name: str = tool_call.function.name # type: ignore
            if tool_name in tools_by_name:
                print(f"===Calling tool {tool_name} on turn {turns}===")
                tool_args = json.loads(tool_call.function.arguments)
                tool_to_call = tools_by_name[tool_name]
                # print(f"TOOL TO CALL: {tool_to_call}, ARGS: {tool_args}")
                tool_result = tool_to_call(**tool_args)
                # print(f"TOOL RESULT: {tool_result}")
                tool_result_str = str(tool_result)
                print(f"TOOL RESULT: {str(tool_result)}")
                if tool_result_str is None:
                    print(f"TOOL RESULT IS NONE")
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": str(tool_result)}
                )

                if tool_name == "return_answer":
                    return tool_result
        turns += 1
        
    
    return None 

if __name__ == "__main__":
    answer = asyncio.run(run_agent("pytorch/pytorch", "Tell me the scenarios where there were memory leaks"))
    print(f"Answer: {answer}")