import asyncio
import json
import logging
import litellm

from textwrap import dedent
from pydantic import BaseModel
from dotenv import load_dotenv
from rich import print
from litellm import acompletion
from tools import read_repo_function, search_repo
from data_types import Function
from langchain_core.utils.function_calling import convert_to_openai_function
from litellm.caching.caching import LiteLLMCacheType, Cache

load_dotenv()

litellm.cache = Cache(type=LiteLLMCacheType.DISK)

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )

class FinalAnswer(BaseModel):
    answer: str
    functions: list[str]
    
MAX_TURNS = 10

async def run_agent(repo: str,input: str) -> FinalAnswer | None:
    SYSTEM_PROMPT = dedent(f"""
        You are a github repo searcher. You will be given a question about the code within the repo.
        You will use the tools provided to search the repo and read functions to answer the question.
        You may operate for up to {MAX_TURNS}, so if your first search doesn't find the answer, you can use different keywords.
    """)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input},
    ]
    
    def search_functions(keywords: list[str]) -> list[dict]:
        """
            Search the repo for functions that match the keywords.
            Return the functions in a list of dictionaries so the LLM can use them.
        """
        return search_repo(repo, keywords)

    def return_answer(answer: str, functions: list[str]) -> FinalAnswer:
        """
            Return the answer and the functions used to answer the question.
        """
        return FinalAnswer(answer=answer, functions=functions)

    def read_function(func_path: str, func_name: str) -> Function:
        """
            Read a function from the repo.
        """
        return read_repo_function(repo, func_path, func_name)
    
    tools = [search_functions, read_function, return_answer]
    tools_by_name = {tool.__name__: tool for tool in tools}
    tools = [
        {
            "type": "function",
            "function": convert_to_openai_function(search_functions)
        },
        {
            "type": "function",
            "function": convert_to_openai_function(read_function)
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
        if response.choices[0] is None:
            print("Response message is None")
        
        messages.append(
            response_message
        )
        
        # Terminate early. We always want tool calls. This indicates an issue.
        if response_message.tool_calls is None:
            logging.error(f"Response message has no tool calls for turn {turns}")
            return None
        
        print(f"RESPONSE: {response}")
        for tool_call in response.choices[0].message.tool_calls:
            tool_name: str = tool_call.function.name # type: ignore
            if tool_name in tools_by_name:
                print(f"===Calling tool {tool_name} on turn {turns}===")
                tool_args = json.loads(tool_call.function.arguments)
                tool_to_call = tools_by_name[tool_name]
                print(f"TOOL TO CALL: {tool_to_call}, ARGS: {tool_args}")
                tool_result = tool_to_call(**tool_args)
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
    answer = asyncio.run(run_agent("deepmind/sonnet", "How can I retrieve all trainable TensorFlow variables defined within a specific Sonnet module?"))
    print(f"Answer: {answer}")