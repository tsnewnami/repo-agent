import asyncio
import json
import logging
import art
import litellm
import wandb

from textwrap import dedent
from pydantic import BaseModel
from dotenv import load_dotenv
from rich import print
from litellm import acompletion
import weave
from judge import judge_answer

from load_data import load_scenarios
from tools import read_repo_function, search_repo
from data_types import Function, Scenario
from langchain_core.utils.function_calling import convert_to_openai_function
from litellm.caching.caching import LiteLLMCacheType, Cache
from art.utils.litellm import convert_litellm_choice_to_openai

load_dotenv()

litellm.cache = Cache(type=LiteLLMCacheType.DISK)

# weave.init("side-project/agent-benchmark")

# Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )


class FinalAnswer(BaseModel):
    answer: str
    functions: list[str]


class ProjectTrajectory(art.Trajectory):
    answer: FinalAnswer | None = None


MAX_TURNS = 10


async def run_agent(model: art.Model, repo: str, question: str) -> ProjectTrajectory:
    trajectory = ProjectTrajectory(reward=0.0, messages_and_choices=[])

    SYSTEM_PROMPT = dedent(
        f"""
        You are a github repo searcher. You will be given a question about the code within the repo.
        You will use the tools provided to search the repo and read functions to answer the question.
        You may operate for up to {MAX_TURNS}, so if your first search doesn't find the answer, you can use different keywords.
    """
    )

    trajectory.messages_and_choices = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    def search_functions(keywords: list[str]) -> list[dict]:
        """
        Search the repo for functions that match the keywords.
        Return the functions in a list of dictionaries so the LLM can use them.
        """
        return search_repo(repo, keywords)

    def read_function(func_path: str, func_name: str) -> Function:
        """
        Read a function from the repo.
        """
        return read_repo_function(repo, func_path, func_name)

    def return_answer(answer: str, functions: list[str]) -> FinalAnswer:
        """
        Return the answer and the functions used to answer the question.
        """
        return FinalAnswer(answer=answer, functions=functions)

    tools = [search_functions, read_function, return_answer]
    tools_by_name = {tool.__name__: tool for tool in tools}
    trajectory.tools = [
        {"type": "function", "function": convert_to_openai_function(search_functions)},
        {"type": "function", "function": convert_to_openai_function(read_function)},
        {"type": "function", "function": convert_to_openai_function(return_answer)},
    ]

    if model.trainable:
        litellm_model_name = f"hosted_vllm/{model.name}"
    else:
        litellm_model_name = model.name

    turns = 0
    logging.info(f"Running agent with input: {question}")
    while turns < MAX_TURNS:
        logging.info(f"Turn {turns + 1}:")
        response = await acompletion(
            model=litellm_model_name,
            base_url=model.inference_base_url,
            api_key=model.inference_api_key,
            temperature=1,
            messages=trajectory.messages(),
            tools=trajectory.tools,
            caching=False,
        )

        response_message = response.choices[0].message
        if response.choices[0] is None:
            logging.error(f"Response message is None for turn {turns}")

        trajectory.messages_and_choices.append(
            convert_litellm_choice_to_openai(response.choices[0])
        )

        # Terminate early. We always want tool calls. This indicates an issue.
        if response_message.tool_calls is None:
            logging.error(f"Response message has no tool calls for turn {turns}")
            return trajectory

        try:
            for tool_call in response.choices[0].message.tool_calls:
                tool_name: str = tool_call.function.name  # type: ignore
                if tool_name in tools_by_name:
                    tool_args = json.loads(tool_call.function.arguments)
                    tool_to_call = tools_by_name[tool_name]
                    tool_result = tool_to_call(**tool_args)
                    logging.info(f"Tool {tool_name} called with args {tool_args}")
                    logging.info(f"Tool result: {tool_result}")
                    trajectory.messages_and_choices.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": (
                                "" if tool_result is None else str(tool_result)
                            ),
                        }
                    )

                    if tool_name == "return_answer":
                        trajectory.answer = tool_result
                        logging.info(f"Returning answer: {tool_result}")
                        return trajectory

        except Exception as e:
            logging.error(f"Error in tool call: {e}")
            return trajectory

        turns += 1

    return trajectory


@weave.op()
async def run_agent_and_score(
    model: art.Model, scenario: Scenario
) -> ProjectTrajectory:
    trajectory = await run_agent(model, scenario.repo, scenario.question)
    if trajectory.answer is None:
        logging.warning(
            f"Agent could not find an answer for scenario {scenario.question}"
        )
        return trajectory

    score = await judge_answer(scenario.question, scenario.answer, trajectory.answer)
    trajectory.reward = float(score.is_correct)

    return trajectory


if __name__ == "__main__":
    model_name = "openrouter/qwen/qwen3-32b"

    scenarios = load_scenarios(
        "JamesSED/synthetic_QA_code_search_net", split="train", limit=1000, shuffle=False
    )
    model = art.Model(name=model_name, project="rl-agent")
    step = 0
    scores = []
    for scenario in scenarios:
        print("--------------------------------")
        print(f"Step {step}: {scenario.question}")
        score = asyncio.run(run_agent_and_score(model, scenario))
        print(f"Score: {score.reward}")
        scores.append(score.reward)
        if step == 50:
            break
        step += 1

    print(f"Average score: {sum(scores)/len(scores)}")
