from textwrap import dedent
from litellm import acompletion
from pydantic import BaseModel
import dotenv
from rich import print

dotenv.load_dotenv()

class JudgeAnswer(BaseModel):
    reasoning: str # Reasoning why answer is correct
    is_correct: bool # Whether the answer is correct

async def judge_answer(question: str, ref_answer: str, answer: str) -> JudgeAnswer:
    SYSTEM_PROMPT = dedent(f"""
        You are a judge that will be given a question, a reference answer, and a model answer.
        You will need to judge whether the model answer is correct or not.
        You will need to provide a reasoning for your answer.
    """)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": 
            f"Question: {question} \n"
            f"Reference Answer: {ref_answer} \n"
            f"Anser: {answer}"
        }
    ]
    
    resp = await acompletion(
        model="gpt-4.1",
        messages=messages,
        cache=True,
        response_format=JudgeAnswer
    )
    
    content = resp["choices"][0]["message"]["content"]  # type: ignore
    judge_answer = JudgeAnswer.model_validate_json(content)
    
    return judge_answer
    
if __name__ == "__main__":
    import asyncio
    res = asyncio.run(judge_answer("What is the capital of France?", "Paris", "Paris"))
    print(res)