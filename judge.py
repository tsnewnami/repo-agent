import logging
from textwrap import dedent
from litellm import acompletion
from pydantic import BaseModel, Field
import dotenv
from rich import print

dotenv.load_dotenv()

class JudgeAnswer(BaseModel):
    reasoning: str = Field(description="Reasoning why answer is correct")
    is_correct: bool = Field(description="Whether the answer is correct")

async def judge_answer(question: str, ref_answer: str, answer: str) -> JudgeAnswer:
    SYSTEM_PROMPT = dedent(
        """
        You will be given a question and two different answers to the question, the correct answer and the answer given by an AI. 
        
        Your job is to determine if the answer given by the AI is correct. 

        You return True the if the AI answer contains the relevant information from the correct answer. You should return False if the AI answer is missing information relevant to the question, or if it contradicts the correct answer.
        
        --------------------------------------------------------------------------------
        JSON response format (no additional keys):
        {
          "reasoning": "<concise explanation of your judgement>",
          "is_correct": <true | false>
        }
        """
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": 
            f"Question: {question} \n"  
            f"Correct answer: {ref_answer} \n"
            f"AI answer: {answer}"
        }
    ]
    
    try:
        resp = await acompletion(
            model="gpt-4.1",
            messages=messages,
            caching=True,
        )
    except Exception as e:
        logging.error(f"Error in acompletion call: {e}")
        logging.error(f"Error type: {type(e)}")
        return JudgeAnswer(reasoning=f"Error in acompletion: {e}", is_correct=False)
    
    try:
        content = resp["choices"][0]["message"]["content"]  # type: ignore
        
        # Try to parse JSON from the content
        import json
        json_data = json.loads(content)
        
        judge_answer = JudgeAnswer(
            reasoning=json_data.get("reasoning", "No reasoning provided"),
            is_correct=json_data.get("is_correct", False)
        )
        
    except Exception as e:
        logging.error(f"Error parsing response: {e}")
        logging.error(f"Content: {content if 'content' in locals() else 'No content'}")
        return JudgeAnswer(reasoning="Error parsing response", is_correct=False)
    
    return judge_answer
    
if __name__ == "__main__":
    import asyncio
    res = asyncio.run(judge_answer("What is the capital of France?", "Paris", "Paris"))
    print(res)