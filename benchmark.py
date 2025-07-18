import asyncio
import json
from pathlib import Path

from agent import run_agent
from judge import judge_answer, JudgeAnswer
from rich import print

def load_data(file_path: str):
    """Load synthetic data from JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

async def test_flow(data_file: str, max_items: int = 40):
    """Test the flow: load data -> run agent -> judge result."""
    
    # Load data
    data = load_data(data_file)
    print(f"Loaded {len(data)} items")
    
    # Track correct answers
    correct_answers = 0
    total_questions = 0
    
    # Test on first few items
    for i, item in enumerate(data[:max_items]):
        print(f"\n--- Item {i+1} ---")
        print(f"Question: {item['question']}")
        print(f"Repo: {item['repo']}")
        print(f"Answer: {item['answer']}")
        print(f"Functions: {item['functions']} \n")
        
        total_questions += 1
        
        # Run agent
        try:
            agent_result = await run_agent(item['repo'], item['question'])
            print(f"Agent Result: {agent_result}")
            if agent_result:                
                # Judge the result
                judge_result = await judge_answer(
                    question=item['question'],
                    ref_answer=item['answer'], 
                    ref_functions=item['functions'],
                    answer=(
                        f"{agent_result.structured_answer.explanation} \n"
                        f"{agent_result.structured_answer.code_snippet} \n"
                        f"{agent_result.structured_answer.code_explanation} \n"
                    )
                )
                print(f"Judge Result: {judge_result}")
                
                if judge_result.is_correct:
                    correct_answers += 1
                    
            else:
                judge_result = JudgeAnswer(reasoning="Agent returned None", is_correct=False)
                print("Agent returned None")
                
        except Exception as e:
            print(f"Error: {e}")
    
    # Calculate and display probability
    if total_questions > 0:
        probability = correct_answers / total_questions
        print(f"\n--- Results ---")
        print(f"Correct answers: {correct_answers}")
        print(f"Total questions: {total_questions}")
        print(f"Probability of correct answer: {probability:.3f} ({probability * 100:.1f}%)")
    else:
        print("No questions processed")

if __name__ == "__main__":
    # Test with train data
    train_file = "synthetic_data/train.jsonl"
    if Path(train_file).exists():
        asyncio.run(test_flow(train_file, max_items=20))
    else:
        print(f"File {train_file} doesn't exist")
