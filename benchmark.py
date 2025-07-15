import asyncio
import json
from pathlib import Path

from agent import run_agent
from judge import judge_answer

def load_data(file_path: str):
    """Load synthetic data from JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

async def test_flow(data_file: str, max_items: int = 1):
    """Test the flow: load data -> run agent -> judge result."""
    
    # Load data
    data = load_data(data_file)
    print(f"Loaded {len(data)} items")
    
    # Test on first few items
    for i, item in enumerate(data[:max_items]):
        print(f"\n--- Item {i+1} ---")
        print(f"Question: {item['question']}")
        print(f"Repo: {item['repo']}")
        print(f"Answer: {item['answer']}")
        
        # Run agent
        try:
            agent_result = await run_agent(item['repo'], item['question'])
            if agent_result:
                print(f"Agent answer: {agent_result.structured_answer.explanation}")
                
                
                
                # Judge the result
                judge_result = await judge_answer(
                    question=item['question'],
                    ref_answer=item['answer'], 
                    answer=(
                        f"{agent_result.structured_answer.explanation} \n"
                        f"{agent_result.structured_answer.code_snippet} \n"
                        f"{agent_result.structured_answer.code_explanation}"
                    )
                )
                print(f"Judge: {judge_result}")
            else:
                print("Agent returned None")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Test with train data
    train_file = "synthetic_data/train.jsonl"
    if Path(train_file).exists():
        asyncio.run(test_flow(train_file))
    else:
        print(f"File {train_file} doesn't exist")
