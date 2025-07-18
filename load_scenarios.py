import json
from pathlib import Path
from typing import Iterator
from data_types import Scenario
from rich import print

def load_scenarios_from_disk(file_path: str) -> Iterator[Scenario]:
    """Load scenarios sequentially from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file containing scenarios
        
    Yields:
        Scenario objects parsed from each line in the file
    """
    path = Path(file_path)
    with path.open('r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                scenario_data = json.loads(line)
                yield Scenario(**scenario_data)

def main():
    """Load and print the first scenario from the training data."""
    scenarios = load_scenarios_from_disk("synthetic_data/train.jsonl")
    first_scenario = next(scenarios)
    print(first_scenario)

if __name__ == "__main__":
    main()
