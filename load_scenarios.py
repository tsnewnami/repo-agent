import json
from pathlib import Path
from typing import Iterator, List, Literal
from data_types import Scenario
from datasets import load_dataset, Dataset
from rich import print

import dotenv
dotenv.load_dotenv()


def download_dataset(name: str, split: Literal["train", "test"], limit: int = 1000, shuffle: bool = False) -> Dataset:
    ds = load_dataset("JamesSED/synthetic_QA_code_search_net", split=split)
    if limit:
        ds = ds.select(range(limit))

    if shuffle:
        ds = ds.shuffle()

    return ds
   
def load_scenarios(name: str, split: Literal["train", "test"], limit: int = 1000, shuffle: bool = False) -> List[Scenario]:
    ds = download_dataset(name, split, limit, shuffle)
    return [Scenario(**row) for row in ds]



if __name__ == "__main__":
    print(load_scenarios("JamesSED/synthetic_QA_code_search_net", "train", limit=5, shuffle=True)[0])
