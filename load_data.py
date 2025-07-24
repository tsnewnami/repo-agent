from typing import List, Literal
from data_types import Scenario
from datasets import load_dataset, Dataset
from rich import print

import dotenv

dotenv.load_dotenv()


def download_dataset(
    name: str, split: Literal["train", "test"], limit: int = 1000, shuffle: bool = False
) -> Dataset:
    ds = load_dataset(name, split=split)
    if limit:
        ds = ds.select(range(limit))

    if shuffle:
        ds = ds.shuffle()

    return ds


def load_scenarios(
    name: str,
    split: Literal["train", "test"],
    limit: int = 1000,
    shuffle: bool = False,
    how_realistic_threshold: float = 0.9,
) -> List[Scenario]:
    ds = download_dataset(name, split, limit, shuffle)
    scenarios = [Scenario(**row) for row in ds]
    scenarios = [
        scenario
        for scenario in scenarios
        if scenario.how_realistic >= how_realistic_threshold
    ]
    return scenarios


if __name__ == "__main__":
    print(
        len(load_scenarios(
            "JamesSED/synthetic_QA_code_search_net", "train", limit=300, shuffle=True
        ))
    )
    print(
        load_scenarios(
            "JamesSED/synthetic_QA_code_search_net", "test", limit=5, shuffle=True
        )[0]
    )
