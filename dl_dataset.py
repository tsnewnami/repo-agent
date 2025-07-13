from datasets import load_dataset
import os

language = "javascript"
# Load the langauge subset of CodeSearchNet
ds = load_dataset("code_search_net", language)

# Explore a few samples
print(ds['train'][0])

# Define local directory to save to
save_dir = f"data/code_search_net_{language}"

# Make sure directory exists
os.makedirs(save_dir, exist_ok=True)

# Save train, valid, and test splits
for split in ds:
    split_path = os.path.join(save_dir, f"{split}.jsonl")
    with open(split_path, "w", encoding="utf-8") as f:
        for example in ds[split]:
            f.write(str(example) + "\n")

print(f"Saved all splits to: {save_dir}")