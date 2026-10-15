from datasets import load_dataset

# Load local images. Folder names become class labels.
dataset = load_dataset(
    "imagefolder",
    data_dir="animals"
)["train"]

print(dataset)
print(dataset.features)

# Create 80% training and 20% testing splits
splits = dataset.train_test_split(
    test_size=0.2,
    seed=42,
    stratify_by_column="label"
)

print(splits)
print("Training images:", len(splits["train"]))
print("Testing images:", len(splits["test"]))

# Upload to your own Hugging Face dataset repository
splits.push_to_hub("iamrafa/animals-split")