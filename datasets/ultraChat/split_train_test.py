import random

if __name__ == "__main__":

    list_examples = []
    for i in [0, ]:
        list_examples.extend(
            open(f"datasets/ultraChat/train_{i}.jsonl", "r", encoding="utf-8").readlines()
        )

    random.shuffle(list_examples)

    with open(f"datasets/ultraChat/train.json", "w", encoding="utf-8") as f:
        for samp in list_examples[: 5000]:
            f.write(samp)
    with open(f"datasets/ultraChat/test.json", "w", encoding="utf-8") as f:
        for samp in list_examples[-500 : ]:
            f.write(samp)

