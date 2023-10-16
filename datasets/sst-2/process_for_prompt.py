import json
import random

if __name__ == "__main__":

    for mode in ["train", "dev", "test"]:

        list_samples = []
        with open(f"internal/instruct_aware_prompt_tuning/datasets/sst-2/{mode}.tsv", "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue

                line = line.split("\t")
                assert len(line) == 2
                label = line[1]
                sent = line[0]
                print("label: ", label)

                prompt = f"{sent}\nThe sentiment of the given sentence is:"
                target = "positive" if label == "1" else "negative"

                list_samples.append(
                    {
                        "instruction": prompt,
                        "response": target,
                    }
                )

        random.shuffle(list_samples)
        with open(f"internal/instruct_aware_prompt_tuning/datasets/sst-2/{mode}.json", "w", encoding="utf-8") as f:
            for samp in list_samples:
                f.write(
                    json.dumps(samp, ensure_ascii=False) + "\n"
                )