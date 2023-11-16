import json

import sys

from tqdm import tqdm

sys.path.append("./")

def multi_turn_to_single_turn(sample):
    id_ = sample["id"]
    data_ = sample["data"]

    num_turns = len(data_) // 2

    list_new_samples = []
    for n_turn in range(1, num_turns + 1):
        data_tmp = data_[: n_turn * 2]

        query = ""
        response = ""
        if n_turn > 1:
            for turn_idx in range(n_turn - 1):
                query_ = data_tmp[2 * turn_idx]
                response_ = data_tmp[2 * turn_idx + 1]
                input_1 = f"<|endoftext|>user:\n{query_}\n"
                input_2 = f"<|endoftext|>assistant:\n{response_}<|endoftext|>"
                query = query + input_1
                query = query + input_2

        query_ = data_tmp[2 * (n_turn - 1)]
        response_ = data_tmp[2 * (n_turn - 1) + 1]
        input_1 = f"<|endoftext|>user:\n{query_}\n<|endoftext|>assistant:\n"
        input_2 = f"{response_}<|endoftext|>"
        query = query + input_1
        response = input_2

        list_new_samples.append(
            {
                "query": query,
                "response": response,
                "id": id_ + "-" + str(n_turn)
            }
        )

    return list_new_samples


if __name__ == "__main__":

    list_examples = []
    for i in [4, 5]:
    # for i in [0]:
        with open(f"datasets/ultraChat/train_{i}.jsonl", "r", encoding="utf-8") as f:
            for line in tqdm(f):
                try:
                    line = json.loads(line.strip())
                except Exception as e:
                    print(e)
                    line = None

                if line is not None:
                    list_examples.append(line)

    list_samples_new = []
    for samp in list_examples:
        list_samples_new.extend(multi_turn_to_single_turn(samp))

    with open(f"datasets/ultraChat/flat_format/train.json", "w", encoding="utf-8") as f:
        for samp in list_samples_new[: - 5000]:
            f.write(json.dumps(samp, ensure_ascii=False) + "\n")

    with open(f"datasets/ultraChat/flat_format/test.json", "w", encoding="utf-8") as f:
        for samp in list_samples_new[- 5000: ]:
            f.write(json.dumps(samp, ensure_ascii=False) + "\n")

