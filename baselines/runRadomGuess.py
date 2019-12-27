import json
import numpy as np

def main():
    predictions = dict()
    with open("../data/dataset/record/dev.json", "r") as f:
        dev = json.load(f)
    data = dev["data"]
    for example in data:
        text = example["passage"]["text"]
        entities = example["passage"]["entities"]
        for queries in example["qas"]:
            choice = np.random.randint(len(entities))
            st = entities[choice]["start"]
            ed = entities[choice]["end"]
            predictions[queries["id"]] = text[st:ed+1]
    with open("../result/random_result.json", "w") as f:
        f.write(json.dumps(predictions, indent=4) + "\n")


if __name__ == "__main__":
    main()